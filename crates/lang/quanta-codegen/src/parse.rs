//! WebIDL parsing front-end.
//!
//! Wraps `weedle` to extract just the pieces B′ cares about today:
//! enum declarations and their string members. Methods/dictionaries
//! get parsed too — we hold onto the AST — but the public surface
//! starts with enums because that's where the lockstep hazard lives.
//!
//! "Project-relevant" enums are those Quanta's existing `ffi.rs` /
//! `web/src/codes.ts` reference. The list is hard-coded below; B″
//! will derive it instead by walking the methods we actually call.

use crate::Result;
use std::collections::HashMap;
use weedle::Definition;
use weedle::argument::Argument;
use weedle::common::Identifier;
use weedle::interface::InterfaceMember;
use weedle::mixin::MixinMember;
use weedle::types::{
    FloatingPointType, IntegerType, NonAnyType, RecordKeyType, ReturnType, SingleType, Type,
    UnionMemberType,
};

/// The set of WebIDL interfaces whose method declarations Quanta
/// mirrors into the Lean spec. Methods on other interfaces are
/// ignored at parse time. Mirrors the call sites in
/// `web/src/webgpu.ts`; when a new interface is wired in, add it here.
pub const PROJECT_RELEVANT_INTERFACES: &[&str] = &[
    "GPU",
    "GPUAdapter",
    "GPUDevice",
    "GPUBuffer",
    "GPUTexture",
    "GPUQueue",
    "GPUCommandEncoder",
    "GPUComputePassEncoder",
    "GPURenderPassEncoder",
];

/// The set of WebIDL enum types Quanta currently uses.
/// Hard-coded list — when we wire a new descriptor, add the relevant
/// IDL enum here so `quanta codegen webgpu` picks it up. Future B″
/// will replace this with a transitive walk from method signatures.
pub const PROJECT_RELEVANT_ENUMS: &[&str] = &[
    "GPUTextureFormat",
    "GPUVertexFormat",
    "GPUPrimitiveTopology",
    "GPUCullMode",
    "GPUFrontFace",
    "GPUBlendFactor",
    "GPUBlendOperation",
    "GPUFilterMode",
    "GPUMipmapFilterMode",
    "GPUAddressMode",
    "GPUCompareFunction",
    "GPUVertexStepMode",
    "GPUIndexFormat",
    "GPULoadOp",
    "GPUStoreOp",
];

/// Subset of the parsed IDL we expose to emitters.
pub struct ParsedIdl {
    /// All enums in the IDL, in source order.
    pub enums: Vec<EnumDecl>,
    /// All operation (method) declarations on
    /// `PROJECT_RELEVANT_INTERFACES`, including the ones contributed by
    /// `partial interface` extensions. Partials are flattened into the
    /// owning interface's method list — Lean does not need to know
    /// which `partial` block a method came from.
    pub methods: Vec<MethodDecl>,
    /// SHA-256 of the IDL source — used for stamping the generated
    /// files so a divergence is loud.
    pub source_hash: String,
}

#[derive(Clone)]
pub struct EnumDecl {
    pub name: String,
    /// String values in source order. e.g. `["r8unorm", "rg8unorm", …]`
    pub values: Vec<String>,
    /// True if this enum is in `PROJECT_RELEVANT_ENUMS`.
    pub project_relevant: bool,
}

/// A WebIDL method declaration on an interface. Carries enough shape
/// for the conformance theorems landed so far:
/// - method-presence (T1711) uses `interface_name` + `method_name`.
/// - method-arity (T1712) uses `required_arity` + `max_arity` +
///   `is_variadic`.
/// - method-param-types (T1713) uses `params`.
#[derive(Clone)]
pub struct MethodDecl {
    pub interface_name: String,
    pub method_name: String,
    /// Number of non-optional, non-variadic parameters.
    pub required_arity: u32,
    /// Number of declared parameters total (optional + required, not
    /// counting variadic since variadic is unbounded). For a variadic
    /// method, this is the count of leading non-variadic params.
    pub max_arity: u32,
    /// True iff the last parameter is variadic (`type... ident`).
    pub is_variadic: bool,
    /// Per-parameter shape, in declaration order. Variadic tail
    /// parameters are not included; T1713 conformance ranges over
    /// the leading `max_arity` entries.
    pub params: Vec<ParamDecl>,
}

/// A WebIDL operation parameter, rendered to a canonical type name
/// string (see `canonicalize_type`). The name preserves typedefs
/// verbatim (so the spec's `GPUSize64` stays `GPUSize64`, not
/// resolved to `unsigned long long`) — typedef stability is the
/// guarantee Quanta-side claims rely on.
#[derive(Clone)]
pub struct ParamDecl {
    pub type_name: String,
    pub optional: bool,
}

pub struct Summary {
    pub enums_total: usize,
    pub enums_kept: usize,
    pub methods_kept: usize,
}

pub fn parse(idl_text: &str) -> Result<ParsedIdl> {
    let definitions = weedle::parse(idl_text).map_err(|e| {
        format!(
            "weedle failed to parse webgpu.idl: {:?} — file may have shifted shape since 2026-04-28",
            e
        )
    })?;

    let mut enums = Vec::new();
    let mut methods = Vec::new();
    // Mixin name → list of operations declared in that mixin
    // (across non-partial + partial blocks). Each operation is the
    // shape we'd otherwise build into a `MethodDecl`, minus the
    // owning interface name (resolved later via `includes` pairs).
    let mut mixin_methods: HashMap<String, Vec<MixinOp>> = HashMap::new();
    // List of `(includer_interface, mixin_name)` pairs from
    // `X includes Y;` statements. Resolved after the main pass so we
    // pick up mixins regardless of their textual order in the IDL.
    let mut includes: Vec<(String, String)> = Vec::new();

    for def in definitions {
        match def {
            Definition::Enum(e) => {
                let name = identifier_to_string(&e.identifier);
                // Path: EnumDefinition.values: Braced<PunctuatedNonEmpty<StringLit, ',' >>.
                // Braced.body is the punctuated list; .list yields the
                // `StringLit`s; `.0` is the &str (no surrounding quotes).
                let values: Vec<String> =
                    e.values.body.list.iter().map(|v| v.0.to_string()).collect();
                let project_relevant = PROJECT_RELEVANT_ENUMS.iter().any(|n| *n == name);
                enums.push(EnumDecl {
                    name,
                    values,
                    project_relevant,
                });
            }
            Definition::Interface(i) => {
                let iname = identifier_to_string(&i.identifier);
                if PROJECT_RELEVANT_INTERFACES.iter().any(|n| *n == iname) {
                    collect_interface_operations(&iname, &i.members.body, &mut methods);
                }
            }
            Definition::PartialInterface(i) => {
                // `partial interface GPUDevice { … };` — methods land on
                // the same interface as the non-partial declaration.
                let iname = identifier_to_string(&i.identifier);
                if PROJECT_RELEVANT_INTERFACES.iter().any(|n| *n == iname) {
                    collect_interface_operations(&iname, &i.members.body, &mut methods);
                }
            }
            Definition::InterfaceMixin(m) => {
                let mname = identifier_to_string(&m.identifier);
                let entry = mixin_methods.entry(mname).or_default();
                collect_mixin_operations(&m.members.body, entry);
            }
            Definition::PartialInterfaceMixin(m) => {
                let mname = identifier_to_string(&m.identifier);
                let entry = mixin_methods.entry(mname).or_default();
                collect_mixin_operations(&m.members.body, entry);
            }
            Definition::IncludesStatement(s) => {
                let lhs = identifier_to_string(&s.lhs_identifier);
                if PROJECT_RELEVANT_INTERFACES.iter().any(|n| *n == lhs) {
                    includes.push((lhs, identifier_to_string(&s.rhs_identifier)));
                }
            }
            _ => {}
        }
    }

    // Resolve mixin includes — flatten each mixin's operations onto its
    // includer. WebIDL mixin inclusion is one level deep (mixins do not
    // include other mixins), so a single pass suffices.
    for (lhs, rhs) in &includes {
        if let Some(ms) = mixin_methods.get(rhs) {
            for op in ms {
                methods.push(MethodDecl {
                    interface_name: lhs.clone(),
                    method_name: op.method_name.clone(),
                    required_arity: op.required_arity,
                    max_arity: op.max_arity,
                    is_variadic: op.is_variadic,
                    params: op.params.clone(),
                });
            }
        }
    }

    let source_hash = sha256_hex(idl_text);

    Ok(ParsedIdl {
        enums,
        methods,
        source_hash,
    })
}

/// One mixin operation, awaiting flattening onto an includer. Same
/// shape as `MethodDecl` minus the owning interface name (which
/// comes from the matching `X includes <mixin>` statement).
struct MixinOp {
    method_name: String,
    required_arity: u32,
    max_arity: u32,
    is_variadic: bool,
    params: Vec<ParamDecl>,
}

/// Walk an interface's members and append each `Operation` (method)
/// to `out`, tagged with its owning interface name.
fn collect_interface_operations(
    iname: &str,
    members: &[InterfaceMember<'_>],
    out: &mut Vec<MethodDecl>,
) {
    for m in members {
        if let InterfaceMember::Operation(op) = m {
            // Anonymous operations (special operations like getter/setter
            // without an identifier) carry no callable name; skip them.
            if let Some(id) = &op.identifier {
                let (req, max, var, params) = arg_shape(&op.args.body.list);
                out.push(MethodDecl {
                    interface_name: iname.to_string(),
                    method_name: identifier_to_string(id),
                    required_arity: req,
                    max_arity: max,
                    is_variadic: var,
                    params,
                });
            }
        }
    }
}

/// Walk a mixin's members and append each named `Operation` to `out`,
/// keyed by the mixin's own name. Caller flattens onto each
/// `X includes <mixin>` pair after the first pass.
fn collect_mixin_operations(members: &[MixinMember<'_>], out: &mut Vec<MixinOp>) {
    for m in members {
        if let MixinMember::Operation(op) = m
            && let Some(id) = &op.identifier
        {
            let (req, max, var, params) = arg_shape(&op.args.body.list);
            out.push(MixinOp {
                method_name: identifier_to_string(id),
                required_arity: req,
                max_arity: max,
                is_variadic: var,
                params,
            });
        }
    }
}

/// Compute `(required_arity, max_arity, is_variadic, params)` for an
/// argument list. WebIDL ordering rule: required arguments come
/// first, then optional, then at most one variadic at the end. The
/// returned `params` list excludes the variadic tail; that's gated
/// by `is_variadic` separately.
fn arg_shape(args: &[Argument<'_>]) -> (u32, u32, bool, Vec<ParamDecl>) {
    let mut required = 0u32;
    let mut max = 0u32;
    let mut variadic = false;
    let mut params = Vec::with_capacity(args.len());
    for a in args {
        match a {
            Argument::Single(s) => {
                let optional = s.optional.is_some();
                if !optional {
                    required += 1;
                }
                max += 1;
                params.push(ParamDecl {
                    type_name: canonicalize_type(&s.type_.type_),
                    optional,
                });
            }
            Argument::Variadic(_) => {
                variadic = true;
                // Variadic arg doesn't bump `max` — `max` is the
                // count of declarable named params, the "infinite"
                // tail is gated by `variadic` instead.
            }
        }
    }
    (required, max, variadic, params)
}

// ────────────────────────────────────────────────────────────────────
// WebIDL → canonical type-name string
// ────────────────────────────────────────────────────────────────────
//
// Used for T1713's parameter conformance discharge. Rules:
// - Typedefs are NOT resolved (so `GPUSize64` stays `"GPUSize64"`).
//   Quanta-side claims must use the same typedef name; the typedef-
//   stability guarantee that f64 is a faithful representation of
//   `GPUSize64` is the residual A11 trust.
// - Numeric/string primitives are rendered in their spec surface
//   form (`"long"`, `"unsigned long long"`, `"USVString"`, …).
// - Generics (`Promise<T>`, `sequence<T>`, `record<K,V>`, …) are
//   rendered recursively.
// - Nullable suffix `?` is appended to the inner rendering.

fn canonicalize_type(t: &Type<'_>) -> String {
    match t {
        Type::Single(s) => render_single(s),
        Type::Union(u) => {
            let inner = u
                .type_
                .body
                .list
                .iter()
                .map(render_union_member)
                .collect::<Vec<_>>()
                .join(" or ");
            let q = if u.q_mark.is_some() { "?" } else { "" };
            format!("({inner}){q}")
        }
    }
}

fn render_single(s: &SingleType<'_>) -> String {
    match s {
        SingleType::Any(_) => "any".to_string(),
        SingleType::NonAny(n) => render_non_any(n),
    }
}

fn render_union_member(m: &UnionMemberType<'_>) -> String {
    match m {
        UnionMemberType::Single(at) => render_non_any(&at.type_),
        UnionMemberType::Union(u) => {
            let inner = u
                .type_
                .body
                .list
                .iter()
                .map(render_union_member)
                .collect::<Vec<_>>()
                .join(" or ");
            let q = if u.q_mark.is_some() { "?" } else { "" };
            format!("({inner}){q}")
        }
    }
}

fn render_non_any(n: &NonAnyType<'_>) -> String {
    match n {
        NonAnyType::Promise(p) => {
            let inner = match &*p.generics.body {
                ReturnType::Undefined(_) => "undefined".to_string(),
                ReturnType::Type(t) => canonicalize_type(t),
            };
            format!("Promise<{inner}>")
        }
        NonAnyType::Integer(m) => with_q(render_integer(&m.type_), m.q_mark.is_some()),
        NonAnyType::FloatingPoint(m) => with_q(render_float(&m.type_), m.q_mark.is_some()),
        NonAnyType::Boolean(m) => with_q("boolean".to_string(), m.q_mark.is_some()),
        NonAnyType::Byte(m) => with_q("byte".to_string(), m.q_mark.is_some()),
        NonAnyType::Octet(m) => with_q("octet".to_string(), m.q_mark.is_some()),
        NonAnyType::ByteString(m) => with_q("ByteString".to_string(), m.q_mark.is_some()),
        NonAnyType::DOMString(m) => with_q("DOMString".to_string(), m.q_mark.is_some()),
        NonAnyType::USVString(m) => with_q("USVString".to_string(), m.q_mark.is_some()),
        NonAnyType::Sequence(m) => with_q(
            format!("sequence<{}>", canonicalize_type(&m.type_.generics.body)),
            m.q_mark.is_some(),
        ),
        NonAnyType::Object(m) => with_q("object".to_string(), m.q_mark.is_some()),
        NonAnyType::Symbol(m) => with_q("symbol".to_string(), m.q_mark.is_some()),
        NonAnyType::Error(m) => with_q("Error".to_string(), m.q_mark.is_some()),
        NonAnyType::ArrayBuffer(m) => with_q("ArrayBuffer".to_string(), m.q_mark.is_some()),
        NonAnyType::DataView(m) => with_q("DataView".to_string(), m.q_mark.is_some()),
        NonAnyType::Int8Array(m) => with_q("Int8Array".to_string(), m.q_mark.is_some()),
        NonAnyType::Int16Array(m) => with_q("Int16Array".to_string(), m.q_mark.is_some()),
        NonAnyType::Int32Array(m) => with_q("Int32Array".to_string(), m.q_mark.is_some()),
        NonAnyType::Uint8Array(m) => with_q("Uint8Array".to_string(), m.q_mark.is_some()),
        NonAnyType::Uint16Array(m) => with_q("Uint16Array".to_string(), m.q_mark.is_some()),
        NonAnyType::Uint32Array(m) => with_q("Uint32Array".to_string(), m.q_mark.is_some()),
        NonAnyType::Uint8ClampedArray(m) => {
            with_q("Uint8ClampedArray".to_string(), m.q_mark.is_some())
        }
        NonAnyType::Float32Array(m) => with_q("Float32Array".to_string(), m.q_mark.is_some()),
        NonAnyType::Float64Array(m) => with_q("Float64Array".to_string(), m.q_mark.is_some()),
        NonAnyType::ArrayBufferView(m) => with_q("ArrayBufferView".to_string(), m.q_mark.is_some()),
        NonAnyType::BufferSource(m) => with_q("BufferSource".to_string(), m.q_mark.is_some()),
        NonAnyType::FrozenArrayType(m) => with_q(
            format!("FrozenArray<{}>", canonicalize_type(&m.type_.generics.body)),
            m.q_mark.is_some(),
        ),
        NonAnyType::ObservableArrayType(m) => with_q(
            format!(
                "ObservableArray<{}>",
                canonicalize_type(&m.type_.generics.body)
            ),
            m.q_mark.is_some(),
        ),
        NonAnyType::RecordType(m) => {
            let (k, _comma, v) = &m.type_.generics.body;
            with_q(
                format!("record<{},{}>", render_record_key(k), canonicalize_type(v)),
                m.q_mark.is_some(),
            )
        }
        NonAnyType::Identifier(m) => with_q(identifier_to_string(&m.type_), m.q_mark.is_some()),
    }
}

fn render_record_key(k: &RecordKeyType<'_>) -> String {
    match k {
        RecordKeyType::Byte(_) => "ByteString".to_string(),
        RecordKeyType::DOM(_) => "DOMString".to_string(),
        RecordKeyType::USV(_) => "USVString".to_string(),
        RecordKeyType::NonAny(n) => render_non_any(n),
    }
}

fn render_integer(t: &IntegerType) -> String {
    match t {
        IntegerType::LongLong(ll) => {
            if ll.unsigned.is_some() {
                "unsigned long long".to_string()
            } else {
                "long long".to_string()
            }
        }
        IntegerType::Long(l) => {
            if l.unsigned.is_some() {
                "unsigned long".to_string()
            } else {
                "long".to_string()
            }
        }
        IntegerType::Short(s) => {
            if s.unsigned.is_some() {
                "unsigned short".to_string()
            } else {
                "short".to_string()
            }
        }
    }
}

fn render_float(t: &FloatingPointType) -> String {
    match t {
        FloatingPointType::Float(f) => {
            if f.unrestricted.is_some() {
                "unrestricted float".to_string()
            } else {
                "float".to_string()
            }
        }
        FloatingPointType::Double(d) => {
            if d.unrestricted.is_some() {
                "unrestricted double".to_string()
            } else {
                "double".to_string()
            }
        }
    }
}

fn with_q(s: String, nullable: bool) -> String {
    if nullable { format!("{s}?") } else { s }
}

pub fn summarize(parsed: &ParsedIdl) -> Summary {
    Summary {
        enums_total: parsed.enums.len(),
        enums_kept: parsed.enums.iter().filter(|e| e.project_relevant).count(),
        methods_kept: parsed.methods.len(),
    }
}

fn identifier_to_string(id: &Identifier<'_>) -> String {
    id.0.to_string()
}

/// Tiny pure-Rust SHA-256 (~80 LOC) so the codegen crate has zero
/// extra runtime deps. The output stamps generated files; if anyone
/// hand-edits the IDL or regenerates without updating headers, the
/// hash diff exposes the drift.
fn sha256_hex(input: &str) -> String {
    let bytes = input.as_bytes();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);

    // Padding: append 0x80, then zeros until length ≡ 56 (mod 64),
    // then 8-byte big-endian bit length.
    let mut padded = Vec::with_capacity(bytes.len() + 64);
    padded.extend_from_slice(bytes);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    // SHA-256 initial hash values (square roots of first 8 primes).
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // SHA-256 round constants (cube roots of first 64 primes).
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, b) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut s = String::with_capacity(64);
    for word in h.iter() {
        s.push_str(&format!("{word:08x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vector_empty() {
        assert_eq!(
            sha256_hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_known_vector_abc() {
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
