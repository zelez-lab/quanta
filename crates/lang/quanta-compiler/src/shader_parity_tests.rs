//! Cross-emitter shader parity corpus.
//!
//! Varyings follow the SHARED-STRUCT model (`ShaderDef::varyings`): one
//! user-declared struct is the explicit vertex↔fragment interface — the
//! vertex body ends in its struct literal (position field → gl_Position /
//! `[[position]]`, each varying stored to its named Output at the
//! field-declaration-order Location), and the fragment reads varyings as
//! `<receiver>.<field>` against the same interface (reading the position
//! field yields the interpolated window position, WGSL FragCoord
//! semantics). Fragment fixtures therefore carry `varyings: sv(..)` with
//! receiver `s`, vertex fixtures either `vv(..)` (tail literal) or `None`
//! (position-only `-> Vec4`); a fragment with a plain VALUE param is a
//! structural rejection on every emitter.
//!
//! The render-shader grammar lives in three hand-maintained emitters —
//! SPIR-V (`emit_spirv`), MSL syn-AST (`emit_msl`), and a lagging WGSL
//! string-replace stub (`quanta_ir::emit_wgsl`). Their divergence has
//! shipped breakage before (invalid MSL from formatted bodies, SPIR-V
//! passthrough fallbacks that misrender on Vulkan), yet no test drove the
//! same body through all three. This table is that safety net: one fixture
//! per grammar construct, an explicit per-emitter verdict, and tests that
//! assert the SPIR-V and MSL verdicts AGREE. A construct that one emitter
//! accepts and the other rejects is a bug the moment it appears here.
//!
//! Bodies are written in a TOKEN-SPACED wire form (`Vec4 :: new ( uv . x )`).
//! What the macro actually ships is `to_token_stream().to_string()`, which
//! under rustc's native token printer reads `Vec4 :: new(uv.x * 0.5, ...)` —
//! `::` spaced, the call paren attached, dots unspaced, trailing commas
//! preserved, and long lines wrapped (the wrap can split `Vec4 ::` and
//! `new(...)` across a newline). The fixture spacing here is token-equivalent
//! to that on both native emitters: the SPIR-V tokenizer splits on whitespace
//! and on `(`/`)`/`,`/`.` within words, and MSL re-parses with `syn`, which
//! is spacing-blind. What is NOT equivalent is the clean-source form
//! (`Vec4::new(...)`, no space around `::`): the SPIR-V tokenizer reads that
//! as one identifier and falls back to a passthrough. A few fixtures carry a
//! clean-source `alt` body to pin that asymmetry, and `ctor_trailing_comma` /
//! `ctor_wrapped_split` pin the two artifacts the printer + rustfmt produce
//! in real builds (a `1.0,)` trailing comma; a `Vec4 ::\nnew(` line split).
//! One deliberate exception: `sample(0` must stay CONTIGUOUS — both native
//! emitters find texture slots by scanning for the literal `sample(N`, which
//! is what the printer emits and what `rewrite_texture_names` normalizes
//! named textures to.
//!
//! Three mechanisms keep the corpus honest:
//!
//! - `KNOWN_DIVERGENCES` is an allowlist of `(fixture, reason)` for cases
//!   where SPIR-V and MSL genuinely disagree today. Every entry is a
//!   surfaced, documented bug — the list exists so the suite stays green
//!   while the divergence is known, not to hide new ones: an unlisted
//!   divergence fails the suite loudly, and an entry that heals also fails
//!   (so the list can't rot).
//!
//! - SPIR-V never errors at the top level: on a body it can't translate it
//!   prints a warning and emits a PASSTHROUGH module (load input[0] → vec4,
//!   or a constant). Real-vs-passthrough is therefore decided by OPCODE
//!   witness — an opcode a passthrough can never contain (a passthrough
//!   uses only OpLoad / OpCompositeExtract / OpCompositeConstruct plus the
//!   interface stores). `SpirvExpect::Real { witness }` means "emitted AND
//!   at least one witness opcode present".
//!
//! - WGSL now walks the body with its OWN hand-rolled tokenizer + recursive-
//!   descent walker (`quanta_ir::emit_wgsl::{shader_tokenizer, shader_walker}`),
//!   at construct parity with the SPIR-V walker. `&T` uniforms bind as
//!   `@group(0) @binding(N) var<uniform>` at their shared decl-index; `&[T]`
//!   slices bind as `var<storage, read>` arrays with a `name[u32(index)]`
//!   access; textures bind as a `texture_2d<f32>` + `sampler` pair at
//!   `8+2*slot` with `sample(N, uv)` → `textureSample(tex_N, smp_N, uv)`.
//!   Statement-`if`/`else` is native WGSL; a value-`if` (`let x = if a { b }
//!   else { c }`) lowers to a fresh `var` assigned in each arm; `let mut` →
//!   `var`. Every construct the SPIR-V walker rejects — a method call, a
//!   `while` loop, an unknown intrinsic, an `if`-expression without
//!   `else`, an out-of-range swizzle, indexing a non-slice — the WGSL walker
//!   `Reject`s with the same error substring MSL uses. So each fixture is
//!   `Translates` (with substrings the walker genuinely emits, all validated
//!   by `naga` in `wgsl_validates`) or `Reject` — no `KnownBroken` remains.
//!   Bounded `for i in A .. N` loops now translate on SPIR-V (a real
//!   structured loop with loop-carried OpPhi) and MSL (a native `for` over a
//!   `uint` counter); the WGSL walker still rejects them — a documented gap
//!   like `frag_coord()` and u32 params, so the loop fixtures carry a WGSL
//!   `Reject` verdict rather than a divergence entry.

use quanta_ir::{ShaderDef, ShaderParam, ShaderStage, ShaderType, ShaderVaryings, VaryingField};

/// How a fixture param binds. Mirrors the DSL param classes: a plain value
/// attribute (VERTEX-only under the shared-struct model), a `&T` uniform, or
/// a `&[T]` slice (storage-buffer array).
#[derive(Clone, Copy, PartialEq, Eq)]
enum ParamKind {
    Attr,
    Uniform,
    Slice,
}

/// A fixture's shared-struct varying interface (`ShaderDef::varyings`): the
/// struct name, the `#[position]` field, the varyings in declaration order
/// (field i = Location i), and — on the fragment side — the receiver param.
#[derive(Clone, Copy)]
struct VaryFix {
    struct_name: &'static str,
    position: &'static str,
    fields: &'static [(&'static str, ShaderType)],
    binding: Option<&'static str>,
}

/// The corpus's standard FRAGMENT interface: struct `V`, position field
/// `clip`, receiver `s` — bodies read `s . <field>`.
const fn sv(fields: &'static [(&'static str, ShaderType)]) -> Option<VaryFix> {
    Some(VaryFix {
        struct_name: "V",
        position: "clip",
        fields,
        binding: Some("s"),
    })
}

/// The VERTEX twin of [`sv`]: same struct, no receiver — the body ends in the
/// `V { clip : … , <field> : … }` literal.
const fn vv(fields: &'static [(&'static str, ShaderType)]) -> Option<VaryFix> {
    Some(VaryFix {
        struct_name: "V",
        position: "clip",
        fields,
        binding: None,
    })
}

// ─── SPIR-V opcode numbers (verified against emit_spirv/constants.rs) ────────

const OP_EXT_INST: u16 = 12;
const OP_CONVERT_F_TO_U: u16 = 109;
const OP_VECTOR_SHUFFLE: u16 = 79;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_FADD: u16 = 129;
const OP_FSUB: u16 = 131;
const OP_FMUL: u16 = 133;
const OP_FDIV: u16 = 136;
const OP_MATRIX_TIMES_VECTOR: u16 = 145;
const OP_F_NEGATE: u16 = 127;
const OP_IEQUAL: u16 = 170;
const OP_ULESS_THAN: u16 = 176;
const OP_FORD_EQUAL: u16 = 180;
const OP_FORD_GREATER_THAN: u16 = 186;
const OP_DPDX: u16 = 207;
const OP_FWIDTH: u16 = 209;
const OP_PHI: u16 = 245;
const OP_LOOP_MERGE: u16 = 246;

// ─── Fixture model ───────────────────────────────────────────────────────────

enum SpirvExpect {
    /// Emission succeeds and at least one witness opcode is present, proving
    /// the body genuinely translated (not a passthrough). An empty witness
    /// list waives the opcode check for bodies a passthrough is indistinct
    /// from (a constant color with no inputs) — Ok emission is the assertion.
    Real { witness: &'static [u16] },
    /// The body defeats the SPIR-V translator; the top level emits a
    /// passthrough module. No witness opcode may be present.
    Passthrough,
    /// Emission itself fails with an error containing the substring — a
    /// STRUCTURAL rejection (e.g. the combined-SSBO cap) that happens during
    /// interface setup, before body translation, so there is no module to
    /// passthrough. The build fails on it, mirroring the MSL `Reject`.
    Reject { err_contains: &'static str },
}

enum MslExpect {
    /// `emit_*_shader` returns Ok and every substring is present in the MSL.
    Accept { contains: &'static [&'static str] },
    /// `emit_*_shader` returns Err whose message contains the substring.
    Reject { err_contains: &'static str },
}

enum WgslExpect {
    /// The walker genuinely produces this WGSL — every substring must be
    /// present, and (when `naga` is on PATH) the whole module must validate.
    Translates { contains: &'static [&'static str] },
    /// `emit_*_shader` returns Err whose message contains the substring — a
    /// rejection of a construct the shader grammar does not accept (a method
    /// call, a loop, an unknown intrinsic, an else-less if-expression, an
    /// out-of-range swizzle, a non-slice index) OR the structural combined-SSBO
    /// cap. Mirrors the MSL and SPIR-V `Reject` — the same construct fails
    /// across all three emitters with the same error substring.
    Reject { err_contains: &'static str },
}

struct Fixture {
    name: &'static str,
    stage: ShaderStage,
    params: &'static [(&'static str, ShaderType, ParamKind)],
    /// The shared-struct varying interface, when the fixture has one:
    /// `sv(..)` on fragments (receiver `s`), `vv(..)` on vertices (tail
    /// struct literal), `None` for position-only vertices and input-free
    /// fragments.
    varyings: Option<VaryFix>,
    /// Token-spaced wire form — the bytes the emitters receive at runtime.
    body: &'static str,
    /// Clean-source twin (`Vec4::new(s.uv.x)`) for the string-replace-fiasco
    /// guard. Present on a representative subset; the equivalence test proves
    /// MSL treats both spacings alike and documents that SPIR-V does not.
    alt: Option<&'static str>,
    spirv: SpirvExpect,
    msl: MslExpect,
    wgsl: WgslExpect,
}

fn def(f: &Fixture) -> ShaderDef {
    ShaderDef {
        name: f.name.to_string(),
        stage: f.stage,
        params: f
            .params
            .iter()
            .map(|(name, ty, kind)| ShaderParam {
                name: name.to_string(),
                ty: *ty,
                is_uniform: *kind == ParamKind::Uniform,
                is_slice: *kind == ParamKind::Slice,
            })
            .collect(),
        return_type: ShaderType::Vec4,
        body_source: f.body.to_string(),
        varyings: f.varyings.map(|v| ShaderVaryings {
            struct_name: v.struct_name.to_string(),
            position: v.position.to_string(),
            fields: v
                .fields
                .iter()
                .map(|(name, ty)| VaryingField {
                    name: name.to_string(),
                    ty: *ty,
                })
                .collect(),
            binding: v.binding.map(str::to_string),
        }),
    }
}

/// A copy of `def` with the body swapped for its clean-source twin.
fn def_alt(f: &Fixture) -> Option<ShaderDef> {
    let mut d = def(f);
    d.body_source = f.alt?.to_string();
    Some(d)
}

// ─── Emitter drivers ─────────────────────────────────────────────────────────

fn emit_spirv(d: &ShaderDef) -> Result<Vec<u8>, String> {
    match d.stage {
        ShaderStage::Vertex => crate::emit_spirv::emit_vertex(d),
        ShaderStage::Fragment => crate::emit_spirv::emit_fragment(d),
    }
}

fn emit_msl(d: &ShaderDef) -> Result<String, String> {
    match d.stage {
        ShaderStage::Vertex => crate::emit_msl::emit_vertex_shader(d),
        ShaderStage::Fragment => crate::emit_msl::emit_fragment_shader(d),
    }
}

fn emit_wgsl(d: &ShaderDef) -> Result<String, String> {
    match d.stage {
        ShaderStage::Vertex => quanta_ir::emit_wgsl::emit_vertex_shader(d),
        ShaderStage::Fragment => quanta_ir::emit_wgsl::emit_fragment_shader(d),
    }
}

/// Opcodes present in a SPIR-V binary (5-word header skipped). Mirrors the
/// decoder in `emit_spirv/swizzle_tests.rs`.
fn opcodes(spirv: &[u8]) -> Vec<u16> {
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut ops = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        ops.push((words[i] & 0xFFFF) as u16);
        i += word_count.max(1);
    }
    ops
}

/// Opcodes a passthrough module can NEVER contain — any real body work. A
/// module carrying one of these translated genuinely; a module carrying none
/// is either a passthrough or a body whose only ops (constant construct /
/// single-component extract) coincide with a passthrough's.
///
/// `OpAccessChain` is deliberately EXCLUDED: the uniform storage-block plumbing
/// emits one per uniform param during setup, BEFORE the body is evaluated, and
/// that setup survives into the passthrough — so a uniform-bearing passthrough
/// carries `OpAccessChain` without translating anything. Uniform fixtures pick
/// a body-specific witness (arithmetic / intrinsic) instead.
const REAL_WORK_OPCODES: &[u16] = &[
    OP_EXT_INST,
    OP_CONVERT_F_TO_U,
    OP_VECTOR_SHUFFLE,
    OP_IMAGE_SAMPLE_IMPLICIT_LOD,
    OP_F_NEGATE,
    OP_FADD,
    OP_FSUB,
    OP_FMUL,
    OP_FDIV,
    OP_MATRIX_TIMES_VECTOR,
    OP_IEQUAL,
    OP_ULESS_THAN,
    OP_FORD_GREATER_THAN,
    OP_DPDX,
    OP_FWIDTH,
    OP_PHI,
];

fn has_real_work(spirv: &[u8]) -> bool {
    let ops = opcodes(spirv);
    REAL_WORK_OPCODES.iter().any(|w| ops.contains(w))
}

/// Whether a fixture's body genuinely translated on SPIR-V (vs. fell to a
/// passthrough). Witnessed-`Real` fixtures require their witness opcode;
/// witness-waived `Real` fixtures (constant/extract-only bodies indistinct
/// from a passthrough) require only Ok emission; `Passthrough` fixtures are
/// "translated" iff the module unexpectedly carries real-work ops.
fn spirv_translated(f: &Fixture, d: &ShaderDef) -> bool {
    match (&f.spirv, emit_spirv(d)) {
        (_, Err(_)) => false,
        (SpirvExpect::Real { witness }, Ok(spirv)) => {
            if witness.is_empty() {
                true
            } else {
                let ops = opcodes(&spirv);
                witness.iter().any(|w| ops.contains(w))
            }
        }
        (SpirvExpect::Passthrough, Ok(spirv)) => has_real_work(&spirv),
        // A `Reject` fixture that unexpectedly emitted Ok did NOT reject; it
        // counts as translated iff the module carries real work.
        (SpirvExpect::Reject { .. }, Ok(spirv)) => has_real_work(&spirv),
    }
}

// ─── Divergence allowlist (surfaced bugs only — see module docs) ─────────────

// Empty: the SPIR-V shader grammar now accepts every construct the corpus
// exercises that MSL does. The last entry — an assignment to an outer local
// inside an expression-`if` branch (`expr_if_assigns_outer_local`) — is closed:
// the expression-`if` merge now phis mutated outer locals exactly like the
// statement-level `if`, so SPIR-V honors the write (OP_PHI) and its fixture
// asserts full SPIR-V/MSL agreement. The list is kept so a future divergence
// has a home and the heal-detection stays wired.
const KNOWN_DIVERGENCES: &[(&str, &str)] = &[];

// ─── The fixture table ───────────────────────────────────────────────────────

use ShaderStage::{Fragment, Vertex};
use ShaderType::{F32, Mat4, U32, Vec2, Vec3, Vec4};

fn fixtures() -> Vec<Fixture> {
    vec![
        // ── CORE EXPRESSIONS ──────────────────────────────────────────────
        Fixture {
            name: "lit_solid",
            stage: Fragment,
            params: &[],
            varyings: None,
            body: "{ Vec4 :: new ( 1.0 , 0.0 , 0.0 , 1.0 ) }",
            alt: Some("{ Vec4::new(1.0, 0.0, 0.0, 1.0) }"),
            // A constant color with no inputs is byte-shaped like a
            // passthrough constant module, so no opcode distinguishes real
            // from passthrough here. Witness is waived: Ok emission is the
            // whole assertion. (The draw D-tests cover runtime correctness.)
            spirv: SpirvExpect::Real { witness: &[] },
            msl: MslExpect::Accept {
                contains: &["float4(1.0"],
            },
            wgsl: WgslExpect::Translates {
                contains: &["vec4<f32>(1.0f, 0.0f, 0.0f, 1.0f)"],
            },
        },
        Fixture {
            name: "arith_mix",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ Vec4 :: new ( s . uv . x * 0.5 + 0.25 , s . uv . y / 2.0 - 0.1 , 1.0 - s . uv . x , 1.0 ) }",
            alt: Some("{ Vec4::new(s.uv.x * 0.5 + 0.25, s.uv.y / 2.0 - 0.1, 1.0 - s.uv.x, 1.0) }"),
            spirv: SpirvExpect::Real {
                witness: &[OP_FMUL, OP_FADD, OP_FSUB, OP_FDIV],
            },
            msl: MslExpect::Accept {
                contains: &["V s [[stage_in]]", "s.uv.x * 0.5"],
            },
            wgsl: WgslExpect::Translates {
                contains: &["s.uv.x * 0.5f + 0.25f"],
            },
        },
        Fixture {
            name: "parens_precedence",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ Vec4 :: new ( ( s . uv . x + s . uv . y ) * 0.5 , s . uv . x + s . uv . y * 0.5 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FMUL, OP_FADD],
            },
            msl: MslExpect::Accept {
                contains: &["float4("],
            },
            wgsl: WgslExpect::Translates {
                contains: &["(s.uv.x + s.uv.y) * 0.5f"],
            },
        },
        Fixture {
            name: "unary_minus",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ Vec4 :: new ( - s . uv . x + 1.0 , abs ( - 0.5 ) , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_F_NEGATE],
            },
            msl: MslExpect::Accept {
                contains: &["abs("],
            },
            wgsl: WgslExpect::Translates {
                contains: &["-s.uv.x + 1.0f", "abs(-0.5f)"],
            },
        },
        Fixture {
            name: "expr_if",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let c = if s . uv . x > 0.5 { 1.0 } else { 0.0 } ; Vec4 :: new ( c , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real { witness: &[OP_PHI] },
            msl: MslExpect::Accept {
                contains: &["if (s.uv.x > 0.5)", "} else {"],
            },
            // The value-`if` lowers to a fresh `var` assigned in each arm of a
            // WGSL statement-`if`, with the var name bound to `c`.
            wgsl: WgslExpect::Translates {
                contains: &["if (s.uv.x > 0.5f) {", "let c = _wif0;"],
            },
        },
        Fixture {
            name: "expr_if_nested",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: BAND_EXPR_BODY,
            alt: None,
            spirv: SpirvExpect::Real { witness: &[OP_PHI] },
            msl: MslExpect::Accept {
                contains: &["if (s.uv.x < 0.25)"],
            },
            // Nested value-`if`s lower to nested statement-`if`s with distinct
            // `_wifN` result vars (the shared counter keeps them unique).
            wgsl: WgslExpect::Translates {
                contains: &["if (s.uv.x < 0.25f) {", "let c = _wif0;"],
            },
        },
        Fixture {
            name: "cmp_all",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let a = if s . uv . x > 0.5 { 1.0 } else { 0.0 } ; \
                   let b = if s . uv . y < 0.5 { 1.0 } else { 0.0 } ; \
                   let c = if s . uv . x >= s . uv . y { 1.0 } else { 0.0 } ; \
                   let d = if s . uv . y <= s . uv . x { 0.5 } else { 0.0 } ; \
                   Vec4 :: new ( a , b , c * d , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FORD_GREATER_THAN],
            },
            msl: MslExpect::Accept {
                contains: &[
                    "s.uv.x > 0.5",
                    "s.uv.y < 0.5",
                    "s.uv.x >= s.uv.y",
                    "s.uv.y <= s.uv.x",
                ],
            },
            // Each comparison maps straight through to WGSL (same spelling);
            // each value-`if` binds a distinct `_wifN` result var.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "s.uv.x > 0.5f",
                    "s.uv.y < 0.5f",
                    "s.uv.x >= s.uv.y",
                    "s.uv.y <= s.uv.x",
                    "let a = _wif0;",
                ],
            },
        },
        // ── STATEMENTS ────────────────────────────────────────────────────
        Fixture {
            name: "let_chain",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let a = s . uv . x ; let b = a * 2.0 ; let c = b - a ; Vec4 :: new ( c , a , b , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FMUL, OP_FSUB],
            },
            msl: MslExpect::Accept {
                contains: &["auto a = s.uv.x;", "auto b = a * 2.0;"],
            },
            // Each `let` stays a WGSL `let`; the chain emits as separate stmts.
            wgsl: WgslExpect::Translates {
                contains: &["let a = s.uv.x;", "let b = a * 2.0f;", "let c = b - a;"],
            },
        },
        Fixture {
            name: "let_mut_reassign",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let mut v = s . uv . x ; v = v * 2.0 ; v = v + 0.1 ; Vec4 :: new ( v , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FMUL, OP_FADD],
            },
            msl: MslExpect::Accept {
                contains: &["v = v * 2.0;", "v = v + 0.1;"],
            },
            // `let mut` → `var`; reassignments are separate WGSL statements.
            wgsl: WgslExpect::Translates {
                contains: &["var v = s.uv.x;", "v = v * 2.0f;", "v = v + 0.1f;"],
            },
        },
        Fixture {
            name: "stmt_if_assign",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let mut c = 0.0 ; if s . uv . x > 0.5 { c = 1.0 ; } else { c = 0.25 ; } \
                   Vec4 :: new ( c , 0.0 , 0.0 , 1.0 ) }",
            alt: Some(
                "{ let mut c = 0.0; if s.uv.x > 0.5 { c = 1.0; } else { c = 0.25; } \
                 Vec4::new(c, 0.0, 0.0, 1.0) }",
            ),
            spirv: SpirvExpect::Real { witness: &[OP_PHI] },
            msl: MslExpect::Accept {
                contains: &["if (s.uv.x > 0.5)", "c = 1.0;", "c = 0.25;"],
            },
            // The statement-`if` reassigns the `var c` in each arm (WGSL native
            // control flow); the final ctor is the return value.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "var c = 0.0f;",
                    "if (s.uv.x > 0.5f) {",
                    "c = 1.0f;",
                    "c = 0.25f;",
                ],
            },
        },
        Fixture {
            name: "stmt_if_nested",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: BAND_STMT_BODY,
            alt: None,
            spirv: SpirvExpect::Real { witness: &[OP_PHI] },
            msl: MslExpect::Accept {
                contains: &["if (s.uv.x < 0.25)"],
            },
            // Nested statement-`if`s reassign the outer `var c` in each arm.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "if (s.uv.x < 0.25f) {",
                    "c = vec4<f32>(1.0f, 0.0f, 0.0f, 1.0f);",
                ],
            },
        },
        Fixture {
            name: "stmt_if_then_expr",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            // The image-fragment shape: a statement-if reassigns, then a
            // final tail expression is the return value (TailMode::Route).
            body: "{ let mut g = 0.25 ; if s . uv . x > 0.5 { g = 0.75 ; } else { g = 0.1 ; } \
                   Vec4 :: new ( s . uv . x , g , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real { witness: &[OP_PHI] },
            msl: MslExpect::Accept {
                contains: &["if (s.uv.x > 0.5)", "return float4(s.uv.x, g, 0.0, 1.0);"],
            },
            // Statement-`if` reassigns `var g`; the tail ctor is the return.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "var g = 0.25f;",
                    "if (s.uv.x > 0.5f) {",
                    "return vec4<f32>(s.uv.x, g, 0.0f, 1.0f);",
                ],
            },
        },
        // ── UNIFORMS + DEREF ──────────────────────────────────────────────
        Fixture {
            name: "uniform_deref_star",
            stage: Vertex,
            params: &[
                ("pos", Vec3, ParamKind::Attr),
                ("uv", Vec2, ParamKind::Attr),
                ("shift", Vec2, ParamKind::Uniform),
            ],
            // Position-only vertex: the attrs are pure inputs (nothing is
            // auto-forwarded under the shared-struct model).
            varyings: None,
            body: "{ Vec4 :: new ( pos . x + ( * shift ) . x , pos . y + shift . y , 0.0 , 1.0 ) }",
            alt: Some("{ Vec4::new(pos.x + (*shift).x, pos.y + shift.y, 0.0, 1.0) }"),
            // Witnessed by the `+` (OpFAdd); OpAccessChain is emitted by the
            // uniform-block setup even in a passthrough, so it cannot witness.
            spirv: SpirvExpect::Real {
                witness: &[OP_FADD],
            },
            msl: MslExpect::Accept {
                contains: &["shift"],
            },
            // `shift` binds as a `var<uniform>` at its shared slot; the deref
            // `(*shift).x` is a value no-op → `(shift).x`.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "@group(0) @binding(0) var<uniform> shift: vec2<f32>;",
                    "pos.x + (shift).x",
                ],
            },
        },
        Fixture {
            name: "uniform_frag_tint",
            stage: Fragment,
            params: &[("tint", Vec4, ParamKind::Uniform)],
            // A declared-but-unread varying: the interface is emitted (both
            // ends would match a `uv`-writing vertex) while the body reads
            // only the uniform.
            varyings: sv(&[("uv", Vec2)]),
            body: "{ Vec4 :: new ( tint . x , tint . y , tint . z , 1.0 ) }",
            alt: None,
            // Reading `tint.x` is OpLoad + OpCompositeExtract — the passthrough
            // shape — and the uniform-setup OpAccessChain is present either way,
            // so no opcode distinguishes real from passthrough. Witness waived.
            spirv: SpirvExpect::Real { witness: &[] },
            msl: MslExpect::Accept {
                contains: &["float4(tint.x, tint.y, tint.z, 1.0)"],
            },
            // `tint` binds as a `var<uniform>` at binding 0; the body reads it.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "@group(0) @binding(0) var<uniform> tint: vec4<f32>;",
                    "vec4<f32>(tint.x, tint.y, tint.z, 1.0f)",
                ],
            },
        },
        Fixture {
            name: "uniform_mat4_mul",
            stage: Vertex,
            params: &[
                ("pos", Vec3, ParamKind::Attr),
                ("uv", Vec2, ParamKind::Attr),
                ("mvp", Mat4, ParamKind::Uniform),
            ],
            varyings: None,
            body: "{ mvp * Vec4 :: new ( pos . x , pos . y , pos . z , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_MATRIX_TIMES_VECTOR],
            },
            msl: MslExpect::Accept {
                contains: &["mvp * float4(pos.x, pos.y, pos.z, 1.0)"],
            },
            // `mvp` binds as a `mat4x4<f32>` uniform; WGSL `mat * vec` is native.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "@group(0) @binding(0) var<uniform> mvp: mat4x4<f32>;",
                    "mvp * vec4<f32>(pos.x, pos.y, pos.z, 1.0f)",
                ],
            },
        },
        Fixture {
            name: "uniform_two_frag",
            stage: Fragment,
            params: &[
                ("base", Vec4, ParamKind::Uniform),
                ("gain", Vec4, ParamKind::Uniform),
            ],
            varyings: sv(&[("uv", Vec2)]),
            // A non-commutative combiner (0.75*base + 0.25*gain) so binding
            // order is observable downstream.
            body: "{ Vec4 :: new ( mix ( base . x , gain . x , 0.25 ) , mix ( base . y , gain . y , 0.25 ) , \
                   mix ( base . z , gain . z , 0.25 ) , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_EXT_INST],
            },
            msl: MslExpect::Accept {
                contains: &["mix(base.x, gain.x, 0.25)"],
            },
            // Two uniforms bind at consecutive shared slots (0, 1) in decl
            // order — the non-commutative combiner keeps that order observable.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "@group(0) @binding(0) var<uniform> base: vec4<f32>;",
                    "@group(0) @binding(1) var<uniform> gain: vec4<f32>;",
                    "mix(base.x, gain.x, 0.25f)",
                ],
            },
        },
        Fixture {
            name: "uniform_mixed_order",
            stage: Fragment,
            // Two uniforms with two varyings read between them: proves
            // uniform binding is the index among UNIFORMS (varyings live in
            // their own Location space) on all emitters.
            params: &[
                ("u1", Vec2, ParamKind::Uniform),
                ("u2", Vec4, ParamKind::Uniform),
            ],
            varyings: sv(&[("a", Vec2), ("b", Vec2)]),
            body: "{ Vec4 :: new ( s . a . x + u1 . x , s . b . y + u2 . y , u2 . z , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FADD],
            },
            msl: MslExpect::Accept {
                contains: &["u1", "u2"],
            },
            // Binding is the index among UNIFORMS+slices: u1→binding 0,
            // u2→binding 1; the varyings a/b take Locations 0/1 instead.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "@group(0) @binding(0) var<uniform> u1: vec2<f32>;",
                    "@group(0) @binding(1) var<uniform> u2: vec4<f32>;",
                    "@location(0) a: vec2<f32>,",
                    "@location(1) b: vec2<f32>,",
                    "s.a.x + u1.x",
                ],
            },
        },
        // ── SWIZZLE + CTOR ────────────────────────────────────────────────
        Fixture {
            name: "swizzle_multi",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("v", Vec4)]),
            body: "{ let t = s . v . zw ; Vec4 :: new ( t . x , t . y , s . v . xy . x , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_VECTOR_SHUFFLE],
            },
            msl: MslExpect::Accept {
                contains: &["auto t = s.v.zw;"],
            },
            // WGSL swizzles use the same syntax; `.zw`/`.xy.x` map straight
            // through, validated against the source arity.
            wgsl: WgslExpect::Translates {
                contains: &["let t = s.v.zw;", "vec4<f32>(t.x, t.y, s.v.xy.x, 1.0f)"],
            },
        },
        Fixture {
            name: "swizzle_on_ctor",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ Vec4 :: new ( Vec2 :: new ( s . uv . y , s . uv . x ) . x , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            // `.x` on a Vec2 constructor is a single OpCompositeExtract — the
            // same op a passthrough uses — and the body has no arithmetic,
            // intrinsic, or multi-component swizzle. Nothing distinguishes
            // real from passthrough by opcode, so the witness is waived (Ok
            // emission is the assertion, like lit_solid).
            spirv: SpirvExpect::Real { witness: &[] },
            msl: MslExpect::Accept {
                contains: &["float2(s.uv.y, s.uv.x).x"],
            },
            // The inner ctor lowers to `vec2<f32>(...)` and the `.x` swizzle
            // applies to the ctor result — the postfix-swizzle path.
            wgsl: WgslExpect::Translates {
                contains: &["vec2<f32>(s.uv.y, s.uv.x).x"],
            },
        },
        Fixture {
            name: "ctor_nested",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ Vec4 :: new ( mix ( s . uv . x , s . uv . y , 0.5 ) , min ( s . uv . x , s . uv . y ) , \
                   max ( s . uv . x , s . uv . y ) , 1.0 ) }",
            alt: Some(
                "{ Vec4::new(mix(s.uv.x, s.uv.y, 0.5), min(s.uv.x, s.uv.y), max(s.uv.x, s.uv.y), 1.0) }",
            ),
            spirv: SpirvExpect::Real {
                witness: &[OP_EXT_INST],
            },
            msl: MslExpect::Accept {
                contains: &[
                    "mix(s.uv.x, s.uv.y, 0.5)",
                    "min(s.uv.x, s.uv.y)",
                    "max(s.uv.x, s.uv.y)",
                ],
            },
            wgsl: WgslExpect::Translates {
                contains: &[
                    "mix(s.uv.x, s.uv.y, 0.5f)",
                    "min(s.uv.x, s.uv.y)",
                    "max(s.uv.x, s.uv.y)",
                ],
            },
        },
        Fixture {
            name: "ctor_trailing_comma",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            // The exact artifact a rustfmt-wrapped multi-line call ships: the
            // token printer preserves the trailing comma (`1.0,)`).
            body: "{ Vec4 :: new(s . uv . x * 0.5 , 0.25 , 0.0 , 1.0 ,) }",
            alt: None,
            // The argument parser tolerates the trailing comma before `)`, so
            // the body translates: `s.uv.x * 0.5` is the OpFMul witness.
            spirv: SpirvExpect::Real {
                witness: &[OP_FMUL],
            },
            msl: MslExpect::Accept {
                contains: &["float4(s.uv.x * 0.5, 0.25, 0.0, 1.0)"],
            },
            // The walker's arg parser consumes the trailing comma (rustfmt's
            // wrapped-call artifact), so the WGSL ctor is clean.
            wgsl: WgslExpect::Translates {
                contains: &["vec4<f32>(s.uv.x * 0.5f, 0.25f, 0.0f, 1.0f)"],
            },
        },
        Fixture {
            name: "ctor_wrapped_split",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            // The token printer wraps long lines and can break between
            // `Vec4 ::` and `new(...)`. Both native emitters must treat the
            // newline as ordinary whitespace.
            body: "{ Vec4 ::\nnew(s . uv . x * 0.5 , 0.0 , 0.0 , 1.0) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FMUL],
            },
            msl: MslExpect::Accept {
                contains: &["float4(s.uv.x * 0.5, 0.0, 0.0, 1.0)"],
            },
            // The tokenizer flattens the wrap newline to whitespace, so the
            // split `Vec4 ::` / `new(` reassembles into one `vec4<f32>(` ctor.
            wgsl: WgslExpect::Translates {
                contains: &["vec4<f32>(s.uv.x * 0.5f, 0.0f, 0.0f, 1.0f)"],
            },
        },
        // ── INTRINSICS ────────────────────────────────────────────────────
        Fixture {
            name: "intrinsics_core",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let a = sqrt ( s . uv . x + 0.1 ) ; let b = pow ( s . uv . y , 2.0 ) ; \
                   let c = floor ( s . uv . x * 4.0 ) + fract ( s . uv . y ) ; \
                   let d = clamp ( s . uv . x , 0.0 , 1.0 ) * abs ( s . uv . y - 0.5 ) ; \
                   Vec4 :: new ( a , b , c * 0.25 + d , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_EXT_INST],
            },
            msl: MslExpect::Accept {
                contains: &["sqrt(", "pow(", "floor(", "fract(", "clamp(", "abs("],
            },
            // Core intrinsics keep their WGSL names.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "sqrt(s.uv.x + 0.1f)",
                    "pow(s.uv.y, 2.0f)",
                    "floor(",
                    "fract(",
                    "clamp(",
                    "abs(",
                ],
            },
        },
        Fixture {
            name: "intrinsics_geo",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let d = dot ( s . uv , s . uv ) ; let l = length ( s . uv ) ; \
                   let n = normalize ( Vec2 :: new ( s . uv . x + 0.001 , s . uv . y ) ) . x ; \
                   Vec4 :: new ( d , l , n , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_EXT_INST],
            },
            msl: MslExpect::Accept {
                contains: &["dot(", "length(", "normalize("],
            },
            // dot/length/normalize keep their WGSL names (dot/length scalar).
            wgsl: WgslExpect::Translates {
                contains: &[
                    "dot(s.uv, s.uv)",
                    "length(s.uv)",
                    "normalize(vec2<f32>(s.uv.x + 0.001f, s.uv.y)).x",
                ],
            },
        },
        Fixture {
            name: "intrinsics_map",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let r = inverse_sqrt ( s . uv . x + 1.0 ) ; Vec4 :: new ( r , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_EXT_INST],
            },
            // inverse_sqrt aliases to MSL rsqrt.
            msl: MslExpect::Accept {
                contains: &["rsqrt("],
            },
            // inverse_sqrt aliases to the WGSL builtin `inverseSqrt`.
            wgsl: WgslExpect::Translates {
                contains: &["inverseSqrt(s.uv.x + 1.0f)"],
            },
        },
        Fixture {
            name: "deriv_frag",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let w = fwidth ( s . uv . x ) ; \
                   Vec4 :: new ( w * 100.0 , dpdx ( s . uv . y ) * 100.0 + 0.5 , dpdy ( s . uv . x ) * 100.0 + 0.5 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FWIDTH, OP_DPDX],
            },
            // dpdx/dpdy alias to MSL dfdx/dfdy; fwidth stays fwidth.
            msl: MslExpect::Accept {
                contains: &["fwidth(", "dfdx(", "dfdy("],
            },
            // WGSL keeps fwidth/dpdx/dpdy (the derivatives' native names).
            wgsl: WgslExpect::Translates {
                contains: &["fwidth(s.uv.x)", "dpdx(s.uv.y)", "dpdy(s.uv.x)"],
            },
        },
        Fixture {
            name: "smoothstep_band",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: CIRCLE_SDF_BODY,
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_EXT_INST, OP_FWIDTH],
            },
            msl: MslExpect::Accept {
                contains: &["smoothstep(", "fwidth(", "length("],
            },
            // smoothstep keeps its WGSL name; the negated `-w` bound and the
            // fwidth-scaled band lower faithfully.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "smoothstep(-w, w, d)",
                    "fwidth(d)",
                    "length(vec2<f32>(dx, dy))",
                ],
            },
        },
        // ── SAMPLING ──────────────────────────────────────────────────────
        Fixture {
            name: "sample_basic",
            stage: Fragment,
            // `sample(0` stays contiguous: both native emitters detect the
            // texture slot by the literal substring `sample(N` (naive scan,
            // not tokenized), which is the wire form the macro ships.
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ sample(0 , s . uv ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_IMAGE_SAMPLE_IMPLICIT_LOD],
            },
            msl: MslExpect::Accept {
                contains: &["tex_0.sample("],
            },
            // `sample(0, s.uv)` → `textureSample(tex_0, smp_0, s.uv)`; the
            // texture and sampler bind at 8/9 (past the uniform/slice space).
            wgsl: WgslExpect::Translates {
                contains: &[
                    "@group(0) @binding(8) var tex_0: texture_2d<f32>;",
                    "@group(0) @binding(9) var smp_0: sampler;",
                    "textureSample(tex_0, smp_0, s.uv)",
                ],
            },
        },
        Fixture {
            name: "sample_swizzle",
            stage: Fragment,
            params: &[("tint", Vec4, ParamKind::Uniform)],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let c = sample(0 , s . uv ) . x ; \
                   Vec4 :: new ( c * tint . x , c * tint . y , c * tint . z , 1.0 ) }",
            alt: Some(
                "{ let c = sample(0, s.uv).x; \
                 Vec4::new(c * tint.x, c * tint.y, c * tint.z, 1.0) }",
            ),
            spirv: SpirvExpect::Real {
                witness: &[OP_IMAGE_SAMPLE_IMPLICIT_LOD],
            },
            msl: MslExpect::Accept {
                contains: &["tex_0.sample(", ".x"],
            },
            // The sampled `.x` applies to the `textureSample` result; the tint
            // uniform binds at 0, the texture/sampler at 8/9.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "@group(0) @binding(0) var<uniform> tint: vec4<f32>;",
                    "let c = textureSample(tex_0, smp_0, s.uv).x;",
                ],
            },
        },
        Fixture {
            name: "sample_in_ctor",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ Vec4 :: new ( sample(0 , s . uv ) . x , sample(0 , s . uv ) . y , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_IMAGE_SAMPLE_IMPLICIT_LOD],
            },
            msl: MslExpect::Accept {
                contains: &["tex_0.sample("],
            },
            // Two samples of the same slot each lower and take a swizzle.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "textureSample(tex_0, smp_0, s.uv).x",
                    "textureSample(tex_0, smp_0, s.uv).y",
                ],
            },
        },
        // ── BUILTINS: frag_coord() ────────────────────────────────────────
        // The fragment window-space position: SPIR-V declares an Input vec4
        // decorated `BuiltIn FragCoord` (enum 15) and loads it per call; MSL
        // declares a `float4 _frag_coord [[position]]` parameter the walker
        // lowers the call to. x,y are pixel coordinates (origin upper-left on
        // both backends — OriginUpperLeft is the fragment execution mode),
        // z is depth, w is 1/w. The `/ 64.0` witnesses via OpFDiv — a
        // passthrough never divides. WGSL: the walker does not know the
        // builtin yet and rejects it as an unknown function (documented gap;
        // reading the Varyings POSITION FIELD `s.<position>` is the
        // supported WGSL spelling — see `varyings_position_read_frag`).
        Fixture {
            name: "frag_coord_scaled",
            stage: Fragment,
            params: &[],
            varyings: None,
            body: "{ Vec4 :: new ( frag_coord ( ) . x / 64.0 , frag_coord ( ) . y / 64.0 , 0.0 , 1.0 ) }",
            alt: Some("{ Vec4::new(frag_coord().x / 64.0, frag_coord().y / 64.0, 0.0, 1.0) }"),
            spirv: SpirvExpect::Real {
                witness: &[OP_FDIV],
            },
            msl: MslExpect::Accept {
                contains: &[
                    "float4 _frag_coord [[position]]",
                    "_frag_coord.x / 64.0",
                    "_frag_coord.y / 64.0",
                ],
            },
            wgsl: WgslExpect::Reject {
                err_contains: "frag_coord",
            },
        },
        // frag_coord() in a VERTEX body is a stage error: MSL rejects
        // structurally (the walker would otherwise emit an undeclared
        // `_frag_coord` identifier); SPIR-V errors in the body parser and
        // falls to the passthrough. Both agree on not-accepted, like the
        // other rejection rows.
        Fixture {
            name: "rej_frag_coord_in_vertex",
            stage: Vertex,
            params: &[
                ("pos", Vec3, ParamKind::Attr),
                ("uv", Vec2, ParamKind::Attr),
            ],
            varyings: None,
            body: "{ Vec4 :: new ( frag_coord ( ) . x , pos . y , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "frag_coord",
            },
            // The WGSL walker rejects the call as an unknown function — the
            // same verdict, via the unknown-intrinsic path.
            wgsl: WgslExpect::Reject {
                err_contains: "frag_coord",
            },
        },
        // ── BUILTINS: vertex_id() / instance_id() ─────────────────────────
        // The vertex-stage index builtins, both u32. SPIR-V declares u32
        // Inputs decorated `BuiltIn VertexIndex` (42) / `BuiltIn
        // InstanceIndex` (43) — the Vulkan pair, NOT OpenGL's
        // VertexId(5)/InstanceId(6) — and loads them per call; MSL declares
        // `uint _vertex_id [[vertex_id]]` / `uint _instance_id
        // [[instance_id]]` parameters the walker lowers the calls to. The
        // body is the buffer-free fullscreen triangle (vid 0/1/2 →
        // (-1,-1)/(3,-1)/(-1,3)) — the shape that deletes dija's 6-vertex
        // unit-quad buffer — plus an instance-scaled z so `instance_id()`
        // flows into the result. OpIEqual witnesses the unsigned compare;
        // `instance_id ( ) * 0.25` rides the u32→f32 widening. WGSL: the
        // walker does not know either builtin and rejects them as unknown
        // functions (documented gap; `@builtin(vertex_index)` /
        // `@builtin(instance_index)` are the analogues when they land). The
        // `vertex_index_spirv_builtin_wiring` test below pins the exact
        // decoration contract by decoding the module.
        Fixture {
            name: "vertex_index_fullscreen_tri",
            stage: Vertex,
            params: &[],
            varyings: None,
            body: "{ let x = if vertex_id ( ) == 1u32 { 3.0 } else { - 1.0 } ; \
                   let y = if vertex_id ( ) == 2u32 { 3.0 } else { - 1.0 } ; \
                   let z = instance_id ( ) * 0.25 ; \
                   Vec4 :: new ( x , y , z , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_IEQUAL],
            },
            msl: MslExpect::Accept {
                contains: &[
                    "uint _vertex_id [[vertex_id]]",
                    "uint _instance_id [[instance_id]]",
                    "_vertex_id == 1u",
                    "_vertex_id == 2u",
                    "_instance_id * 0.25",
                ],
            },
            wgsl: WgslExpect::Reject {
                err_contains: "vertex_id",
            },
        },
        // vertex_id() in a FRAGMENT body is a stage error — the polarity
        // flip of `rej_frag_coord_in_vertex`: MSL rejects structurally (the
        // walker would otherwise emit an undeclared `_vertex_id`
        // identifier); SPIR-V errors in the body parser and falls to the
        // passthrough. Both agree on not-accepted.
        Fixture {
            name: "rej_vertex_id_in_fragment",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ Vec4 :: new ( vertex_id ( ) * 0.1 , s . uv . x , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "vertex_id",
            },
            wgsl: WgslExpect::Reject {
                err_contains: "vertex_id",
            },
        },
        // instance_id() gets its own fragment-rejection row so its guard is
        // pinned independently of vertex_id's (the MSL check is per-builtin).
        Fixture {
            name: "rej_instance_id_in_fragment",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ Vec4 :: new ( instance_id ( ) * 0.1 , s . uv . y , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "instance_id",
            },
            wgsl: WgslExpect::Reject {
                err_contains: "instance_id",
            },
        },
        // ── THE SHARED-STRUCT VARYING INTERFACE (flagship pair) ──────────
        // The canonical new-model pair: ONE Varyings struct (`Surface`,
        // position field `clip`, a float varying `uv`, a flat u32 varying
        // `kind`), written by the vertex through the tail struct literal and
        // consumed by the fragment through the receiver param. Locations are
        // field-declaration order (uv→0, kind→1) on every backend; `kind` is
        // Flat/[[flat]] on BOTH interface ends. The
        // `varyings_interface_wiring_spirv` test below decodes both modules
        // and pins Location numbers, Flat decorations, and BuiltIn Position.
        // WGSL: u32 varyings are a documented gap (same as u32 params), so
        // this pair Rejects there; the float-only pair below covers the WGSL
        // face of the model.
        Fixture {
            name: "varyings_struct_vert",
            stage: Vertex,
            params: &[
                ("pos", Vec2, ParamKind::Attr),
                ("in_uv", Vec2, ParamKind::Attr),
                ("in_kind", U32, ParamKind::Attr),
            ],
            varyings: Some(VaryFix {
                struct_name: "Surface",
                position: "clip",
                fields: &[("uv", Vec2), ("kind", U32)],
                binding: None,
            }),
            body: "{ Surface { clip : Vec4 :: new ( pos . x , pos . y , 0.0 , 1.0 ) , \
                   uv : in_uv , kind : in_kind } }",
            alt: None,
            // Loads + a ctor + interface stores are byte-shaped like a
            // passthrough; the decode test below pins the real wiring.
            spirv: SpirvExpect::Real { witness: &[] },
            msl: MslExpect::Accept {
                contains: &[
                    "struct Surface {",
                    "float4 clip [[position]];",
                    "float2 uv [[user(loc0)]];",
                    "uint kind [[user(loc1)]] [[flat]];",
                    "out.clip = float4(pos.x, pos.y, 0.0, 1.0);",
                    "out.uv = in_uv;",
                    "out.kind = in_kind;",
                ],
            },
            wgsl: WgslExpect::Reject {
                err_contains: "u32 shader params are not yet supported",
            },
        },
        Fixture {
            name: "varyings_struct_frag",
            stage: Fragment,
            params: &[],
            varyings: Some(VaryFix {
                struct_name: "Surface",
                position: "clip",
                fields: &[("uv", Vec2), ("kind", U32)],
                binding: Some("s"),
            }),
            body: "{ let c = if s . kind == 1u32 { 1.0 } else { 0.2 } ; \
                   Vec4 :: new ( s . uv . x , s . uv . y , c , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_IEQUAL],
            },
            msl: MslExpect::Accept {
                contains: &[
                    "Surface s [[stage_in]]",
                    "uint kind [[user(loc1)]] [[flat]];",
                    "s.kind == 1u",
                    "s.uv.x",
                ],
            },
            wgsl: WgslExpect::Reject {
                err_contains: "u32 shader params are not yet supported",
            },
        },
        // The float-only pair: proves the WGSL face of the shared-struct
        // model end to end (@builtin(position) member, @location per field,
        // the receiver param, the tail literal as member assignments).
        Fixture {
            name: "varyings_float_pair_vert",
            stage: Vertex,
            params: &[("pos", Vec2, ParamKind::Attr)],
            varyings: vv(&[("uv", Vec2), ("fade", F32)]),
            body: "{ V { clip : Vec4 :: new ( pos . x , pos . y , 0.0 , 1.0 ) , \
                   uv : pos , fade : 0.5 } }",
            alt: None,
            spirv: SpirvExpect::Real { witness: &[] },
            msl: MslExpect::Accept {
                contains: &[
                    "struct V {",
                    "float4 clip [[position]];",
                    "float2 uv [[user(loc0)]];",
                    "float fade [[user(loc1)]];",
                    "out.uv = pos;",
                    "out.fade = 0.5;",
                ],
            },
            wgsl: WgslExpect::Translates {
                contains: &[
                    "struct V {",
                    "@builtin(position) clip: vec4<f32>,",
                    "@location(0) uv: vec2<f32>,",
                    "@location(1) fade: f32,",
                    "var _vout: V;",
                    "_vout.clip = vec4<f32>(pos.x, pos.y, 0.0f, 1.0f);",
                    "_vout.uv = pos;",
                    "_vout.fade = 0.5f;",
                    "return _vout;",
                ],
            },
        },
        Fixture {
            name: "varyings_float_pair_frag",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2), ("fade", F32)]),
            body: "{ Vec4 :: new ( s . uv . x , s . uv . y , s . fade , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real { witness: &[] },
            msl: MslExpect::Accept {
                contains: &["V s [[stage_in]]", "s.uv.x", "s.fade"],
            },
            wgsl: WgslExpect::Translates {
                contains: &[
                    "@builtin(position) clip: vec4<f32>,",
                    "@location(0) uv: vec2<f32>,",
                    "@location(1) fade: f32,",
                    "fn main(s: V) -> @location(0) vec4<f32> {",
                    "vec4<f32>(s.uv.x, s.uv.y, s.fade, 1.0f)",
                ],
            },
        },
        // Field-init shorthand + literal field order ≠ declaration order:
        // Rust allows both; Locations stay declaration-ordered regardless.
        Fixture {
            name: "varyings_shorthand_any_order_vert",
            stage: Vertex,
            params: &[("pos", Vec2, ParamKind::Attr)],
            varyings: vv(&[("uv", Vec2), ("fade", F32)]),
            body: "{ let uv = Vec2 :: new ( pos . x , pos . y ) ; \
                   V { fade : 0.25 , uv , clip : Vec4 :: new ( pos . x , pos . y , 0.0 , 1.0 ) } }",
            alt: None,
            spirv: SpirvExpect::Real { witness: &[] },
            msl: MslExpect::Accept {
                contains: &["out.clip = ", "out.uv = uv;", "out.fade = 0.25;"],
            },
            wgsl: WgslExpect::Translates {
                contains: &["_vout.uv = uv;", "_vout.fade = 0.25f;"],
            },
        },
        // Reading the POSITION FIELD in the fragment: `s.clip` is the
        // interpolated window position — WGSL semantics (equivalent to
        // frag_coord()). SPIR-V loads a `BuiltIn FragCoord` Input; MSL/WGSL
        // read the `[[position]]` / `@builtin(position)` struct member.
        Fixture {
            name: "varyings_position_read_frag",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ Vec4 :: new ( s . clip . x / 64.0 , s . uv . y , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FDIV],
            },
            msl: MslExpect::Accept {
                contains: &["float4 clip [[position]];", "s.clip.x / 64.0"],
            },
            wgsl: WgslExpect::Translates {
                contains: &["s.clip.x / 64.0f"],
            },
        },
        // ── NEW-MODEL REJECTIONS ─────────────────────────────────────────
        // A fragment with a plain VALUE param has no meaning under the
        // shared-struct model — every emitter rejects it structurally with
        // the same wording (the SPIR-V side errors during interface setup,
        // like the SSBO cap, so there is no module to passthrough).
        Fixture {
            name: "rej_value_param_frag",
            stage: Fragment,
            params: &[("uv", Vec2, ParamKind::Attr)],
            varyings: None,
            body: "{ Vec4 :: new ( uv . x , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Reject {
                err_contains: "Varyings",
            },
            msl: MslExpect::Reject {
                err_contains: "Varyings",
            },
            wgsl: WgslExpect::Reject {
                err_contains: "Varyings",
            },
        },
        // A varyings vertex whose tail is NOT the struct literal: the
        // interface must be built exactly once, in tail position.
        Fixture {
            name: "rej_vert_missing_struct_tail",
            stage: Vertex,
            params: &[("pos", Vec2, ParamKind::Attr)],
            varyings: vv(&[("uv", Vec2)]),
            body: "{ Vec4 :: new ( pos . x , pos . y , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "struct literal",
            },
            wgsl: WgslExpect::Reject {
                err_contains: "struct literal",
            },
        },
        // A literal that misses a declared field: every field is stored
        // explicitly, so an absent one is an error, not an implicit zero.
        Fixture {
            name: "rej_vert_missing_field",
            stage: Vertex,
            params: &[("pos", Vec2, ParamKind::Attr)],
            varyings: vv(&[("uv", Vec2)]),
            body: "{ V { clip : Vec4 :: new ( pos . x , pos . y , 0.0 , 1.0 ) } }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "missing field `uv`",
            },
            wgsl: WgslExpect::Reject {
                err_contains: "missing field `uv`",
            },
        },
        // Reading a field the struct does not declare.
        Fixture {
            name: "rej_frag_unknown_varying_field",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ Vec4 :: new ( s . bogus , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "no field `bogus`",
            },
            wgsl: WgslExpect::Reject {
                err_contains: "no field `bogus`",
            },
        },
        // ── U32 SCALARS (integer attribute → flat varying → comparison) ──
        // The dija `shape_type` shape under the shared-struct model: a u32
        // vertex attribute written into a u32 varying FIELD by the tail
        // literal, branched on in the fragment with an integer compare —
        // replacing the f32-smuggled `> 0.5` hack. SPIR-V: the varying is
        // `Flat` on BOTH the vertex Output and the fragment Input (a non-Flat
        // int interpolant is invalid), the vertex-attribute Input is NOT Flat,
        // and comparisons use the UNSIGNED opcode family (OpIEqual /
        // OpULessThan), never the FOrd float ops. MSL: `uint` members carry
        // `[[flat]]` in the shared struct, and literals spell as `Nu`. WGSL:
        // documented gap — u32 fields reject with a named error (flat
        // interpolation + u32 literals not wired in the walker). The
        // `u32_flat_wiring_spirv` test below pins the decorations by
        // decoding the modules.
        Fixture {
            name: "u32_flat_varying_vert",
            stage: Vertex,
            params: &[
                ("pos", Vec2, ParamKind::Attr),
                ("in_uv", Vec2, ParamKind::Attr),
                ("in_shape", U32, ParamKind::Attr),
            ],
            varyings: vv(&[("uv", Vec2), ("shape_type", U32)]),
            body: "{ V { clip : Vec4 :: new ( pos . x , pos . y , 0.0 , 1.0 ) , \
                   uv : in_uv , shape_type : in_shape } }",
            alt: None,
            // Loads + a constant ctor are byte-shaped like a passthrough, so
            // the witness is waived — Ok emission plus the decoration test
            // below are the assertions.
            spirv: SpirvExpect::Real { witness: &[] },
            msl: MslExpect::Accept {
                contains: &[
                    "uint in_shape [[attribute(2)]];",
                    "float2 uv [[user(loc0)]];",
                    "uint shape_type [[user(loc1)]] [[flat]];",
                    "out.shape_type = in_shape;",
                ],
            },
            wgsl: WgslExpect::Reject {
                err_contains: "u32 shader params are not yet supported",
            },
        },
        // Fragment consuming the flat u32 varying with `== 3u32` — the
        // explicit-suffix literal form. OpIEqual witnesses the integer
        // compare; the phi carries the branch.
        Fixture {
            name: "u32_eq_suffixed_literal_frag",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2), ("shape_type", U32)]),
            body: "{ let c = if s . shape_type == 3u32 { 1.0 } else { 0.2 } ; \
                   Vec4 :: new ( c , s . uv . x , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_IEQUAL],
            },
            msl: MslExpect::Accept {
                contains: &[
                    "uint shape_type [[user(loc1)]] [[flat]];",
                    "s.shape_type == 3u",
                ],
            },
            wgsl: WgslExpect::Reject {
                err_contains: "u32 shader params are not yet supported",
            },
        },
        // The BARE-integer-literal twin (`kind < 2`): the tokenizer types a
        // bare `2` as f32, so the comparison coerces it with OpConvertFToU
        // and still compares UNSIGNED (OpULessThan). MSL spells the literal
        // `2u` so its comparison is integer too.
        Fixture {
            name: "u32_cmp_bare_literal_frag",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2), ("kind", U32)]),
            body: "{ let c = if s . kind < 2 { 0.9 } else { 0.1 } ; \
                   Vec4 :: new ( c , c , c , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_ULESS_THAN],
            },
            msl: MslExpect::Accept {
                contains: &["uint kind [[user(loc1)]] [[flat]];", "s.kind < 2u"],
            },
            wgsl: WgslExpect::Reject {
                err_contains: "u32 shader params are not yet supported",
            },
        },
        // ── SLICE PARAMS (`&[T]` storage-buffer arrays) ───────────────────
        // A literal index into a `&[Vec4]` slice. SPIR-V lowers `stops[0]` to
        // OpConvertFToU (index → uint) + OpAccessChain[member0, idx] + OpLoad —
        // the ConvertFToU witnesses genuine slice access (a passthrough never
        // converts f→u). MSL declares `const device float4* stops [[buffer(0)]]`
        // and indexes `stops[(uint)(0.0)]`. WGSL binds a read-only runtime array
        // at the same slot and indexes `stops[u32(0)]` — full parity.
        Fixture {
            name: "slice_index_literal",
            stage: Fragment,
            params: &[("stops", Vec4, ParamKind::Slice)],
            varyings: None,
            body: "{ Vec4 :: new ( stops [ 0 ] . x , stops [ 0 ] . y , stops [ 0 ] . z , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_CONVERT_F_TO_U],
            },
            msl: MslExpect::Accept {
                contains: &[
                    "const device float4* stops [[buffer(0)]]",
                    "stops[(uint)(0.0)]",
                ],
            },
            wgsl: WgslExpect::Translates {
                contains: &[
                    "@group(0) @binding(0) var<storage, read> stops: array<vec4<f32>>;",
                    "stops[u32(0.0f)]",
                ],
            },
        },
        // A computed index (from uv arithmetic). The `s.uv.x * 4.0` is OpFMul;
        // the index still truncates through OpConvertFToU. MSL:
        // `stops[(uint)(i)]`; WGSL: `stops[u32(i)]` (the same truncation).
        Fixture {
            name: "slice_index_computed",
            stage: Fragment,
            params: &[("stops", Vec4, ParamKind::Slice)],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let i = s . uv . x * 4.0 ; Vec4 :: new ( stops [ i ] . x , stops [ i ] . y , stops [ i ] . z , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FMUL, OP_CONVERT_F_TO_U],
            },
            msl: MslExpect::Accept {
                contains: &["stops[(uint)(i)]"],
            },
            wgsl: WgslExpect::Translates {
                contains: &[
                    "@group(0) @binding(0) var<storage, read> stops: array<vec4<f32>>;",
                    "stops[u32(i)]",
                ],
            },
        },
        // A slice DECLARED between two uniforms proves uniform and slice params
        // share ONE decl-index binding space: u_a→buffer(0), stops→buffer(1),
        // u_b→buffer(2). The MSL `[[buffer(N)]]` substrings pin the bindings;
        // the SPIR-V OpFAdd (u_a.x + stops[0].x) + OpConvertFToU witness the
        // translation (opcodes can't encode a binding number, exactly as
        // `uniform_mixed_order` proves uniform order via substrings + OpFAdd).
        //
        // WGSL binds the slice at the SHARED index (binding 1, past the u_a
        // uniform's index 0) with a `u32`-indexed access, AND the two `&T`
        // uniforms bind as `var<uniform>` at bindings 0 and 2 — so the
        // shared decl-index space is proven across all three: u_a→0, stops→1,
        // u_b→2, byte-identical to the MSL `[[buffer(N)]]` slots.
        Fixture {
            name: "slice_mixed_decl_order",
            stage: Fragment,
            params: &[
                ("u_a", Vec4, ParamKind::Uniform),
                ("stops", Vec4, ParamKind::Slice),
                ("u_b", Vec4, ParamKind::Uniform),
            ],
            varyings: None,
            body: "{ Vec4 :: new ( u_a . x + stops [ 0 ] . x , u_b . y , stops [ 1 ] . z , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FADD, OP_CONVERT_F_TO_U],
            },
            msl: MslExpect::Accept {
                contains: &[
                    "constant float4& u_a [[buffer(0)]]",
                    "const device float4* stops [[buffer(1)]]",
                    "constant float4& u_b [[buffer(2)]]",
                ],
            },
            // Slice at shared index 1 (`u32`-rewritten access), uniforms at 0
            // and 2 — the full shared decl-index space is now emitted.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "@group(0) @binding(0) var<uniform> u_a: vec4<f32>;",
                    "@group(0) @binding(1) var<storage, read> stops: array<vec4<f32>>;",
                    "@group(0) @binding(2) var<uniform> u_b: vec4<f32>;",
                    "u_a.x + stops[u32(0.0f)].x",
                ],
            },
        },
        // ── REJECTIONS: SLICE + CAP ───────────────────────────────────────
        // Indexing a NON-slice value (a Vec2 local). SPIR-V rejects it
        // (→ passthrough); MSL rejects with a "non-slice" error. The two
        // agree on rejection.
        Fixture {
            name: "rej_index_non_slice",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let t = s . uv ; Vec4 :: new ( t [ 0 ] . x , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "non-slice",
            },
            // The walker rejects indexing a non-slice value — same as MSL.
            wgsl: WgslExpect::Reject {
                err_contains: "non-slice",
            },
        },
        // 9 combined uniform + slice params overrun the cap of 8 (texture
        // bindings start at 8). ALL THREE emitters fail with a structural error
        // that names the count and the cap — SPIR-V errors during interface
        // setup (no module to passthrough), MSL and WGSL likewise before body
        // lowering, with byte-identical "cap of 8" wording.
        Fixture {
            name: "rej_ssbo_cap",
            stage: Fragment,
            params: &[
                ("u0", Vec4, ParamKind::Uniform),
                ("u1", Vec4, ParamKind::Uniform),
                ("u2", Vec4, ParamKind::Uniform),
                ("u3", Vec4, ParamKind::Uniform),
                ("s4", Vec4, ParamKind::Slice),
                ("u5", Vec4, ParamKind::Uniform),
                ("u6", Vec4, ParamKind::Uniform),
                ("u7", Vec4, ParamKind::Uniform),
                ("s8", Vec4, ParamKind::Slice),
            ],
            varyings: None,
            body: "{ Vec4 :: new ( u0 . x , u1 . y , s4 [ 0 ] . z , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Reject {
                err_contains: "cap of 8",
            },
            msl: MslExpect::Reject {
                err_contains: "cap of 8",
            },
            wgsl: WgslExpect::Reject {
                err_contains: "cap of 8",
            },
        },
        // ── SPACING-ROBUST TEXTURE SCAN (A4) ──────────────────────────────
        // The spaced twin of `sample_basic`: `sample ( 0 , s . uv )`. Both
        // native emitters must find the slot despite whitespace between
        // `sample`, `(`, and the digit — a non-macro producer or a printer
        // change could space them apart. (syn is spacing-blind on the MSL
        // side; the SPIR-V scan and the MSL scan/rewrite are the sites made
        // whitespace-tolerant.)
        Fixture {
            name: "sample_spaced_scan",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ sample ( 0 , s . uv ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_IMAGE_SAMPLE_IMPLICIT_LOD],
            },
            msl: MslExpect::Accept {
                contains: &["tex_0.sample("],
            },
            // The walker's `sample(N` scan and the `sample(...)` parse both
            // tolerate whitespace, so the spaced twin binds and lowers too.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "@group(0) @binding(8) var tex_0: texture_2d<f32>;",
                    "textureSample(tex_0, smp_0, s.uv)",
                ],
            },
        },
        // ── REAL DIJA BODIES (the shared-struct pairs) ────────────────────
        Fixture {
            name: "dija_rect_frag",
            stage: Fragment,
            params: &[],
            varyings: sv(&[
                ("corner", Vec2),
                ("size", Vec2),
                ("color", Vec4),
                ("border_color", Vec4),
                ("corner_radii", Vec4),
                ("border_width", F32),
                ("shape_type", F32),
            ]),
            body: DIJA_RECT_FRAG_BODY,
            alt: None,
            // The real rounded-rect fragment. Its `let dist = if s.shape_type >
            // 0.5 { let nx = …; … } else { … }` drives the branch-local-`let`
            // path: the expression-`if` branches run through the statement
            // walker, so the body translates on SPIR-V. OpPhi witnesses the
            // if-merge; the ExtInst witnesses the smoothstep/length/min/mix
            // intrinsics — neither can appear in a passthrough.
            spirv: SpirvExpect::Real {
                witness: &[OP_PHI, OP_EXT_INST],
            },
            msl: MslExpect::Accept {
                contains: &["smoothstep(", "length(float2("],
            },
            // The real rounded-rect fragment: nested value-`if`s (each a
            // `_wifN` result var), branch-local `let`s, and smoothstep/length/
            // min/mix intrinsics all lower to valid WGSL.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "smoothstep(",
                    "length(vec2<f32>(nx, ny))",
                    "let dist = _wif0;",
                ],
            },
        },
        Fixture {
            name: "dija_glyph_frag",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("corner", Vec2), ("uv_rect", Vec4), ("color", Vec4)]),
            body: DIJA_GLYPH_FRAG_BODY,
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_IMAGE_SAMPLE_IMPLICIT_LOD],
            },
            msl: MslExpect::Accept {
                contains: &["tex_0.sample(", "mix("],
            },
            // The glyph fragment: mix over the uv-rect, a coverage sample, and
            // a color multiply — all straight-through WGSL.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "mix(s.uv_rect.x, s.uv_rect.z, s.corner.x)",
                    "let coverage = textureSample(tex_0, smp_0, vec2<f32>(u, v)).x;",
                ],
            },
        },
        Fixture {
            name: "dija_image_frag",
            stage: Fragment,
            params: &[],
            varyings: sv(&[
                ("corner", Vec2),
                ("size", Vec2),
                ("uv_rect", Vec4),
                ("opacity", F32),
                ("border_radius", F32),
            ]),
            body: DIJA_IMAGE_FRAG_BODY,
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_IMAGE_SAMPLE_IMPLICIT_LOD, OP_PHI],
            },
            msl: MslExpect::Accept {
                contains: &["tex_0.sample(", "if (s.border_radius > 0.0) {"],
            },
            // The image fragment ends in a statement-`if`/`else` whose branches
            // both yield a color (TailMode::Route) — lowered to a `_wifN` result
            // var returned as the body value.
            wgsl: WgslExpect::Translates {
                contains: &["if (s.border_radius > 0.0f) {", "return _wif0;"],
            },
        },
        // The rect VERTEX under the shared-struct model: NDC math, then the
        // tail literal forwards every attribute into the interface
        // explicitly — the real dija pair's write side.
        Fixture {
            name: "dija_rect_vert",
            stage: Vertex,
            params: &[
                ("pos", Vec2, ParamKind::Attr),
                ("corner", Vec2, ParamKind::Attr),
                ("size", Vec2, ParamKind::Attr),
                ("color", Vec4, ParamKind::Attr),
                ("border_color", Vec4, ParamKind::Attr),
                ("corner_radii", Vec4, ParamKind::Attr),
                ("border_width", F32, ParamKind::Attr),
                ("shape_type", F32, ParamKind::Attr),
                ("viewport", Vec2, ParamKind::Uniform),
            ],
            varyings: vv(&[
                ("corner", Vec2),
                ("size", Vec2),
                ("color", Vec4),
                ("border_color", Vec4),
                ("corner_radii", Vec4),
                ("border_width", F32),
                ("shape_type", F32),
            ]),
            body: DIJA_RECT_VERT_BODY,
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FMUL, OP_FSUB],
            },
            msl: MslExpect::Accept {
                contains: &[
                    "viewport",
                    "out.corner = corner;",
                    "out.shape_type = shape_type;",
                ],
            },
            // The real rect vertex: the `viewport` uniform binds at 0, the
            // NDC math lowers to statements, and the tail literal assigns
            // every interface member.
            wgsl: WgslExpect::Translates {
                contains: &[
                    "@group(0) @binding(0) var<uniform> viewport: vec2<f32>;",
                    "let ndc_x = px / viewport.x * 2.0f - 1.0f;",
                    "_vout.clip = vec4<f32>(ndc_x, ndc_y, 0.0f, 1.0f);",
                    "_vout.corner = corner;",
                ],
            },
        },
        // ── EXPRESSION-IF OUTER ASSIGNMENT (A3 phi-merge) ─────────────────
        // Assignment to an OUTER local inside an if-expression branch. The
        // expression-`if` merge phis mutated outer locals exactly like the
        // statement-level `if`, so SPIR-V honors the write — `acc` reads 1.0 on
        // the then-branch and 0.0 on the else — witnessed by OP_PHI. MSL always
        // honored it (the branch lowers to an assignment), so the two agree.
        Fixture {
            name: "expr_if_assigns_outer_local",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let mut acc = 0.0 ; let v = if s . uv . x > 0.5 { acc = 1.0 ; s . uv . x } else { s . uv . y } ; Vec4 :: new ( v , acc , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real { witness: &[OP_PHI] },
            msl: MslExpect::Accept {
                contains: &["acc = 1.0;"],
            },
            // The value-`if` lowers to a statement-`if`, so an assignment to
            // the OUTER `acc` in the then-branch persists naturally (WGSL
            // mutation), and the branch tail is the `_wifN` value bound to `v`.
            wgsl: WgslExpect::Translates {
                contains: &["var acc = 0.0f;", "acc = 1.0f;", "let v = _wif0;"],
            },
        },
        Fixture {
            name: "rej_method_call",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ Vec4 :: new ( s . uv . length ( ) , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "method",
            },
            // `.length()` is a method call the shader grammar rejects.
            wgsl: WgslExpect::Reject {
                err_contains: "method",
            },
        },
        // ── BOUNDED FOR LOOPS (`for i in A .. N`, compile-time bounds) ────
        // The counted-loop accumulator: a `let mut` carried across iterations
        // and the u32 counter joining float arithmetic. SPIR-V lowers to the
        // canonical structured loop — header with the counter and accumulator
        // OpPhi, OpLoopMerge, OpULessThan bound check — witnessed by
        // OP_ULESS_THAN (nothing else in this float body compares unsigned);
        // `for_loop_spirv_structured_wiring` below pins the block shape by
        // decoding the module. MSL emits a native `for` over a `uint`
        // counter. The clean-source twin pins MSL spacing-blindness; on
        // SPIR-V the clean form still falls to a passthrough (at the
        // unspaced `Vec4::new`, as everywhere). WGSL: documented gap — the
        // walker has no loops and rejects the `for` construct.
        Fixture {
            name: "for_loop_accum",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let mut a = 0.0 ; for i in 0 .. 4 { a = a + s . uv . x + i * 0.5 ; } \
                   Vec4 :: new ( a * 0.25 , 0.0 , 0.0 , 1.0 ) }",
            alt: Some(
                "{ let mut a = 0.0; for i in 0..4 { a = a + s.uv.x + i * 0.5; } \
                 Vec4::new(a * 0.25, 0.0, 0.0, 1.0) }",
            ),
            spirv: SpirvExpect::Real {
                witness: &[OP_ULESS_THAN],
            },
            msl: MslExpect::Accept {
                contains: &[
                    "for (uint i = 0u; i < 4u; ++i) {",
                    "a = a + s.uv.x + i * 0.5;",
                ],
            },
            wgsl: WgslExpect::Reject {
                err_contains: "for",
            },
        },
        // The dija gradient stop-search shape this feature exists for: a
        // loop-carried Vec4 (`c`), a nested statement-`if` whose then-branch
        // reassigns it, and the u32 counter indexing a `&[Vec4]` slice
        // directly (no f32 round-trip — `stops[i]` uses the counter as-is;
        // the literal `stops[0]` still truncates via OpConvertFToU). The
        // range is deliberately the CONTIGUOUS printer form `0..8u32` —
        // rustc's token printer keeps `..` attached to its operands, so this
        // is the shape the macro actually ships; the SPIR-V tokenizer splits
        // it (the spaced `0 .. 4` twin is `for_loop_accum`).
        Fixture {
            name: "for_loop_stop_search",
            stage: Fragment,
            params: &[("stops", Vec4, ParamKind::Slice)],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let mut c = stops [ 0 ] ; \
                   for i in 0..8u32 { if s . uv . x > stops [ i ] . w { c = stops [ i ] ; } else { } } \
                   Vec4 :: new ( c . x , c . y , c . z , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_ULESS_THAN],
            },
            msl: MslExpect::Accept {
                contains: &[
                    "for (uint i = 0u; i < 8u; ++i) {",
                    "if (s.uv.x > stops[(uint)(i)].w) {",
                    "c = stops[(uint)(i)];",
                ],
            },
            wgsl: WgslExpect::Reject {
                err_contains: "for",
            },
        },
        // NESTED loops — the 13-tap-blur-in-two-axes shape, and the stress
        // case for the SPIR-V emitter's scratch-buffer wiring: the inner
        // loop's structured construct nests inside the outer body, the inner
        // accumulator `b` re-initializes from the OUTER body block each
        // outer iteration (its header phi's preheader edge), the inner body
        // reads the outer counter, and the outer carry `a` picks up the
        // inner loop's exit phi.
        Fixture {
            name: "for_loop_nested",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let mut a = 0.0 ; \
                   for i in 0 .. 3 { let mut b = s . uv . y ; \
                   for j in 0 .. 2 { b = b + i * 1.0 + j * 0.5 ; } \
                   a = a + b ; } \
                   Vec4 :: new ( a * 0.1 , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_ULESS_THAN],
            },
            msl: MslExpect::Accept {
                contains: &[
                    "for (uint i = 0u; i < 3u; ++i) {",
                    "for (uint j = 0u; j < 2u; ++j) {",
                    "b = b + i * 1.0 + j * 0.5;",
                    "a = a + b;",
                ],
            },
            wgsl: WgslExpect::Reject {
                err_contains: "for",
            },
        },
        // A RUNTIME loop bound (`0 .. s . t`, `t` a varying) must be refused
        // everywhere: an unbounded shader loop is invalid. SPIR-V's body
        // parser errors ("compile-time constant") and falls to the
        // passthrough — the loop is never emitted; MSL rejects the build
        // outright with the same wording.
        Fixture {
            name: "rej_for_nonconst_bound",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2), ("t", F32)]),
            body: "{ let mut a = 0.0 ; for i in 0 .. s . t { a = a + 1.0 ; } Vec4 :: new ( a , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "compile-time constant",
            },
            wgsl: WgslExpect::Reject {
                err_contains: "for",
            },
        },
        // The inclusive range `0 ..= 4` is rejected (only half-open ascending
        // ranges are supported; `..=` is one off-by-one trap the DSL refuses
        // to inherit).
        Fixture {
            name: "rej_for_inclusive_range",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let mut a = 0.0 ; for i in 0 ..= 4 { a = a + 1.0 ; } Vec4 :: new ( a , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "inclusive",
            },
            wgsl: WgslExpect::Reject {
                err_contains: "for",
            },
        },
        Fixture {
            name: "rej_unknown_intrinsic",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ Vec4 :: new ( bogus_fn ( s . uv . x ) , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "bogus_fn",
            },
            // An unknown function name is a rejection (names the offender).
            wgsl: WgslExpect::Reject {
                err_contains: "bogus_fn",
            },
        },
        Fixture {
            name: "rej_if_no_else",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            // An EXPRESSION-if without else. (A statement-if without else is
            // handled asymmetrically — SPIR-V translates it, MSL rejects it —
            // so it is deliberately NOT a fixture here; that asymmetry is a
            // reported finding, not a parity row.)
            body: "{ let c = if s . uv . x > 0.5 { 1.0 } ; Vec4 :: new ( c , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "else",
            },
            // A value-`if` without `else` has no second branch to yield — the
            // walker rejects it, same as MSL.
            wgsl: WgslExpect::Reject {
                err_contains: "else",
            },
        },
        Fixture {
            name: "rej_out_of_range_component",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("v", Vec4)]),
            body: "{ Vec4 :: new ( s . v . q , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject { err_contains: "q" },
            // `.q` is not a component letter — an unknown field, rejected.
            wgsl: WgslExpect::Reject { err_contains: "q" },
        },
        Fixture {
            name: "rej_while_loop",
            stage: Fragment,
            params: &[],
            varyings: sv(&[("uv", Vec2)]),
            body: "{ let mut a = 0.0 ; while a < 4.0 { a = a + 1.0 ; } Vec4 :: new ( a , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "while",
            },
            // The grammar has no loops; `while` is not a recognized construct.
            wgsl: WgslExpect::Reject {
                err_contains: "while",
            },
        },
    ]
}

// ─── Long fixture bodies (token-spaced wire form) ────────────────────────────

/// 4-band expression-if on s.uv.x (mirrors draw D1). Constant per column.
const BAND_EXPR_BODY: &str = "{ let c = if s . uv . x < 0.25 { Vec4 :: new ( 1.0 , 0.0 , 0.0 , 1.0 ) } \
    else { if s . uv . x < 0.5 { Vec4 :: new ( 0.0 , 1.0 , 0.0 , 1.0 ) } \
    else { if s . uv . x < 0.75 { Vec4 :: new ( 0.0 , 0.0 , 1.0 , 1.0 ) } \
    else { Vec4 :: new ( 1.0 , 1.0 , 1.0 , 1.0 ) } } } ; c }";

/// 4-band statement-if on s.uv.x with a reassigned `let mut` (mirrors D2).
const BAND_STMT_BODY: &str = "{ let mut c = Vec4 :: new ( 1.0 , 1.0 , 1.0 , 1.0 ) ; \
    if s . uv . x < 0.25 { c = Vec4 :: new ( 1.0 , 0.0 , 0.0 , 1.0 ) ; } \
    else { if s . uv . x < 0.5 { c = Vec4 :: new ( 0.0 , 1.0 , 0.0 , 1.0 ) ; } \
    else { if s . uv . x < 0.75 { c = Vec4 :: new ( 0.0 , 0.0 , 1.0 , 1.0 ) ; } else { } } } c }";

/// Circle SDF around the uv center with an fwidth-scaled smoothstep band
/// (mirrors draw D5). Symmetric → orientation-free.
const CIRCLE_SDF_BODY: &str = "{ let dx = s . uv . x - 0.5 ; let dy = s . uv . y - 0.5 ; \
    let d = length ( Vec2 :: new ( dx , dy ) ) - 0.3 ; let w = fwidth ( d ) ; \
    let a = 1.0 - smoothstep ( - w , w , d ) ; Vec4 :: new ( 1.0 , 1.0 , 1.0 , a ) }";

// The dija bodies mirror the real UI shaders in `emit_msl/shader_tests.rs`
// (rect/glyph/image fragment + rect vertex), in the token-spaced wire form
// under the shared-struct model: fragments read varyings as `s . <field>`;
// the vertex ends in the interface struct literal. They are the real dija
// render shaders and must translate on both natives.
const DIJA_RECT_FRAG_BODY: &str = "{ let px = s . corner . x * s . size . x ; let py = s . corner . y * s . size . y ; \
    let half_x = s . size . x * 0.5 ; let half_y = s . size . y * 0.5 ; let cpx = px - half_x ; let cpy = py - half_y ; \
    let dist = if s . shape_type > 0.5 { let nx = cpx / half_x ; let ny = cpy / half_y ; \
    let d = length ( Vec2 :: new ( nx , ny ) ) - 1.0 ; d * min ( half_x , half_y ) } \
    else { let r_right = if cpy > 0.0 { s . corner_radii . z } else { s . corner_radii . y } ; \
    let r_left = if cpy > 0.0 { s . corner_radii . w } else { s . corner_radii . x } ; \
    let r0 = if cpx > 0.0 { r_right } else { r_left } ; let r = min ( r0 , min ( half_x , half_y ) ) ; \
    let qx = abs ( cpx ) - half_x + r ; let qy = abs ( cpy ) - half_y + r ; \
    let outside = length ( Vec2 :: new ( max ( qx , 0.0 ) , max ( qy , 0.0 ) ) ) ; \
    min ( max ( qx , qy ) , 0.0 ) + outside - r } ; \
    let shape_alpha = 1.0 - smoothstep ( - 0.75 , 0.75 , dist ) ; \
    let fill = if s . border_width > 0.0 { let inner_alpha = 1.0 - smoothstep ( - 0.75 , 0.75 , dist + s . border_width ) ; \
    Vec4 :: new ( mix ( s . border_color . x , s . color . x , inner_alpha ) , mix ( s . border_color . y , s . color . y , inner_alpha ) , \
    mix ( s . border_color . z , s . color . z , inner_alpha ) , mix ( s . border_color . w , s . color . w , inner_alpha ) ) } \
    else { s . color } ; \
    Vec4 :: new ( fill . x * shape_alpha , fill . y * shape_alpha , fill . z * shape_alpha , fill . w * shape_alpha ) }";

const DIJA_GLYPH_FRAG_BODY: &str = "{ let u = mix ( s . uv_rect . x , s . uv_rect . z , s . corner . x ) ; \
    let v = mix ( s . uv_rect . y , s . uv_rect . w , s . corner . y ) ; \
    let coverage = sample(0 , Vec2 :: new ( u , v ) ) . x ; \
    Vec4 :: new ( s . color . x * coverage , s . color . y * coverage , s . color . z * coverage , s . color . w * coverage ) }";

const DIJA_IMAGE_FRAG_BODY: &str = "{ let u = mix ( s . uv_rect . x , s . uv_rect . z , s . corner . x ) ; \
    let v = mix ( s . uv_rect . y , s . uv_rect . w , s . corner . y ) ; \
    let sampled = sample(0 , Vec2 :: new ( u , v ) ) ; \
    let color = Vec4 :: new ( sampled . x * s . opacity , sampled . y * s . opacity , sampled . z * s . opacity , sampled . w * s . opacity ) ; \
    if s . border_radius > 0.0 { let px = s . corner . x * s . size . x ; let py = s . corner . y * s . size . y ; \
    let half_x = s . size . x * 0.5 ; let half_y = s . size . y * 0.5 ; let r = min ( s . border_radius , min ( half_x , half_y ) ) ; \
    let cpx = px - half_x ; let cpy = py - half_y ; let qx = abs ( cpx ) - half_x + r ; let qy = abs ( cpy ) - half_y + r ; \
    let outside = length ( Vec2 :: new ( max ( qx , 0.0 ) , max ( qy , 0.0 ) ) ) ; \
    let dist = min ( max ( qx , qy ) , 0.0 ) + outside - r ; let mask = 1.0 - smoothstep ( - 0.75 , 0.75 , dist ) ; \
    Vec4 :: new ( color . x * mask , color . y * mask , color . z * mask , color . w * mask ) } else { color } }";

const DIJA_RECT_VERT_BODY: &str = "{ let px = pos . x + corner . x * size . x ; \
    let py = pos . y + corner . y * size . y ; let ndc_x = px / viewport . x * 2.0 - 1.0 ; \
    let ndc_y = 1.0 - py / viewport . y * 2.0 ; \
    V { clip : Vec4 :: new ( ndc_x , ndc_y , 0.0 , 1.0 ) , corner : corner , size : size , \
    color : color , border_color : border_color , corner_radii : corner_radii , \
    border_width : border_width , shape_type : shape_type } }";

// ─── spirv-val gate (skip-if-missing) ────────────────────────────────────────

fn spirv_val_path() -> Option<&'static str> {
    ["spirv-val", "/opt/homebrew/bin/spirv-val"]
        .into_iter()
        .find(|cand| {
            std::process::Command::new(cand)
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        })
}

/// Run spirv-val on a module via stdin; returns Err(stderr) on rejection.
fn spirv_val(tool: &str, spirv: &[u8]) -> Result<(), String> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new(tool)
        .args(["--target-env", "vulkan1.3", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn spirv-val: {e}"))?;
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(spirv)
        .map_err(|e| format!("write spirv-val stdin: {e}"))?;
    let out = child
        .wait_with_output()
        .map_err(|e| format!("spirv-val run: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

fn xcrun_present() -> bool {
    std::process::Command::new("xcrun")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ─── naga gate for the WGSL twin (skip-if-missing) ───────────────────────────

/// Locate a `naga` CLI on PATH (the wgpu project's shader translator/validator).
/// `naga --version` prints the version and exits 0 when present.
fn naga_path() -> Option<&'static str> {
    ["naga"].into_iter().find(|cand| {
        std::process::Command::new(cand)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// Validate a WGSL module with `naga`. The CLI validates when given a single
/// `.wgsl` input file with no output file, exiting non-zero (and printing the
/// parse/validation error to stderr) on rejection. Returns Err(stderr) on
/// failure. Writes to a per-run temp file so parallel test binaries don't race.
fn naga_validate(tool: &str, wgsl: &str) -> Result<(), String> {
    use std::io::Write;
    let path = std::env::temp_dir().join(format!(
        "quanta_parity_{}_{}.wgsl",
        std::process::id(),
        // A cheap unique-per-call suffix so two modules in one process differ.
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let mut f = std::fs::File::create(&path).map_err(|e| format!("create temp wgsl: {e}"))?;
    f.write_all(wgsl.as_bytes())
        .map_err(|e| format!("write temp wgsl: {e}"))?;
    drop(f);
    let out = std::process::Command::new(tool)
        .arg(&path)
        .output()
        .map_err(|e| format!("run naga: {e}"))?;
    let _ = std::fs::remove_file(&path);
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// The core parity assertion: for every fixture, the MSL acceptance verdict
/// (Ok) must equal the SPIR-V real-translation verdict (Ok + witness). A
/// construct one native emitter handles and the other silently drops (SPIR-V
/// passthrough) or rejects (MSL Err) is exactly the class of bug this corpus
/// exists to catch. Each side is also checked against its declared
/// expectation. All mismatches are collected and reported together so one
/// fixture cannot mask the rest.
#[test]
fn acceptance_parity_msl_vs_spirv() {
    let mut failures = Vec::new();

    for f in fixtures() {
        let d = def(&f);
        let translated = spirv_translated(&f, &d);
        let msl_res = emit_msl(&d);
        let msl_accepts = msl_res.is_ok();

        // An allowlisted fixture asserts its documented divergent state and
        // fails if the divergence heals (so the allowlist can't go stale).
        if let Some((_, reason)) = KNOWN_DIVERGENCES.iter().find(|(n, _)| *n == f.name) {
            if msl_accepts == translated {
                failures.push(format!(
                    "[{}] listed in KNOWN_DIVERGENCES ({reason}) but msl_accepts == spirv \
                     ({msl_accepts}) — the divergence healed; remove the allowlist entry",
                    f.name
                ));
            }
            continue;
        }

        // Core parity: MSL acceptance must equal SPIR-V real translation.
        if msl_accepts != translated {
            failures.push(format!(
                "[{}] PARITY BREAK: msl_accepts={msl_accepts}, spirv_translated={translated}. \
                 msl={:?}",
                f.name,
                msl_res.as_ref().err()
            ));
        }

        // SPIR-V matches its declared expectation.
        match &f.spirv {
            SpirvExpect::Real { witness } => {
                if !translated {
                    failures.push(format!(
                        "[{}] expected SPIR-V Real (witness {witness:?}) but got passthrough/none. \
                         present opcodes: {:?}",
                        f.name,
                        emit_spirv(&d).map(|s| opcodes(&s)).unwrap_or_default()
                    ));
                }
            }
            SpirvExpect::Passthrough => {
                if translated {
                    failures.push(format!(
                        "[{}] expected SPIR-V Passthrough but real-work opcodes were present",
                        f.name
                    ));
                }
            }
            SpirvExpect::Reject { err_contains } => match emit_spirv(&d) {
                Ok(_) => failures.push(format!(
                    "[{}] expected SPIR-V Reject ({err_contains:?}), got Ok",
                    f.name
                )),
                Err(e) => {
                    if !e.contains(err_contains) {
                        failures.push(format!(
                            "[{}] SPIR-V error {e:?} does not contain {err_contains:?}",
                            f.name
                        ));
                    }
                }
            },
        }

        // MSL matches its declared expectation.
        match &f.msl {
            MslExpect::Accept { contains } => match &msl_res {
                Ok(msl) => {
                    for needle in *contains {
                        if !msl.contains(needle) {
                            failures.push(format!(
                                "[{}] MSL missing {needle:?}.\n--- MSL ---\n{msl}",
                                f.name
                            ));
                        }
                    }
                }
                Err(e) => failures.push(format!("[{}] expected MSL Accept, got Err: {e}", f.name)),
            },
            MslExpect::Reject { err_contains } => match &msl_res {
                Ok(msl) => failures.push(format!(
                    "[{}] expected MSL Reject ({err_contains:?}), got Ok:\n{msl}",
                    f.name
                )),
                Err(e) => {
                    if !e.contains(err_contains) {
                        failures.push(format!(
                            "[{}] MSL error {e:?} does not contain {err_contains:?}",
                            f.name
                        ));
                    }
                }
            },
        }
    }

    assert!(
        failures.is_empty(),
        "{} parity failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Every fixture's emitted SPIR-V — real translation OR passthrough — must
/// pass spirv-val. The passthrough fallback now rebuilds on a fresh emitter, so
/// a failed body no longer leaks ids across sections and every passthrough is
/// id-consistent by construction; a rejection from EITHER kind of module is a
/// hard failure. Skips if spirv-val is absent.
#[test]
fn spirv_validates() {
    let Some(tool) = spirv_val_path() else {
        eprintln!("SKIP spirv_validates: spirv-val not found");
        return;
    };

    let mut failures = Vec::new();
    for f in fixtures() {
        // A `Reject` fixture fails emission by contract (structural error, no
        // module) — its error text is asserted in the parity test instead.
        if matches!(f.spirv, SpirvExpect::Reject { .. }) {
            continue;
        }
        let d = def(&f);
        let spirv = match emit_spirv(&d) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("[{}] SPIR-V emission errored: {e}", f.name));
                continue;
            }
        };
        let kind = if has_real_work(&spirv) || matches!(f.spirv, SpirvExpect::Real { witness: &[] })
        {
            "real"
        } else {
            "passthrough"
        };
        if let Err(e) = spirv_val(tool, &spirv) {
            failures.push(format!(
                "[{}] spirv-val rejected a {kind} module:\n{e}",
                f.name
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} spirv-val failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Per-fixture WGSL verdicts: `Translates` fixtures must contain their
/// substrings; `Reject` fixtures must fail emission with the right error.
#[test]
fn wgsl_verdicts_hold() {
    let mut failures = Vec::new();
    for f in fixtures() {
        let d = def(&f);
        let wgsl_res = emit_wgsl(&d);
        // A `Reject` fixture asserts the emission error (a grammar rejection or
        // the SSBO cap), so it is handled before the Ok expectation of
        // `Translates`.
        if let WgslExpect::Reject { err_contains } = &f.wgsl {
            match &wgsl_res {
                Ok(w) => failures.push(format!(
                    "[{}] expected WGSL Reject ({err_contains:?}), got Ok:\n{w}",
                    f.name
                )),
                Err(e) => {
                    if !e.contains(err_contains) {
                        failures.push(format!(
                            "[{}] WGSL error {e:?} does not contain {err_contains:?}",
                            f.name
                        ));
                    }
                }
            }
            continue;
        }
        let wgsl = match wgsl_res {
            Ok(w) => w,
            Err(e) => {
                // A `Translates` fixture must emit Ok; an Err here means the
                // walker rejected a body it should handle.
                failures.push(format!("[{}] emit_wgsl unexpectedly errored: {e}", f.name));
                continue;
            }
        };
        match &f.wgsl {
            WgslExpect::Translates { contains } => {
                for needle in *contains {
                    if !wgsl.contains(needle) {
                        failures.push(format!(
                            "[{}] WGSL Translates missing {needle:?}.\n--- WGSL ---\n{wgsl}",
                            f.name
                        ));
                    }
                }
            }
            // Handled above; unreachable here.
            WgslExpect::Reject { .. } => {}
        }
    }
    assert!(
        failures.is_empty(),
        "{} WGSL verdict failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Every WGSL-`Translates` fixture must round-trip through the real `naga`
/// validator — catching "structurally emitted but invalid WGSL", the twin of
/// `spirv_validates`/`msl_compiles_to_metallib` for the WGSL emitter. A
/// `Reject` fixture fails emission by contract (no module to validate) and is
/// skipped. Gated on `naga` being on PATH — skips with an eprintln otherwise.
#[test]
fn wgsl_validates() {
    let Some(tool) = naga_path() else {
        eprintln!("SKIP wgsl_validates: naga not found");
        return;
    };

    let mut failures = Vec::new();
    for f in fixtures() {
        // Only `Translates` fixtures produce a module; a `Reject` fixture errors
        // at emission (its error text is asserted in `wgsl_verdicts_hold`).
        if matches!(f.wgsl, WgslExpect::Reject { .. }) {
            continue;
        }
        let d = def(&f);
        let wgsl = match emit_wgsl(&d) {
            Ok(w) => w,
            Err(e) => {
                failures.push(format!("[{}] WGSL emission errored: {e}", f.name));
                continue;
            }
        };
        if let Err(e) = naga_validate(tool, &wgsl) {
            failures.push(format!(
                "[{}] naga rejected the module:\n{e}\n--- WGSL ---\n{wgsl}",
                f.name
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} naga validation failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Every MSL-accepted fixture must round-trip through the real metallib
/// compiler — catching "structurally emitted but invalid MSL". xcrun-gated.
#[test]
fn msl_compiles_to_metallib() {
    if !xcrun_present() {
        eprintln!("SKIP msl_compiles_to_metallib: no xcrun");
        return;
    }

    let mut failures = Vec::new();
    for f in fixtures() {
        if !matches!(f.msl, MslExpect::Accept { .. }) {
            continue;
        }
        let d = def(&f);
        let msl = match emit_msl(&d) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!(
                    "[{}] MSL Accept expected but emission errored: {e}",
                    f.name
                ));
                continue;
            }
        };
        match crate::metallib::compile_msl_to_metallib(&msl) {
            Ok(Some(bytes)) if bytes.is_empty() => {
                failures.push(format!("[{}] empty metallib", f.name))
            }
            Ok(Some(_)) => {}
            Ok(None) => failures.push(format!("[{}] xcrun vanished mid-run", f.name)),
            Err(e) => failures.push(format!(
                "[{}] metallib compile failed:\n{e}\n--- MSL ---\n{msl}",
                f.name
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "{} metallib failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The clean-source and token-spaced wire forms must reach the SAME MSL
/// acceptance verdict (syn is spacing-blind). The SPIR-V tokenizer, by
/// contrast, is whitespace-delimited — the wire form is a real translation
/// while the clean form falls to a passthrough. This test pins both
/// contracts: MSL parity across spacings, and the SPIR-V wire-form's
/// real-vs-clean-form's-passthrough split. (Byte-identical output is not
/// required.)
#[test]
fn clean_and_wire_forms_agree_on_msl() {
    let mut failures = Vec::new();
    let mut checked = 0usize;

    for f in fixtures() {
        let Some(clean) = def_alt(&f) else {
            continue;
        };
        checked += 1;
        let wire = def(&f);

        // MSL: both spacings agree.
        let wire_msl = emit_msl(&wire).is_ok();
        let clean_msl = emit_msl(&clean).is_ok();
        if wire_msl != clean_msl {
            failures.push(format!(
                "[{}] MSL verdict differs by spacing: wire={wire_msl}, clean={clean_msl} \
                 (clean err: {:?})",
                f.name,
                emit_msl(&clean).err()
            ));
        }

        // SPIR-V: the wire form is real; the clean form falls to passthrough
        // (the tokenizer needs standalone `::`/`.` tokens). Only assert this
        // for witnessed-Real fixtures — a witness-waived body is a passthrough
        // look-alike either way, so the split is unobservable.
        if let SpirvExpect::Real { witness } = &f.spirv
            && !witness.is_empty()
        {
            if !spirv_translated(&f, &wire) {
                failures.push(format!(
                    "[{}] wire form expected SPIR-V Real but got passthrough",
                    f.name
                ));
            }
            let clean_real = emit_spirv(&clean)
                .map(|s| {
                    let ops = opcodes(&s);
                    witness.iter().any(|w| ops.contains(w))
                })
                .unwrap_or(false);
            if clean_real {
                failures.push(format!(
                    "[{}] clean form unexpectedly produced a real SPIR-V translation — the \
                     tokenizer may now be whitespace-insensitive; revisit the wire-form note",
                    f.name
                ));
            }
        }
    }

    assert!(
        checked >= 4,
        "expected at least 4 clean-source twins, found {checked}"
    );
    assert!(
        failures.is_empty(),
        "{} clean/wire-form failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The frag_coord fixture's SPIR-V must carry the FragCoord contract
/// explicitly: exactly one id decorated `BuiltIn FragCoord`, whose OpVariable
/// sits in the Input storage class, listed in the OpEntryPoint interface.
/// spirv-val checks well-formedness (`spirv_validates` covers this fixture
/// too); this pins the specific builtin wiring so a regression to, say, a
/// Location-decorated input cannot slip through while staying "valid".
#[test]
fn frag_coord_spirv_builtin_wiring() {
    const OP_ENTRY_POINT: u16 = 15;
    const OP_VARIABLE: u16 = 59;
    const OP_DECORATE: u16 = 71;
    const DECORATION_BUILTIN: u32 = 11;
    const BUILTIN_FRAG_COORD: u32 = 15;
    const STORAGE_CLASS_INPUT: u32 = 1;

    let f = fixtures()
        .into_iter()
        .find(|f| f.name == "frag_coord_scaled")
        .expect("frag_coord_scaled fixture present");
    let d = def(&f);
    let spirv = emit_spirv(&d).expect("frag_coord fixture must emit");
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // One instruction-walk collecting the three facts.
    let mut frag_coord_ids = Vec::new();
    let mut input_var_ids = Vec::new();
    let mut entry_operand_words: Vec<u32> = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let word_count = ((words[i] >> 16) as usize).max(1);
        let op = (words[i] & 0xFFFF) as u16;
        let operands = &words[i + 1..i + word_count];
        match op {
            OP_DECORATE
                if operands.len() == 3
                    && operands[1] == DECORATION_BUILTIN
                    && operands[2] == BUILTIN_FRAG_COORD =>
            {
                frag_coord_ids.push(operands[0]);
            }
            // OpVariable: [result_type, result_id, storage_class, (init)]
            OP_VARIABLE if operands.len() >= 3 && operands[2] == STORAGE_CLASS_INPUT => {
                input_var_ids.push(operands[1]);
            }
            OP_ENTRY_POINT => entry_operand_words.extend_from_slice(operands),
            _ => {}
        }
        i += word_count;
    }

    assert_eq!(
        frag_coord_ids.len(),
        1,
        "expected exactly one BuiltIn FragCoord decoration, found {frag_coord_ids:?}"
    );
    let id = frag_coord_ids[0];
    assert!(
        input_var_ids.contains(&id),
        "FragCoord id %{id} is not an Input-storage OpVariable (Input vars: {input_var_ids:?})"
    );
    // Interface ids trail the packed entry-point name; the id values are tiny
    // next to ASCII-packed name words, so a plain contains is unambiguous.
    assert!(
        entry_operand_words.contains(&id),
        "FragCoord id %{id} missing from the OpEntryPoint interface"
    );
}

/// The u32 interface contract, pinned by decoding the modules (spirv-val
/// checks well-formedness in `spirv_validates`; this pins the SPECIFIC
/// wiring so a regression cannot slip through while staying "valid"):
///
/// - vertex: the u32 VARYING Output is decorated `Flat`; the u32
///   vertex-attribute Input is NOT (Flat is invalid on vertex inputs); the
///   float varying Output is NOT (floats stay smooth).
/// - fragment: the u32 Input is decorated `Flat`; the float input is NOT.
/// - fragment: the `== 3u32` comparison lowers to the INTEGER OpIEqual, and
///   no float FOrdEqual appears.
#[test]
fn u32_flat_wiring_spirv() {
    const OP_TYPE_INT: u16 = 21;
    const OP_TYPE_POINTER: u16 = 32;
    const OP_VARIABLE: u16 = 59;
    const OP_DECORATE: u16 = 71;
    const DECORATION_FLAT: u32 = 14;
    const STORAGE_CLASS_INPUT: u32 = 1;
    const STORAGE_CLASS_OUTPUT: u32 = 3;

    /// Decode a module into (opcode, operands) instructions.
    fn instrs(spirv: &[u8]) -> Vec<(u16, Vec<u32>)> {
        let words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut out = Vec::new();
        let mut i = 5;
        while i < words.len() {
            let word_count = ((words[i] >> 16) as usize).max(1);
            out.push((
                (words[i] & 0xFFFF) as u16,
                words[i + 1..i + word_count].to_vec(),
            ));
            i += word_count;
        }
        out
    }

    /// The variables of storage class `class` whose pointee type is / is not
    /// the module's `OpTypeInt 32 0`, plus the set of Flat-decorated ids.
    struct Wiring {
        uint_vars: Vec<u32>,
        other_vars: Vec<u32>,
        flat_ids: Vec<u32>,
    }
    fn wiring(instrs: &[(u16, Vec<u32>)], class: u32) -> Wiring {
        let uint_ty: Vec<u32> = instrs
            .iter()
            .filter(|(op, a)| *op == OP_TYPE_INT && a.len() == 3 && a[1] == 32 && a[2] == 0)
            .map(|(_, a)| a[0])
            .collect();
        // OpTypePointer: [result, storage_class, pointee]
        let ptr_to = |uint: bool| -> Vec<u32> {
            instrs
                .iter()
                .filter(|(op, a)| {
                    *op == OP_TYPE_POINTER
                        && a.len() == 3
                        && a[1] == class
                        && uint_ty.contains(&a[2]) == uint
                })
                .map(|(_, a)| a[0])
                .collect()
        };
        let (uint_ptrs, other_ptrs) = (ptr_to(true), ptr_to(false));
        // OpVariable: [result_type, result_id, storage_class]
        let vars_of = |ptrs: &[u32]| -> Vec<u32> {
            instrs
                .iter()
                .filter(|(op, a)| {
                    *op == OP_VARIABLE && a.len() >= 3 && a[2] == class && ptrs.contains(&a[0])
                })
                .map(|(_, a)| a[1])
                .collect()
        };
        Wiring {
            uint_vars: vars_of(&uint_ptrs),
            other_vars: vars_of(&other_ptrs),
            flat_ids: instrs
                .iter()
                .filter(|(op, a)| *op == OP_DECORATE && a.len() == 2 && a[1] == DECORATION_FLAT)
                .map(|(_, a)| a[0])
                .collect(),
        }
    }

    // ── Vertex: u32 varying Output Flat; u32 attribute Input NOT Flat. ──
    let vert = fixtures()
        .into_iter()
        .find(|f| f.name == "u32_flat_varying_vert")
        .expect("u32_flat_varying_vert fixture present");
    let spirv = emit_spirv(&def(&vert)).expect("vertex fixture must emit");
    let vi = instrs(&spirv);

    let outputs = wiring(&vi, STORAGE_CLASS_OUTPUT);
    assert_eq!(
        outputs.uint_vars.len(),
        1,
        "vertex: expected exactly one u32 varying Output, found {:?}",
        outputs.uint_vars
    );
    assert!(
        outputs.flat_ids.contains(&outputs.uint_vars[0]),
        "vertex: the u32 varying Output %{} must be decorated Flat (Flat ids: {:?})",
        outputs.uint_vars[0],
        outputs.flat_ids
    );
    for float_out in &outputs.other_vars {
        assert!(
            !outputs.flat_ids.contains(float_out),
            "vertex: float varying Output %{float_out} must NOT be Flat (smooth interpolation)"
        );
    }

    let inputs = wiring(&vi, STORAGE_CLASS_INPUT);
    assert_eq!(
        inputs.uint_vars.len(),
        1,
        "vertex: expected exactly one u32 attribute Input, found {:?}",
        inputs.uint_vars
    );
    assert!(
        !inputs.flat_ids.contains(&inputs.uint_vars[0]),
        "vertex: the u32 ATTRIBUTE Input %{} must NOT be Flat — the decoration is \
         invalid on vertex-stage Inputs",
        inputs.uint_vars[0]
    );

    // ── Fragment: u32 Input Flat, float input smooth, integer compare. ──
    let frag = fixtures()
        .into_iter()
        .find(|f| f.name == "u32_eq_suffixed_literal_frag")
        .expect("u32_eq_suffixed_literal_frag fixture present");
    let spirv = emit_spirv(&def(&frag)).expect("fragment fixture must emit");
    let fi = instrs(&spirv);

    let inputs = wiring(&fi, STORAGE_CLASS_INPUT);
    assert_eq!(
        inputs.uint_vars.len(),
        1,
        "fragment: expected exactly one u32 Input, found {:?}",
        inputs.uint_vars
    );
    assert!(
        inputs.flat_ids.contains(&inputs.uint_vars[0]),
        "fragment: the u32 varying Input %{} must be decorated Flat (Flat ids: {:?})",
        inputs.uint_vars[0],
        inputs.flat_ids
    );
    for float_in in &inputs.other_vars {
        assert!(
            !inputs.flat_ids.contains(float_in),
            "fragment: float Input %{float_in} must NOT be Flat"
        );
    }

    let ops: Vec<u16> = fi.iter().map(|(op, _)| *op).collect();
    assert!(
        ops.contains(&OP_IEQUAL),
        "fragment: `shape_type == 3u32` must lower to the integer OpIEqual"
    );
    assert!(
        !ops.contains(&OP_FORD_EQUAL),
        "fragment: no float FOrdEqual may appear for a u32 comparison"
    );
}

/// The shared-struct interface contract, pinned by decoding the flagship
/// pair's modules (`varyings_struct_vert` / `varyings_struct_frag` — struct
/// `Surface { #[position] clip: Vec4, uv: Vec2, kind: u32 }`), plus the
/// position-reading fragment (`varyings_position_read_frag`). spirv-val
/// checks well-formedness; this pins the SPECIFIC wiring so a regression
/// cannot slip through while staying "valid":
///
/// - vertex: exactly one Output decorated `BuiltIn Position` (the `clip`
///   field); the `uv` Output at Location 0 (vec2, smooth); the `kind` Output
///   at Location 1 (uint, Flat) — Locations are FIELD-DECLARATION order.
/// - fragment: Inputs mirror the vertex Outputs exactly — Location 0 vec2
///   smooth, Location 1 uint Flat — so the two modules link by construction.
/// - fragment reading `s.<position>`: exactly one Input decorated
///   `BuiltIn FragCoord` (the WGSL window-position semantics).
#[test]
fn varyings_interface_wiring_spirv() {
    const OP_TYPE_INT: u16 = 21;
    const OP_TYPE_FLOAT: u16 = 22;
    const OP_TYPE_VECTOR: u16 = 23;
    const OP_TYPE_POINTER: u16 = 32;
    const OP_VARIABLE: u16 = 59;
    const OP_DECORATE: u16 = 71;
    const DECORATION_BUILTIN: u32 = 11;
    const DECORATION_FLAT: u32 = 14;
    const DECORATION_LOCATION: u32 = 30;
    const BUILTIN_POSITION: u32 = 0;
    const BUILTIN_FRAG_COORD: u32 = 15;
    const STORAGE_CLASS_INPUT: u32 = 1;
    const STORAGE_CLASS_OUTPUT: u32 = 3;

    /// Decode a module into (opcode, operands) instructions.
    fn instrs(spirv: &[u8]) -> Vec<(u16, Vec<u32>)> {
        let words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut out = Vec::new();
        let mut i = 5;
        while i < words.len() {
            let word_count = ((words[i] >> 16) as usize).max(1);
            out.push((
                (words[i] & 0xFFFF) as u16,
                words[i + 1..i + word_count].to_vec(),
            ));
            i += word_count;
        }
        out
    }

    /// The interface variables of storage class `class`, as
    /// (var_id, location, is_flat, pointee_kind) where pointee_kind is
    /// "uint" / "vec2" / "other". Variables without a Location (builtins)
    /// are skipped.
    fn located_vars(instrs: &[(u16, Vec<u32>)], class: u32) -> Vec<(u32, u32, bool, String)> {
        let uint_ty: Vec<u32> = instrs
            .iter()
            .filter(|(op, a)| *op == OP_TYPE_INT && a.len() == 3 && a[1] == 32 && a[2] == 0)
            .map(|(_, a)| a[0])
            .collect();
        let float_ty: Vec<u32> = instrs
            .iter()
            .filter(|(op, a)| *op == OP_TYPE_FLOAT && a.len() == 2 && a[1] == 32)
            .map(|(_, a)| a[0])
            .collect();
        let vec2_ty: Vec<u32> = instrs
            .iter()
            .filter(|(op, a)| {
                *op == OP_TYPE_VECTOR && a.len() == 3 && float_ty.contains(&a[1]) && a[2] == 2
            })
            .map(|(_, a)| a[0])
            .collect();
        // OpTypePointer: [result, storage_class, pointee]
        let kind_of_ptr = |ptr: u32| -> Option<String> {
            instrs.iter().find_map(|(op, a)| {
                if *op == OP_TYPE_POINTER && a.len() == 3 && a[0] == ptr && a[1] == class {
                    Some(if uint_ty.contains(&a[2]) {
                        "uint".to_string()
                    } else if vec2_ty.contains(&a[2]) {
                        "vec2".to_string()
                    } else {
                        "other".to_string()
                    })
                } else {
                    None
                }
            })
        };
        let location_of = |id: u32| -> Option<u32> {
            instrs.iter().find_map(|(op, a)| {
                if *op == OP_DECORATE && a.len() == 3 && a[0] == id && a[1] == DECORATION_LOCATION {
                    Some(a[2])
                } else {
                    None
                }
            })
        };
        let flat_ids: Vec<u32> = instrs
            .iter()
            .filter(|(op, a)| *op == OP_DECORATE && a.len() == 2 && a[1] == DECORATION_FLAT)
            .map(|(_, a)| a[0])
            .collect();
        instrs
            .iter()
            .filter(|(op, a)| *op == OP_VARIABLE && a.len() >= 3 && a[2] == class)
            .filter_map(|(_, a)| {
                let kind = kind_of_ptr(a[0])?;
                let loc = location_of(a[1])?;
                Some((a[1], loc, flat_ids.contains(&a[1]), kind))
            })
            .collect()
    }

    /// Ids decorated `BuiltIn <builtin>`.
    fn builtin_ids(instrs: &[(u16, Vec<u32>)], builtin: u32) -> Vec<u32> {
        instrs
            .iter()
            .filter(|(op, a)| {
                *op == OP_DECORATE && a.len() == 3 && a[1] == DECORATION_BUILTIN && a[2] == builtin
            })
            .map(|(_, a)| a[0])
            .collect()
    }

    /// The Location→(flat, kind) table of one stage's interface, asserted to
    /// be exactly {0: smooth vec2, 1: Flat uint} — field-declaration order.
    fn assert_uv_kind_interface(stage: &str, vars: &[(u32, u32, bool, String)]) {
        assert_eq!(
            vars.len(),
            2,
            "{stage}: expected exactly two located varying variables, got {vars:?}"
        );
        let at = |loc: u32| {
            vars.iter()
                .find(|(_, l, _, _)| *l == loc)
                .unwrap_or_else(|| panic!("{stage}: no varying at Location {loc}: {vars:?}"))
        };
        let (_, _, uv_flat, uv_kind) = at(0);
        assert_eq!(uv_kind, "vec2", "{stage}: Location 0 must be the vec2 `uv`");
        assert!(!uv_flat, "{stage}: the float varying must stay smooth");
        let (_, _, kind_flat, kind_kind) = at(1);
        assert_eq!(
            kind_kind, "uint",
            "{stage}: Location 1 must be the uint `kind`"
        );
        assert!(
            kind_flat,
            "{stage}: the uint varying must be Flat on this interface end"
        );
    }

    // ── Vertex: BuiltIn Position + Outputs in field-declaration order. ──
    let vert = fixtures()
        .into_iter()
        .find(|f| f.name == "varyings_struct_vert")
        .expect("varyings_struct_vert fixture present");
    let spirv = emit_spirv(&def(&vert)).expect("varyings vertex must emit");
    let vi = instrs(&spirv);
    let position_ids = builtin_ids(&vi, BUILTIN_POSITION);
    assert_eq!(
        position_ids.len(),
        1,
        "vertex: expected exactly one BuiltIn Position, found {position_ids:?}"
    );
    assert_uv_kind_interface("vertex Outputs", &located_vars(&vi, STORAGE_CLASS_OUTPUT));

    // ── Fragment: Inputs mirror the vertex Outputs, Location for Location. ──
    let frag = fixtures()
        .into_iter()
        .find(|f| f.name == "varyings_struct_frag")
        .expect("varyings_struct_frag fixture present");
    let spirv = emit_spirv(&def(&frag)).expect("varyings fragment must emit");
    let fi = instrs(&spirv);
    assert_uv_kind_interface("fragment Inputs", &located_vars(&fi, STORAGE_CLASS_INPUT));

    // ── Fragment reading `s.<position>`: BuiltIn FragCoord declared. ──
    let posread = fixtures()
        .into_iter()
        .find(|f| f.name == "varyings_position_read_frag")
        .expect("varyings_position_read_frag fixture present");
    let spirv = emit_spirv(&def(&posread)).expect("position-reading fragment must emit");
    let pi = instrs(&spirv);
    let frag_coord_ids = builtin_ids(&pi, BUILTIN_FRAG_COORD);
    assert_eq!(
        frag_coord_ids.len(),
        1,
        "fragment: reading the position field must declare exactly one \
         BuiltIn FragCoord Input, found {frag_coord_ids:?}"
    );
    // And a fragment that does NOT read the position field declares none.
    let plain = fixtures()
        .into_iter()
        .find(|f| f.name == "varyings_struct_frag")
        .expect("varyings_struct_frag fixture present");
    let spirv = emit_spirv(&def(&plain)).expect("plain varyings fragment must emit");
    let ni = instrs(&spirv);
    assert!(
        builtin_ids(&ni, BUILTIN_FRAG_COORD).is_empty(),
        "fragment: a body that never reads the position field must not \
         declare FragCoord"
    );
}

/// The vertex-index fixture's SPIR-V must carry the builtin contract
/// explicitly: exactly one id decorated `BuiltIn VertexIndex` (42) and
/// exactly one `BuiltIn InstanceIndex` (43) — the Vulkan-flavoured pair,
/// NOT OpenGL's VertexId(5)/InstanceId(6) — each an Input-storage
/// OpVariable listed in the OpEntryPoint interface, and NEITHER Flat- nor
/// Location-decorated (a builtin vertex Input takes only the BuiltIn
/// decoration; Flat belongs to interpolants, Location to user attributes).
/// spirv-val checks well-formedness (`spirv_validates` covers this fixture
/// too); this pins the specific wiring so a regression to, say, the OpenGL
/// builtin values cannot slip through while staying structurally "valid"
/// under a laxer target. Mirrors `frag_coord_spirv_builtin_wiring`.
#[test]
fn vertex_index_spirv_builtin_wiring() {
    const OP_ENTRY_POINT: u16 = 15;
    const OP_VARIABLE: u16 = 59;
    const OP_DECORATE: u16 = 71;
    const DECORATION_BUILTIN: u32 = 11;
    const DECORATION_FLAT: u32 = 14;
    const DECORATION_LOCATION: u32 = 30;
    const BUILTIN_VERTEX_INDEX: u32 = 42;
    const BUILTIN_INSTANCE_INDEX: u32 = 43;
    const STORAGE_CLASS_INPUT: u32 = 1;

    let f = fixtures()
        .into_iter()
        .find(|f| f.name == "vertex_index_fullscreen_tri")
        .expect("vertex_index_fullscreen_tri fixture present");
    let d = def(&f);
    let spirv = emit_spirv(&d).expect("vertex_index fixture must emit");
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // One instruction-walk collecting the facts.
    let mut vertex_index_ids = Vec::new();
    let mut instance_index_ids = Vec::new();
    let mut flat_or_location_ids = Vec::new();
    let mut input_var_ids = Vec::new();
    let mut entry_operand_words: Vec<u32> = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let word_count = ((words[i] >> 16) as usize).max(1);
        let op = (words[i] & 0xFFFF) as u16;
        let operands = &words[i + 1..i + word_count];
        match op {
            OP_DECORATE if operands.len() == 3 && operands[1] == DECORATION_BUILTIN => {
                match operands[2] {
                    BUILTIN_VERTEX_INDEX => vertex_index_ids.push(operands[0]),
                    BUILTIN_INSTANCE_INDEX => instance_index_ids.push(operands[0]),
                    _ => {}
                }
            }
            OP_DECORATE
                if operands.len() >= 2
                    && (operands[1] == DECORATION_FLAT || operands[1] == DECORATION_LOCATION) =>
            {
                flat_or_location_ids.push(operands[0]);
            }
            // OpVariable: [result_type, result_id, storage_class, (init)]
            OP_VARIABLE if operands.len() >= 3 && operands[2] == STORAGE_CLASS_INPUT => {
                input_var_ids.push(operands[1]);
            }
            OP_ENTRY_POINT => entry_operand_words.extend_from_slice(operands),
            _ => {}
        }
        i += word_count;
    }

    for (builtin, ids) in [
        ("VertexIndex", &vertex_index_ids),
        ("InstanceIndex", &instance_index_ids),
    ] {
        assert_eq!(
            ids.len(),
            1,
            "expected exactly one BuiltIn {builtin} decoration, found {ids:?}"
        );
        let id = ids[0];
        assert!(
            input_var_ids.contains(&id),
            "{builtin} id %{id} is not an Input-storage OpVariable (Input vars: {input_var_ids:?})"
        );
        // Interface ids trail the packed entry-point name; the id values are
        // tiny next to ASCII-packed name words, so a plain contains is
        // unambiguous.
        assert!(
            entry_operand_words.contains(&id),
            "{builtin} id %{id} missing from the OpEntryPoint interface"
        );
        assert!(
            !flat_or_location_ids.contains(&id),
            "{builtin} id %{id} must carry ONLY the BuiltIn decoration — \
             never Flat (interpolants only) or Location (user attributes only)"
        );
    }
}

/// The for-loop fixture's SPIR-V must carry the structured-loop contract
/// explicitly, pinned by decoding the module (spirv-val checks
/// well-formedness in `spirv_validates`; this pins the SPECIFIC shape so a
/// regression — say, an unrolled body or a merge-less branch cycle — cannot
/// slip through while staying "valid" somewhere laxer):
///
/// - exactly one `OpLoopMerge merge continue` in the module;
/// - it sits in a header block terminated by `OpBranchConditional cond body
///   merge` (the bound check exits to the merge block);
/// - the header opens with OpPhi instructions — the u32 counter AND the
///   loop-carried accumulator — each naming the continue block as its
///   back-edge predecessor;
/// - the continue block ends with `OpBranch header` (the back edge);
/// - the bound check is the UNSIGNED OpULessThan (u32 counter, never float).
#[test]
fn for_loop_spirv_structured_wiring() {
    const OP_LABEL: u16 = 248;
    const OP_BRANCH: u16 = 249;
    const OP_BRANCH_CONDITIONAL: u16 = 250;

    /// A decoded instruction: (opcode, operand words).
    type Inst = (u16, Vec<u32>);

    /// Decode a module into (opcode, operands) instructions.
    fn instrs(spirv: &[u8]) -> Vec<Inst> {
        let words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut out = Vec::new();
        let mut i = 5;
        while i < words.len() {
            let word_count = ((words[i] >> 16) as usize).max(1);
            out.push((
                (words[i] & 0xFFFF) as u16,
                words[i + 1..i + word_count].to_vec(),
            ));
            i += word_count;
        }
        out
    }

    let f = fixtures()
        .into_iter()
        .find(|f| f.name == "for_loop_accum")
        .expect("for_loop_accum fixture present");
    let spirv = emit_spirv(&def(&f)).expect("for_loop fixture must emit");
    let all = instrs(&spirv);

    // Group the function's instructions into blocks by their OpLabel.
    let mut blocks: Vec<(u32, Vec<&Inst>)> = Vec::new();
    for inst in &all {
        if inst.0 == OP_LABEL {
            blocks.push((inst.1[0], Vec::new()));
        } else if let Some((_, body)) = blocks.last_mut() {
            body.push(inst);
        }
    }
    let block_of = |label: u32| -> &Vec<&Inst> {
        &blocks
            .iter()
            .find(|(l, _)| *l == label)
            .unwrap_or_else(|| panic!("no block labeled %{label}"))
            .1
    };

    // Exactly one loop in the module; its merge/continue targets.
    let loop_merges: Vec<&Vec<u32>> = all
        .iter()
        .filter(|(op, _)| *op == OP_LOOP_MERGE)
        .map(|(_, a)| a)
        .collect();
    assert_eq!(
        loop_merges.len(),
        1,
        "expected exactly one OpLoopMerge, found {}",
        loop_merges.len()
    );
    let (merge_label, continue_label) = (loop_merges[0][0], loop_merges[0][1]);

    // The header block: the one containing the OpLoopMerge.
    let (header_label, header) = blocks
        .iter()
        .find(|(_, body)| body.iter().any(|(op, _)| *op == OP_LOOP_MERGE))
        .expect("a block contains the OpLoopMerge");

    // Terminator: OpBranchConditional whose FALSE target is the merge block
    // (the bound check exits the loop).
    let (term_op, term_args) = header.last().expect("header has a terminator");
    assert_eq!(
        *term_op, OP_BRANCH_CONDITIONAL,
        "the loop header must end in OpBranchConditional"
    );
    assert_eq!(
        term_args[2], merge_label,
        "the bound check's false edge must exit to the merge block"
    );

    // Header phis: they open the block, and BOTH the counter and the carried
    // accumulator `a` are present — each phi's back-edge parent is the
    // continue block. OpPhi operands: [ty, result, v1, blk1, v2, blk2].
    let phis: Vec<&&Inst> = header.iter().take_while(|(op, _)| *op == OP_PHI).collect();
    assert!(
        phis.len() >= 2,
        "expected >= 2 leading header phis (counter + carried local), found {}",
        phis.len()
    );
    for (_, args) in &phis {
        let parents: Vec<u32> = args[2..].chunks(2).map(|p| p[1]).collect();
        assert!(
            parents.contains(&continue_label),
            "each loop phi must name the continue block %{continue_label} as a \
             predecessor, got parents {parents:?}"
        );
    }

    // The bound check is unsigned and feeds the terminator from this block.
    assert!(
        header.iter().any(|(op, _)| *op == OP_ULESS_THAN),
        "the loop bound check must be the unsigned OpULessThan in the header"
    );

    // The continue block ends with the back edge to the header.
    let continue_block = block_of(continue_label);
    let (back_op, back_args) = continue_block.last().expect("continue has a terminator");
    assert_eq!(*back_op, OP_BRANCH, "continue must end in OpBranch");
    assert_eq!(
        back_args[0], *header_label,
        "the continue block's branch must be the back edge to the header"
    );
}
