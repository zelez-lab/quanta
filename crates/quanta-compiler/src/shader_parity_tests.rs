//! Cross-emitter shader parity corpus.
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
//! - WGSL is the honesty layer for the lagging twin. Its stub is a naive
//!   string replace (`Vec4 :: new (` → `vec4<f32>(`, `let mut ` → `var `)
//!   that ALWAYS returns Ok, drops uniforms entirely, and passes
//!   statement-`if`, `let`, `sample(`, deref-`*`, and un-aliased `Vec2 :: new
//!   (` through untranslated. `Translates` is claimed only for the bodies it
//!   genuinely handles; every other fixture is `KnownBroken`, which asserts
//!   the known wrongness is STILL observable (a Rust `if`/`let` verbatim
//!   inside a `return`, an undeclared uniform name, a raw `sample(`). When
//!   someone fixes `emit_wgsl`, those assertions fail and the fixture must
//!   flip to `Translates` — the designed trip-wire.

use quanta_ir::{ShaderDef, ShaderParam, ShaderStage, ShaderType};

// ─── SPIR-V opcode numbers (verified against emit_spirv/constants.rs) ────────

const OP_EXT_INST: u16 = 12;
const OP_VECTOR_SHUFFLE: u16 = 79;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_FADD: u16 = 129;
const OP_FSUB: u16 = 131;
const OP_FMUL: u16 = 133;
const OP_FDIV: u16 = 136;
const OP_MATRIX_TIMES_VECTOR: u16 = 145;
const OP_F_NEGATE: u16 = 127;
const OP_FORD_GREATER_THAN: u16 = 186;
const OP_DPDX: u16 = 207;
const OP_FWIDTH: u16 = 209;
const OP_PHI: u16 = 245;

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
}

enum MslExpect {
    /// `emit_*_shader` returns Ok and every substring is present in the MSL.
    Accept { contains: &'static [&'static str] },
    /// `emit_*_shader` returns Err whose message contains the substring.
    Reject { err_contains: &'static str },
}

enum WgslExpect {
    /// The string-replace stub genuinely produces this WGSL — every
    /// substring must be present.
    Translates { contains: &'static [&'static str] },
    /// The stub mistranslates; each residue substring pins a still-observable
    /// wrongness. When emit_wgsl is fixed these assertions break and the
    /// fixture must move to `Translates`.
    KnownBroken { residue: &'static [&'static str] },
}

struct Fixture {
    name: &'static str,
    stage: ShaderStage,
    params: &'static [(&'static str, ShaderType, bool)],
    /// Token-spaced wire form — the bytes the emitters receive at runtime.
    body: &'static str,
    /// Clean-source twin (`Vec4::new(uv.x)`) for the string-replace-fiasco
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
            .map(|(name, ty, is_uniform)| ShaderParam {
                name: name.to_string(),
                ty: *ty,
                is_uniform: *is_uniform,
            })
            .collect(),
        return_type: ShaderType::Vec4,
        body_source: f.body.to_string(),
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
    OP_VECTOR_SHUFFLE,
    OP_IMAGE_SAMPLE_IMPLICIT_LOD,
    OP_F_NEGATE,
    OP_FADD,
    OP_FSUB,
    OP_FMUL,
    OP_FDIV,
    OP_MATRIX_TIMES_VECTOR,
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
    }
}

// ─── Divergence allowlist (surfaced bugs only — see module docs) ─────────────

const KNOWN_DIVERGENCES: &[(&str, &str)] = &[
    // The real dija rounded-rect fragment: MSL accepts it (syn handles the
    // `let`-bindings nested inside its `if`-expression branches), but the
    // SPIR-V expression-`if` parser rejects a branch-local `let` and falls to
    // a passthrough — so this shader misrenders on Vulkan while it is correct
    // on Metal. Documented so the parity suite stays green; removing the
    // SPIR-V limitation makes this heal and the entry must then be deleted.
    (
        "dija_rect_frag",
        "SPIR-V expr-if parser rejects branch-local `let`; MSL (syn) accepts",
    ),
    // rustfmt puts a trailing comma on every wrapped multi-line call, the
    // token printer preserves it, and the SPIR-V argument parser leaves it
    // unconsumed ("unexpected token: Comma") → passthrough. MSL re-parses
    // with syn and re-emits without the comma. Any rustfmt-wrapped call in a
    // user shader silently misrenders on Vulkan until this is fixed.
    (
        "ctor_trailing_comma",
        "trailing comma in a call defeats the SPIR-V arg parser; MSL (syn) accepts",
    ),
];

// ─── The fixture table ───────────────────────────────────────────────────────

use ShaderStage::{Fragment, Vertex};
use ShaderType::{F32, Mat4, Vec2, Vec3, Vec4};

fn fixtures() -> Vec<Fixture> {
    vec![
        // ── CORE EXPRESSIONS ──────────────────────────────────────────────
        Fixture {
            name: "lit_solid",
            stage: Fragment,
            params: &[],
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
                contains: &["vec4<f32>( 1.0 , 0.0 , 0.0 , 1.0 )"],
            },
        },
        Fixture {
            name: "arith_mix",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ Vec4 :: new ( uv . x * 0.5 + 0.25 , uv . y / 2.0 - 0.1 , 1.0 - uv . x , 1.0 ) }",
            alt: Some("{ Vec4::new(uv.x * 0.5 + 0.25, uv.y / 2.0 - 0.1, 1.0 - uv.x, 1.0) }"),
            spirv: SpirvExpect::Real {
                witness: &[OP_FMUL, OP_FADD, OP_FSUB, OP_FDIV],
            },
            msl: MslExpect::Accept {
                contains: &["* 0.5"],
            },
            wgsl: WgslExpect::Translates {
                contains: &["uv . x * 0.5 + 0.25"],
            },
        },
        Fixture {
            name: "parens_precedence",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ Vec4 :: new ( ( uv . x + uv . y ) * 0.5 , uv . x + uv . y * 0.5 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FMUL, OP_FADD],
            },
            msl: MslExpect::Accept {
                contains: &["float4("],
            },
            wgsl: WgslExpect::Translates {
                contains: &["( uv . x + uv . y ) * 0.5"],
            },
        },
        Fixture {
            name: "unary_minus",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ Vec4 :: new ( - uv . x + 1.0 , abs ( - 0.5 ) , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_F_NEGATE],
            },
            msl: MslExpect::Accept {
                contains: &["abs("],
            },
            wgsl: WgslExpect::Translates {
                contains: &["- uv . x + 1.0", "abs ( - 0.5 )"],
            },
        },
        Fixture {
            name: "expr_if",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ let c = if uv . x > 0.5 { 1.0 } else { 0.0 } ; Vec4 :: new ( c , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real { witness: &[OP_PHI] },
            msl: MslExpect::Accept {
                contains: &["if (uv.x > 0.5)", "} else {"],
            },
            // `let c = if …` becomes `return let c = …;` — a Rust `let`
            // binding verbatim inside a WGSL return.
            wgsl: WgslExpect::KnownBroken {
                residue: &["return let c ="],
            },
        },
        Fixture {
            name: "expr_if_nested",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: BAND_EXPR_BODY,
            alt: None,
            spirv: SpirvExpect::Real { witness: &[OP_PHI] },
            msl: MslExpect::Accept {
                contains: &["if (uv.x < 0.25)"],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["return let c ="],
            },
        },
        Fixture {
            name: "cmp_all",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ let a = if uv . x > 0.5 { 1.0 } else { 0.0 } ; \
                   let b = if uv . y < 0.5 { 1.0 } else { 0.0 } ; \
                   let c = if uv . x >= uv . y { 1.0 } else { 0.0 } ; \
                   let d = if uv . y <= uv . x { 0.5 } else { 0.0 } ; \
                   Vec4 :: new ( a , b , c * d , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FORD_GREATER_THAN],
            },
            msl: MslExpect::Accept {
                contains: &["uv.x > 0.5", "uv.y < 0.5", "uv.x >= uv.y", "uv.y <= uv.x"],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["return let a ="],
            },
        },
        // ── STATEMENTS ────────────────────────────────────────────────────
        Fixture {
            name: "let_chain",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ let a = uv . x ; let b = a * 2.0 ; let c = b - a ; Vec4 :: new ( c , a , b , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FMUL, OP_FSUB],
            },
            msl: MslExpect::Accept {
                contains: &["auto a = uv.x;", "auto b = a * 2.0;"],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["return let a ="],
            },
        },
        Fixture {
            name: "let_mut_reassign",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ let mut v = uv . x ; v = v * 2.0 ; v = v + 0.1 ; Vec4 :: new ( v , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FMUL, OP_FADD],
            },
            msl: MslExpect::Accept {
                contains: &["v = v * 2.0;", "v = v + 0.1;"],
            },
            // `let mut ` → `var `, but the trailing statements and the final
            // constructor still land inside one `return`.
            wgsl: WgslExpect::KnownBroken {
                residue: &["return var v ="],
            },
        },
        Fixture {
            name: "stmt_if_assign",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ let mut c = 0.0 ; if uv . x > 0.5 { c = 1.0 ; } else { c = 0.25 ; } \
                   Vec4 :: new ( c , 0.0 , 0.0 , 1.0 ) }",
            alt: Some(
                "{ let mut c = 0.0; if uv.x > 0.5 { c = 1.0; } else { c = 0.25; } \
                 Vec4::new(c, 0.0, 0.0, 1.0) }",
            ),
            spirv: SpirvExpect::Real { witness: &[OP_PHI] },
            msl: MslExpect::Accept {
                contains: &["if (uv.x > 0.5)", "c = 1.0;", "c = 0.25;"],
            },
            // Statement-`if` passes through verbatim — a Rust `if` block
            // inside a WGSL `return`.
            wgsl: WgslExpect::KnownBroken {
                residue: &["if uv . x > 0.5 {", "return var c ="],
            },
        },
        Fixture {
            name: "stmt_if_nested",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: BAND_STMT_BODY,
            alt: None,
            spirv: SpirvExpect::Real { witness: &[OP_PHI] },
            msl: MslExpect::Accept {
                contains: &["if (uv.x < 0.25)"],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["if uv . x < 0.25 {"],
            },
        },
        Fixture {
            name: "stmt_if_then_expr",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            // The image-fragment shape: a statement-if reassigns, then a
            // final tail expression is the return value (TailMode::Route).
            body: "{ let mut g = 0.25 ; if uv . x > 0.5 { g = 0.75 ; } else { g = 0.1 ; } \
                   Vec4 :: new ( uv . x , g , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real { witness: &[OP_PHI] },
            msl: MslExpect::Accept {
                contains: &["if (uv.x > 0.5)", "return float4(uv.x, g, 0.0, 1.0);"],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["if uv . x > 0.5 {"],
            },
        },
        // ── UNIFORMS + DEREF ──────────────────────────────────────────────
        Fixture {
            name: "uniform_deref_star",
            stage: Vertex,
            params: &[
                ("pos", Vec3, false),
                ("uv", Vec2, false),
                ("shift", Vec2, true),
            ],
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
            // The uniform `shift` is dropped from the WGSL entirely, and the
            // deref-`*` leaks: `( * shift ) . x` is not WGSL, and `shift` is
            // undeclared.
            wgsl: WgslExpect::KnownBroken {
                residue: &["( * shift ) . x"],
            },
        },
        Fixture {
            name: "uniform_frag_tint",
            stage: Fragment,
            params: &[("uv", Vec2, false), ("tint", Vec4, true)],
            body: "{ Vec4 :: new ( tint . x , tint . y , tint . z , 1.0 ) }",
            alt: None,
            // Reading `tint.x` is OpLoad + OpCompositeExtract — the passthrough
            // shape — and the uniform-setup OpAccessChain is present either way,
            // so no opcode distinguishes real from passthrough. Witness waived.
            spirv: SpirvExpect::Real { witness: &[] },
            msl: MslExpect::Accept {
                contains: &["float4(tint.x, tint.y, tint.z, 1.0)"],
            },
            // `tint` is a uniform → dropped from the stage inputs, so the
            // body references an undeclared `tint`.
            wgsl: WgslExpect::KnownBroken {
                residue: &["tint . x"],
            },
        },
        Fixture {
            name: "uniform_mat4_mul",
            stage: Vertex,
            params: &[
                ("pos", Vec3, false),
                ("uv", Vec2, false),
                ("mvp", Mat4, true),
            ],
            body: "{ mvp * Vec4 :: new ( pos . x , pos . y , pos . z , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_MATRIX_TIMES_VECTOR],
            },
            msl: MslExpect::Accept {
                contains: &["mvp * float4(pos.x, pos.y, pos.z, 1.0)"],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["mvp * vec4<f32>"],
            },
        },
        Fixture {
            name: "uniform_two_frag",
            stage: Fragment,
            params: &[
                ("uv", Vec2, false),
                ("base", Vec4, true),
                ("gain", Vec4, true),
            ],
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
            wgsl: WgslExpect::KnownBroken {
                residue: &["mix ( base . x , gain . x , 0.25 )"],
            },
        },
        Fixture {
            name: "uniform_mixed_order",
            stage: Fragment,
            // Two non-uniform params interleaved with two uniforms: proves
            // uniform binding is the index among UNIFORMS, not among all
            // params, on both emitters.
            params: &[
                ("a", Vec2, false),
                ("u1", Vec2, true),
                ("b", Vec2, false),
                ("u2", Vec4, true),
            ],
            body: "{ Vec4 :: new ( a . x + u1 . x , b . y + u2 . y , u2 . z , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FADD],
            },
            msl: MslExpect::Accept {
                contains: &["u1", "u2"],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["u1 . x"],
            },
        },
        // ── SWIZZLE + CTOR ────────────────────────────────────────────────
        Fixture {
            name: "swizzle_multi",
            stage: Fragment,
            params: &[("v", Vec4, false)],
            body: "{ let s = v . zw ; Vec4 :: new ( s . x , s . y , v . xy . x , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_VECTOR_SHUFFLE],
            },
            msl: MslExpect::Accept {
                contains: &["auto s = v.zw;"],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["return let s ="],
            },
        },
        Fixture {
            name: "swizzle_on_ctor",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ Vec4 :: new ( Vec2 :: new ( uv . y , uv . x ) . x , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            // `.x` on a Vec2 constructor is a single OpCompositeExtract — the
            // same op a passthrough uses — and the body has no arithmetic,
            // intrinsic, or multi-component swizzle. Nothing distinguishes
            // real from passthrough by opcode, so the witness is waived (Ok
            // emission is the assertion, like lit_solid).
            spirv: SpirvExpect::Real { witness: &[] },
            msl: MslExpect::Accept {
                contains: &["float2(uv.y, uv.x).x"],
            },
            // The stub aliases `Vec2 :: new(` but not the spaced-paren
            // `Vec2 :: new (`, so the inner constructor leaks untranslated.
            wgsl: WgslExpect::KnownBroken {
                residue: &["Vec2 :: new ("],
            },
        },
        Fixture {
            name: "ctor_nested",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ Vec4 :: new ( mix ( uv . x , uv . y , 0.5 ) , min ( uv . x , uv . y ) , \
                   max ( uv . x , uv . y ) , 1.0 ) }",
            alt: Some("{ Vec4::new(mix(uv.x, uv.y, 0.5), min(uv.x, uv.y), max(uv.x, uv.y), 1.0) }"),
            spirv: SpirvExpect::Real {
                witness: &[OP_EXT_INST],
            },
            msl: MslExpect::Accept {
                contains: &["mix(uv.x, uv.y, 0.5)", "min(uv.x, uv.y)", "max(uv.x, uv.y)"],
            },
            wgsl: WgslExpect::Translates {
                contains: &["mix ( uv . x , uv . y , 0.5 )"],
            },
        },
        Fixture {
            name: "ctor_trailing_comma",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            // The exact artifact a rustfmt-wrapped multi-line call ships: the
            // token printer preserves the trailing comma (`1.0,)`).
            body: "{ Vec4 :: new(uv . x * 0.5 , 0.25 , 0.0 , 1.0 ,) }",
            alt: None,
            // DIVERGENCE (allowlisted): after the fourth argument the SPIR-V
            // constructor parser expects `)` and leaves the trailing `,`
            // unconsumed, failing the body ("unexpected token: Comma") →
            // passthrough. See KNOWN_DIVERGENCES.
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Accept {
                contains: &["float4(uv.x * 0.5, 0.25, 0.0, 1.0)"],
            },
            // The stub replaces the constructor name and passes the trailing
            // comma through — legal in WGSL argument lists.
            wgsl: WgslExpect::Translates {
                contains: &["1.0 ,)"],
            },
        },
        Fixture {
            name: "ctor_wrapped_split",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            // The token printer wraps long lines and can break between
            // `Vec4 ::` and `new(...)`. Both native emitters must treat the
            // newline as ordinary whitespace.
            body: "{ Vec4 ::\nnew(uv . x * 0.5 , 0.0 , 0.0 , 1.0) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FMUL],
            },
            msl: MslExpect::Accept {
                contains: &["float4(uv.x * 0.5, 0.0, 0.0, 1.0)"],
            },
            // The stub's replace patterns require a single space in
            // `Vec4 :: new(`, so the newline-split constructor leaks
            // verbatim into the WGSL.
            wgsl: WgslExpect::KnownBroken {
                residue: &["Vec4 ::"],
            },
        },
        // ── INTRINSICS ────────────────────────────────────────────────────
        Fixture {
            name: "intrinsics_core",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ let a = sqrt ( uv . x + 0.1 ) ; let b = pow ( uv . y , 2.0 ) ; \
                   let c = floor ( uv . x * 4.0 ) + fract ( uv . y ) ; \
                   let d = clamp ( uv . x , 0.0 , 1.0 ) * abs ( uv . y - 0.5 ) ; \
                   Vec4 :: new ( a , b , c * 0.25 + d , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_EXT_INST],
            },
            msl: MslExpect::Accept {
                contains: &["sqrt(", "pow(", "floor(", "fract(", "clamp(", "abs("],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["return let a ="],
            },
        },
        Fixture {
            name: "intrinsics_geo",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ let d = dot ( uv , uv ) ; let l = length ( uv ) ; \
                   let n = normalize ( Vec2 :: new ( uv . x + 0.001 , uv . y ) ) . x ; \
                   Vec4 :: new ( d , l , n , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_EXT_INST],
            },
            msl: MslExpect::Accept {
                contains: &["dot(", "length(", "normalize("],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["return let d ="],
            },
        },
        Fixture {
            name: "intrinsics_map",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ let r = inverse_sqrt ( uv . x + 1.0 ) ; Vec4 :: new ( r , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_EXT_INST],
            },
            // inverse_sqrt aliases to MSL rsqrt.
            msl: MslExpect::Accept {
                contains: &["rsqrt("],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["return let r ="],
            },
        },
        Fixture {
            name: "deriv_frag",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ let w = fwidth ( uv . x ) ; \
                   Vec4 :: new ( w * 100.0 , dpdx ( uv . y ) * 100.0 + 0.5 , dpdy ( uv . x ) * 100.0 + 0.5 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FWIDTH, OP_DPDX],
            },
            // dpdx/dpdy alias to MSL dfdx/dfdy; fwidth stays fwidth.
            msl: MslExpect::Accept {
                contains: &["fwidth(", "dfdx(", "dfdy("],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["return let w ="],
            },
        },
        Fixture {
            name: "smoothstep_band",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: CIRCLE_SDF_BODY,
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_EXT_INST, OP_FWIDTH],
            },
            msl: MslExpect::Accept {
                contains: &["smoothstep(", "fwidth(", "length("],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["return let dx ="],
            },
        },
        // ── SAMPLING ──────────────────────────────────────────────────────
        Fixture {
            name: "sample_basic",
            stage: Fragment,
            // `sample(0` stays contiguous: both native emitters detect the
            // texture slot by the literal substring `sample(N` (naive scan,
            // not tokenized), which is the wire form the macro ships.
            params: &[("uv", Vec2, false)],
            body: "{ sample(0 , uv ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_IMAGE_SAMPLE_IMPLICIT_LOD],
            },
            msl: MslExpect::Accept {
                contains: &["tex_0.sample("],
            },
            // `sample(` has no WGSL translation in the stub; it passes
            // through raw.
            wgsl: WgslExpect::KnownBroken {
                residue: &["sample(0 , uv )"],
            },
        },
        Fixture {
            name: "sample_swizzle",
            stage: Fragment,
            params: &[("uv", Vec2, false), ("tint", Vec4, true)],
            body: "{ let c = sample(0 , uv ) . x ; \
                   Vec4 :: new ( c * tint . x , c * tint . y , c * tint . z , 1.0 ) }",
            alt: Some(
                "{ let c = sample(0, uv).x; \
                 Vec4::new(c * tint.x, c * tint.y, c * tint.z, 1.0) }",
            ),
            spirv: SpirvExpect::Real {
                witness: &[OP_IMAGE_SAMPLE_IMPLICIT_LOD],
            },
            msl: MslExpect::Accept {
                contains: &["tex_0.sample(", ".x"],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["return let c ="],
            },
        },
        Fixture {
            name: "sample_in_ctor",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ Vec4 :: new ( sample(0 , uv ) . x , sample(0 , uv ) . y , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_IMAGE_SAMPLE_IMPLICIT_LOD],
            },
            msl: MslExpect::Accept {
                contains: &["tex_0.sample("],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["sample(0 , uv ) . x"],
            },
        },
        // ── REAL DIJA BODIES (verbatim shape from emit_msl/shader_tests.rs) ─
        Fixture {
            name: "dija_rect_frag",
            stage: Fragment,
            params: &[
                ("corner", Vec2, false),
                ("size", Vec2, false),
                ("color", Vec4, false),
                ("border_color", Vec4, false),
                ("corner_radii", Vec4, false),
                ("border_width", F32, false),
                ("shape_type", F32, false),
            ],
            body: DIJA_RECT_FRAG_BODY,
            alt: None,
            // DIVERGENCE (allowlisted): the SPIR-V expression-`if` parser does
            // not accept `let` bindings inside a branch block, so this body's
            // `let dist = if … { let nx = …; … } else { … }` fails translation
            // and the real rect fragment falls to a passthrough on Vulkan.
            // MSL's syn walker handles branch-local lets. See KNOWN_DIVERGENCES.
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Accept {
                contains: &["smoothstep(", "length(float2("],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["return let px ="],
            },
        },
        Fixture {
            name: "dija_glyph_frag",
            stage: Fragment,
            params: &[
                ("corner", Vec2, false),
                ("uv_rect", Vec4, false),
                ("color", Vec4, false),
            ],
            body: DIJA_GLYPH_FRAG_BODY,
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_IMAGE_SAMPLE_IMPLICIT_LOD],
            },
            msl: MslExpect::Accept {
                contains: &["tex_0.sample(", "mix("],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["return let u ="],
            },
        },
        Fixture {
            name: "dija_image_frag",
            stage: Fragment,
            params: &[
                ("corner", Vec2, false),
                ("size", Vec2, false),
                ("uv_rect", Vec4, false),
                ("opacity", F32, false),
                ("border_radius", F32, false),
            ],
            body: DIJA_IMAGE_FRAG_BODY,
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_IMAGE_SAMPLE_IMPLICIT_LOD, OP_PHI],
            },
            msl: MslExpect::Accept {
                contains: &["tex_0.sample(", "if (border_radius > 0.0) {"],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["return let u ="],
            },
        },
        Fixture {
            name: "dija_rect_vert",
            stage: Vertex,
            params: &[
                ("pos", Vec2, false),
                ("corner", Vec2, false),
                ("size", Vec2, false),
                ("color", Vec4, false),
                ("border_color", Vec4, false),
                ("corner_radii", Vec4, false),
                ("border_width", F32, false),
                ("shape_type", F32, false),
                ("viewport", Vec2, true),
            ],
            body: DIJA_RECT_VERT_BODY,
            alt: None,
            spirv: SpirvExpect::Real {
                witness: &[OP_FMUL, OP_FSUB],
            },
            msl: MslExpect::Accept {
                contains: &["viewport"],
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["let pos_result = let px ="],
            },
        },
        // ── REJECTIONS ────────────────────────────────────────────────────
        Fixture {
            name: "rej_method_call",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ Vec4 :: new ( uv . length ( ) , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "method",
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["uv . length ( )"],
            },
        },
        Fixture {
            name: "rej_for_loop",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ let mut a = 0.0 ; for i in 0 .. 4 { a = a + 1.0 ; } Vec4 :: new ( a , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "for",
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["for i in 0 .. 4"],
            },
        },
        Fixture {
            name: "rej_unknown_intrinsic",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ Vec4 :: new ( bogus_fn ( uv . x ) , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "bogus_fn",
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["bogus_fn ( uv . x )"],
            },
        },
        Fixture {
            name: "rej_if_no_else",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            // An EXPRESSION-if without else. (A statement-if without else is
            // handled asymmetrically — SPIR-V translates it, MSL rejects it —
            // so it is deliberately NOT a fixture here; that asymmetry is a
            // reported finding, not a parity row.)
            body: "{ let c = if uv . x > 0.5 { 1.0 } ; Vec4 :: new ( c , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "else",
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["return let c ="],
            },
        },
        Fixture {
            name: "rej_out_of_range_component",
            stage: Fragment,
            params: &[("v", Vec4, false)],
            body: "{ Vec4 :: new ( v . q , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject { err_contains: "q" },
            wgsl: WgslExpect::KnownBroken {
                residue: &["v . q"],
            },
        },
        Fixture {
            name: "rej_while_loop",
            stage: Fragment,
            params: &[("uv", Vec2, false)],
            body: "{ let mut a = 0.0 ; while a < 4.0 { a = a + 1.0 ; } Vec4 :: new ( a , 0.0 , 0.0 , 1.0 ) }",
            alt: None,
            spirv: SpirvExpect::Passthrough,
            msl: MslExpect::Reject {
                err_contains: "while",
            },
            wgsl: WgslExpect::KnownBroken {
                residue: &["while a < 4.0"],
            },
        },
    ]
}

// ─── Long fixture bodies (token-spaced wire form) ────────────────────────────

/// 4-band expression-if on uv.x (mirrors draw D1). Constant per column.
const BAND_EXPR_BODY: &str = "{ let c = if uv . x < 0.25 { Vec4 :: new ( 1.0 , 0.0 , 0.0 , 1.0 ) } \
    else { if uv . x < 0.5 { Vec4 :: new ( 0.0 , 1.0 , 0.0 , 1.0 ) } \
    else { if uv . x < 0.75 { Vec4 :: new ( 0.0 , 0.0 , 1.0 , 1.0 ) } \
    else { Vec4 :: new ( 1.0 , 1.0 , 1.0 , 1.0 ) } } } ; c }";

/// 4-band statement-if on uv.x with a reassigned `let mut` (mirrors D2).
const BAND_STMT_BODY: &str = "{ let mut c = Vec4 :: new ( 1.0 , 1.0 , 1.0 , 1.0 ) ; \
    if uv . x < 0.25 { c = Vec4 :: new ( 1.0 , 0.0 , 0.0 , 1.0 ) ; } \
    else { if uv . x < 0.5 { c = Vec4 :: new ( 0.0 , 1.0 , 0.0 , 1.0 ) ; } \
    else { if uv . x < 0.75 { c = Vec4 :: new ( 0.0 , 0.0 , 1.0 , 1.0 ) ; } else { } } } c }";

/// Circle SDF around uv center with an fwidth-scaled smoothstep band
/// (mirrors draw D5). Symmetric → orientation-free.
const CIRCLE_SDF_BODY: &str = "{ let dx = uv . x - 0.5 ; let dy = uv . y - 0.5 ; \
    let d = length ( Vec2 :: new ( dx , dy ) ) - 0.3 ; let w = fwidth ( d ) ; \
    let a = 1.0 - smoothstep ( - w , w , d ) ; Vec4 :: new ( 1.0 , 1.0 , 1.0 , a ) }";

// The dija bodies mirror the real UI shaders in `emit_msl/shader_tests.rs`
// (rect/glyph/image fragment + rect vertex), in the token-spaced wire form.
// They are the real dija render shaders and must translate on both natives.
const DIJA_RECT_FRAG_BODY: &str = "{ let px = corner . x * size . x ; let py = corner . y * size . y ; \
    let half_x = size . x * 0.5 ; let half_y = size . y * 0.5 ; let cpx = px - half_x ; let cpy = py - half_y ; \
    let dist = if shape_type > 0.5 { let nx = cpx / half_x ; let ny = cpy / half_y ; \
    let d = length ( Vec2 :: new ( nx , ny ) ) - 1.0 ; d * min ( half_x , half_y ) } \
    else { let r_right = if cpy > 0.0 { corner_radii . z } else { corner_radii . y } ; \
    let r_left = if cpy > 0.0 { corner_radii . w } else { corner_radii . x } ; \
    let r0 = if cpx > 0.0 { r_right } else { r_left } ; let r = min ( r0 , min ( half_x , half_y ) ) ; \
    let qx = abs ( cpx ) - half_x + r ; let qy = abs ( cpy ) - half_y + r ; \
    let outside = length ( Vec2 :: new ( max ( qx , 0.0 ) , max ( qy , 0.0 ) ) ) ; \
    min ( max ( qx , qy ) , 0.0 ) + outside - r } ; \
    let shape_alpha = 1.0 - smoothstep ( - 0.75 , 0.75 , dist ) ; \
    let fill = if border_width > 0.0 { let inner_alpha = 1.0 - smoothstep ( - 0.75 , 0.75 , dist + border_width ) ; \
    Vec4 :: new ( mix ( border_color . x , color . x , inner_alpha ) , mix ( border_color . y , color . y , inner_alpha ) , \
    mix ( border_color . z , color . z , inner_alpha ) , mix ( border_color . w , color . w , inner_alpha ) ) } \
    else { color } ; \
    Vec4 :: new ( fill . x * shape_alpha , fill . y * shape_alpha , fill . z * shape_alpha , fill . w * shape_alpha ) }";

const DIJA_GLYPH_FRAG_BODY: &str = "{ let u = mix ( uv_rect . x , uv_rect . z , corner . x ) ; \
    let v = mix ( uv_rect . y , uv_rect . w , corner . y ) ; \
    let coverage = sample(0 , Vec2 :: new ( u , v ) ) . x ; \
    Vec4 :: new ( color . x * coverage , color . y * coverage , color . z * coverage , color . w * coverage ) }";

const DIJA_IMAGE_FRAG_BODY: &str = "{ let u = mix ( uv_rect . x , uv_rect . z , corner . x ) ; \
    let v = mix ( uv_rect . y , uv_rect . w , corner . y ) ; \
    let sampled = sample(0 , Vec2 :: new ( u , v ) ) ; \
    let color = Vec4 :: new ( sampled . x * opacity , sampled . y * opacity , sampled . z * opacity , sampled . w * opacity ) ; \
    if border_radius > 0.0 { let px = corner . x * size . x ; let py = corner . y * size . y ; \
    let half_x = size . x * 0.5 ; let half_y = size . y * 0.5 ; let r = min ( border_radius , min ( half_x , half_y ) ) ; \
    let cpx = px - half_x ; let cpy = py - half_y ; let qx = abs ( cpx ) - half_x + r ; let qy = abs ( cpy ) - half_y + r ; \
    let outside = length ( Vec2 :: new ( max ( qx , 0.0 ) , max ( qy , 0.0 ) ) ) ; \
    let dist = min ( max ( qx , qy ) , 0.0 ) + outside - r ; let mask = 1.0 - smoothstep ( - 0.75 , 0.75 , dist ) ; \
    Vec4 :: new ( color . x * mask , color . y * mask , color . z * mask , color . w * mask ) } else { color } }";

const DIJA_RECT_VERT_BODY: &str = "{ let px = pos . x + corner . x * size . x ; \
    let py = pos . y + corner . y * size . y ; let ndc_x = px / viewport . x * 2.0 - 1.0 ; \
    let ndc_y = 1.0 - py / viewport . y * 2.0 ; Vec4 :: new ( ndc_x , ndc_y , 0.0 , 1.0 ) }";

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

/// Every fixture's REAL SPIR-V translation must pass spirv-val. Passthrough
/// modules are validated too, but leniently: the passthrough fallback emitter
/// currently produces spirv-val-invalid modules (a duplicate-id / ID-bound
/// defect independent of the parity contract), so a passthrough rejection is
/// reported as a notice rather than failing the suite. A real translation
/// that fails to validate IS a hard failure. Skips if spirv-val is absent.
#[test]
fn spirv_validates() {
    let Some(tool) = spirv_val_path() else {
        eprintln!("SKIP spirv_validates: spirv-val not found");
        return;
    };

    let mut failures = Vec::new();
    for f in fixtures() {
        let d = def(&f);
        let spirv = match emit_spirv(&d) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("[{}] SPIR-V emission errored: {e}", f.name));
                continue;
            }
        };
        let real = has_real_work(&spirv) || matches!(f.spirv, SpirvExpect::Real { witness: &[] });
        match (spirv_val(tool, &spirv), real) {
            (Ok(()), _) => {}
            (Err(e), true) => failures.push(format!(
                "[{}] spirv-val rejected a REAL module:\n{e}",
                f.name
            )),
            (Err(e), false) => eprintln!(
                "NOTE [{}] passthrough module fails spirv-val (known passthrough-emitter \
                 defect, not a parity break): {}",
                f.name,
                e.lines().next().unwrap_or("").trim()
            ),
        }
    }
    assert!(
        failures.is_empty(),
        "{} spirv-val failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Per-fixture WGSL honesty assertions: `Translates` fixtures must contain
/// their substrings; `KnownBroken` fixtures must still exhibit their residue.
#[test]
fn wgsl_verdicts_hold() {
    let mut failures = Vec::new();
    for f in fixtures() {
        let d = def(&f);
        let wgsl = match emit_wgsl(&d) {
            Ok(w) => w,
            Err(e) => {
                // The stub is documented as always-Ok; an Err is a finding.
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
            WgslExpect::KnownBroken { residue } => {
                for needle in *residue {
                    if !wgsl.contains(needle) {
                        failures.push(format!(
                            "[{}] WGSL KnownBroken residue {needle:?} is gone — emit_wgsl may \
                             have been fixed; flip this fixture to Translates.\n--- WGSL ---\n{wgsl}",
                            f.name
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} WGSL verdict failure(s):\n{}",
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
