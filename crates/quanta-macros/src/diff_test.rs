//! Differential test: legacy syn-AST parser vs the new wasm32 lowering
//! route should agree on the kernels they both can handle.
//!
//! The legacy parser is `crate::parse::parse_kernel(&ItemFn) -> KernelDef`.
//! The new pipeline is `quanta_wasm_lowering::lower(wasm_bytes,
//! &SideTable) -> KernelDef`. The two op streams aren't byte-equal —
//! legacy assigns regs in source-walk order while wasm-lowering walks
//! the WASM stack machine — but they should agree on the observable
//! shape of the kernel: same name, same param shapes, same workgroup
//! size, and the same multiset of "structural" ops (Loads/Stores/BinOps
//! tagged by field/slot/scalar-type/op-kind, ignoring register numbers
//! and bookkeeping `Const` / `Cast` materialization).
//!
//! Once the WASM route is the only route, this test goes away — it
//! exists to prove the cutover doesn't regress kernel semantics.

use quanta_ir::{BinOp, KernelDef, KernelOp, KernelParam, ScalarType};
use quanta_wasm_lowering::{ParamKind, ParamSlot, SideTable, lower};
use syn::ItemFn;

use crate::parse::parse_kernel;

const VECTOR_ADD_WASM: &[u8] = include_bytes!("../../quanta-wasm-lowering/tests/hello_quanta.wasm");

/// Source for the kernel both pipelines lower. Mirrors
/// `examples/hello_quanta.rs` so the WASM and the syn-AST come from
/// the same Rust function.
const VECTOR_ADD_SRC: &str = r#"
fn vector_add(d: &VecAdd) {
    let i = quark_id();
    d.result[i] = d.a[i] + d.b[i];
}
"#;

fn vector_add_side_table() -> SideTable {
    SideTable {
        kernel_name: "vector_add".to_string(),
        params: vec![
            ParamSlot {
                wasm_index: 0,
                slot: 0,
                kind: ParamKind::BufferRead,
                scalar: ScalarType::F32,
            },
            ParamSlot {
                wasm_index: 1,
                slot: 1,
                kind: ParamKind::BufferRead,
                scalar: ScalarType::F32,
            },
            ParamSlot {
                wasm_index: 2,
                slot: 2,
                kind: ParamKind::BufferWrite,
                scalar: ScalarType::F32,
            },
        ],
        workgroup_size: [64, 1, 1],
    }
}

/// Reduce a `KernelParam` to (slot, kind-tag, scalar_type) — the
/// observable shape independent of the cosmetic field name. Legacy
/// uses the user's struct field name; wasm-lowering synthesises
/// `buf{slot}` from the side table.
fn param_shape(p: &KernelParam) -> (u32, u8, ScalarType) {
    match p {
        KernelParam::FieldRead {
            slot, scalar_type, ..
        } => (*slot, 0, *scalar_type),
        KernelParam::FieldWrite {
            slot, scalar_type, ..
        } => (*slot, 1, *scalar_type),
        KernelParam::Constant {
            slot, scalar_type, ..
        } => (*slot, 2, *scalar_type),
        KernelParam::Texture2DRead {
            slot, scalar_type, ..
        } => (*slot, 3, *scalar_type),
        KernelParam::Texture2DWrite {
            slot, scalar_type, ..
        } => (*slot, 4, *scalar_type),
        KernelParam::Texture3DRead {
            slot, scalar_type, ..
        } => (*slot, 5, *scalar_type),
    }
}

/// Map an op to a structural fingerprint string, ignoring register
/// numbers. Bookkeeping ops (`Const`, `Copy`, `Cast`) are filtered
/// out by the caller before fingerprints are zipped — they exist to
/// materialize values for the underlying stack machine and the two
/// pipelines emit them at different points.
fn op_fingerprint(op: &KernelOp) -> Option<String> {
    Some(match op {
        KernelOp::QuarkId { .. } => "QuarkId".to_string(),
        KernelOp::QuarkCount { .. } => "QuarkCount".to_string(),
        KernelOp::ProtonId { .. } => "ProtonId".to_string(),
        KernelOp::NucleusId { .. } => "NucleusId".to_string(),
        KernelOp::ProtonSize { .. } => "ProtonSize".to_string(),
        KernelOp::Load { field, ty, .. } => format!("Load(f={field},ty={ty:?})"),
        KernelOp::Store { field, ty, .. } => format!("Store(f={field},ty={ty:?})"),
        KernelOp::BinOp { op, ty, .. } => format!("BinOp({op:?},{ty:?})"),
        KernelOp::UnaryOp { op, ty, .. } => format!("UnaryOp({op:?},{ty:?})"),
        KernelOp::Cmp { op, ty, .. } => format!("Cmp({op:?},{ty:?})"),
        KernelOp::Branch { .. } => "Branch".to_string(),
        KernelOp::Loop { .. } => "Loop".to_string(),
        KernelOp::Barrier => "Barrier".to_string(),
        KernelOp::Break => "Break".to_string(),
        KernelOp::MathCall { func, ty, .. } => format!("MathCall({func:?},{ty:?})"),
        // Filtered: bookkeeping that one pipeline emits and the
        // other doesn't, but is functionally inert when isolated.
        KernelOp::Const { .. } | KernelOp::Copy { .. } | KernelOp::Cast { .. } => return None,
        // Anything else: include the variant name so a mismatch
        // surfaces with enough context to investigate.
        other => format!("{:?}", core::mem::discriminant(other)),
    })
}

fn body_fingerprint(body: &[KernelOp]) -> Vec<String> {
    body.iter().filter_map(op_fingerprint).collect()
}

fn parse_legacy(src: &str) -> KernelDef {
    let item: ItemFn = syn::parse_str(src).expect("parse Rust source");
    parse_kernel(&item).expect("legacy parse_kernel")
}

#[test]
fn legacy_and_wasm_agree_on_vector_add_param_shape() {
    let legacy = parse_legacy(VECTOR_ADD_SRC);
    let wasm =
        lower(VECTOR_ADD_WASM, &vector_add_side_table()).expect("wasm lowering of vector_add");

    assert_eq!(legacy.name, wasm.name, "kernel name divergence");
    assert_eq!(
        legacy.params.len(),
        wasm.params.len(),
        "param count divergence"
    );

    let mut legacy_shapes: Vec<_> = legacy.params.iter().map(param_shape).collect();
    let mut wasm_shapes: Vec<_> = wasm.params.iter().map(param_shape).collect();
    // ScalarType is not Ord, so sort by the slot/kind tuple alone —
    // each (slot, kind) pair is unique per param so the secondary
    // ordering on scalar_type isn't needed for canonicalisation.
    legacy_shapes.sort_by_key(|(slot, kind, _)| (*slot, *kind));
    wasm_shapes.sort_by_key(|(slot, kind, _)| (*slot, *kind));
    assert_eq!(
        legacy_shapes, wasm_shapes,
        "param (slot, kind, scalar_type) shape divergence"
    );
}

#[test]
fn legacy_and_wasm_agree_on_vector_add_body_fingerprint() {
    let legacy = parse_legacy(VECTOR_ADD_SRC);
    let wasm =
        lower(VECTOR_ADD_WASM, &vector_add_side_table()).expect("wasm lowering of vector_add");

    let legacy_fp = body_fingerprint(&legacy.body);
    let wasm_fp = body_fingerprint(&wasm.body);

    // Both pipelines must witness the canonical vector_add shape.
    let must_contain = [
        "QuarkId".to_string(),
        format!("Load(f=0,ty={:?})", ScalarType::F32),
        format!("Load(f=1,ty={:?})", ScalarType::F32),
        format!("Store(f=2,ty={:?})", ScalarType::F32),
        format!("BinOp({:?},{:?})", BinOp::Add, ScalarType::F32),
    ];

    for needle in &must_contain {
        assert!(
            legacy_fp.contains(needle),
            "legacy fingerprint missing `{needle}`: {legacy_fp:?}"
        );
        assert!(
            wasm_fp.contains(needle),
            "wasm fingerprint missing `{needle}`: {wasm_fp:?}"
        );
    }

    // Stronger: filtered fingerprints are equal as sequences. If this
    // fails, print both side-by-side so the divergence is obvious.
    assert_eq!(
        legacy_fp, wasm_fp,
        "body fingerprint divergence\n  legacy: {legacy_fp:?}\n  wasm:   {wasm_fp:?}",
    );
}
