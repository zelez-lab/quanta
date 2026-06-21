//! Main WGSL emit function and kernel setup.
//!
//! WGSL has no fast-math mode — all float operations follow strict IEEE 754
//! semantics per the WebGPU spec.

use crate::*;
use std::collections::{HashMap, HashSet};

use super::helpers::translate_device_fn_to_wgsl;
use super::ops::emit_op;

/// Emit WGSL source from a [`KernelDef`].
///
/// The output is a complete WGSL module: `enable` directives (where
/// required), storage/uniform bindings, module-scope `var<workgroup>`
/// shared declarations, and the `@compute` entry point.
pub fn emit(kernel: &KernelDef) -> Result<String, String> {
    let mut out = String::new();

    // Enable directives — emit defensively for any feature the kernel uses.
    let needs_f16 = kernel_uses_f16(kernel);
    let needs_subgroups = kernel_uses_subgroups(kernel);
    if needs_f16 {
        out.push_str("enable f16;\n");
    }
    if needs_subgroups {
        out.push_str("enable subgroups;\n");
    }
    if needs_f16 || needs_subgroups {
        out.push('\n');
    }

    // fp8 conversion helpers (one set per format used at a Load/Store).
    for (eb, mb) in crate::dtype_codegen::kernel_fp8_formats(kernel) {
        out.push_str(&crate::dtype_codegen::wgsl_fp8_helpers(eb, mb));
        out.push('\n');
    }

    // Device helper functions — these come before atomics because some helpers
    // operate on plain values, not atomic-wrapped storage cells.
    for src in &kernel.device_sources {
        out.push_str(&translate_device_fn_to_wgsl(src));
        out.push('\n');
    }

    // Identify which fields are accessed atomically — those need to be wrapped
    // in `atomic<T>` in WGSL. The IR makes this distinction at the op level.
    let atomic_fields = collect_atomic_fields(&kernel.body);
    // Same for shared-memory slots: a slot touched by SharedAtomicOp
    // must be declared `array<atomic<T>>`, and its plain loads/stores
    // must go through atomicLoad/atomicStore (WGSL forbids bare
    // access to atomic<T> cells).
    let atomic_shared = collect_atomic_shared(&kernel.body);

    let mut slot_names: HashMap<u32, String> = HashMap::new();

    for param in &kernel.params {
        match param {
            KernelParam::FieldRead {
                name,
                slot,
                scalar_type,
            } => {
                out.push_str(&format!(
                    "@group(0) @binding({}) var<storage, read> {}: array<{}>;\n",
                    slot,
                    name,
                    scalar_type.wgsl_storage_name()
                ));
                slot_names.insert(*slot, name.clone());
            }
            KernelParam::FieldWrite {
                name,
                slot,
                scalar_type,
            } => {
                let elem = if atomic_fields.contains(slot) {
                    format!("atomic<{}>", scalar_type.wgsl_storage_name())
                } else {
                    scalar_type.wgsl_storage_name().to_string()
                };
                out.push_str(&format!(
                    "@group(0) @binding({}) var<storage, read_write> {}: array<{}>;\n",
                    slot, name, elem
                ));
                slot_names.insert(*slot, name.clone());
            }
            KernelParam::Constant {
                name,
                slot,
                scalar_type,
            } => {
                slot_names.insert(*slot, name.clone());
                out.push_str(&format!(
                    "@group(0) @binding({}) var<uniform> {}: {};\n",
                    slot,
                    name,
                    scalar_type.wgsl_name()
                ));
            }
            KernelParam::Texture2DRead {
                name,
                slot,
                scalar_type,
            } => {
                out.push_str(&format!(
                    "@group(0) @binding({}) var {}: texture_2d<{}>;\n",
                    slot,
                    name,
                    scalar_type.wgsl_name()
                ));
                slot_names.insert(*slot, name.clone());
            }
            KernelParam::Texture2DWrite {
                name,
                slot,
                scalar_type: _,
            } => {
                out.push_str(&format!(
                    "@group(0) @binding({}) var {}: texture_storage_2d<rgba8unorm, write>;\n",
                    slot, name,
                ));
                slot_names.insert(*slot, name.clone());
            }
            KernelParam::Texture3DRead {
                name,
                slot,
                scalar_type,
            } => {
                out.push_str(&format!(
                    "@group(0) @binding({}) var {}: texture_3d<{}>;\n",
                    slot,
                    name,
                    scalar_type.wgsl_name()
                ));
                slot_names.insert(*slot, name.clone());
            }
        }
    }

    // Module-scope shared memory declarations (WGSL requires `var<workgroup>`
    // at module scope, not inside a function).
    collect_shared_decls(&mut out, &kernel.body, &atomic_shared);

    // Compute entry point.
    let [wgx, wgy, wgz] = kernel.workgroup_size;
    out.push_str(&format!(
        "\n@compute @workgroup_size({}, {}, {})\nfn {}(\n",
        wgx, wgy, wgz, kernel.name
    ));
    out.push_str("    @builtin(global_invocation_id) gid: vec3<u32>,\n");
    out.push_str("    @builtin(local_invocation_id) lid: vec3<u32>,\n");
    out.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n");
    out.push_str("    @builtin(num_workgroups) ngroups: vec3<u32>,\n");
    out.push_str(") {\n");
    out.push_str("    let _quark_id = gid.x;\n");
    out.push_str("    let _proton_id = lid.x;\n");
    out.push_str("    let _nucleus_id = wid.x;\n");
    out.push_str(&format!(
        "    let _proton_size: u32 = {}u;\n",
        wgx * wgy * wgz
    ));
    out.push_str("    let _quark_count = ngroups.x * _proton_size;\n");

    for op in &kernel.body {
        emit_op(&mut out, op, 1, &slot_names, &atomic_fields, &atomic_shared);
    }

    out.push_str("}\n");
    Ok(out)
}

fn kernel_uses_f16(kernel: &KernelDef) -> bool {
    kernel.params.iter().any(|p| {
        matches!(
            p,
            KernelParam::FieldRead {
                scalar_type: ScalarType::F16,
                ..
            } | KernelParam::FieldWrite {
                scalar_type: ScalarType::F16,
                ..
            } | KernelParam::Constant {
                scalar_type: ScalarType::F16,
                ..
            }
        )
    }) || body_uses_f16(&kernel.body)
}

fn body_uses_f16(ops: &[KernelOp]) -> bool {
    ops.iter().any(|op| match op {
        KernelOp::Const {
            value: ConstValue::F16(_),
            ..
        } => true,
        KernelOp::Branch {
            then_ops, else_ops, ..
        } => body_uses_f16(then_ops) || body_uses_f16(else_ops),
        KernelOp::Loop { body, .. } => body_uses_f16(body),
        _ => false,
    })
}

fn kernel_uses_subgroups(kernel: &KernelDef) -> bool {
    body_uses_subgroups(&kernel.body)
}

fn body_uses_subgroups(ops: &[KernelOp]) -> bool {
    ops.iter().any(|op| {
        matches!(
            op,
            KernelOp::WaveShuffle { .. }
                | KernelOp::WaveBallot { .. }
                | KernelOp::WaveAny { .. }
                | KernelOp::WaveAll { .. }
                | KernelOp::SubgroupSize { .. }
                | KernelOp::SubgroupReduceAdd { .. }
                | KernelOp::SubgroupReduceMin { .. }
                | KernelOp::SubgroupReduceMax { .. }
                | KernelOp::SubgroupExclusiveAdd { .. }
                | KernelOp::SubgroupInclusiveAdd { .. }
        ) || match op {
            KernelOp::Branch {
                then_ops, else_ops, ..
            } => body_uses_subgroups(then_ops) || body_uses_subgroups(else_ops),
            KernelOp::Loop { body, .. } => body_uses_subgroups(body),
            _ => false,
        }
    })
}

/// Collect every field slot used in an Atomic{Op,Cas} op.
fn collect_atomic_fields(ops: &[KernelOp]) -> std::collections::HashSet<u32> {
    let mut acc = std::collections::HashSet::new();
    walk_atomic_fields(ops, &mut acc);
    acc
}

fn walk_atomic_fields(ops: &[KernelOp], acc: &mut std::collections::HashSet<u32>) {
    for op in ops {
        match op {
            KernelOp::AtomicOp { field, .. } | KernelOp::AtomicCas { field, .. } => {
                acc.insert(*field);
            }
            KernelOp::Branch {
                then_ops, else_ops, ..
            } => {
                walk_atomic_fields(then_ops, acc);
                walk_atomic_fields(else_ops, acc);
            }
            KernelOp::Loop { body, .. } => walk_atomic_fields(body, acc),
            _ => {}
        }
    }
}

/// Collect the shared-memory slots touched by `SharedAtomicOp` —
/// those declare as `array<atomic<T>>` and route every plain
/// load/store through atomicLoad/atomicStore.
fn collect_atomic_shared(ops: &[KernelOp]) -> HashSet<u32> {
    let mut acc = HashSet::new();
    walk_atomic_shared(ops, &mut acc);
    acc
}

fn walk_atomic_shared(ops: &[KernelOp], acc: &mut HashSet<u32>) {
    for op in ops {
        match op {
            KernelOp::SharedAtomicOp { slot, .. } => {
                acc.insert(*slot);
            }
            KernelOp::Branch {
                then_ops, else_ops, ..
            } => {
                walk_atomic_shared(then_ops, acc);
                walk_atomic_shared(else_ops, acc);
            }
            KernelOp::Loop { body, .. } => walk_atomic_shared(body, acc),
            _ => {}
        }
    }
}

fn collect_shared_decls(out: &mut String, ops: &[KernelOp], atomic_shared: &HashSet<u32>) {
    for op in ops {
        match op {
            KernelOp::SharedDecl { id, ty, count } => {
                let elem = if atomic_shared.contains(id) {
                    format!("atomic<{}>", ty.wgsl_name())
                } else {
                    ty.wgsl_name().to_string()
                };
                out.push_str(&format!(
                    "var<workgroup> shared_{}: array<{}, {}>;\n",
                    id, elem, count
                ));
            }
            KernelOp::SharedDeclDyn { id, ty } => {
                // Dynamic-sized workgroup memory is not natively expressible
                // in WGSL; emit a fixed-but-large array as a placeholder.
                // Step 050 will tune this against actual dispatch needs.
                let elem = if atomic_shared.contains(id) {
                    format!("atomic<{}>", ty.wgsl_name())
                } else {
                    ty.wgsl_name().to_string()
                };
                out.push_str(&format!(
                    "var<workgroup> shared_{}: array<{}, 1024>;\n",
                    id, elem
                ));
            }
            KernelOp::Branch {
                then_ops, else_ops, ..
            } => {
                collect_shared_decls(out, then_ops, atomic_shared);
                collect_shared_decls(out, else_ops, atomic_shared);
            }
            KernelOp::Loop { body, .. } => collect_shared_decls(out, body, atomic_shared),
            _ => {}
        }
    }
}
