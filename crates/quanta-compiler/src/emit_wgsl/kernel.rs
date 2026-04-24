//! Main WGSL emit function and kernel setup.
//!
//! WGSL has no fast-math mode -- all float operations use strict IEEE 754 semantics.
//! This is a known limitation of the WebGPU spec.

use quanta_ir::*;
use std::collections::HashMap;

use super::helpers::translate_device_fn_to_wgsl;
use super::ops::emit_op;

pub fn emit(kernel: &KernelDef) -> Result<String, String> {
    let mut out = String::new();

    // Emit device helper functions (from inner fn definitions)
    for src in &kernel.device_sources {
        out.push_str(&translate_device_fn_to_wgsl(src));
        out.push('\n');
    }

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
                    scalar_type.wgsl_name()
                ));
                slot_names.insert(*slot, name.clone());
            }
            KernelParam::FieldWrite {
                name,
                slot,
                scalar_type,
            } => {
                out.push_str(&format!(
                    "@group(0) @binding({}) var<storage, read_write> {}: array<{}>;\n",
                    slot,
                    name,
                    scalar_type.wgsl_name()
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
            _ => {}
        }
    }

    out.push_str(&format!(
        "\n@compute @workgroup_size(64)\nfn {}(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let _quark_id = gid.x;\n",
        kernel.name));

    for op in &kernel.body {
        emit_op(&mut out, op, 1, &slot_names);
    }

    out.push_str("}\n");
    Ok(out)
}
