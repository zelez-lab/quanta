//! Main emit() function — kernel signature, device function translation.

use crate::*;
use std::collections::HashMap;

use super::ops::emit_op;

pub fn emit(kernel: &KernelDef) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    // Emit device helper functions (from inner fn definitions)
    for src in &kernel.device_sources {
        out.push_str(&translate_device_fn_to_msl(src));
        out.push('\n');
    }

    // Kernel signature with max_total_threads_per_threadgroup attribute
    let max_threads =
        kernel.workgroup_size[0] * kernel.workgroup_size[1] * kernel.workgroup_size[2];
    out.push_str(&format!(
        "[[max_total_threads_per_threadgroup({})]]\nkernel void {}(\n",
        max_threads, kernel.name
    ));

    let mut param_lines = Vec::new();
    let mut slot_names: HashMap<u32, String> = HashMap::new();

    for param in &kernel.params {
        match param {
            KernelParam::FieldRead {
                name,
                slot,
                scalar_type,
            } => {
                param_lines.push(format!(
                    "    device const {}* {} [[buffer({})]]",
                    scalar_type.msl_name(),
                    name,
                    slot
                ));
                slot_names.insert(*slot, name.clone());
            }
            KernelParam::FieldWrite {
                name,
                slot,
                scalar_type,
            } => {
                param_lines.push(format!(
                    "    device {}* {} [[buffer({})]]",
                    scalar_type.msl_name(),
                    name,
                    slot
                ));
                slot_names.insert(*slot, name.clone());
            }
            KernelParam::Constant {
                name,
                slot,
                scalar_type,
            } => {
                param_lines.push(format!(
                    "    constant {}& {} [[buffer({})]]",
                    scalar_type.msl_name(),
                    name,
                    slot
                ));
                slot_names.insert(*slot, name.clone());
            }
            _ => {}
        }
    }
    // Check if kernel uses debug print — if so, add a debug buffer parameter
    let uses_debug_print = kernel
        .body
        .iter()
        .any(|op| matches!(op, KernelOp::DebugPrint { .. }));
    if uses_debug_print {
        param_lines.push("    device uint* _debug_buf [[buffer(30)]]".to_string());
    }

    param_lines.push("    uint _quark_id [[thread_position_in_grid]]".to_string());
    param_lines.push("    uint _proton_id [[thread_position_in_threadgroup]]".to_string());
    param_lines.push("    uint _nucleus_id [[threadgroup_position_in_grid]]".to_string());
    param_lines.push("    uint _proton_size [[threads_per_threadgroup]]".to_string());
    param_lines.push("    uint _simd_width [[threads_per_simdgroup]]".to_string());

    out.push_str(&param_lines.join(",\n"));
    out.push_str("\n) {\n");

    for op in &kernel.body {
        emit_op(&mut out, op, 1, &slot_names);
    }

    out.push_str("}\n");
    Ok(out)
}

/// Translate a Rust device function source to MSL.
///
/// Rewrites the function signature (return type, parameter types) and body
/// using string substitutions. This is the Phase 1 text-based approach;
/// Phase 2 will walk KernelOps for device function bodies too.
fn translate_device_fn_to_msl(rust_source: &str) -> String {
    // Map Rust return types to MSL return types. The `fn name(...) -> T` form
    // becomes `T name(...)` in MSL.
    let type_map: &[(&str, &str)] = &[
        ("f32", "float"),
        ("f64", "double"),
        ("u32", "uint"),
        ("u64", "ulong"),
        ("i32", "int"),
        ("i64", "long"),
        ("bool", "bool"),
    ];

    let mut s = rust_source.to_string();

    // Replace return type: "-> f32" -> "" (moved to front)
    let mut ret_msl = "void";
    for &(rust_ty, msl_ty) in type_map {
        let arrow = format!("-> {}", rust_ty);
        if s.contains(&arrow) {
            ret_msl = msl_ty;
            s = s.replace(&arrow, "");
            break;
        }
    }

    // Replace "fn name" with "inline <ret_type> name"
    if let Some(pos) = s.find("fn ") {
        s = format!("{}inline {} {}", &s[..pos], ret_msl, &s[pos + 3..]);
    }

    // Replace parameter types
    for &(rust_ty, msl_ty) in type_map {
        let param_pattern = format!(": {}", rust_ty);
        let param_replacement = format!(": {}", msl_ty);
        // Only replace parameter annotations (": type" patterns), not
        // occurrences inside the body. Since parameter annotations come before
        // the opening brace, this is safe with a simple replace.
        s = s.replace(&param_pattern, &param_replacement);
    }

    // Body translations
    s = s.replace("let mut ", "auto ");
    s = s.replace("let ", "auto ");
    s = s.replace(" as f32", "");
    s = s.replace(" as u32", "");
    s = s.replace(" as i32", "");

    s
}
