//! Cutover bridge: ItemFn + struct-ref analysis → KernelDef via the
//! WASM route.
//!
//! The pipeline:
//! 1. Render the kernel as a `pub unsafe extern "C" fn` Rust source
//!    string with struct-ref accesses flattened to raw-pointer ops
//!    (`d.field[i]` → `*field.add(i as usize)`, `d.scalar` → `scalar`).
//!    Uses `crate::quanta::intrinsics::*` to match the wrapper crate
//!    `wasm_compile` builds.
//! 2. Hand that source to `wasm_compile::compile_kernel_to_wasm` which
//!    runs `rustc --target wasm32-unknown-unknown --crate-type=cdylib`
//!    and caches the resulting WASM bytes.
//! 3. Build a `quanta_wasm_lowering::SideTable` from the kernel-
//!    signature analysis output (slot ordering + read/write/indexed
//!    bits) plus caller-supplied scalar types.
//! 4. Call `quanta_wasm_lowering::lower_module` to translate WASM →
//!    `KernelDef`.
//!
//! What this module does NOT solve: deciding the per-field scalar
//! types. Today the legacy parser walks the kernel body to infer them
//! (e.g., body sees `d.scale * 0.5f32` → `scale: f32`). The cutover
//! plan still depends on that inference layer; this bridge takes
//! scalar types as an explicit caller input. Slice 5d will figure out
//! how to keep the inference machinery alive while replacing the
//! KernelOp emission with the WASM route.

#![allow(dead_code)]

use proc_macro2::{Span, TokenStream};
use quanta_ir::{KernelDef, ScalarType};
use quanta_wasm_lowering::{ParamKind, ParamSlot, SideTable, lower};
use quote::{format_ident, quote};
use syn::visit_mut::{self, VisitMut};
use syn::{Expr, ExprField, ExprIndex, ExprPath, ItemFn, Member};

use crate::kernel_signature::{StructFieldAccess, StructRefParam};
use crate::wasm_compile::compile_kernel_to_wasm;

/// Inputs needed to drive the WASM route on a struct-ref kernel.
pub(crate) struct StructRefKernelInputs<'a> {
    /// The original kernel `fn item` as written by the user.
    pub func: &'a ItemFn,
    /// Result of `kernel_signature::detect_struct_ref_param`.
    pub struct_ref: &'a StructRefParam,
    /// Result of `kernel_signature::scan_struct_field_accesses`,
    /// **with `scalar_type_name` filled in** by the caller. We don't
    /// run scalar-type inference here — see module docstring.
    pub field_accesses: Vec<StructFieldAccess>,
    /// Workgroup dims; carried verbatim into the SideTable so the
    /// lowering pass can place it on the resulting KernelDef.
    pub workgroup_size: [u32; 3],
}

/// Drive the full WASM-route pipeline for a struct-ref kernel.
pub(crate) fn compile_struct_ref_kernel_via_wasm(
    inputs: &StructRefKernelInputs<'_>,
) -> Result<KernelDef, String> {
    let source = emit_kernel_source(inputs)?;
    let wasm_bytes = compile_kernel_to_wasm(&source)?;
    let side_table = build_side_table(inputs);
    lower(&wasm_bytes, &side_table).map_err(|e| format!("WASM lowering failed: {e}"))
}

/// Emit the wasm-compilable extern "C" source for a struct-ref kernel.
/// Public for testing — callers normally go through
/// `compile_struct_ref_kernel_via_wasm`.
pub(crate) fn emit_kernel_source(inputs: &StructRefKernelInputs<'_>) -> Result<String, String> {
    let kernel_name = &inputs.func.sig.ident;

    let (mut buffer_fields, mut scalar_fields): (Vec<&StructFieldAccess>, Vec<&StructFieldAccess>) =
        inputs.field_accesses.iter().partition(|f| f.is_indexed);
    buffer_fields.sort_by_key(|f| f.slot);
    scalar_fields.sort_by_key(|f| f.slot);

    let mut params = Vec::new();
    for f in &buffer_fields {
        let ident = format_ident!("{}", f.name);
        let ty = scalar_to_rust_ty(&f.scalar_type_name)?;
        if f.is_written {
            params.push(quote! { #ident: *mut #ty });
        } else {
            params.push(quote! { #ident: *const #ty });
        }
    }
    for f in &scalar_fields {
        let ident = format_ident!("{}", f.name);
        let ty = scalar_to_rust_ty(&f.scalar_type_name)?;
        params.push(quote! { #ident: #ty });
    }

    // Rewrite body: `d.field[idx]` → `*field.add(idx as usize)`,
    // `d.scalar` → `scalar`. Same algorithm as `wasm_twin` but using
    // the local intrinsics path.
    let mut rewriter = StructRefRewriter {
        param_name: inputs.struct_ref.param_name.clone(),
        buffer_field_names: buffer_fields.iter().map(|f| f.name.clone()).collect(),
        scalar_field_names: scalar_fields.iter().map(|f| f.name.clone()).collect(),
    };
    let mut body_block = inputs.func.block.clone();
    rewriter.visit_block_mut(&mut body_block);

    let stream: TokenStream = quote! {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #kernel_name(#(#params),*) {
            #[allow(unused_imports)]
            use crate::quanta::intrinsics::*;
            unsafe { #body_block }
        }
    };
    Ok(stream.to_string())
}

fn build_side_table(inputs: &StructRefKernelInputs<'_>) -> SideTable {
    let (mut buffer_fields, mut scalar_fields): (Vec<&StructFieldAccess>, Vec<&StructFieldAccess>) =
        inputs.field_accesses.iter().partition(|f| f.is_indexed);
    buffer_fields.sort_by_key(|f| f.slot);
    scalar_fields.sort_by_key(|f| f.slot);

    let mut params = Vec::new();
    let mut wasm_index: u32 = 0;
    for f in &buffer_fields {
        let kind = if f.is_written {
            ParamKind::BufferWrite
        } else {
            ParamKind::BufferRead
        };
        params.push(ParamSlot {
            wasm_index,
            slot: f.slot as u32,
            kind,
            scalar: name_to_scalar_type(&f.scalar_type_name).unwrap_or(ScalarType::F32),
        });
        wasm_index += 1;
    }
    for f in &scalar_fields {
        params.push(ParamSlot {
            wasm_index,
            slot: f.slot as u32,
            kind: ParamKind::Scalar,
            scalar: name_to_scalar_type(&f.scalar_type_name).unwrap_or(ScalarType::U32),
        });
        wasm_index += 1;
    }

    SideTable {
        kernel_name: inputs.func.sig.ident.to_string(),
        params,
        workgroup_size: inputs.workgroup_size,
    }
}

fn scalar_to_rust_ty(name: &str) -> Result<TokenStream, String> {
    Ok(match name {
        "u8" => quote! { u8 },
        "u16" => quote! { u16 },
        "u32" => quote! { u32 },
        "u64" => quote! { u64 },
        "i8" => quote! { i8 },
        "i16" => quote! { i16 },
        "i32" => quote! { i32 },
        "i64" => quote! { i64 },
        // f16 lowers to f32 on wasm32 — rustc has no stable f16 on the
        // wasm32 target. Caller is responsible for any host-side
        // bit-pattern reinterpretation.
        "f16" => quote! { f32 },
        "f32" => quote! { f32 },
        "f64" => quote! { f64 },
        "bool" => quote! { bool },
        other => return Err(format!("unsupported scalar type for wasm-twin: {other}")),
    })
}

fn name_to_scalar_type(name: &str) -> Option<ScalarType> {
    Some(match name {
        "u8" => ScalarType::U8,
        "u16" => ScalarType::U16,
        "u32" => ScalarType::U32,
        "u64" => ScalarType::U64,
        "i8" => ScalarType::I8,
        "i16" => ScalarType::I16,
        "i32" => ScalarType::I32,
        "i64" => ScalarType::I64,
        "f16" => ScalarType::F16,
        "f32" => ScalarType::F32,
        "f64" => ScalarType::F64,
        "bool" => ScalarType::Bool,
        _ => return None,
    })
}

struct StructRefRewriter {
    param_name: String,
    buffer_field_names: Vec<String>,
    scalar_field_names: Vec<String>,
}

impl StructRefRewriter {
    fn is_buffer_field(&self, name: &str) -> bool {
        self.buffer_field_names.iter().any(|n| n == name)
    }
    fn is_scalar_field(&self, name: &str) -> bool {
        self.scalar_field_names.iter().any(|n| n == name)
    }
    fn extract_field(&self, expr: &Expr) -> Option<String> {
        let Expr::Field(ExprField {
            base,
            member: Member::Named(ident),
            ..
        }) = expr
        else {
            return None;
        };
        let Expr::Path(ExprPath { path, .. }) = base.as_ref() else {
            return None;
        };
        let seg = path.segments.last()?;
        if seg.ident != self.param_name.as_str() {
            return None;
        }
        Some(ident.to_string())
    }
}

impl VisitMut for StructRefRewriter {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        visit_mut::visit_expr_mut(self, expr);

        if let Expr::Index(ExprIndex {
            expr: base, index, ..
        }) = expr
            && let Some(field_name) = self.extract_field(base)
            && self.is_buffer_field(&field_name)
        {
            let ident = format_ident!("{}", field_name);
            let idx = index.clone();
            *expr = syn::parse_quote! { *(#ident.add((#idx) as usize)) };
            return;
        }

        if let Expr::Field(ExprField {
            member: Member::Named(_),
            ..
        }) = expr
            && let Some(field_name) = self.extract_field(expr)
            && self.is_scalar_field(&field_name)
        {
            let ident = format_ident!("{}", field_name);
            *expr = Expr::Path(syn::ExprPath {
                attrs: Vec::new(),
                qself: None,
                path: syn::Path::from(ident),
            });
        }
    }
}

#[allow(dead_code)]
fn _unused_span_anchor() -> Span {
    Span::call_site()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_signature::{detect_struct_ref_param, scan_struct_field_accesses};
    use quanta_ir::KernelOp;

    /// Mandelbrot kernel source — same as `examples/cookbook_mandelbrot.rs`
    /// but stripped to just the function so we can parse it standalone.
    const MANDELBROT_SRC: &str = r#"
        fn mandelbrot(d: &MandelbrotData) {
            let idx = quark_id();
            let px = idx % d.width;
            let py = idx / d.width;
            let x0 = (px as f32 / d.width as f32) * 3.5f32 - 2.5f32;
            let y0 = (py as f32 / d.height as f32) * 2.0f32 - 1.0f32;
            let mut x = 0.0f32;
            let mut y = 0.0f32;
            let mut iter = 0u32;
            while x * x + y * y <= 4.0f32 && iter < d.max_iter {
                let tmp = x * x - y * y + x0;
                y = 2.0f32 * x * y + y0;
                x = tmp;
                iter += 1u32;
            }
            d.output[idx] = iter;
        }
    "#;

    fn parse_mandelbrot() -> ItemFn {
        syn::parse_str::<ItemFn>(MANDELBROT_SRC).expect("parse Mandelbrot")
    }

    fn mandelbrot_inputs(func: &ItemFn) -> StructRefKernelInputs<'_> {
        let sr = detect_struct_ref_param(func).expect("Mandelbrot is struct-ref");
        let mut accesses = scan_struct_field_accesses(func, &sr.param_name);
        // The Mandelbrot struct: output: Vec<u32>, width/height/max_iter: u32.
        // Slice 5c takes scalar types as an explicit input (see module
        // docstring); slice 5d will plug in real inference.
        for a in accesses.iter_mut() {
            a.scalar_type_name = "u32".to_string();
        }
        // Leak the StructRefParam so the borrow lives for the test.
        let sr = Box::leak(Box::new(sr));
        StructRefKernelInputs {
            func: Box::leak(Box::new(func.clone())),
            struct_ref: sr,
            field_accesses: accesses,
            workgroup_size: [64, 1, 1],
        }
    }

    #[test]
    fn emits_kernel_source_with_local_intrinsics_path() {
        let func = parse_mandelbrot();
        let inputs = mandelbrot_inputs(&func);
        let source = emit_kernel_source(&inputs).expect("emit source");

        assert!(
            source.contains("crate :: quanta :: intrinsics"),
            "intrinsics path must point at the wasm_compile wrapper's local mod, got:\n{source}"
        );
        assert!(
            source.contains("extern \"C\" fn mandelbrot"),
            "kernel must be emitted as extern C fn, got:\n{source}"
        );
        // Buffer write rewrite must have landed.
        assert!(
            source.contains(". add"),
            "expected `output.add(idx as usize)` rewrite, got:\n{source}"
        );
    }

    #[test]
    fn builds_side_table_from_field_accesses() {
        let func = parse_mandelbrot();
        let inputs = mandelbrot_inputs(&func);
        let st = build_side_table(&inputs);
        assert_eq!(st.kernel_name, "mandelbrot");
        assert_eq!(st.workgroup_size, [64, 1, 1]);
        // 1 buffer (output) + 3 scalars (width, height, max_iter).
        assert_eq!(st.params.len(), 4);
        assert!(matches!(st.params[0].kind, ParamKind::BufferWrite));
        assert_eq!(st.params[0].scalar, ScalarType::U32);
        for p in &st.params[1..] {
            assert!(matches!(p.kind, ParamKind::Scalar));
            assert_eq!(p.scalar, ScalarType::U32);
        }
    }

    #[test]
    fn end_to_end_via_wasm_route() {
        if !crate::wasm_compile::wasm32_target_available() {
            eprintln!(
                "[compile_via_wasm] skipping end_to_end: wasm32-unknown-unknown not installed"
            );
            return;
        }
        let func = parse_mandelbrot();
        let inputs = mandelbrot_inputs(&func);
        let kernel_def = compile_struct_ref_kernel_via_wasm(&inputs)
            .expect("Mandelbrot should round-trip via the WASM route");

        assert_eq!(kernel_def.name, "mandelbrot");
        assert_eq!(kernel_def.params.len(), 4);

        let mut saw_quark_id = false;
        let mut saw_loop = false;
        let mut saw_store = false;
        for op in flatten_ops(&kernel_def.body) {
            if matches!(op, KernelOp::QuarkId { .. }) {
                saw_quark_id = true;
            }
            if matches!(op, KernelOp::Loop { .. }) {
                saw_loop = true;
            }
            if matches!(op, KernelOp::Store { .. }) {
                saw_store = true;
            }
        }
        assert!(saw_quark_id, "Mandelbrot must contain QuarkId");
        assert!(saw_loop, "Mandelbrot must contain Loop");
        assert!(saw_store, "Mandelbrot must contain Store");
    }

    fn flatten_ops(ops: &[KernelOp]) -> Vec<&KernelOp> {
        let mut out = Vec::new();
        for op in ops {
            out.push(op);
            match op {
                KernelOp::Loop { body, .. } => out.extend(flatten_ops(body)),
                KernelOp::Branch {
                    then_ops, else_ops, ..
                } => {
                    out.extend(flatten_ops(then_ops));
                    out.extend(flatten_ops(else_ops));
                }
                _ => {}
            }
        }
        out
    }
}
