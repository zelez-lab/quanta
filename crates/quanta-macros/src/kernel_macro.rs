//! Implementation body for `#[quanta::kernel]`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::visit_mut::VisitMut;
use syn::{Expr, ItemFn, Lit, parse::Parser};

use crate::auto_dispatch;
use crate::compile_via_wasm::{
    FlatParamKernelInputs, StructRefKernelInputs, compile_flat_param_kernel_via_wasm,
    compile_struct_ref_kernel_via_wasm, emit_host_oracle_flat, emit_host_oracle_struct_ref,
};
use crate::compiler;
use crate::device_macro::QualifiedDeviceCallRewriter;
use crate::kernel_signature::{
    StructRefParam, detect_struct_ref_param, scan_struct_field_accesses,
};
use crate::kernel_type_inference::infer_kernel;
use crate::validate;

/// Outer `#[quanta::kernel]` entry. Walks the kernel body for
/// qualified device-fn calls (`quanta_rand::foo(...)`), rewrites
/// them to bare-name calls (`foo(...)`), and emits sibling
/// `<crate>::<fn>_src!()` macro invocations so the device-fn
/// source registers in this crate's macro process before the real
/// kernel work happens. The rewritten kernel is then re-emitted
/// with `#[quanta::__kernel_inner]`, the proc-macro that does the
/// existing kernel compilation work.
///
/// Two-pass design because:
/// - Auto-discovering the device-fn imports means emitting
///   `_src!()` invocations as siblings of the kernel item.
/// - Those `_src!()` invocations must expand BEFORE the kernel
///   compiles (it needs the device-fn source in the registry).
/// - But `#[quanta::kernel]` can't directly emit `<crate>::<fn>_src!()`
///   then continue with kernel work — it would recurse on itself.
///
/// Splitting into two attributes solves it: `kernel` does the
/// discovery + emit + delegate, `__kernel_inner` does the
/// compilation. Macro expansion proceeds outside-in, so the
/// `_src!()` siblings expand and register sources before
/// `__kernel_inner` fires.
pub(crate) fn expand_kernel(attr: TokenStream, mut func: ItemFn) -> TokenStream {
    // Phase 1: rewrite the body.
    let mut rewriter = QualifiedDeviceCallRewriter::new();
    rewriter.visit_block_mut(&mut func.block);

    // If nothing to do, fall through to the original path directly —
    // no point in the extra delegation hop.
    if rewriter.paths.is_empty() {
        return expand_kernel_core(attr, func);
    }

    // Emit one `<path>_src!();` per recorded qualified call. The
    // _src macros are at the crate root of the library that owns
    // the device fn (because `#[quanta::device]` emits them with
    // `#[macro_export]`). So we rewrite the path to keep the
    // crate name but append `_src` to the final segment.
    let src_invocations: Vec<TokenStream2> = rewriter
        .paths
        .into_iter()
        .map(|mut path| {
            let last_idx = path.segments.len() - 1;
            let last = &mut path.segments[last_idx];
            let new_name = format!("{}_src", last.ident);
            last.ident = syn::Ident::new(&new_name, last.ident.span());
            quote! { #path!(); }
        })
        .collect();

    // Re-emit the kernel with the inner attribute. Keep the original
    // attr args verbatim so workgroup/opt/jit settings transfer.
    let attr_ts: TokenStream2 = attr.into();
    let func_ts: TokenStream2 = quote! { #func };
    let expanded = quote! {
        #(#src_invocations)*

        #[::quanta::__kernel_inner(#attr_ts)]
        #func_ts
    };
    expanded.into()
}

/// Core implementation of the `#[quanta::kernel]` attribute macro,
/// called via the inner-attribute delegation in `expand_kernel`.
/// At this point the body has been rewritten to use bare-name
/// calls for any cross-crate device fns, and the matching
/// `_src!()` macros have already expanded and registered the
/// device-fn source in this crate's macro process.
pub(crate) fn expand_kernel_core(attr: TokenStream, func: ItemFn) -> TokenStream {
    let attr_str = attr.to_string();
    let is_jit = attr_str.contains("jit");
    let kernel_attrs = match parse_kernel_attrs(attr.clone()) {
        Ok(a) => a,
        Err(err) => return err.to_compile_error().into(),
    };

    if let Err(err) = validate::validate_kernel(&func) {
        return err.to_compile_error().into();
    }

    // Detect struct-ref parameter: single param typed as `p: &MyStruct`
    let struct_ref = detect_struct_ref_param(&func);

    let mut kernel_def = match infer_kernel(&func) {
        Ok(def) => def,
        Err(err) => return err.to_compile_error().into(),
    };
    kernel_def.opt_level = kernel_attrs.opt_level;
    kernel_def.workgroup_size = kernel_attrs.workgroup_size;
    kernel_def.subgroup_size = kernel_attrs.subgroup_size;

    // WASM-route cutover (complete). `infer_kernel` above produced
    // `kernel_def.params` with inferred scalar types (the focused
    // type-inference walker in `kernel_type_inference.rs` — all that
    // survives of the legacy parser; its body translator was deleted
    // in 754fd19). Those params bridge into the SideTable for the
    // WASM lowerer, and the body is derived from `rustc → wasm32 →
    // KernelOps`. Struct-ref and flat-param kernels dispatch to
    // their respective emitters inside `swap_body_via_wasm_route`.
    let host_oracle = match swap_body_via_wasm_route(&mut kernel_def, &func, struct_ref.as_ref()) {
        Ok(oracle) => oracle.unwrap_or_default(),
        Err(err) => {
            let msg = format!("WASM route failed: {err}");
            return syn::Error::new_spanned(&func.sig.ident, msg)
                .to_compile_error()
                .into();
        }
    };

    if is_jit {
        return emit_jit_kernel(&func, &kernel_def);
    }

    let outputs = match compiler::compile_kernel(&kernel_def) {
        Ok(outputs) => outputs,
        Err(err) => {
            let msg = format!("quanta compiler error: {}", err);
            return syn::Error::new_spanned(&func.sig.ident, msg)
                .to_compile_error()
                .into();
        }
    };

    let func_name = &func.sig.ident;

    // For struct-ref kernels, the wave function is named `{name}_wave` and
    // the auto-dispatch wrapper takes the original name.
    let wave_fn_name = if struct_ref.is_some() {
        format_ident!("{}_wave", func_name)
    } else {
        func_name.clone()
    };

    let binary_name = syn::Ident::new(
        &format!("{}_BINARY", func_name.to_string().to_uppercase()),
        func_name.span(),
    );

    let nvidia_expr = match &outputs.nvidia {
        Some(bytes) => {
            let lit = proc_macro2::Literal::byte_string(bytes);
            quote! { Some(#lit as &[u8]) }
        }
        None => quote! { None },
    };
    let amd_expr = match &outputs.amd {
        Some(bytes) => {
            let lit = proc_macro2::Literal::byte_string(bytes);
            quote! { Some(#lit as &[u8]) }
        }
        None => quote! { None },
    };
    let spirv_expr = match &outputs.spirv {
        Some(bytes) => {
            let lit = proc_macro2::Literal::byte_string(bytes);
            quote! { Some(#lit as &[u8]) }
        }
        None => quote! { None },
    };
    let metallib_expr = match &outputs.metallib {
        Some(bytes) => {
            let lit = proc_macro2::Literal::byte_string(bytes);
            quote! { Some(#lit as &[u8]) }
        }
        None => quote! { None },
    };
    let wgsl_expr = match &outputs.wgsl {
        Some(s) => quote! { Some(#s) },
        None => quote! { None },
    };

    let wg_x = kernel_attrs.workgroup_size[0];
    let wg_y = kernel_attrs.workgroup_size[1];
    let wg_z = kernel_attrs.workgroup_size[2];

    // Const generics: extract from the function signature and generate set_value calls.
    // For struct-ref kernels, the slot offset is the number of struct fields (not func params).
    let generics = &func.sig.generics;
    let mut const_setters = Vec::new();
    let num_field_params = kernel_def.params.len()
        - func
            .sig
            .generics
            .params
            .iter()
            .filter(|g| matches!(g, syn::GenericParam::Const(_)))
            .count();
    for (i, generic) in func.sig.generics.params.iter().enumerate() {
        if let syn::GenericParam::Const(cp) = generic {
            let ident = &cp.ident;
            let slot = (num_field_params + i) as u32;
            const_setters.push(quote! {
                wave.set_value(#slot, #ident as u32);
            });
        }
    }
    let const_generic_setters = quote! { #(#const_setters)* };

    let serialized_ir = quanta_ir::serialize_kernel(&kernel_def);
    let ir_lit = proc_macro2::Literal::byte_string(&serialized_ir);
    let ir_static_name = syn::Ident::new(
        &format!("__{}_IR", wave_fn_name.to_string().to_uppercase()),
        wave_fn_name.span(),
    );

    let wave_fn = quote! {
        pub static #binary_name: ::quanta::KernelBinary = ::quanta::KernelBinary {
            amd: #amd_expr,
            nvidia: #nvidia_expr,
            spirv: #spirv_expr,
            metallib: #metallib_expr,
            wgsl: #wgsl_expr,
        };

        // Embedded KernelDef IR — used as JIT fallback when the
        // device's vendor isn't in the precompiled binary table
        // (lavapipe, niche drivers, etc.).
        pub static #ir_static_name: &[u8] = #ir_lit;

        pub fn #wave_fn_name #generics (device: &::quanta::Gpu) -> Result<::quanta::Wave, ::quanta::QuantaError> {
            let mut wave = match #binary_name.for_vendor(device.caps().vendor) {
                Some(binary) => device.wave(binary)?,
                // No precompiled binary for this vendor — JIT-compile
                // from the embedded IR.
                None => device.wave_jit(#ir_static_name)?,
            };
            wave.workgroup_size = [#wg_x, #wg_y, #wg_z];
            #const_generic_setters
            Ok(wave)
        }
    };

    // For struct-ref kernels, also generate the auto-dispatch wrapper
    // and the wasm-twin (roadmap step 058 phase 1.2). The twin is a
    // `#[cfg(target_arch = "wasm32")] extern "C" fn` that rustc lowers
    // to wasm32 — the future WASM → KernelOps lowering pass consumes
    // it. Today nothing reads it; emitting it now keeps the kernel
    // surface honest (any kernel that can't be flattened to raw
    // pointers fails the build immediately) and gives step 2.2 working
    // input on day one.
    if let Some(sr) = struct_ref {
        let field_accesses = scan_struct_field_accesses(&func, &sr.param_name);
        let dispatch_info = build_dispatch_info(&sr, &field_accesses, &kernel_def);
        let dispatch_fn = auto_dispatch::emit_auto_dispatch(&func, &dispatch_info, &wave_fn_name);

        let expanded = quote! {
            #wave_fn
            #dispatch_fn
            #host_oracle
        };
        return expanded.into();
    }

    let expanded = quote! {
        #wave_fn
        #host_oracle
    };
    expanded.into()
}

/// Build the auto_dispatch::StructParamInfo from kernel_signature
/// outputs, filling in scalar_type_name from the inferred KernelDef.
fn build_dispatch_info(
    sr: &crate::kernel_signature::StructRefParam,
    field_accesses: &[crate::kernel_signature::StructFieldAccess],
    kernel_def: &quanta_ir::KernelDef,
) -> auto_dispatch::StructParamInfo {
    let fields = field_accesses
        .iter()
        .map(|fa| {
            // Look up the scalar type from the KernelDef params by slot
            let scalar_type_name = kernel_def
                .params
                .get(fa.slot)
                .map(|p| scalar_type_to_name(param_scalar_type(p)))
                .unwrap_or_else(|| "f32".to_string());

            auto_dispatch::StructFieldAccess {
                name: fa.name.clone(),
                slot: fa.slot,
                is_indexed: fa.is_indexed,
                is_read: fa.is_read,
                is_written: fa.is_written,
                scalar_type_name,
            }
        })
        .collect();

    auto_dispatch::StructParamInfo {
        param_name: sr.param_name.clone(),
        type_name: sr.type_name.clone(),
        type_tokens: sr.type_tokens.clone(),
        fields,
    }
}

/// Extract the ScalarType from any KernelParam variant.
fn param_scalar_type(p: &quanta_ir::KernelParam) -> quanta_ir::ScalarType {
    match p {
        quanta_ir::KernelParam::FieldRead { scalar_type, .. }
        | quanta_ir::KernelParam::FieldWrite { scalar_type, .. }
        | quanta_ir::KernelParam::Constant { scalar_type, .. }
        | quanta_ir::KernelParam::Texture2DRead { scalar_type, .. }
        | quanta_ir::KernelParam::Texture2DWrite { scalar_type, .. }
        | quanta_ir::KernelParam::Texture3DRead { scalar_type, .. } => *scalar_type,
    }
}

/// Convert a ScalarType to its Rust type name string.
fn scalar_type_to_name(ty: quanta_ir::ScalarType) -> String {
    match ty {
        quanta_ir::ScalarType::F16 => "f16",
        quanta_ir::ScalarType::BF16 => "bf16",
        quanta_ir::ScalarType::FP8E5M2 => "fp8e5m2",
        quanta_ir::ScalarType::FP8E4M3 => "fp8e4m3",
        quanta_ir::ScalarType::F32 => "f32",
        quanta_ir::ScalarType::F64 => "f64",
        quanta_ir::ScalarType::U8 => "u8",
        quanta_ir::ScalarType::U16 => "u16",
        quanta_ir::ScalarType::U32 => "u32",
        quanta_ir::ScalarType::U64 => "u64",
        quanta_ir::ScalarType::I8 => "i8",
        quanta_ir::ScalarType::I16 => "i16",
        quanta_ir::ScalarType::I32 => "i32",
        quanta_ir::ScalarType::I64 => "i64",
        quanta_ir::ScalarType::Bool => "bool",
    }
    .to_string()
}

/// Emit JIT kernel: serialize KernelDef and embed it, generate runtime
/// compilation function via `wave_jit`.
fn emit_jit_kernel(func: &ItemFn, kernel_def: &quanta_ir::KernelDef) -> TokenStream {
    let func_name = &func.sig.ident;
    let def_name = syn::Ident::new(
        &format!("{}_DEF", func_name.to_string().to_uppercase()),
        func_name.span(),
    );

    let serialized = quanta_ir::serialize_kernel(kernel_def);
    let def_lit = proc_macro2::Literal::byte_string(&serialized);

    let wg_x = kernel_def.workgroup_size[0];
    let wg_y = kernel_def.workgroup_size[1];
    let wg_z = kernel_def.workgroup_size[2];

    let expanded = quote! {
        pub static #def_name: &[u8] = #def_lit;

        pub fn #func_name(device: &::quanta::Gpu) -> Result<::quanta::Wave, ::quanta::QuantaError> {
            let mut wave = device.wave_jit(#def_name)?;
            wave.workgroup_size = [#wg_x, #wg_y, #wg_z];
            Ok(wave)
        }
    };

    expanded.into()
}

/// Parsed kernel attributes from `#[quanta::kernel(...)]`.
struct KernelAttrs {
    opt_level: u8,
    workgroup_size: [u32; 3],
    subgroup_size: Option<u32>,
}

impl Default for KernelAttrs {
    fn default() -> Self {
        Self {
            opt_level: 3,
            workgroup_size: [64, 1, 1],
            subgroup_size: None,
        }
    }
}

/// Parse kernel attributes: `opt = "O2"`, `workgroup = [16, 16, 1]`, `jit`.
///
/// Supports:
/// - `#[quanta::kernel]`                           -> defaults
/// - `#[quanta::kernel(opt = "O2")]`               -> opt only
/// - `#[quanta::kernel(workgroup = [256])]`        -> [256, 1, 1]
/// - `#[quanta::kernel(workgroup = [16, 16])]`     -> [16, 16, 1]
/// - `#[quanta::kernel(workgroup = [16, 16, 1])]`  -> explicit 3D
/// - `#[quanta::kernel(workgroup = [256], opt = "O2")]` -> both
// Recognised kernel-attribute names. Anything outside this set
// is a compile error. Typos like `workgroup_size` used to
// silently fall back to the default and produce kernels with
// the wrong thread count.
const KNOWN_KERNEL_ATTRS: &[&str] = &["opt", "workgroup", "subgroup", "jit"];

fn parse_kernel_attrs(attr: TokenStream) -> Result<KernelAttrs, syn::Error> {
    let mut attrs = KernelAttrs::default();

    if attr.is_empty() {
        return Ok(attrs);
    }

    // Try parsing as a punctuated list of name = value pairs.
    // We use syn to parse the token stream as comma-separated meta items.
    let parser = syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated;
    let parsed = match parser.parse(attr.clone()) {
        Ok(p) => p,
        Err(_) => {
            // Fall back: might be just `jit` or a single `opt = "O2"`.
            // Try single MetaNameValue parse for backward compat.
            if let Ok(nv) = syn::parse::<syn::MetaNameValue>(attr)
                && nv.path.is_ident("opt")
                && let Expr::Lit(expr_lit) = &nv.value
                && let Lit::Str(s) = &expr_lit.lit
            {
                attrs.opt_level = parse_opt_str(&s.value());
            }
            return Ok(attrs);
        }
    };

    for meta in &parsed {
        let path = meta.path();
        let ident = path.get_ident().map(|i| i.to_string()).unwrap_or_default();

        if !KNOWN_KERNEL_ATTRS.contains(&ident.as_str()) {
            let known = KNOWN_KERNEL_ATTRS.join(", ");
            return Err(syn::Error::new_spanned(
                path,
                format!(
                    "unknown kernel attribute `{ident}` — recognised attributes are: {known}.\n\
                     hint: workgroup size is `workgroup = [256]`, not `workgroup_size = [...]`."
                ),
            ));
        }

        match meta {
            syn::Meta::NameValue(nv) if nv.path.is_ident("opt") => {
                if let Expr::Lit(expr_lit) = &nv.value
                    && let Lit::Str(s) = &expr_lit.lit
                {
                    attrs.opt_level = parse_opt_str(&s.value());
                }
            }
            syn::Meta::NameValue(nv) if nv.path.is_ident("workgroup") => {
                if let Some(wg) = parse_workgroup_expr(&nv.value) {
                    attrs.workgroup_size = wg;
                }
            }
            syn::Meta::NameValue(nv) if nv.path.is_ident("subgroup") => {
                if let Expr::Lit(expr_lit) = &nv.value
                    && let Lit::Int(i) = &expr_lit.lit
                    && let Ok(v) = i.base10_parse::<u32>()
                {
                    attrs.subgroup_size = Some(v);
                }
            }
            // `Meta::Path` for bare flags like `jit`. Already
            // accepted by the KNOWN_KERNEL_ATTRS gate above; the
            // value side is consumed by the `attr_str.contains`
            // check in `expand_kernel_core`.
            _ => {}
        }
    }

    Ok(attrs)
}

fn parse_opt_str(s: &str) -> u8 {
    match s {
        "O0" | "0" => 0,
        "O1" | "1" => 1,
        "O2" | "2" => 2,
        "O3" | "3" => 3,
        _ => 3,
    }
}

/// Parse `[256]`, `[16, 16]`, or `[16, 16, 1]` from an expression.
fn parse_workgroup_expr(expr: &Expr) -> Option<[u32; 3]> {
    if let Expr::Array(arr) = expr {
        let elems: Vec<u32> = arr
            .elems
            .iter()
            .filter_map(|e| {
                if let Expr::Lit(lit) = e
                    && let Lit::Int(int_lit) = &lit.lit
                {
                    return int_lit.base10_parse::<u32>().ok();
                }
                None
            })
            .collect();
        match elems.len() {
            1 => return Some([elems[0], 1, 1]),
            2 => return Some([elems[0], elems[1], 1]),
            3 => return Some([elems[0], elems[1], elems[2]]),
            _ => {}
        }
    }
    None
}

/// Re-derive `kernel_def.body` (and `next_reg`) by routing the kernel
/// through `rustc → wasm32 → KernelOps`. The legacy parser already
/// produced `kernel_def.params` with inferred scalar types — those
/// bridge into the SideTable for the WASM lowerer. Dispatches to the
/// struct-ref or flat-param emitter based on the kernel signature.
fn swap_body_via_wasm_route(
    kernel_def: &mut quanta_ir::KernelDef,
    func: &ItemFn,
    sr: Option<&StructRefParam>,
) -> Result<Option<proc_macro2::TokenStream>, String> {
    let (wasm_def, host_oracle) = match sr {
        Some(sr) => {
            let mut accesses = scan_struct_field_accesses(func, &sr.param_name);
            for access in accesses.iter_mut() {
                let ty = kernel_def
                    .params
                    .iter()
                    .find(|p| param_slot(p) == access.slot as u32)
                    .map(param_scalar_type)
                    .ok_or_else(|| {
                        format!(
                            "no legacy KernelParam for slot {} (field `{}`); cannot \
                             bridge scalar type into WASM SideTable",
                            access.slot, access.name
                        )
                    })?;
                access.scalar_type_name = scalar_type_to_name(ty);
            }
            let inputs = StructRefKernelInputs {
                func,
                struct_ref: sr,
                field_accesses: accesses,
                workgroup_size: kernel_def.workgroup_size,
            };
            let def = compile_struct_ref_kernel_via_wasm(&inputs)?;
            let oracle = emit_host_oracle_struct_ref(&inputs)?;
            (def, oracle)
        }
        None => {
            let inputs = FlatParamKernelInputs {
                func,
                params: kernel_def.params.clone(),
                workgroup_size: kernel_def.workgroup_size,
            };
            let def = compile_flat_param_kernel_via_wasm(&inputs)?;
            let oracle = emit_host_oracle_flat(&inputs)?;
            (def, oracle)
        }
    };

    kernel_def.body = wasm_def.body;
    kernel_def.next_reg = wasm_def.next_reg;
    Ok(host_oracle)
}

fn param_slot(p: &quanta_ir::KernelParam) -> u32 {
    match p {
        quanta_ir::KernelParam::FieldRead { slot, .. }
        | quanta_ir::KernelParam::FieldWrite { slot, .. }
        | quanta_ir::KernelParam::Constant { slot, .. }
        | quanta_ir::KernelParam::Texture2DRead { slot, .. }
        | quanta_ir::KernelParam::Texture2DWrite { slot, .. }
        | quanta_ir::KernelParam::Texture3DRead { slot, .. } => *slot,
    }
}
