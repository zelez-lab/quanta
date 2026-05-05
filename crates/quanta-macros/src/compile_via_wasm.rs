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
use quanta_ir::{KernelDef, KernelOp, ScalarType};
use quanta_wasm_lowering::{ParamKind, ParamSlot, SideTable, lower};
use quote::{format_ident, quote};
use syn::visit_mut::{self, VisitMut};
use syn::{Expr, ExprField, ExprIndex, ExprPath, ItemFn, Member};

use crate::kernel_signature::{StructFieldAccess, StructRefParam};
use crate::wasm_compile::compile_kernel_to_wasm;

use quanta_ir::KernelParam;
use std::collections::HashMap;
use syn::{Attribute, FnArg, Local, Pat, Stmt, Type};

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
    // Pre-pass: harvest `#[quanta::shared] let NAME: [TY; N];` decls
    // from the body so we can (a) strip them from the source we hand
    // to rustc and rewrite `NAME[idx]` accesses to extern calls, and
    // (b) inject `KernelOp::SharedDecl` ops at the head of the
    // resulting KernelDef body.
    let mut func = inputs.func.clone();
    let shared_decls = harvest_and_rewrite_shared(&mut func)?;
    let local_inputs = StructRefKernelInputs {
        func: &func,
        struct_ref: inputs.struct_ref,
        field_accesses: inputs.field_accesses.clone(),
        workgroup_size: inputs.workgroup_size,
    };

    let source = emit_kernel_source(&local_inputs)?;
    let wasm_bytes = compile_kernel_to_wasm(&source)?;
    let side_table = build_side_table(&local_inputs);
    let mut def =
        lower(&wasm_bytes, &side_table).map_err(|e| format!("WASM lowering failed: {e}"))?;
    prepend_shared_decls(&mut def, &shared_decls);
    Ok(def)
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

    // Drop f16 casts/type-annotations — see F16CastEliminator docs.
    F16CastEliminator.visit_block_mut(&mut body_block);

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

// ── Workgroup-shared memory ────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SharedDeclInfo {
    name: String,
    id: u32,
    ty: ScalarType,
    count: u32,
}

/// Walk the function body, collect every `#[quanta::shared] let NAME:
/// [TY; N];` declaration, strip those `let` statements from the body,
/// and rewrite every `NAME[idx]` access to a call to the matching
/// `shared_load_<ty>` / `shared_store_<ty>` extern. Returns the
/// collected declarations so the caller can prepend `KernelOp::SharedDecl`
/// ops to the lowered KernelDef body.
fn harvest_and_rewrite_shared(func: &mut ItemFn) -> Result<Vec<SharedDeclInfo>, String> {
    let mut decls: Vec<SharedDeclInfo> = Vec::new();
    let mut name_to_info: HashMap<String, (u32, ScalarType)> = HashMap::new();
    let mut next_id: u32 = 0;

    // Scan + strip in a single pass over the top-level statement list.
    func.block.stmts.retain_mut(|stmt| {
        if let Stmt::Local(local) = stmt
            && has_shared_attr(&local.attrs)
        {
            match parse_shared_decl(local) {
                Ok((name, ty, count)) => {
                    let id = next_id;
                    next_id += 1;
                    decls.push(SharedDeclInfo {
                        name: name.clone(),
                        id,
                        ty,
                        count,
                    });
                    name_to_info.insert(name, (id, ty));
                    return false;
                }
                Err(_e) => {
                    // Leave the statement in place so rustc errors with
                    // a clear span. The downstream wasm compile will
                    // surface the resulting rustc error to the user.
                    return true;
                }
            }
        }
        true
    });

    if !name_to_info.is_empty() {
        let mut rewriter = SharedAccessRewriter { name_to_info };
        rewriter.visit_block_mut(&mut func.block);
    }

    Ok(decls)
}

fn has_shared_attr(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| {
        let segs: Vec<String> = a
            .path()
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect();
        // Match `#[quanta::shared]` or `#[shared]`. We deliberately
        // don't try to handle `#[quanta::shared(dyn)]` here — dynamic
        // shared memory has its own IR op (`SharedDeclDyn`) and isn't
        // supported by the WASM route yet.
        matches!(segs.as_slice(), [a] if a == "shared")
            || matches!(segs.as_slice(), [a, b] if a == "quanta" && b == "shared")
    })
}

/// Parse a `let NAME: [TY; COUNT];` decl into its component pieces.
fn parse_shared_decl(local: &Local) -> Result<(String, ScalarType, u32), String> {
    // Pattern: either `Pat::Ident` or `Pat::Type { pat: Ident, ty }`.
    let (name, ty_ref) = match &local.pat {
        Pat::Type(pat_type) => {
            let name = match pat_type.pat.as_ref() {
                Pat::Ident(id) => id.ident.to_string(),
                _ => return Err("shared-memory variable must be a simple ident".into()),
            };
            (name, pat_type.ty.as_ref())
        }
        _ => {
            return Err(
                "shared-memory let must have an explicit type: `let NAME: [TY; N];`".into(),
            );
        }
    };
    let (scalar, count) = parse_array_type(ty_ref)?;
    Ok((name, scalar, count))
}

fn parse_array_type(ty: &Type) -> Result<(ScalarType, u32), String> {
    let arr = match ty {
        Type::Array(a) => a,
        _ => return Err("expected an array type `[TY; N]`".into()),
    };
    let elem_ty = match arr.elem.as_ref() {
        Type::Path(p) => p
            .path
            .segments
            .last()
            .ok_or("empty type path")?
            .ident
            .to_string(),
        _ => return Err("array element type must be a primitive name".into()),
    };
    let scalar = name_to_scalar_type(&elem_ty)
        .ok_or_else(|| format!("unsupported shared-memory element type: {elem_ty}"))?;
    let count = match &arr.len {
        Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Int(i) => i
                .base10_parse::<u32>()
                .map_err(|e| format!("array length must be u32: {e}"))?,
            _ => return Err("array length must be an integer literal".into()),
        },
        _ => return Err("array length must be a const integer literal".into()),
    };
    Ok((scalar, count))
}

struct SharedAccessRewriter {
    name_to_info: HashMap<String, (u32, ScalarType)>,
}

impl SharedAccessRewriter {
    fn intrinsic_call(&self, name: &str, args: TokenStream) -> Expr {
        let ident = format_ident!("{}", name);
        syn::parse_quote! { #ident( #args ) }
    }
}

impl VisitMut for SharedAccessRewriter {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        // Special-case `NAME[idx] = val` BEFORE recursing — otherwise
        // the recursion rewrites the LHS `NAME[idx]` into a load call
        // and we lose the chance to recognize it as a store.
        if let Expr::Assign(assign) = expr
            && let Expr::Index(ExprIndex {
                expr: base, index, ..
            }) = assign.left.as_ref()
            && let Expr::Path(ExprPath { path, .. }) = base.as_ref()
            && let Some(seg) = path.segments.last()
            && let Some((id, ty)) = self.name_to_info.get(&seg.ident.to_string()).copied()
        {
            let store_fn = format_ident!("{}", shared_store_fn_name(ty));
            let id_lit = id;
            let idx = index.clone();
            let mut val = assign.right.clone();
            // The RHS may itself contain shared-load reads — recurse
            // on it before splicing into the call expression.
            self.visit_expr_mut(&mut val);
            // The index expression too.
            let mut idx_expr: Expr = (*idx).clone();
            self.visit_expr_mut(&mut idx_expr);
            *expr = syn::parse_quote! {
                #store_fn(#id_lit, (#idx_expr) as u32, #val)
            };
            return;
        }

        // Recurse for everything else so nested shared reads get
        // rewritten before we look at the parent.
        visit_mut::visit_expr_mut(self, expr);

        // `NAME[idx]`: shared-slot read (encountered outside the
        // Assign-LHS context handled above).
        if let Expr::Index(ExprIndex {
            expr: base, index, ..
        }) = expr
            && let Expr::Path(ExprPath { path, .. }) = base.as_ref()
            && let Some(seg) = path.segments.last()
            && let Some((id, ty)) = self.name_to_info.get(&seg.ident.to_string()).copied()
        {
            let load_fn = format_ident!("{}", shared_load_fn_name(ty));
            let id_lit = id;
            let idx = index.clone();
            *expr = syn::parse_quote! {
                #load_fn(#id_lit, (#idx) as u32)
            };
        }

        let _ = SharedAccessRewriter::intrinsic_call;
    }
}

fn shared_load_fn_name(ty: ScalarType) -> &'static str {
    match ty {
        ScalarType::F32 => "shared_load_f32",
        ScalarType::U32 => "shared_load_u32",
        ScalarType::I32 => "shared_load_i32",
        // Other types fall back to f32 — the lowerer rejects with a
        // clear error if the kernel actually instantiates them.
        _ => "shared_load_f32",
    }
}

fn shared_store_fn_name(ty: ScalarType) -> &'static str {
    match ty {
        ScalarType::F32 => "shared_store_f32",
        ScalarType::U32 => "shared_store_u32",
        ScalarType::I32 => "shared_store_i32",
        _ => "shared_store_f32",
    }
}

/// Prepend `KernelOp::SharedDecl` ops to the lowered body so emitters
/// (Metal/SPIR-V/WGSL/MSL) know which shared arrays to declare. The
/// IR's `id` field matches the call-time slot constant we wove into
/// the rewritten body.
fn prepend_shared_decls(def: &mut KernelDef, decls: &[SharedDeclInfo]) {
    if decls.is_empty() {
        return;
    }
    let mut prefix: Vec<KernelOp> = decls
        .iter()
        .map(|d| KernelOp::SharedDecl {
            id: d.id,
            ty: d.ty,
            count: d.count,
        })
        .collect();
    prefix.append(&mut def.body);
    def.body = prefix;
}

// ── Flat-param kernels ─────────────────────────────────────────────────

/// Inputs needed to drive the WASM route on a flat-param kernel —
/// `fn k(a: &[f32], b: &mut [f32], n: u32)` shape.
pub(crate) struct FlatParamKernelInputs<'a> {
    /// The original kernel `fn item` as written by the user.
    pub func: &'a ItemFn,
    /// Typed kernel params (slot + scalar type), one per fn arg.
    /// Comes from the legacy parser; carries scalar-type inference
    /// the WASM lowerer needs to build a SideTable. Order MUST
    /// match `func.sig.inputs`.
    pub params: Vec<KernelParam>,
    /// Workgroup dims; carried verbatim into the SideTable.
    pub workgroup_size: [u32; 3],
}

/// Drive the full WASM-route pipeline for a flat-param kernel.
pub(crate) fn compile_flat_param_kernel_via_wasm(
    inputs: &FlatParamKernelInputs<'_>,
) -> Result<KernelDef, String> {
    let mut func = inputs.func.clone();
    let shared_decls = harvest_and_rewrite_shared(&mut func)?;
    let local_inputs = FlatParamKernelInputs {
        func: &func,
        params: inputs.params.clone(),
        workgroup_size: inputs.workgroup_size,
    };

    let source = emit_flat_param_source(&local_inputs)?;
    let wasm_bytes = compile_kernel_to_wasm(&source)?;
    let side_table = build_flat_side_table(&local_inputs);
    let mut def =
        lower(&wasm_bytes, &side_table).map_err(|e| format!("WASM lowering failed: {e}"))?;
    prepend_shared_decls(&mut def, &shared_decls);
    Ok(def)
}

fn emit_flat_param_source(inputs: &FlatParamKernelInputs<'_>) -> Result<String, String> {
    let kernel_name = &inputs.func.sig.ident;

    // Iterate fn args in declaration order. Slices become raw
    // pointers; scalars stay; textures are STRIPPED (they're bound
    // by slot at dispatch time, not passed at runtime — see
    // TextureCallRewriter below). We also collect each arg's name so
    // the body rewriter can recognize `a[i]` / `a[i] = …` patterns
    // and `texture_load_2d(t, x, y)` calls.
    let mut params_emitted = Vec::new();
    let mut slice_param_names = Vec::new();
    let mut texture_params: Vec<(String, u32, ScalarType, TextureKind)> = Vec::new();
    let arg_count = inputs.func.sig.inputs.len();
    for (i, arg) in inputs.func.sig.inputs.iter().enumerate() {
        let FnArg::Typed(pat_ty) = arg else {
            return Err("flat-param kernel cannot take `self`".into());
        };
        let name = match pat_ty.pat.as_ref() {
            Pat::Ident(id) => id.ident.to_string(),
            _ => return Err("flat-param kernel arg pattern must be a plain ident".into()),
        };
        let ident = format_ident!("{}", name);
        let p = inputs.params.get(i).ok_or_else(|| {
            format!(
                "flat-param input #{i} `{name}` has no matching KernelParam in scalar-type bridge"
            )
        })?;
        match p {
            KernelParam::Texture2DRead {
                slot, scalar_type, ..
            } => {
                texture_params.push((name, *slot, *scalar_type, TextureKind::Tex2DRead));
                continue;
            }
            KernelParam::Texture2DWrite {
                slot, scalar_type, ..
            } => {
                texture_params.push((name, *slot, *scalar_type, TextureKind::Tex2DWrite));
                continue;
            }
            KernelParam::Texture3DRead {
                slot, scalar_type, ..
            } => {
                texture_params.push((name, *slot, *scalar_type, TextureKind::Tex3DRead));
                continue;
            }
            _ => {}
        }
        let ty_str = scalar_type_to_short_name(scalar_type_of(p));
        let ty = scalar_to_rust_ty(ty_str)?;
        match (&*pat_ty.ty, p) {
            (Type::Reference(r), KernelParam::FieldRead { .. }) if is_slice(&r.elem) => {
                params_emitted.push(quote! { #ident: *const #ty });
                slice_param_names.push(name);
            }
            (Type::Reference(r), KernelParam::FieldWrite { .. }) if is_slice(&r.elem) => {
                params_emitted.push(quote! { #ident: *mut #ty });
                slice_param_names.push(name);
            }
            (_, KernelParam::Constant { .. }) => {
                params_emitted.push(quote! { #ident: #ty });
            }
            _ => {
                return Err(format!(
                    "flat-param arg `{name}` has an unsupported shape — expected `&[T]`, `&mut [T]`, `&Texture2D<T>`, `&mut Texture2D<T>`, `&Texture3D<T>`, or a scalar; KernelParam was {:?}",
                    p
                ));
            }
        }
    }

    // Append const-generic params as runtime scalars. The legacy
    // dispatch glue calls `wave.set_value(slot, VAL)` so they arrive
    // at the kernel as push-constants, identical in shape to a
    // regular scalar param. The wasm-twin must declare them as
    // explicit u32 args so rustc resolves the body's `VAL`
    // identifier — kernels written `fn k<const VAL: u32>(...)` use
    // VAL as a value.
    for (gi, generic) in inputs.func.sig.generics.params.iter().enumerate() {
        let cp = match generic {
            syn::GenericParam::Const(c) => c,
            _ => continue,
        };
        let ident = cp.ident.clone();
        let slot_idx = arg_count + gi;
        let p = inputs.params.get(slot_idx).ok_or_else(|| {
            format!(
                "const generic `{ident}` (#{slot_idx}) has no matching KernelParam in scalar-type bridge"
            )
        })?;
        let ty_str = scalar_type_to_short_name(scalar_type_of(p));
        let ty = scalar_to_rust_ty(ty_str)?;
        params_emitted.push(quote! { #ident: #ty });
    }

    // Rewrite body: `slice_name[idx]` → `*slice_name.add(idx as usize)`.
    let mut rewriter = FlatSliceRewriter {
        slice_names: slice_param_names,
    };
    let mut body_block = inputs.func.block.clone();
    rewriter.visit_block_mut(&mut body_block);

    // Drop f16 casts/type-annotations — see F16CastEliminator docs.
    F16CastEliminator.visit_block_mut(&mut body_block);

    // Rewrite texture API calls. `texture_load_2d(tex, x, y)` becomes
    // `texture_load_2d_f32(<slot>, x, y)`; the `tex` ident
    // disappears since it doesn't appear in the emitted WASM
    // signature anymore. Same shape for sample/write/3D variants.
    if !texture_params.is_empty() {
        let mut tex_rewriter = TextureCallRewriter {
            params: texture_params,
        };
        tex_rewriter.visit_block_mut(&mut body_block);
    }

    let stream: TokenStream = quote! {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #kernel_name(#(#params_emitted),*) {
            #[allow(unused_imports)]
            use crate::quanta::intrinsics::*;
            unsafe { #body_block }
        }
    };
    Ok(stream.to_string())
}

fn build_flat_side_table(inputs: &FlatParamKernelInputs<'_>) -> SideTable {
    let mut params = Vec::new();
    let mut wasm_index: u32 = 0;
    for p in inputs.params.iter() {
        let (slot, kind, scalar) = match p {
            KernelParam::FieldRead {
                slot, scalar_type, ..
            } => (*slot, ParamKind::BufferRead, *scalar_type),
            KernelParam::FieldWrite {
                slot, scalar_type, ..
            } => (*slot, ParamKind::BufferWrite, *scalar_type),
            KernelParam::Constant {
                slot, scalar_type, ..
            } => (*slot, ParamKind::Scalar, *scalar_type),
            // Textures are bound by slot at dispatch time, not passed
            // to the kernel as runtime args. They're stripped from the
            // emitted WASM signature, so they don't appear in the
            // SideTable either (lower_module's arity check enforces
            // SideTable.params.len() == WASM sig.params.len()).
            KernelParam::Texture2DRead { .. }
            | KernelParam::Texture2DWrite { .. }
            | KernelParam::Texture3DRead { .. } => continue,
        };
        params.push(ParamSlot {
            wasm_index,
            slot,
            kind,
            scalar,
        });
        wasm_index += 1;
    }
    SideTable {
        kernel_name: inputs.func.sig.ident.to_string(),
        params,
        workgroup_size: inputs.workgroup_size,
    }
}

fn scalar_type_of(p: &KernelParam) -> ScalarType {
    match p {
        KernelParam::FieldRead { scalar_type, .. }
        | KernelParam::FieldWrite { scalar_type, .. }
        | KernelParam::Constant { scalar_type, .. }
        | KernelParam::Texture2DRead { scalar_type, .. }
        | KernelParam::Texture2DWrite { scalar_type, .. }
        | KernelParam::Texture3DRead { scalar_type, .. } => *scalar_type,
    }
}

fn scalar_type_to_short_name(ty: ScalarType) -> &'static str {
    match ty {
        ScalarType::F16 => "f16",
        ScalarType::F32 => "f32",
        ScalarType::F64 => "f64",
        ScalarType::U8 => "u8",
        ScalarType::U16 => "u16",
        ScalarType::U32 => "u32",
        ScalarType::U64 => "u64",
        ScalarType::I8 => "i8",
        ScalarType::I16 => "i16",
        ScalarType::I32 => "i32",
        ScalarType::I64 => "i64",
        ScalarType::Bool => "bool",
    }
}

fn is_slice(ty: &Type) -> bool {
    matches!(ty, Type::Slice(_))
}

/// Rewrites `slice_name[idx]` → `*slice_name.add(idx as usize)` for
/// every slice param. Mirrors the struct-ref rewriter but matches on
/// a plain Path expression instead of `param.field`.
struct FlatSliceRewriter {
    slice_names: Vec<String>,
}

impl FlatSliceRewriter {
    fn is_slice_ident(&self, name: &str) -> bool {
        self.slice_names.iter().any(|n| n == name)
    }
}

impl VisitMut for FlatSliceRewriter {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        visit_mut::visit_expr_mut(self, expr);

        if let Expr::Index(ExprIndex {
            expr: base, index, ..
        }) = expr
            && let Expr::Path(ExprPath { path, .. }) = base.as_ref()
            && let Some(seg) = path.segments.last()
            && self.is_slice_ident(&seg.ident.to_string())
        {
            let ident = seg.ident.clone();
            let idx = index.clone();
            *expr = syn::parse_quote! { *(#ident.add((#idx) as usize)) };
        }
    }
}

/// Walk the kernel body and replace `expr as f16` casts and `f16`
/// type annotations with their f32 equivalents. `f16` is unstable
/// on stable rustc (issue #116909), so the wasm-twin source can't
/// use it directly — but kernels written for native compilation
/// commonly use `as f16` casts to model half-precision arithmetic
/// (input/output buffers stay f32; the f16 just rounds intermediate
/// values).
///
/// Today the WASM route downgrades f16 to f32 — kernels run at full
/// precision rather than half, sacrificing the precision-truncation
/// behavior. This is acceptable because (a) f16 kernels in the
/// workspace today (gpu_f16) only test that arithmetic stays within
/// f16-typical tolerance, which f32 trivially satisfies; (b) on-GPU
/// f16 codegen still happens via the legacy parser's typed
/// KernelOps until rustc stabilizes f16. Once rustc supports f16 in
/// stable, we can drop this rewrite and let f16 flow through end-to-
/// end.
struct F16CastEliminator;

impl F16CastEliminator {
    fn rewrite_type(ty: &mut Type) {
        if let Type::Path(p) = ty
            && let Some(seg) = p.path.segments.last_mut()
            && seg.ident == "f16"
        {
            seg.ident = format_ident!("f32");
        }
    }
}

impl VisitMut for F16CastEliminator {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        // `expr as f16` → `expr as f32`. Recurse first so the inner
        // expression has its own f16 casts dealt with.
        visit_mut::visit_expr_mut(self, expr);
        if let Expr::Cast(c) = expr {
            Self::rewrite_type(&mut c.ty);
        }
    }

    fn visit_local_mut(&mut self, local: &mut Local) {
        // `let x: f16 = …;` → `let x: f32 = …;`.
        if let Pat::Type(pat_ty) = &mut local.pat {
            Self::rewrite_type(&mut pat_ty.ty);
        }
        syn::visit_mut::visit_local_mut(self, local);
    }
}

#[derive(Copy, Clone, Debug)]
enum TextureKind {
    Tex2DRead,
    Tex2DWrite,
    Tex3DRead,
}

/// Rewrites texture API calls so the kernel source compiles without
/// the `Texture2D<T>` placeholder type. The kernel writes
/// `texture_load_2d(tex, x, y)` etc. with `tex` being a function arg
/// of texture-ref type. The flat-param emitter strips the texture
/// arg from the WASM signature (textures are bound by slot, not
/// passed at runtime); this rewriter likewise replaces every
/// `texture_<op>(<tex_arg>, …)` call with `texture_<op>_<ty>(<slot>,
/// …)`, where `<slot>` is a `u32` literal so the lowerer can lift it
/// into the IR's `texture: u32` field.
///
/// `params` carries `(arg_name, slot, scalar_type, kind)` for every
/// texture param of the kernel. Only the canonical free-function
/// call shapes are matched — method-style `tex.load(x, y)` is not
/// rewritten yet (kernels in the workspace today use the free-function
/// API exclusively).
struct TextureCallRewriter {
    params: Vec<(String, u32, ScalarType, TextureKind)>,
}

impl TextureCallRewriter {
    fn lookup<'a>(&'a self, name: &str) -> Option<&'a (String, u32, ScalarType, TextureKind)> {
        self.params.iter().find(|(n, _, _, _)| n == name)
    }
}

impl VisitMut for TextureCallRewriter {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        // Recurse first so nested patterns get rewritten before we
        // check the parent.
        visit_mut::visit_expr_mut(self, expr);

        let Expr::Call(call) = expr else {
            return;
        };
        let Expr::Path(ExprPath { path, .. }) = call.func.as_ref() else {
            return;
        };
        let Some(seg) = path.segments.last() else {
            return;
        };
        let fn_name = seg.ident.to_string();

        // Match the canonical free-function texture API:
        //   texture_load_2d   → texture_load_2d_<ty>
        //   texture_sample_2d → texture_sample_2d_<ty>
        //   texture_load_3d   → texture_load_3d_<ty>
        //   texture_write_2d  → texture_write_2d_<ty>
        let (suffix_kind, expects_kind): (&str, TextureKind) = match fn_name.as_str() {
            "texture_load_2d" => ("texture_load_2d", TextureKind::Tex2DRead),
            "texture_sample_2d" => ("texture_sample_2d", TextureKind::Tex2DRead),
            "texture_load_3d" => ("texture_load_3d", TextureKind::Tex3DRead),
            "texture_write_2d" => ("texture_write_2d", TextureKind::Tex2DWrite),
            _ => return,
        };
        // First arg must be the texture-param ident.
        let Some(first) = call.args.first() else {
            return;
        };
        let tex_name = match first {
            Expr::Path(ExprPath { path, .. }) => path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default(),
            // `&tex` reference form.
            Expr::Reference(r) => match r.expr.as_ref() {
                Expr::Path(ExprPath { path, .. }) => path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default(),
                _ => return,
            },
            _ => return,
        };
        let Some((_, slot, scalar_ty, kind)) = self.lookup(&tex_name) else {
            return;
        };
        // Sanity: the call shape must match the texture's declared
        // kind (read vs write). Mismatch leaves the call alone so
        // rustc surfaces a clear error.
        if !kinds_compatible(*kind, expects_kind) {
            return;
        }

        let suffix = scalar_type_to_short_name(*scalar_ty);
        let new_fn = format_ident!("{}_{}", suffix_kind, suffix);
        let slot_val = *slot;
        let rest_args: Vec<Expr> = call.args.iter().skip(1).cloned().collect();
        *expr = syn::parse_quote! {
            #new_fn(#slot_val, #(#rest_args),*)
        };
    }
}

fn kinds_compatible(decl: TextureKind, used: TextureKind) -> bool {
    matches!(
        (decl, used),
        (TextureKind::Tex2DRead, TextureKind::Tex2DRead)
            | (TextureKind::Tex2DWrite, TextureKind::Tex2DWrite)
            | (TextureKind::Tex3DRead, TextureKind::Tex3DRead)
            // A `Tex2DWrite` texture can also be read in some APIs;
            // accept Tex2DRead lookups against a Write-declared
            // texture as a permissive default.
            | (TextureKind::Tex2DWrite, TextureKind::Tex2DRead)
    )
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
