//! Graphics pipeline creation for Metal.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::{Pipeline, QuantaError, SpecValue};

use super::super::ffi;
use super::super::{
    MetalDevice, blend_factor_to_metal, blend_op_to_metal, compare_to_metal, format_to_metal,
    stencil_op_to_metal,
};

impl MetalDevice {
    pub(crate) fn pipeline_create_impl(
        &self,
        desc: &crate::PipelineDesc,
    ) -> Result<Pipeline, QuantaError> {
        // Step 063 slice 5 — gate deferred render-pipeline features
        // up-front rather than silently dropping the request when
        // the render-pipeline integration hasn't been wired yet.
        // The typed wrappers (TessellationPipeline, MeshPipeline)
        // continue to exercise the Lean refinement contract via
        // their own create methods; setting these on the standard
        // PipelineDesc is what gets gated.
        if desc.tessellation.is_some() {
            return Err(QuantaError::not_supported(
                "Metal render pipelines: tessellation (drawIndexedPatches + tessellationFactorBuffer) integration pending — set PipelineDesc.tessellation = None or use the typed TessellationPipeline wrapper",
            ));
        }
        if desc.mesh_shader.is_some() {
            return Err(QuantaError::not_supported(
                "Metal render pipelines: MTLMeshRenderPipelineDescriptor integration pending — use dispatch_mesh on the typed MeshPipeline wrapper",
            ));
        }
        if desc.conservative_rasterization {
            return Err(QuantaError::not_supported(
                "Metal render pipelines: conservative rasterization is not exposed by Metal",
            ));
        }
        // Build MTLFunctionConstantValues if specialization constants are present.
        let fcv = if !desc.specialization.is_empty() {
            unsafe {
                let fcv = ffi::msg_id(
                    ffi::cls(b"MTLFunctionConstantValues\0") as ffi::Id,
                    b"new\0",
                );
                for (index, sc) in desc.specialization.iter().enumerate() {
                    match sc.value {
                        SpecValue::F32(v) => {
                            ffi::msg_set_constant_value(
                                fcv,
                                &v as *const f32 as *const _,
                                ffi::MTL_DATA_TYPE_FLOAT,
                                index as u64,
                            );
                        }
                        SpecValue::I32(v) => {
                            ffi::msg_set_constant_value(
                                fcv,
                                &v as *const i32 as *const _,
                                ffi::MTL_DATA_TYPE_INT,
                                index as u64,
                            );
                        }
                        SpecValue::U32(v) => {
                            ffi::msg_set_constant_value(
                                fcv,
                                &v as *const u32 as *const _,
                                ffi::MTL_DATA_TYPE_UINT,
                                index as u64,
                            );
                        }
                        SpecValue::Bool(v) => {
                            let b: u8 = if v { 1 } else { 0 };
                            ffi::msg_set_constant_value(
                                fcv,
                                &b as *const u8 as *const _,
                                ffi::MTL_DATA_TYPE_BOOL,
                                index as u64,
                            );
                        }
                    }
                }
                Some(fcv)
            }
        } else {
            None
        };

        // Compile shader source(s) into Metal library/libraries.
        let (vert_fn, frag_fn) = unsafe {
            if let Some(combined) = desc.source {
                let src = std::str::from_utf8(combined).map_err(|_| {
                    QuantaError::compilation_failed("invalid UTF-8 in shader source")
                })?;
                let mut src_bytes: Vec<u8> = src.bytes().collect();
                src_bytes.push(0);
                let ns_src = ffi::nsstring(&src_bytes);
                let (lib, error) = ffi::msg_new_library_with_source(self.device, ns_src, ffi::NIL);
                if lib.is_null() {
                    let msg = error_string(error);
                    return Err(QuantaError::compilation_failed(format!("shader: {msg}")));
                }
                let vf = get_function_maybe_specialized(lib, desc.vertex_entry, fcv)?;
                let ff = get_function_maybe_specialized(lib, desc.fragment_entry, fcv)?;
                (vf, ff)
            } else {
                // Load vertex shader: metallib binary (MTLB) or MSL text
                let vert_lib = if desc.vertex.len() >= 4 && &desc.vertex[..4] == b"MTLB" {
                    let (lib, error) = ffi::msg_new_library_with_data(
                        self.device,
                        desc.vertex.as_ptr() as *const _,
                        desc.vertex.len() as u64,
                    );
                    if lib.is_null() {
                        let msg = error_string(error);
                        return Err(QuantaError::compilation_failed(format!(
                            "vertex metallib: {msg}"
                        )));
                    }
                    lib
                } else {
                    let src = std::str::from_utf8(desc.vertex).map_err(|_| {
                        QuantaError::compilation_failed("invalid UTF-8 in vertex shader")
                    })?;
                    let mut vb: Vec<u8> = src.bytes().collect();
                    vb.push(0);
                    let ns_vert = ffi::nsstring(&vb);
                    let (lib, err) =
                        ffi::msg_new_library_with_source(self.device, ns_vert, ffi::NIL);
                    if lib.is_null() {
                        let msg = error_string(err);
                        return Err(QuantaError::compilation_failed(format!("vertex: {msg}")));
                    }
                    lib
                };

                // Load fragment shader: metallib binary (MTLB) or MSL text
                let frag_lib = if desc.fragment.len() >= 4 && &desc.fragment[..4] == b"MTLB" {
                    let (lib, error) = ffi::msg_new_library_with_data(
                        self.device,
                        desc.fragment.as_ptr() as *const _,
                        desc.fragment.len() as u64,
                    );
                    if lib.is_null() {
                        let msg = error_string(error);
                        return Err(QuantaError::compilation_failed(format!(
                            "fragment metallib: {msg}"
                        )));
                    }
                    lib
                } else {
                    let src = std::str::from_utf8(desc.fragment).map_err(|_| {
                        QuantaError::compilation_failed("invalid UTF-8 in fragment shader")
                    })?;
                    let mut fb: Vec<u8> = src.bytes().collect();
                    fb.push(0);
                    let ns_frag = ffi::nsstring(&fb);
                    let (lib, err) =
                        ffi::msg_new_library_with_source(self.device, ns_frag, ffi::NIL);
                    if lib.is_null() {
                        let msg = error_string(err);
                        return Err(QuantaError::compilation_failed(format!("fragment: {msg}")));
                    }
                    lib
                };

                let vf = get_function_maybe_specialized(vert_lib, desc.vertex_entry, fcv)?;
                let ff = get_function_maybe_specialized(frag_lib, desc.fragment_entry, fcv)?;
                (vf, ff)
            }
        };

        unsafe {
            let pipe_desc = ffi::msg_id(
                ffi::cls(b"MTLRenderPipelineDescriptor\0") as ffi::Id,
                b"new\0",
            );
            ffi::msg_void_id(pipe_desc, b"setVertexFunction:\0", vert_fn);
            ffi::msg_void_id(pipe_desc, b"setFragmentFunction:\0", frag_fn);

            // Color attachments
            let attachments = ffi::msg_id(pipe_desc, b"colorAttachments\0");
            for (i, fmt) in desc.color_formats.iter().enumerate() {
                let ca = ffi::msg_id_u64(attachments, b"objectAtIndexedSubscript:\0", i as u64);
                ffi::msg_void_u64(ca, b"setPixelFormat:\0", format_to_metal(*fmt));
                if desc.blend.enabled {
                    ffi::msg_void_bool(ca, b"setBlendingEnabled:\0", true);
                    ffi::msg_void_u64(
                        ca,
                        b"setSourceRGBBlendFactor:\0",
                        blend_factor_to_metal(desc.blend.src_rgb),
                    );
                    ffi::msg_void_u64(
                        ca,
                        b"setDestinationRGBBlendFactor:\0",
                        blend_factor_to_metal(desc.blend.dst_rgb),
                    );
                    ffi::msg_void_u64(
                        ca,
                        b"setSourceAlphaBlendFactor:\0",
                        blend_factor_to_metal(desc.blend.src_alpha),
                    );
                    ffi::msg_void_u64(
                        ca,
                        b"setDestinationAlphaBlendFactor:\0",
                        blend_factor_to_metal(desc.blend.dst_alpha),
                    );
                    ffi::msg_void_u64(
                        ca,
                        b"setRgbBlendOperation:\0",
                        blend_op_to_metal(desc.blend.op_rgb),
                    );
                    ffi::msg_void_u64(
                        ca,
                        b"setAlphaBlendOperation:\0",
                        blend_op_to_metal(desc.blend.op_alpha),
                    );
                }
            }

            if let Some(depth_fmt) = desc.depth_format {
                ffi::msg_void_u64(
                    pipe_desc,
                    b"setDepthAttachmentPixelFormat:\0",
                    format_to_metal(depth_fmt),
                );
            }

            // Vertex descriptor — maps VertexLayout to MTLVertexDescriptor
            if !desc.vertex_layouts.is_empty() {
                let vtx_desc = ffi::msg_id(ffi::cls(b"MTLVertexDescriptor\0") as ffi::Id, b"new\0");
                let layouts_array = ffi::msg_id(vtx_desc, b"layouts\0");
                let attrs_array = ffi::msg_id(vtx_desc, b"attributes\0");

                for (buf_idx, layout) in desc.vertex_layouts.iter().enumerate() {
                    let layout_obj = ffi::msg_id_u64(
                        layouts_array,
                        b"objectAtIndexedSubscript:\0",
                        buf_idx as u64,
                    );
                    ffi::msg_void_u64(layout_obj, b"setStride:\0", layout.stride as u64);
                    let step_fn = match layout.step {
                        crate::StepMode::Vertex => ffi::MTL_VERTEX_STEP_FUNCTION_PER_VERTEX,
                        crate::StepMode::Instance => ffi::MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE,
                    };
                    ffi::msg_void_u64(layout_obj, b"setStepFunction:\0", step_fn);

                    for attr in &layout.attributes {
                        let attr_obj = ffi::msg_id_u64(
                            attrs_array,
                            b"objectAtIndexedSubscript:\0",
                            attr.location as u64,
                        );
                        ffi::msg_void_u64(
                            attr_obj,
                            b"setFormat:\0",
                            attr_format_to_metal(attr.format),
                        );
                        ffi::msg_void_u64(attr_obj, b"setOffset:\0", attr.offset as u64);
                        ffi::msg_void_u64(attr_obj, b"setBufferIndex:\0", buf_idx as u64);
                    }
                }
                ffi::msg_void_id(pipe_desc, b"setVertexDescriptor:\0", vtx_desc);
            }

            ffi::msg_void_u64(pipe_desc, b"setSampleCount:\0", desc.sample_count as u64);

            let (pipeline_state, error) = ffi::msg_new_render_pipeline(self.device, pipe_desc);
            if pipeline_state.is_null() {
                let msg = error_string(error);
                return Err(QuantaError::compilation_failed(format!(
                    "render pipeline: {msg}"
                )));
            }

            // Depth/stencil state
            let ds_desc = ffi::msg_id(
                ffi::cls(b"MTLDepthStencilDescriptor\0") as ffi::Id,
                b"new\0",
            );
            if desc.depth_stencil.depth_test {
                ffi::msg_void_u64(
                    ds_desc,
                    b"setDepthCompareFunction:\0",
                    compare_to_metal(desc.depth_stencil.depth_compare),
                );
                ffi::msg_void_bool(
                    ds_desc,
                    b"setDepthWriteEnabled:\0",
                    desc.depth_stencil.depth_write,
                );
            }
            if let Some(ref front) = desc.depth_stencil.stencil_front {
                let s = ffi::msg_id(ffi::cls(b"MTLStencilDescriptor\0") as ffi::Id, b"new\0");
                ffi::msg_void_u64(
                    s,
                    b"setStencilFailureOperation:\0",
                    stencil_op_to_metal(front.fail),
                );
                ffi::msg_void_u64(
                    s,
                    b"setDepthFailureOperation:\0",
                    stencil_op_to_metal(front.depth_fail),
                );
                ffi::msg_void_u64(
                    s,
                    b"setDepthStencilPassOperation:\0",
                    stencil_op_to_metal(front.pass),
                );
                ffi::msg_void_u64(
                    s,
                    b"setStencilCompareFunction:\0",
                    compare_to_metal(front.compare),
                );
                ffi::msg_void_u32(s, b"setReadMask:\0", front.read_mask);
                ffi::msg_void_u32(s, b"setWriteMask:\0", front.write_mask);
                ffi::msg_void_id(ds_desc, b"setFrontFaceStencil:\0", s);
            }
            if let Some(ref back) = desc.depth_stencil.stencil_back {
                let s = ffi::msg_id(ffi::cls(b"MTLStencilDescriptor\0") as ffi::Id, b"new\0");
                ffi::msg_void_u64(
                    s,
                    b"setStencilFailureOperation:\0",
                    stencil_op_to_metal(back.fail),
                );
                ffi::msg_void_u64(
                    s,
                    b"setDepthFailureOperation:\0",
                    stencil_op_to_metal(back.depth_fail),
                );
                ffi::msg_void_u64(
                    s,
                    b"setDepthStencilPassOperation:\0",
                    stencil_op_to_metal(back.pass),
                );
                ffi::msg_void_u64(
                    s,
                    b"setStencilCompareFunction:\0",
                    compare_to_metal(back.compare),
                );
                ffi::msg_void_u32(s, b"setReadMask:\0", back.read_mask);
                ffi::msg_void_u32(s, b"setWriteMask:\0", back.write_mask);
                ffi::msg_void_id(ds_desc, b"setBackFaceStencil:\0", s);
            }
            let ds_state = ffi::msg_id_id(
                self.device,
                b"newDepthStencilStateWithDescriptor:\0",
                ds_desc,
            );

            let handle = self.alloc_handle();
            self.render_pipelines
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?
                .insert(handle, pipeline_state);
            self.depth_stencil_states
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?
                .insert(handle, ds_state);
            Ok(Pipeline {
                handle,
                drop_fn: None,
            })
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

pub(super) unsafe fn error_string(error: ffi::Id) -> String {
    if !error.is_null() {
        unsafe {
            let desc = ffi::msg_id(error, b"localizedDescription\0");
            let cstr = ffi::msg_utf8_string(desc);
            std::ffi::CStr::from_ptr(cstr as *const _)
                .to_string_lossy()
                .into_owned()
        }
    } else {
        "unknown error".into()
    }
}

unsafe fn get_named_function(library: ffi::Id, name: &str) -> Result<ffi::Id, QuantaError> {
    let mut name_bytes: Vec<u8> = name.bytes().collect();
    name_bytes.push(0);
    let ns_name = ffi::nsstring(&name_bytes);
    let func = unsafe { ffi::msg_get_function(library, ns_name) };
    if func.is_null() {
        return Err(QuantaError::compilation_failed(format!(
            "function '{}' not found",
            name
        )));
    }
    Ok(func)
}

/// Get a function from a library, optionally with specialization constants.
/// When `constants` is `Some`, uses `newFunctionWithName:constantValues:error:`.
/// When `None`, falls back to `newFunctionWithName:`.
unsafe fn get_function_maybe_specialized(
    library: ffi::Id,
    name: &str,
    constants: Option<ffi::Id>,
) -> Result<ffi::Id, QuantaError> {
    match constants {
        Some(fcv) => {
            let mut name_bytes: Vec<u8> = name.bytes().collect();
            name_bytes.push(0);
            let ns_name = ffi::nsstring(&name_bytes);
            let (func, error) =
                unsafe { ffi::msg_new_function_with_constants(library, ns_name, fcv) };
            if func.is_null() {
                let msg = unsafe { error_string(error) };
                return Err(QuantaError::compilation_failed(format!(
                    "function '{}' with constants: {}",
                    name, msg
                )));
            }
            Ok(func)
        }
        None => unsafe { get_named_function(library, name) },
    }
}

pub(super) fn attr_format_to_metal(fmt: crate::AttributeFormat) -> ffi::NSUInteger {
    match fmt {
        crate::AttributeFormat::Float => ffi::MTL_VERTEX_FORMAT_FLOAT,
        crate::AttributeFormat::Float2 => ffi::MTL_VERTEX_FORMAT_FLOAT2,
        crate::AttributeFormat::Float3 => ffi::MTL_VERTEX_FORMAT_FLOAT3,
        crate::AttributeFormat::Float4 => ffi::MTL_VERTEX_FORMAT_FLOAT4,
        crate::AttributeFormat::Int => ffi::MTL_VERTEX_FORMAT_INT,
        crate::AttributeFormat::Int2 => ffi::MTL_VERTEX_FORMAT_INT2,
        crate::AttributeFormat::Int3 => ffi::MTL_VERTEX_FORMAT_INT3,
        crate::AttributeFormat::Int4 => ffi::MTL_VERTEX_FORMAT_INT4,
        crate::AttributeFormat::UInt => ffi::MTL_VERTEX_FORMAT_UINT,
        crate::AttributeFormat::UInt2 => ffi::MTL_VERTEX_FORMAT_UINT2,
        crate::AttributeFormat::UInt3 => ffi::MTL_VERTEX_FORMAT_UINT3,
        crate::AttributeFormat::UInt4 => ffi::MTL_VERTEX_FORMAT_UINT4,
        crate::AttributeFormat::UByte4Norm => ffi::MTL_VERTEX_FORMAT_UCHAR4_NORMALIZED,
    }
}
