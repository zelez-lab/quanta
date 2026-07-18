//! Reader struct and top-level decode functions (KernelDef, CompilerOutput,
//! ShaderDef, ShaderOutput).

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::{DeviceFnDef, KernelDef};

use super::helpers::{read_kernel_param, read_scalar_type};
use super::ops::read_kernel_ops;

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Zero-copy binary reader over a byte slice.
pub(crate) struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(crate) fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], &'static str> {
        if self.remaining() < n {
            return Err("unexpected end of input");
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    // -- primitives --

    pub(crate) fn u8(&mut self) -> Result<u8, &'static str> {
        let b = self.take(1)?;
        Ok(b[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16, &'static str> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, &'static str> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, &'static str> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub(crate) fn i32(&mut self) -> Result<i32, &'static str> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub(crate) fn i64(&mut self) -> Result<i64, &'static str> {
        let b = self.take(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub(crate) fn f32(&mut self) -> Result<f32, &'static str> {
        let bits = self.u32()?;
        Ok(f32::from_bits(bits))
    }

    pub(crate) fn f64(&mut self) -> Result<f64, &'static str> {
        let bits = self.u64()?;
        Ok(f64::from_bits(bits))
    }

    pub(crate) fn bool_val(&mut self) -> Result<bool, &'static str> {
        let v = self.u8()?;
        match v {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err("invalid bool tag"),
        }
    }

    // -- composites --

    pub(crate) fn str(&mut self) -> Result<String, &'static str> {
        let len = self.u32()? as usize;
        let b = self.take(len)?;
        core::str::from_utf8(b)
            .map(String::from)
            .map_err(|_| "invalid utf-8 in string")
    }

    pub(crate) fn bytes(&mut self) -> Result<Vec<u8>, &'static str> {
        let len = self.u32()? as usize;
        let b = self.take(len)?;
        Ok(b.to_vec())
    }

    pub(crate) fn option_str(&mut self) -> Result<Option<String>, &'static str> {
        let tag = self.u8()?;
        match tag {
            0 => Ok(None),
            1 => self.str().map(Some),
            _ => Err("invalid option tag"),
        }
    }

    pub(crate) fn option_bytes(&mut self) -> Result<Option<Vec<u8>>, &'static str> {
        let tag = self.u8()?;
        match tag {
            0 => Ok(None),
            1 => self.bytes().map(Some),
            _ => Err("invalid option tag"),
        }
    }
}

// ---------------------------------------------------------------------------
// DeviceFnDef
// ---------------------------------------------------------------------------

fn read_device_fn_def(r: &mut Reader) -> Result<DeviceFnDef, &'static str> {
    let name = r.str()?;
    let param_count = r.u32()? as usize;
    let mut params = Vec::with_capacity(param_count);
    for _ in 0..param_count {
        let pname = r.str()?;
        let ty = read_scalar_type(r)?;
        params.push((pname, ty));
    }
    let return_type = read_scalar_type(r)?;
    let body = read_kernel_ops(r)?;
    let next_reg = r.u32()?;
    Ok(DeviceFnDef {
        name,
        params,
        return_type,
        body,
        next_reg,
    })
}

// ---------------------------------------------------------------------------
// KernelDef
// ---------------------------------------------------------------------------

pub(crate) fn read_kernel_def(r: &mut Reader) -> Result<KernelDef, &'static str> {
    let name = r.str()?;
    let param_count = r.u32()? as usize;
    let mut params = Vec::with_capacity(param_count);
    for _ in 0..param_count {
        params.push(read_kernel_param(r)?);
    }
    let body = read_kernel_ops(r)?;
    let body_source = r.option_str()?;
    let next_reg = r.u32()?;
    let opt_level = r.u8()?;
    // device_sources: Vec<String> — appended after opt_level.
    // If there are no remaining bytes (old format), default to empty.
    let device_sources = if r.remaining() > 0 {
        let count = r.u32()? as usize;
        let mut v = Vec::with_capacity(count);
        for _ in 0..count {
            v.push(r.str()?);
        }
        v
    } else {
        Vec::new()
    };
    // device_functions: Vec<DeviceFnDef> — appended after device_sources.
    let device_functions = if r.remaining() > 0 {
        let count = r.u32()? as usize;
        let mut v = Vec::with_capacity(count);
        for _ in 0..count {
            v.push(read_device_fn_def(r)?);
        }
        v
    } else {
        Vec::new()
    };
    // workgroup_size: [u32; 3] — appended after device_functions.
    // Default [64, 1, 1] for backward compatibility with old format.
    let workgroup_size = if r.remaining() >= 12 {
        [r.u32()?, r.u32()?, r.u32()?]
    } else {
        [64, 1, 1]
    };
    // subgroup_size: Option<u32> — appended after workgroup_size.
    let subgroup_size = if r.remaining() > 0 {
        let tag = r.u8()?;
        match tag {
            0 => None,
            1 => Some(r.u32()?),
            _ => return Err("invalid subgroup_size option tag"),
        }
    } else {
        None
    };
    // dynamic_shared_bytes: u32 — appended after subgroup_size.
    let dynamic_shared_bytes = if r.remaining() >= 4 { r.u32()? } else { 0 };
    Ok(KernelDef {
        name,
        params,
        body,
        body_source,
        next_reg,
        opt_level,
        device_sources,
        device_functions,
        workgroup_size,
        subgroup_size,
        dynamic_shared_bytes,
    })
}

// ---------------------------------------------------------------------------
// CompilerOutput
// ---------------------------------------------------------------------------

pub(crate) fn read_compiler_output(r: &mut Reader) -> Result<crate::CompilerOutput, &'static str> {
    let amd = r.option_bytes()?;
    let nvidia = r.option_bytes()?;
    let spirv = r.option_bytes()?;
    let metallib = r.option_bytes()?;
    // iOS metallib variants — positionally after the macOS one, mirroring
    // write_compiler_output. Producer/consumer move together under the
    // build-rev handshake, so an unguarded read is correct.
    let metallib_ios = r.option_bytes()?;
    let metallib_ios_sim = r.option_bytes()?;
    let wgsl = r.option_str()?;
    Ok(crate::CompilerOutput {
        amd,
        nvidia,
        spirv,
        metallib,
        metallib_ios,
        metallib_ios_sim,
        wgsl,
    })
}

// ---------------------------------------------------------------------------
// ShaderDef / ShaderOutput
// ---------------------------------------------------------------------------

fn read_shader_stage(tag: u8) -> Result<crate::ShaderStage, &'static str> {
    match tag {
        0 => Ok(crate::ShaderStage::Vertex),
        1 => Ok(crate::ShaderStage::Fragment),
        _ => Err("invalid ShaderStage tag"),
    }
}

fn read_shader_type(tag: u8) -> Result<crate::ShaderType, &'static str> {
    match tag {
        0 => Ok(crate::ShaderType::F32),
        1 => Ok(crate::ShaderType::Vec2),
        2 => Ok(crate::ShaderType::Vec3),
        3 => Ok(crate::ShaderType::Vec4),
        4 => Ok(crate::ShaderType::Mat4),
        5 => Ok(crate::ShaderType::Mat3),
        6 => Ok(crate::ShaderType::U32),
        _ => Err("invalid ShaderType tag"),
    }
}

pub(crate) fn read_shader_def(r: &mut Reader) -> Result<crate::ShaderDef, &'static str> {
    let name = r.str()?;
    let stage = read_shader_stage(r.u8()?)?;
    let param_count = r.u32()? as usize;
    let mut params = Vec::with_capacity(param_count);
    for _ in 0..param_count {
        let pname = r.str()?;
        let ty = read_shader_type(r.u8()?)?;
        let is_uniform = r.bool_val()?;
        let is_slice = r.bool_val()?;
        params.push(crate::ShaderParam {
            name: pname,
            ty,
            is_uniform,
            is_slice,
        });
    }
    let return_type = read_shader_type(r.u8()?)?;
    let body_source = r.str()?;
    Ok(crate::ShaderDef {
        name,
        stage,
        params,
        return_type,
        body_source,
    })
}

pub(crate) fn read_shader_output(r: &mut Reader) -> Result<crate::ShaderOutput, &'static str> {
    let spirv = r.option_bytes()?;
    let metallib = r.option_bytes()?;
    // iOS metallib variants — positionally after the macOS one, mirroring
    // write_shader_output.
    let metallib_ios = r.option_bytes()?;
    let metallib_ios_sim = r.option_bytes()?;
    let wgsl = r.option_str()?;
    Ok(crate::ShaderOutput {
        spirv,
        metallib,
        metallib_ios,
        metallib_ios_sim,
        wgsl,
    })
}
