//! Writer struct and top-level encode functions (KernelDef, CompilerOutput,
//! ShaderDef, ShaderOutput).

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::{DeviceFnDef, KernelDef};

use super::helpers::{write_kernel_param, write_scalar_type};
use super::ops::write_kernel_ops;

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Append-only binary writer backed by a `Vec<u8>`.
pub(crate) struct Writer {
    buf: Vec<u8>,
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

impl Writer {
    pub(crate) fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub(crate) fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
        }
    }

    pub(crate) fn finish(self) -> Vec<u8> {
        self.buf
    }

    // -- primitives --

    pub(crate) fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub(crate) fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    pub(crate) fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    pub(crate) fn bool_val(&mut self, v: bool) {
        self.buf.push(v as u8);
    }

    // -- composites --

    pub(crate) fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }

    pub(crate) fn bytes(&mut self, b: &[u8]) {
        self.u32(b.len() as u32);
        self.buf.extend_from_slice(b);
    }

    pub(crate) fn option_str(&mut self, v: &Option<String>) {
        match v {
            None => self.u8(0),
            Some(s) => {
                self.u8(1);
                self.str(s);
            }
        }
    }

    pub(crate) fn option_bytes(&mut self, v: &Option<Vec<u8>>) {
        match v {
            None => self.u8(0),
            Some(b) => {
                self.u8(1);
                self.bytes(b);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DeviceFnDef
// ---------------------------------------------------------------------------

fn write_device_fn_def(w: &mut Writer, f: &DeviceFnDef) {
    w.str(&f.name);
    w.u32(f.params.len() as u32);
    for (name, ty) in &f.params {
        w.str(name);
        write_scalar_type(w, ty);
    }
    write_scalar_type(w, &f.return_type);
    write_kernel_ops(w, &f.body);
    w.u32(f.next_reg);
}

// ---------------------------------------------------------------------------
// KernelDef
// ---------------------------------------------------------------------------

pub(crate) fn write_kernel_def(w: &mut Writer, k: &KernelDef) {
    w.str(&k.name);
    w.u32(k.params.len() as u32);
    for p in &k.params {
        write_kernel_param(w, p);
    }
    write_kernel_ops(w, &k.body);
    w.option_str(&k.body_source);
    w.u32(k.next_reg);
    w.u8(k.opt_level);
    // device_sources: Vec<String>
    w.u32(k.device_sources.len() as u32);
    for s in &k.device_sources {
        w.str(s);
    }
    // device_functions: Vec<DeviceFnDef>
    w.u32(k.device_functions.len() as u32);
    for f in &k.device_functions {
        write_device_fn_def(w, f);
    }
    // workgroup_size: [u32; 3]
    w.u32(k.workgroup_size[0]);
    w.u32(k.workgroup_size[1]);
    w.u32(k.workgroup_size[2]);
    // subgroup_size: Option<u32>
    match k.subgroup_size {
        None => w.u8(0),
        Some(s) => {
            w.u8(1);
            w.u32(s);
        }
    }
    // dynamic_shared_bytes: u32
    w.u32(k.dynamic_shared_bytes);
}

// ---------------------------------------------------------------------------
// CompilerOutput
// ---------------------------------------------------------------------------

pub(crate) fn write_compiler_output(w: &mut Writer, o: &crate::CompilerOutput) {
    w.option_bytes(&o.amd);
    w.option_bytes(&o.nvidia);
    w.option_bytes(&o.spirv);
    w.option_bytes(&o.metallib);
    // iOS metallib variants ride after the macOS one. Safe positional
    // growth: the git-derived build-rev handshake forces the compiler and
    // the DSL-macro reader to move together, so producer and consumer
    // always agree on the field count.
    w.option_bytes(&o.metallib_ios);
    w.option_bytes(&o.metallib_ios_sim);
    w.option_str(&o.wgsl);
}

// ---------------------------------------------------------------------------
// ShaderDef / ShaderOutput
// ---------------------------------------------------------------------------

pub(crate) fn write_shader_def(w: &mut Writer, s: &crate::ShaderDef) {
    w.str(&s.name);
    w.u8(s.stage as u8);
    w.u32(s.params.len() as u32);
    for p in &s.params {
        w.str(&p.name);
        w.u8(p.ty as u8);
        w.bool_val(p.is_uniform);
        w.bool_val(p.is_slice);
    }
    w.u8(s.return_type as u8);
    w.str(&s.body_source);
}

pub(crate) fn write_shader_output(w: &mut Writer, o: &crate::ShaderOutput) {
    w.option_bytes(&o.spirv);
    w.option_bytes(&o.metallib);
    // iOS metallib variants ride after the macOS one — same handshake
    // guarantee as CompilerOutput above.
    w.option_bytes(&o.metallib_ios);
    w.option_bytes(&o.metallib_ios_sim);
    w.option_str(&o.wgsl);
}
