//! MetalDevice struct definition and device discovery.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::{Caps, GpuDevice, Vendor};
use std::collections::HashMap;
use std::sync::RwLock;

use super::ffi;

/// Metal-backed GPU device.
pub struct MetalDevice {
    pub(crate) device: ffi::Id,
    pub(crate) queue: ffi::Id,
    pub(crate) caps: Caps,
    // Resource storage — keyed by handle.
    // RwLock: dispatch/render paths take read locks; alloc/free take write locks.
    pub(crate) buffers: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) textures: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) compute_pipelines: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) render_pipelines: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) depth_stencil_states: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) samplers: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) queues: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) next_handle: AtomicU64,
}

// Safety: Metal objects (MTLDevice, MTLCommandQueue, etc.) are thread-safe.
// All mutable state is protected by RwLock.
unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

impl MetalDevice {
    pub(crate) fn alloc_handle(&self) -> u64 {
        self.next_handle.fetch_add(1, Ordering::Relaxed) + 1
    }
}

/// Discover Metal devices on this system.
pub fn discover() -> Vec<Box<dyn GpuDevice>> {
    let device = unsafe { ffi::MTLCreateSystemDefaultDevice() };
    if device.is_null() {
        return Vec::new();
    }

    let name = unsafe {
        let ns_name = ffi::msg_id(device, b"name\0");
        let cstr = ffi::msg_utf8_string(ns_name);
        std::ffi::CStr::from_ptr(cstr as *const _)
            .to_string_lossy()
            .into_owned()
    };

    let max_threads = unsafe { ffi::msg_mtlsize(device, b"maxThreadsPerThreadgroup\0") };
    let memory_bytes = unsafe { ffi::msg_u64(device, b"recommendedMaxWorkingSetSize\0") };

    let caps = Caps {
        nuclei: (max_threads.width as u32 / 32).max(1),
        protons_per_nucleus: 32,
        quarks_per_proton: 32,
        memory_bytes,
        max_quarks_per_dispatch: u32::MAX,
        max_groups: [u32::MAX, u32::MAX, u32::MAX],
        vendor: Vendor::Apple,
        name,
    };

    let queue = unsafe { ffi::msg_id(device, b"newCommandQueue\0") };

    vec![Box::new(MetalDevice {
        device,
        queue,
        caps,
        buffers: RwLock::new(HashMap::new()),
        textures: RwLock::new(HashMap::new()),
        compute_pipelines: RwLock::new(HashMap::new()),
        render_pipelines: RwLock::new(HashMap::new()),
        depth_stencil_states: RwLock::new(HashMap::new()),
        samplers: RwLock::new(HashMap::new()),
        queues: RwLock::new(HashMap::new()),
        next_handle: AtomicU64::new(0),
    })]
}
