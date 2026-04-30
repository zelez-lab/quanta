//! CpuDevice — software GPU device implementation.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
use crate::{
    Caps, FieldUsage, GpuDevice, Pipeline, Pulse, QuantaError, RenderPass, Texture, TextureDesc,
    Vendor, Wave,
};
use quanta_ir::{KernelDef, KernelOp};

use super::exec::{ExecCtx, execute_ops};
use super::value::Value;

// ── CPU Device ───────────────────────────────────────────────────────────────

/// Internal buffer allocation.
struct CpuBuffer {
    data: Vec<u8>,
}

/// Stored kernel ready for dispatch.
struct CpuKernel {
    def: KernelDef,
    /// Pre-computed barrier segment ranges into def.body (start, end).
    /// Computed once at wave_jit time, reused every dispatch.
    segments: Vec<(usize, usize)>,
}

/// One recorded ICB command. Mirrors the Lean
/// `Quanta.Icb.Command` sum type (Dispatch | Draw); the CPU
/// device's `indirect_buffer_execute` folds over this list, which
/// is the direct refinement of T7000.
#[allow(clippy::large_enum_variant)]
enum RecordedCommand {
    Dispatch {
        wave_handle: u64,
        bindings: [u64; crate::api::wave::MAX_BINDINGS],
        binding_count: u8,
        texture_bindings: [u64; crate::api::wave::MAX_TEXTURES],
        texture_count: u8,
        push_data: [u8; crate::api::wave::PUSH_DATA_CAP],
        push_len: u16,
        push_mask: u16,
        workgroup_size: [u32; 3],
        groups: [u32; 3],
    },
    /// Render-path draw command. The CPU device has no rasterizer —
    /// recording snapshots the parameters and `execute` replays
    /// them as no-ops. The proof contract (T7006: record extends
    /// the recorded sequence by exactly that command) is satisfied;
    /// observable rendering side-effects belong to GPU backends.
    Draw {
        pipeline: u64,
        vertex_count: u32,
        instance_count: u32,
    },
}

/// CPU-side ICB state — sized capacity + recorded commands.
struct CpuIcb {
    cap: u32,
    commands: Vec<RecordedCommand>,
}

/// CPU bindless texture/buffer array. Mirrors `Quanta.Bindless.Array`:
/// fixed capacity at create, slot updates via `set`, destroy
/// invalidates the handle. Texture and buffer arrays share the
/// same shape; backend differentiates by which trait method is called.
struct CpuBindlessArray {
    cap: u32,
    entries: Vec<u64>,
}

/// CPU tessellation pipeline state. Mirrors
/// `Quanta.Tessellation.Pipeline`: fixed topology and control-point
/// count at create, mutable inner / outer factor lists.
struct CpuTessPipeline {
    outer: Vec<u32>,
    inner: Vec<u32>,
}

/// CPU software device — executes GPU kernel IR without hardware.
pub struct CpuDevice {
    caps: Caps,
    next_handle: Mutex<u64>,
    buffers: Mutex<HashMap<u64, CpuBuffer>>,
    kernels: Mutex<HashMap<u64, CpuKernel>>,
    /// Indirect command buffers indexed by handle.
    icbs: Mutex<HashMap<u64, CpuIcb>>,
    /// Bindless texture arrays indexed by handle.
    bindless_textures: Mutex<HashMap<u64, CpuBindlessArray>>,
    /// Bindless buffer arrays indexed by handle.
    bindless_buffers: Mutex<HashMap<u64, CpuBindlessArray>>,
    /// Tessellation pipelines indexed by handle.
    tess_pipelines: Mutex<HashMap<u64, CpuTessPipeline>>,
}

impl CpuDevice {
    /// Create a new CPU software device.
    pub fn new() -> Self {
        Self {
            caps: Caps {
                nuclei: 1,
                protons_per_nucleus: 1,
                quarks_per_proton: 1,
                memory_bytes: 1024 * 1024 * 1024, // 1 GB virtual
                max_quarks_per_dispatch: u32::MAX,
                max_groups: [u32::MAX; 3],
                vendor: Vendor::Software,
                name: String::from("Quanta CPU (software)"),
            },
            next_handle: Mutex::new(1),
            buffers: Mutex::new(HashMap::new()),
            kernels: Mutex::new(HashMap::new()),
            icbs: Mutex::new(HashMap::new()),
            bindless_textures: Mutex::new(HashMap::new()),
            bindless_buffers: Mutex::new(HashMap::new()),
            tess_pipelines: Mutex::new(HashMap::new()),
        }
    }

    fn alloc_handle(&self) -> u64 {
        let mut h = self.next_handle.lock().unwrap();
        let handle = *h;
        *h += 1;
        handle
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuDevice for CpuDevice {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    // === Fields ===

    fn field_alloc(&self, size: usize, _usage: FieldUsage) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        let buf = CpuBuffer {
            data: vec![0u8; size],
        };
        self.buffers.lock().unwrap().insert(handle, buf);
        Ok(handle)
    }

    fn field_free(&self, handle: u64) {
        self.buffers.lock().unwrap().remove(&handle);
    }

    fn field_write_bytes(&self, handle: u64, data: &[u8]) -> Result<(), QuantaError> {
        let mut bufs = self.buffers.lock().unwrap();
        let buf = bufs
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::invalid_param("field handle not found"))?;
        let len = data.len().min(buf.data.len());
        buf.data[..len].copy_from_slice(&data[..len]);
        Ok(())
    }

    fn field_read_bytes(&self, handle: u64, size: usize) -> Result<Vec<u8>, QuantaError> {
        let bufs = self.buffers.lock().unwrap();
        let buf = bufs
            .get(&handle)
            .ok_or_else(|| QuantaError::invalid_param("field handle not found"))?;
        let len = size.min(buf.data.len());
        Ok(buf.data[..len].to_vec())
    }

    fn field_copy_bytes(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError> {
        let mut bufs = self.buffers.lock().unwrap();
        // Copy src data first to avoid borrow conflict
        let src_data = {
            let src_buf = bufs
                .get(&src)
                .ok_or_else(|| QuantaError::invalid_param("src field not found"))?;
            let len = size.min(src_buf.data.len());
            src_buf.data[..len].to_vec()
        };
        let dst_buf = bufs
            .get_mut(&dst)
            .ok_or_else(|| QuantaError::invalid_param("dst field not found"))?;
        let len = src_data.len().min(dst_buf.data.len());
        dst_buf.data[..len].copy_from_slice(&src_data[..len]);
        Ok(())
    }

    fn field_map(&self, handle: u64, _size: usize) -> Result<*mut u8, QuantaError> {
        let mut bufs = self.buffers.lock().unwrap();
        let buf = bufs
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::invalid_param("field handle not found"))?;
        Ok(buf.data.as_mut_ptr())
    }

    fn field_unmap(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(()) // CPU memory is always mapped
    }

    fn field_create_mapped(
        &self,
        size: usize,
        _usage: FieldUsage,
    ) -> Result<(u64, *mut u8), QuantaError> {
        let handle = self.alloc_handle();
        let buf = CpuBuffer {
            data: vec![0u8; size],
        };
        self.buffers.lock().unwrap().insert(handle, buf);
        let ptr = self
            .buffers
            .lock()
            .unwrap()
            .get_mut(&handle)
            .unwrap()
            .data
            .as_mut_ptr();
        Ok((handle, ptr))
    }

    // === Textures (minimal stubs) ===

    fn texture_create(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        let handle = self.alloc_handle();
        let size = (desc.width * desc.height) as usize * desc.format.bytes_per_pixel();
        self.buffers.lock().unwrap().insert(
            handle,
            CpuBuffer {
                data: vec![0u8; size],
            },
        );
        Ok(Texture {
            handle,
            width: desc.width,
            height: desc.height,
            format: desc.format,
            device: None,
        })
    }

    fn texture_write(&self, texture: &Texture, data: &[u8]) -> Result<(), QuantaError> {
        self.field_write_bytes(texture.handle(), data)
    }

    fn texture_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        let size =
            (texture.width() * texture.height()) as usize * texture.format().bytes_per_pixel();
        self.field_read_bytes(texture.handle(), size)
    }

    fn sampler_create(
        &self,
        _desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        Ok(crate::Sampler {
            handle: self.alloc_handle(),
            drop_fn: None,
        })
    }

    fn generate_mipmaps(&self, _texture: &Texture) -> Result<(), QuantaError> {
        Ok(()) // no-op on CPU
    }

    // === Compute ===

    fn wave(&self, _kernel: &[u8]) -> Result<Wave, QuantaError> {
        Err(QuantaError::invalid_param(
            "CPU device only supports JIT path (wave_jit). \
             Pre-compiled binaries cannot be executed on CPU.",
        ))
    }

    fn wave_jit(&self, kernel_def_bytes: &[u8]) -> Result<Wave, QuantaError> {
        let def = quanta_ir::deserialize_kernel(kernel_def_bytes)
            .map_err(|e| QuantaError::compilation_failed(e.to_string()))?;
        let handle = self.alloc_handle();
        let workgroup_size = def.workgroup_size;
        let segments = barrier_segment_ranges(&def.body);
        self.kernels
            .lock()
            .unwrap()
            .insert(handle, CpuKernel { def, segments });
        Ok(Wave {
            handle,
            bindings: [0u64; 16],
            binding_count: 0,
            texture_bindings: [0u64; 16],
            texture_count: 0,
            push_data: [0u8; 256],
            push_len: 0,
            push_mask: 0,
            workgroup_size,
            drop_fn: None,
        })
    }

    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        let kernels = self.kernels.lock().unwrap();
        let kernel = kernels
            .get(&wave.handle)
            .ok_or_else(|| QuantaError::invalid_param("wave handle not found"))?;

        let wg = kernel.def.workgroup_size;
        let total_groups = groups[0] as u64 * groups[1] as u64 * groups[2] as u64;
        let threads_per_group = wg[0] as u64 * wg[1] as u64 * wg[2] as u64;
        let total_threads = total_groups * threads_per_group;

        // Snapshot bound buffer data into fixed-size array (max 16 bindings)
        let mut field_data: [Option<Vec<u8>>; 16] = Default::default();
        {
            let bufs = self.buffers.lock().unwrap();
            for (i, slot) in field_data
                .iter_mut()
                .enumerate()
                .take(wave.binding_count as usize)
            {
                let handle = wave.bindings[i];
                if handle != 0
                    && let Some(buf) = bufs.get(&handle)
                {
                    *slot = Some(buf.data.clone());
                }
            }
        }

        let group_size_x = wg[0];

        // Use pre-computed barrier segment ranges (zero-copy slices)
        let segments = &kernel.segments;

        // Allocate workgroup state once, reuse via clear()
        let mut shared: HashMap<u32, Vec<u8>> = HashMap::new();
        let mut thread_regs: Vec<HashMap<u32, Value>> =
            (0..threads_per_group).map(|_| HashMap::new()).collect();

        for gid in 0..total_groups {
            shared.clear();
            for reg_map in thread_regs.iter_mut() {
                reg_map.clear();
            }

            for &(seg_start, seg_end) in segments {
                let segment = &kernel.def.body[seg_start..seg_end];
                for lid in 0..threads_per_group {
                    let quark_id = (gid * threads_per_group + lid) as u32;
                    let local_id = lid as u32;
                    let group_id = gid as u32;

                    let mut ctx = ExecCtx {
                        quark_id,
                        local_id,
                        group_id,
                        group_size: group_size_x,
                        quark_count: total_threads as u32,
                        regs: core::mem::take(&mut thread_regs[lid as usize]),
                        fields: &mut field_data,
                        shared: &mut shared,
                    };

                    execute_ops(&mut ctx, segment).map_err(|e| {
                        QuantaError::compilation_failed(format!(
                            "CPU execution error (quark {quark_id}): {e}"
                        ))
                    })?;

                    thread_regs[lid as usize] = ctx.regs;
                }
            }
        }

        // Write back modified buffer data
        {
            let mut bufs = self.buffers.lock().unwrap();
            for (i, slot) in field_data
                .iter_mut()
                .enumerate()
                .take(wave.binding_count as usize)
            {
                let handle = wave.bindings[i];
                if handle != 0
                    && let Some(modified) = slot.take()
                    && let Some(buf) = bufs.get_mut(&handle)
                {
                    buf.data = modified;
                }
            }
        }

        Ok(Pulse {
            handle: 0,
            completed: true,
            wait_fn: None,
        })
    }

    fn wave_dispatch_indirect(
        &self,
        _wave: &Wave,
        _buffer: u64,
        _offset: u64,
    ) -> Result<Pulse, QuantaError> {
        Err(QuantaError::invalid_param(
            "indirect dispatch not supported on CPU device",
        ))
    }

    // === Render (stubs) ===

    fn pipeline_create(&self, _desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        Ok(Pipeline {
            handle: self.alloc_handle(),
            drop_fn: None,
        })
    }

    fn render_begin(&self, _target: &Texture) -> Result<RenderPass, QuantaError> {
        Err(QuantaError::invalid_param(
            "render passes not supported on CPU device",
        ))
    }

    fn render_end(&self, _pass: RenderPass) -> Result<Pulse, QuantaError> {
        Err(QuantaError::invalid_param(
            "render passes not supported on CPU device",
        ))
    }

    // === Sync ===

    fn pulse_wait(&self, pulse: &mut Pulse) -> Result<(), QuantaError> {
        pulse.completed = true;
        Ok(())
    }

    fn pulse_poll(&self, _pulse: &Pulse) -> bool {
        true // CPU execution is synchronous
    }

    // === M4.2: Mesh shaders ===

    fn dispatch_mesh(&self, _pipeline: u64, _groups: [u32; 3]) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
            "mesh shaders not supported on CPU device",
        ))
    }

    // === M4.3: Ray tracing ===

    fn build_acceleration_structure(&self, _geometry: &[GeometryDesc]) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param(
            "ray tracing not supported on CPU device",
        ))
    }

    fn create_ray_tracing_pipeline(
        &self,
        _desc: &RayTracingPipelineDesc,
    ) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param(
            "ray tracing not supported on CPU device",
        ))
    }

    fn dispatch_rays(&self, _pipeline: u64, _width: u32, _height: u32) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
            "ray tracing not supported on CPU device",
        ))
    }

    fn destroy_acceleration_structure(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    // === M5.1: Sparse textures ===

    fn sparse_texture_create(&self, _desc: &TextureDesc) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param(
            "sparse textures not supported on CPU device",
        ))
    }

    fn sparse_map_tile(
        &self,
        _texture: u64,
        _mip: u32,
        _x: u32,
        _y: u32,
        _backing: u64,
    ) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
            "sparse textures not supported on CPU device",
        ))
    }

    fn sparse_unmap_tile(
        &self,
        _texture: u64,
        _mip: u32,
        _x: u32,
        _y: u32,
    ) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
            "sparse textures not supported on CPU device",
        ))
    }

    // === M5.2: Indirect command buffers (steps 032 + 033) ===
    //
    // CPU implementation refines the abstract `Quanta.Icb` model:
    // - create allocates a fresh handle + empty Vec sized to capacity.
    // - icb_record_dispatch snapshots the wave + group counts.
    // - indirect_buffer_execute replays the first `count` recorded
    //   dispatches sequentially through the existing `wave_dispatch`
    //   path, satisfying the Lean T7000 equivalence theorem.

    fn indirect_buffer_create(&self, max_commands: u32) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        self.icbs.lock().unwrap().insert(
            handle,
            CpuIcb {
                cap: max_commands,
                commands: Vec::with_capacity(max_commands as usize),
            },
        );
        Ok(handle)
    }

    fn icb_record_dispatch(
        &self,
        handle: u64,
        index: u32,
        wave: &Wave,
        groups: [u32; 3],
    ) -> Result<(), QuantaError> {
        let mut icbs = self.icbs.lock().unwrap();
        let icb = icbs
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::invalid_param("ICB handle not found"))?;
        if index != icb.commands.len() as u32 {
            return Err(QuantaError::invalid_param(
                "ICB record index must equal current length",
            ));
        }
        if index >= icb.cap {
            return Err(QuantaError::invalid_param("ICB index >= capacity"));
        }
        icb.commands.push(RecordedCommand::Dispatch {
            wave_handle: wave.handle,
            bindings: wave.bindings,
            binding_count: wave.binding_count,
            texture_bindings: wave.texture_bindings,
            texture_count: wave.texture_count,
            push_data: wave.push_data,
            push_len: wave.push_len,
            push_mask: wave.push_mask,
            workgroup_size: wave.workgroup_size,
            groups,
        });
        Ok(())
    }

    fn icb_record_draw(
        &self,
        handle: u64,
        index: u32,
        pipeline: u64,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<(), QuantaError> {
        let mut icbs = self.icbs.lock().unwrap();
        let icb = icbs
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::invalid_param("ICB handle not found"))?;
        if index != icb.commands.len() as u32 {
            return Err(QuantaError::invalid_param(
                "ICB record index must equal current length",
            ));
        }
        if index >= icb.cap {
            return Err(QuantaError::invalid_param("ICB index >= capacity"));
        }
        icb.commands.push(RecordedCommand::Draw {
            pipeline,
            vertex_count,
            instance_count,
        });
        Ok(())
    }

    fn indirect_buffer_execute(&self, handle: u64, count: u32) -> Result<(), QuantaError> {
        // Snapshot the recorded commands while holding the ICB lock,
        // then drop the lock before re-entering wave_dispatch (which
        // takes its own locks on buffers/kernels).
        let snapshot: Vec<RecordedCommand> = {
            let icbs = self.icbs.lock().unwrap();
            let icb = icbs
                .get(&handle)
                .ok_or_else(|| QuantaError::invalid_param("ICB handle not found"))?;
            if count as usize > icb.commands.len() {
                return Err(QuantaError::invalid_param(
                    "ICB execute count exceeds recorded length",
                ));
            }
            icb.commands[..count as usize]
                .iter()
                .map(|r| match r {
                    RecordedCommand::Dispatch {
                        wave_handle,
                        bindings,
                        binding_count,
                        texture_bindings,
                        texture_count,
                        push_data,
                        push_len,
                        push_mask,
                        workgroup_size,
                        groups,
                    } => RecordedCommand::Dispatch {
                        wave_handle: *wave_handle,
                        bindings: *bindings,
                        binding_count: *binding_count,
                        texture_bindings: *texture_bindings,
                        texture_count: *texture_count,
                        push_data: *push_data,
                        push_len: *push_len,
                        push_mask: *push_mask,
                        workgroup_size: *workgroup_size,
                        groups: *groups,
                    },
                    RecordedCommand::Draw {
                        pipeline,
                        vertex_count,
                        instance_count,
                    } => RecordedCommand::Draw {
                        pipeline: *pipeline,
                        vertex_count: *vertex_count,
                        instance_count: *instance_count,
                    },
                })
                .collect()
        };
        for rec in &snapshot {
            match rec {
                RecordedCommand::Dispatch {
                    wave_handle,
                    bindings,
                    binding_count,
                    texture_bindings,
                    texture_count,
                    push_data,
                    push_len,
                    push_mask,
                    workgroup_size,
                    groups,
                } => {
                    // Reconstruct a transient Wave from the snapshot.
                    // Drop is a no-op (drop_fn = None), so this does
                    // not double-free any underlying kernel handle.
                    let wave = Wave {
                        handle: *wave_handle,
                        bindings: *bindings,
                        binding_count: *binding_count,
                        texture_bindings: *texture_bindings,
                        texture_count: *texture_count,
                        push_data: *push_data,
                        push_len: *push_len,
                        push_mask: *push_mask,
                        workgroup_size: *workgroup_size,
                        drop_fn: None,
                    };
                    let mut pulse = self.wave_dispatch(&wave, *groups)?;
                    pulse.wait()?;
                }
                RecordedCommand::Draw {
                    pipeline: _,
                    vertex_count: _,
                    instance_count: _,
                } => {
                    // CPU device has no rasterizer — recorded draws
                    // replay as no-ops. T7006 (record-draw appends
                    // to the sequence) is still satisfied; backends
                    // with a real raster path provide the visible
                    // side-effect.
                }
            }
        }
        Ok(())
    }

    fn indirect_buffer_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.icbs.lock().unwrap().remove(&handle);
        Ok(())
    }

    // === M5.3: Bindless resources ===

    fn bind_texture_array(&self, _textures: &[u64]) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param(
            "bindless resources not supported on CPU device",
        ))
    }

    fn bind_buffer_array(&self, _buffers: &[u64]) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param(
            "bindless resources not supported on CPU device",
        ))
    }

    // === Bindless typed wrappers (steps 034 + 035) ===
    //
    // CPU implementation refines the abstract `Quanta.Bindless.Array`:
    // - create allocates a Vec<u64> sized to cap, all zeroed.
    // - set updates one slot, bounds-checked.
    // - destroy removes the handle.

    fn bindless_texture_create(&self, cap: u32) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        self.bindless_textures.lock().unwrap().insert(
            handle,
            CpuBindlessArray {
                cap,
                entries: vec![0u64; cap as usize],
            },
        );
        Ok(handle)
    }

    fn bindless_texture_set(
        &self,
        handle: u64,
        index: u32,
        texture: u64,
    ) -> Result<(), QuantaError> {
        let mut arrays = self.bindless_textures.lock().unwrap();
        let arr = arrays
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::invalid_param("bindless texture array not found"))?;
        if index >= arr.cap {
            return Err(QuantaError::invalid_param(
                "bindless texture index >= capacity",
            ));
        }
        arr.entries[index as usize] = texture;
        Ok(())
    }

    fn bindless_texture_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.bindless_textures.lock().unwrap().remove(&handle);
        Ok(())
    }

    fn bindless_buffer_create(&self, cap: u32) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        self.bindless_buffers.lock().unwrap().insert(
            handle,
            CpuBindlessArray {
                cap,
                entries: vec![0u64; cap as usize],
            },
        );
        Ok(handle)
    }

    fn bindless_buffer_set(&self, handle: u64, index: u32, buffer: u64) -> Result<(), QuantaError> {
        let mut arrays = self.bindless_buffers.lock().unwrap();
        let arr = arrays
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::invalid_param("bindless buffer array not found"))?;
        if index >= arr.cap {
            return Err(QuantaError::invalid_param(
                "bindless buffer index >= capacity",
            ));
        }
        arr.entries[index as usize] = buffer;
        Ok(())
    }

    fn bindless_buffer_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.bindless_buffers.lock().unwrap().remove(&handle);
        Ok(())
    }

    // === Tessellation pipelines (steps 022 + 023) ===
    //
    // CPU implementation refines `Quanta.Tessellation.Pipeline`:
    // - create allocates outer/inner Vecs sized to topology, all 1s.
    // - set_{outer,inner} updates one slot, bounds-checked. The typed
    //   wrapper has already clamped the factor into [1, MAX_TESS_LEVEL].
    // - destroy removes the handle.

    fn tessellation_pipeline_create(
        &self,
        topology: u8,
        _control_points: u32,
    ) -> Result<u64, QuantaError> {
        let (outer_count, inner_count) = match topology {
            0 => (3usize, 1usize),
            1 => (4usize, 2usize),
            _ => {
                return Err(QuantaError::invalid_param(
                    "tessellation topology must be 0 (triangle) or 1 (quad)",
                ));
            }
        };
        let _ = topology;
        let handle = self.alloc_handle();
        self.tess_pipelines.lock().unwrap().insert(
            handle,
            CpuTessPipeline {
                outer: vec![1u32; outer_count],
                inner: vec![1u32; inner_count],
            },
        );
        Ok(handle)
    }

    fn tessellation_set_outer(
        &self,
        handle: u64,
        index: u32,
        factor: u32,
    ) -> Result<(), QuantaError> {
        let mut pipes = self.tess_pipelines.lock().unwrap();
        let pipe = pipes
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::invalid_param("tessellation pipeline not found"))?;
        if (index as usize) >= pipe.outer.len() {
            return Err(QuantaError::invalid_param(
                "tessellation outer index out of range",
            ));
        }
        pipe.outer[index as usize] = factor;
        Ok(())
    }

    fn tessellation_set_inner(
        &self,
        handle: u64,
        index: u32,
        factor: u32,
    ) -> Result<(), QuantaError> {
        let mut pipes = self.tess_pipelines.lock().unwrap();
        let pipe = pipes
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::invalid_param("tessellation pipeline not found"))?;
        if (index as usize) >= pipe.inner.len() {
            return Err(QuantaError::invalid_param(
                "tessellation inner index out of range",
            ));
        }
        pipe.inner[index as usize] = factor;
        Ok(())
    }

    fn tessellation_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.tess_pipelines.lock().unwrap().remove(&handle);
        Ok(())
    }
}

/// Compute barrier segment ranges as (start, end) indices into the ops slice.
///
/// Each range covers the ops between consecutive barriers.
/// The barrier ops themselves are excluded. Zero allocation at dispatch
/// time — segments are sliced from the original body via `&ops[start..end]`.
fn barrier_segment_ranges(ops: &[KernelOp]) -> Vec<(usize, usize)> {
    let mut segments = Vec::new();
    let mut start = 0;

    for (i, op) in ops.iter().enumerate() {
        if matches!(op, KernelOp::Barrier) {
            segments.push((start, i));
            start = i + 1;
        }
    }
    segments.push((start, ops.len()));
    segments
}

/// Discover CPU devices. Always returns exactly one.
pub fn discover() -> Vec<Box<dyn GpuDevice>> {
    vec![Box::new(CpuDevice::new())]
}

#[cfg(test)]
mod tests {
    use super::super::eval::{eval_binop, eval_cmp};
    use super::super::value::{Value, f16_to_f32, f32_to_f16, read_scalar, write_scalar};
    use super::*;
    use quanta_ir::{BinOp, CmpOp, ScalarType};

    #[test]
    fn value_conversions() {
        assert_eq!(Value::U32(42).as_u32(), 42);
        assert_eq!(Value::U32(42).as_f32(), 42.0);
        assert_eq!(Value::F32(3.14).as_u32(), 3);
        assert!(Value::U32(1).as_bool());
        assert!(!Value::U32(0).as_bool());
        assert!(Value::Bool(true).as_bool());
    }

    #[test]
    fn scalar_read_write_roundtrip() {
        let mut buf = vec![0u8; 16];
        write_scalar(&mut buf, 0, Value::F32(3.14), &ScalarType::F32);
        let v = read_scalar(&buf, 0, &ScalarType::F32);
        assert!((v.as_f32() - 3.14).abs() < 1e-6);

        write_scalar(&mut buf, 1, Value::U32(42), &ScalarType::U32);
        let v = read_scalar(&buf, 1, &ScalarType::U32);
        assert_eq!(v.as_u32(), 42);
    }

    #[test]
    fn binop_add() {
        let r = eval_binop(Value::U32(3), Value::U32(4), &BinOp::Add, &ScalarType::U32);
        assert_eq!(r.as_u32(), 7);

        let r = eval_binop(
            Value::F32(1.5),
            Value::F32(2.5),
            &BinOp::Add,
            &ScalarType::F32,
        );
        assert!((r.as_f32() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn binop_div_by_zero() {
        let r = eval_binop(Value::U32(10), Value::U32(0), &BinOp::Div, &ScalarType::U32);
        assert_eq!(r.as_u32(), 0);
    }

    #[test]
    fn cmp_ops() {
        let r = eval_cmp(Value::U32(3), Value::U32(5), &CmpOp::Lt, &ScalarType::U32);
        assert!(r.as_bool());

        let r = eval_cmp(Value::U32(5), Value::U32(3), &CmpOp::Lt, &ScalarType::U32);
        assert!(!r.as_bool());
    }

    #[test]
    fn f16_roundtrip() {
        let original = 1.5f32;
        let bits = f32_to_f16(original);
        let back = f16_to_f32(bits);
        assert!((back - original).abs() < 1e-3);
    }

    #[test]
    fn cpu_device_field_alloc_write_read() {
        let dev = CpuDevice::new();
        let handle = dev.field_alloc(16, FieldUsage::default_compute()).unwrap();
        dev.field_write_bytes(handle, &[1, 2, 3, 4]).unwrap();
        let data = dev.field_read_bytes(handle, 16).unwrap();
        assert_eq!(&data[..4], &[1, 2, 3, 4]);
        assert_eq!(&data[4..], &[0; 12]);
        dev.field_free(handle);
    }

    #[test]
    fn cpu_device_caps() {
        let dev = CpuDevice::new();
        assert_eq!(dev.caps().vendor, Vendor::Software);
        assert_eq!(dev.caps().name, "Quanta CPU (software)");
    }

    #[test]
    fn cpu_device_wave_rejects_binary() {
        let dev = CpuDevice::new();
        let result = dev.wave(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn cpu_device_pulse_is_synchronous() {
        let dev = CpuDevice::new();
        let pulse = Pulse {
            handle: 0,
            completed: false,
            wait_fn: None,
        };
        assert!(dev.pulse_poll(&pulse));
    }
}
