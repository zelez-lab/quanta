//! Per-device cache of compiled wave pipelines, keyed by kernel bytes.
//!
//! [`Gpu::wave`](crate::Gpu::wave) / [`Gpu::wave_jit`](crate::Gpu::wave_jit)
//! build a driver-side shader module and compute pipeline on every
//! call, and the sci/ml layers create a wave once per op — on
//! backends where pipeline construction dominates (lavapipe: ~80% of
//! the synchronous per-op cost) that is the whole overhead story. The
//! cache keys the compiled pipeline by the exact kernel bytes so a
//! repeat creation of the same kernel returns a fresh [`Wave`] over
//! the already-built pipeline instead of compiling another one.
//!
//! Ownership: the driver registry entry is co-owned through
//! [`SharedPipeline`] — the cache slot holds one `Arc`, every
//! handed-out [`Wave`] holds another (`Wave::shared`), and the LAST
//! owner to drop releases the entry through `wave_destroy`, which
//! carries the driver's in-flight protection (Vulkan parks
//! destruction behind the submission serial / batch pins; Metal's
//! encoder retains the pipeline object). Evicting an entry that
//! outstanding `Wave`s still reference is therefore safe at any
//! moment: the pipeline lives until the last of them drops.
//!
//! Sharing races nothing: the driver-side entry is immutable after
//! creation on all four backends (Vulkan `VkComputePipeline`, Metal's
//! pipeline pointer, the CPU `CpuKernel`, WebGPU's `WaveEntry` —
//! whose `bindings` map is write-only bookkeeping), and every
//! per-dispatch mutable value (bindings, push constants, workgroup
//! size) lives in the per-call `Wave`.
//!
//! Cached pipelines REMAIN driver-registry entries, so
//! `debug_registry_counts` counts them (the field-pool precedent) —
//! absolute-count tests drain via `Gpu::__wave_cache_drain` before
//! asserting.

use alloc::sync::Arc;

use crate::GpuDevice;

/// Co-owned driver-side compiled pipeline (a wave registry entry).
/// The last owner — cache slot or outstanding [`crate::Wave`] — to
/// drop releases the entry, through the same `wave_destroy` an
/// uncached wave's `Drop` uses.
pub(crate) struct SharedPipeline {
    pub(crate) handle: u64,
    /// Keeps the device alive for the destroy: the cache slot or a
    /// parked `Wave` may outlive the `Gpu` clone that created the
    /// entry (the parked-`Batch` precedent).
    device: Arc<dyn GpuDevice>,
}

impl SharedPipeline {
    pub(crate) fn new(handle: u64, device: Arc<dyn GpuDevice>) -> Self {
        SharedPipeline { handle, device }
    }
}

impl Drop for SharedPipeline {
    fn drop(&mut self) {
        let _ = self.device.wave_destroy(self.handle);
    }
}

#[cfg(feature = "std")]
pub(crate) use cache::{WaveCache, WaveKind};

#[cfg(feature = "std")]
mod cache {
    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::SharedPipeline;

    /// Which creation path the bytes belong to. AOT bytes are a
    /// compiled binary (SPIR-V, metallib), JIT bytes a serialized
    /// `KernelDef` — the formats never collide today (distinct magic
    /// numbers), but keying them apart costs nothing and keeps that
    /// a non-assumption.
    #[derive(Clone, Copy)]
    pub(crate) enum WaveKind {
        Aot,
        Jit,
    }

    /// Retained distinct kernels per device. The distinct-kernel count
    /// is a property of program structure (the sci/ml layers build a
    /// few dozen shapes), so this is headroom, not a working-set
    /// tuner; LRU eviction keeps a pathological stream of unique
    /// kernels bounded instead of growing without limit.
    const CAP: usize = 256;

    struct Entry {
        shared: Arc<SharedPipeline>,
        /// Template values a fresh driver creation would have
        /// stamped on the returned `Wave` — functions of the kernel
        /// bytes, replayed on every cache hit.
        workgroup_size: [u32; 3],
        write_mask: u16,
        /// `tick` at last hand-out — the LRU ordering key.
        last_use: u64,
    }

    struct Inner {
        aot: HashMap<Box<[u8]>, Entry>,
        jit: HashMap<Box<[u8]>, Entry>,
        tick: u64,
    }

    impl Inner {
        fn map(&mut self, kind: WaveKind) -> &mut HashMap<Box<[u8]>, Entry> {
            match kind {
                WaveKind::Aot => &mut self.aot,
                WaveKind::Jit => &mut self.jit,
            }
        }
    }

    /// One device's compiled-wave cache. Lives in the
    /// `DeviceContext` beside the deferred lane; every `Gpu` clone
    /// shares it, and isolated devices get their own by construction.
    pub(crate) struct WaveCache {
        inner: Mutex<Inner>,
    }

    impl Default for WaveCache {
        fn default() -> Self {
            WaveCache {
                inner: Mutex::new(Inner {
                    aot: HashMap::new(),
                    jit: HashMap::new(),
                    tick: 0,
                }),
            }
        }
    }

    impl WaveCache {
        /// Look up the compiled pipeline for `bytes`. A hit bumps the
        /// LRU stamp and returns the co-ownership Arc plus the wave
        /// template values.
        pub(crate) fn get(
            &self,
            kind: WaveKind,
            bytes: &[u8],
        ) -> Option<(Arc<SharedPipeline>, [u32; 3], u16)> {
            let mut inner = self.inner.lock().expect("wave cache mutex poisoned");
            inner.tick += 1;
            let tick = inner.tick;
            let entry = inner.map(kind).get_mut(bytes)?;
            entry.last_use = tick;
            Some((entry.shared.clone(), entry.workgroup_size, entry.write_mask))
        }

        /// Adopt a freshly compiled pipeline. If a concurrent creation
        /// of the same bytes won the slot, the incumbent stays and the
        /// caller's wave keeps its solo Arc (it releases its own
        /// driver entry on drop — one redundant compile, no aliasing).
        /// Beyond [`CAP`] entries the least-recently-used one is
        /// evicted; its driver entry is released outside the cache
        /// lock (`wave_destroy` may take driver locks or park work in
        /// the retire bin — never call it under a cache-level lock).
        pub(crate) fn insert(
            &self,
            kind: WaveKind,
            bytes: &[u8],
            shared: Arc<SharedPipeline>,
            workgroup_size: [u32; 3],
            write_mask: u16,
        ) {
            let evicted = {
                let mut inner = self.inner.lock().expect("wave cache mutex poisoned");
                inner.tick += 1;
                let tick = inner.tick;
                if inner.map(kind).contains_key(bytes) {
                    return;
                }
                inner.map(kind).insert(
                    Box::from(bytes),
                    Entry {
                        shared,
                        workgroup_size,
                        write_mask,
                        last_use: tick,
                    },
                );
                if inner.aot.len() + inner.jit.len() > CAP {
                    Self::evict_lru(&mut inner)
                } else {
                    None
                }
            };
            drop(evicted);
        }

        /// Remove the least-recently-used entry across both maps and
        /// return its Arc for the caller to drop outside the lock.
        fn evict_lru(inner: &mut Inner) -> Option<Arc<SharedPipeline>> {
            let oldest = |m: &HashMap<Box<[u8]>, Entry>| {
                m.iter()
                    .min_by_key(|(_, e)| e.last_use)
                    .map(|(k, e)| (k.clone(), e.last_use))
            };
            let a = oldest(&inner.aot);
            let j = oldest(&inner.jit);
            let (kind, key) = match (a, j) {
                (Some((ka, ta)), Some((kj, tj))) => {
                    if ta <= tj {
                        (WaveKind::Aot, ka)
                    } else {
                        (WaveKind::Jit, kj)
                    }
                }
                (Some((ka, _)), None) => (WaveKind::Aot, ka),
                (None, Some((kj, _))) => (WaveKind::Jit, kj),
                (None, None) => return None,
            };
            inner.map(kind).remove(&key).map(|e| e.shared)
        }

        /// Drop every cache-held Arc, returning how many entries were
        /// released. Entries still referenced by outstanding `Wave`s
        /// free when those drop. Driver releases happen outside the
        /// cache lock (see `insert`).
        pub(crate) fn drain(&self) -> usize {
            let (aot, jit) = {
                let mut inner = self.inner.lock().expect("wave cache mutex poisoned");
                (
                    core::mem::take(&mut inner.aot),
                    core::mem::take(&mut inner.jit),
                )
            };
            aot.len() + jit.len()
        }
    }
}
