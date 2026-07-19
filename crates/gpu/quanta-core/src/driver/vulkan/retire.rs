//! Fence-deferred resource destruction.
//!
//! Render submissions are asynchronous (`submit_and_wait` returns a live
//! Pulse), so a wrapper `Drop` can reach a destroy method while the GPU
//! still references the resource. Destroying immediately is
//! VUID-vkDestroyImage-image-01000 / -vkDestroyImageView-imageView-01026
//! territory and a real device loss on Intel (the dija Glass/Surface
//! report). Instead of per-resource fences, destruction is gated on a
//! monotonic submission serial:
//!
//! - every `vkQueueSubmit` on the device's one queue takes the next
//!   serial, assigned under the queue lock so serial order IS submission
//!   order;
//! - a fence wait that returns `VK_SUCCESS` for the submission with
//!   serial `s` proves every serial `<= s` complete — a fence signal
//!   operation on a queue happens-after all previously submitted work on
//!   that queue;
//! - a destroy that arrives while serials are outstanding parks the raw
//!   handles here, stamped with the newest serial; the bin drains on
//!   every successful fence wait and at every queue-idle point.
//!
//! On a failed wait (device loss) nothing is destroyed and no serial
//! advances — leaking handles on a dead device beats freeing memory the
//! GPU may still touch.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use super::ffi;

/// Raw Vulkan handles whose destruction is deferred behind a serial.
pub(super) enum Retired {
    Image {
        image: ffi::VkImage,
        view: ffi::VkImageView,
        memory: ffi::VkDeviceMemory,
    },
    Buffer {
        buffer: ffi::VkBuffer,
        memory: ffi::VkDeviceMemory,
    },
    View(ffi::VkImageView),
    Sampler(ffi::VkSampler),
}

// Raw Vulkan handles are plain pointers. The bin only ever holds its
// owning device's handles, every mutation sits behind the Mutex or an
// atomic, and destruction happens on whichever thread completes a fence
// wait — the same Send/Sync justification as `FenceWaiter`.
unsafe impl Send for RetireBin {}
unsafe impl Sync for RetireBin {}

pub(super) struct RetireBin {
    /// Serial handed to the most recent `vkQueueSubmit` (0 = none yet).
    submit_serial: AtomicU64,
    /// Highest serial whose fence wait returned `VK_SUCCESS`.
    completed_serial: AtomicU64,
    /// (must-complete-serial, resources) — destroyed once
    /// `completed_serial` reaches the stamp.
    retired: Mutex<Vec<(u64, Retired)>>,
}

impl RetireBin {
    pub(super) fn new() -> Self {
        Self {
            submit_serial: AtomicU64::new(0),
            completed_serial: AtomicU64::new(0),
            retired: Mutex::new(Vec::new()),
        }
    }

    /// Assign the next submission serial. Call under the queue lock,
    /// immediately after a successful `vkQueueSubmit`, so that serial
    /// order matches submission order.
    pub(super) fn next_serial(&self) -> u64 {
        self.submit_serial.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// A fence wait for `serial` returned `VK_SUCCESS`: every submission
    /// with serial `<= serial` has completed. Destroy everything parked
    /// at or below the new watermark.
    pub(super) fn complete(&self, device: ffi::VkDevice, serial: u64) {
        self.completed_serial.fetch_max(serial, Ordering::SeqCst);
        self.drain(device);
    }

    /// The queue went fully idle (`vkQueueWaitIdle` returned
    /// `VK_SUCCESS`): everything submitted so far is complete.
    pub(super) fn complete_all(&self, device: ffi::VkDevice) {
        let newest = self.submit_serial.load(Ordering::SeqCst);
        self.complete(device, newest);
    }

    /// Destroy now if the queue provably has nothing outstanding, else
    /// park behind the newest serial. Callers must have removed the
    /// resource from its registry first, so no later-recorded submission
    /// can pick it up — a racing submit can only make us wait longer,
    /// never destroy earlier.
    pub(super) fn retire(&self, device: ffi::VkDevice, resources: Retired) {
        let newest = self.submit_serial.load(Ordering::SeqCst);
        if self.completed_serial.load(Ordering::SeqCst) >= newest {
            unsafe { destroy(device, resources) };
        } else if let Ok(mut bin) = self.retired.lock() {
            bin.push((newest, resources));
        }
        // Poisoned lock: leak rather than risk a destroy-in-flight.
    }

    fn drain(&self, device: ffi::VkDevice) {
        let done = self.completed_serial.load(Ordering::SeqCst);
        let Ok(mut bin) = self.retired.lock() else {
            return;
        };
        let mut i = 0;
        while i < bin.len() {
            if bin[i].0 <= done {
                let (_, resources) = bin.swap_remove(i);
                unsafe { destroy(device, resources) };
            } else {
                i += 1;
            }
        }
    }
}

unsafe fn destroy(device: ffi::VkDevice, resources: Retired) {
    unsafe {
        match resources {
            Retired::Image {
                image,
                view,
                memory,
            } => {
                ffi::vkDestroyImageView(device, view, core::ptr::null());
                ffi::vkDestroyImage(device, image, core::ptr::null());
                ffi::vkFreeMemory(device, memory, core::ptr::null());
            }
            Retired::Buffer { buffer, memory } => {
                ffi::vkDestroyBuffer(device, buffer, core::ptr::null());
                ffi::vkFreeMemory(device, memory, core::ptr::null());
            }
            Retired::View(view) => {
                ffi::vkDestroyImageView(device, view, core::ptr::null());
            }
            Retired::Sampler(sampler) => {
                ffi::vkDestroySampler(device, sampler, core::ptr::null());
            }
        }
    }
}
