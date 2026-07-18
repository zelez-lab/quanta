//! Vulkan presentation surfaces — `VkSurfaceKHR` + `VkSwapchainKHR`.
//!
//! Mirrors the Metal surface contract: `surface_acquire` registers the
//! swapchain image as an ordinary texture (fresh handle per acquire) so
//! the render path targets it, `Timeout` maps from an acquire that
//! produced no image in time, `SurfaceOutdated` maps from
//! `VK_ERROR_OUT_OF_DATE_KHR`, and present is ordered after the
//! submitted render work by a semaphore signaled from the pre-present
//! layout transition (`COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR`).
//!
//! All WSI entry points are extension procs resolved at device
//! discovery (`SurfaceProcs::resolve`); when the loader lacks
//! `VK_KHR_surface`/`VK_KHR_swapchain` the device simply reports
//! `supports_surface_present() == false`.

use alloc::format;
use alloc::vec::Vec;

use crate::{PresentMode, QuantaError, SurfaceConfig, SurfaceTarget, Texture};

use super::VulkanDevice;
use super::device::VkTexture;
use super::ffi;
use super::helpers::{format_to_vulkan, vulkan_to_format};

/// WSI extension entry points, resolved once at device discovery.
/// Present only when the instance offers `VK_KHR_surface` and the
/// physical device offers `VK_KHR_swapchain` and every proc resolved.
pub(super) struct SurfaceProcs {
    pub destroy_surface: ffi::PfnVkDestroySurfaceKHR,
    pub create_headless: Option<ffi::PfnVkCreateHeadlessSurfaceEXT>,
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub create_xlib: Option<ffi::PfnVkCreateXlibSurfaceKHR>,
    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    pub create_android: Option<ffi::PfnVkCreateAndroidSurfaceKHR>,
    #[cfg_attr(not(target_os = "windows"), allow(dead_code))]
    pub create_win32: Option<ffi::PfnVkCreateWin32SurfaceKHR>,
    pub surface_support: ffi::PfnVkGetPhysicalDeviceSurfaceSupportKHR,
    pub surface_caps: ffi::PfnVkGetPhysicalDeviceSurfaceCapabilitiesKHR,
    pub surface_formats: ffi::PfnVkGetPhysicalDeviceSurfaceFormatsKHR,
    pub present_modes: ffi::PfnVkGetPhysicalDeviceSurfacePresentModesKHR,
    pub create_swapchain: ffi::PfnVkCreateSwapchainKHR,
    pub destroy_swapchain: ffi::PfnVkDestroySwapchainKHR,
    pub get_images: ffi::PfnVkGetSwapchainImagesKHR,
    pub acquire_next: ffi::PfnVkAcquireNextImageKHR,
    pub queue_present: ffi::PfnVkQueuePresentKHR,
}

impl SurfaceProcs {
    /// Resolve every WSI proc; `None` if any REQUIRED proc is missing
    /// (headless/xlib surface creation stay optional per-target).
    pub(super) fn resolve(
        instance: ffi::VkInstance,
        device: ffi::VkDevice,
        has_headless_ext: bool,
        has_xlib_ext: bool,
        has_android_ext: bool,
        has_win32_ext: bool,
    ) -> Option<Self> {
        unsafe fn ipfn(instance: ffi::VkInstance, name: &[u8]) -> *const core::ffi::c_void {
            unsafe { ffi::vkGetInstanceProcAddr(instance, name.as_ptr() as *const _) }
        }
        unsafe fn dpfn(device: ffi::VkDevice, name: &[u8]) -> *const core::ffi::c_void {
            unsafe { ffi::vkGetDeviceProcAddr(device, name.as_ptr() as *const _) }
        }
        macro_rules! resolve {
            (inst $name:literal as $ty:ty) => {{
                let p = unsafe { ipfn(instance, $name) };
                if p.is_null() {
                    return None;
                }
                // SAFETY: non-null proc of the documented signature.
                unsafe { core::mem::transmute::<*const core::ffi::c_void, $ty>(p) }
            }};
            (dev $name:literal as $ty:ty) => {{
                let p = unsafe { dpfn(device, $name) };
                if p.is_null() {
                    return None;
                }
                // SAFETY: non-null proc of the documented signature.
                unsafe { core::mem::transmute::<*const core::ffi::c_void, $ty>(p) }
            }};
            (inst opt $name:literal as $ty:ty) => {{
                let p = unsafe { ipfn(instance, $name) };
                if p.is_null() {
                    None
                } else {
                    // SAFETY: non-null proc of the documented signature.
                    Some(unsafe { core::mem::transmute::<*const core::ffi::c_void, $ty>(p) })
                }
            }};
        }

        Some(SurfaceProcs {
            destroy_surface: resolve!(inst b"vkDestroySurfaceKHR\0" as ffi::PfnVkDestroySurfaceKHR),
            create_headless: if has_headless_ext {
                resolve!(inst opt b"vkCreateHeadlessSurfaceEXT\0" as ffi::PfnVkCreateHeadlessSurfaceEXT)
            } else {
                None
            },
            create_xlib: if has_xlib_ext {
                resolve!(inst opt b"vkCreateXlibSurfaceKHR\0" as ffi::PfnVkCreateXlibSurfaceKHR)
            } else {
                None
            },
            create_android: if has_android_ext {
                resolve!(inst opt b"vkCreateAndroidSurfaceKHR\0" as ffi::PfnVkCreateAndroidSurfaceKHR)
            } else {
                None
            },
            create_win32: if has_win32_ext {
                resolve!(inst opt b"vkCreateWin32SurfaceKHR\0" as ffi::PfnVkCreateWin32SurfaceKHR)
            } else {
                None
            },
            surface_support: resolve!(inst b"vkGetPhysicalDeviceSurfaceSupportKHR\0"
                as ffi::PfnVkGetPhysicalDeviceSurfaceSupportKHR),
            surface_caps: resolve!(inst b"vkGetPhysicalDeviceSurfaceCapabilitiesKHR\0"
                as ffi::PfnVkGetPhysicalDeviceSurfaceCapabilitiesKHR),
            surface_formats: resolve!(inst b"vkGetPhysicalDeviceSurfaceFormatsKHR\0"
                as ffi::PfnVkGetPhysicalDeviceSurfaceFormatsKHR),
            present_modes: resolve!(inst b"vkGetPhysicalDeviceSurfacePresentModesKHR\0"
                as ffi::PfnVkGetPhysicalDeviceSurfacePresentModesKHR),
            create_swapchain: resolve!(dev b"vkCreateSwapchainKHR\0"
                as ffi::PfnVkCreateSwapchainKHR),
            destroy_swapchain: resolve!(dev b"vkDestroySwapchainKHR\0"
                as ffi::PfnVkDestroySwapchainKHR),
            get_images: resolve!(dev b"vkGetSwapchainImagesKHR\0"
                as ffi::PfnVkGetSwapchainImagesKHR),
            acquire_next: resolve!(dev b"vkAcquireNextImageKHR\0"
                as ffi::PfnVkAcquireNextImageKHR),
            queue_present: resolve!(dev b"vkQueuePresentKHR\0" as ffi::PfnVkQueuePresentKHR),
        })
    }
}

/// A live surface: the `VkSurfaceKHR`, its swapchain, per-image views,
/// and the per-surface sync/transition objects.
pub(super) struct VkSurfaceEntry {
    pub surface: ffi::VkSurfaceKHR,
    pub swapchain: ffi::VkSwapchainKHR,
    pub images: Vec<ffi::VkImage>,
    pub views: Vec<ffi::VkImageView>,
    pub width: u32,
    pub height: u32,
    pub format: crate::Format,
    pub vk_format: u32,
    pub present_mode: PresentMode,
    /// Set when a frame was discarded un-presented: Vulkan has no
    /// un-present, so the image only returns through a swapchain
    /// rebuild — done lazily on the next acquire.
    pub needs_rebuild: bool,
    /// Reused for every acquire (one frame in flight at a time per
    /// surface, matching the Metal drawable model).
    pub acquire_fence: ffi::VkFence,
    /// Signaled by the pre-present transition submit; waited by
    /// `vkQueuePresentKHR`.
    pub present_sem: ffi::VkSemaphore,
    /// One dedicated transition command buffer + fence, recycled with
    /// a bounded wait on the PREVIOUS frame's transition (depth-1
    /// pipelining — present never CPU-waits the current frame).
    pub present_cb: ffi::VkCommandBuffer,
    pub present_fence: ffi::VkFence,
}

/// An acquired, not-yet-presented frame.
pub(super) struct VkSurfaceFrame {
    pub surface: u64,
    pub image_index: u32,
    pub texture_handle: u64,
}

impl VulkanDevice {
    fn procs(&self) -> Result<&SurfaceProcs, QuantaError> {
        self.surface_procs.as_ref().ok_or_else(|| {
            QuantaError::not_supported(
                "surface presentation needs VK_KHR_surface + VK_KHR_swapchain, \
                 which this Vulkan environment does not offer",
            )
        })
    }

    pub(crate) fn surface_create_impl(
        &self,
        target: &SurfaceTarget,
        config: &SurfaceConfig,
    ) -> Result<u64, QuantaError> {
        let procs = self.procs()?;

        // 1. Create the VkSurfaceKHR for the requested target.
        let mut surface: ffi::VkSurfaceKHR = ffi::null_handle();
        match target {
            SurfaceTarget::Headless => {
                let create = procs.create_headless.ok_or_else(|| {
                    QuantaError::not_supported(
                        "headless surfaces need VK_EXT_headless_surface (not offered here)",
                    )
                })?;
                let info = ffi::VkHeadlessSurfaceCreateInfoEXT {
                    s_type: ffi::VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
                    p_next: core::ptr::null(),
                    flags: 0,
                };
                let r = unsafe { create(self.instance, &info, core::ptr::null(), &mut surface) };
                if r != ffi::VK_SUCCESS {
                    return Err(QuantaError::internal("vkCreateHeadlessSurfaceEXT failed"));
                }
            }
            SurfaceTarget::Xlib { display, window } => {
                let create = procs.create_xlib.ok_or_else(|| {
                    QuantaError::not_supported(
                        "Xlib surfaces need VK_KHR_xlib_surface (not offered here)",
                    )
                })?;
                if display.is_null() {
                    return Err(QuantaError::invalid_param(
                        "SurfaceTarget::Xlib display pointer is null",
                    ));
                }
                let info = ffi::VkXlibSurfaceCreateInfoKHR {
                    s_type: ffi::VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                    p_next: core::ptr::null(),
                    flags: 0,
                    dpy: *display,
                    window: *window,
                };
                let r = unsafe { create(self.instance, &info, core::ptr::null(), &mut surface) };
                if r != ffi::VK_SUCCESS {
                    return Err(QuantaError::internal("vkCreateXlibSurfaceKHR failed"));
                }
            }
            SurfaceTarget::AndroidWindow { a_native_window } => {
                let create = procs.create_android.ok_or_else(|| {
                    QuantaError::not_supported(
                        "Android surfaces need VK_KHR_android_surface (not offered here)",
                    )
                })?;
                if a_native_window.is_null() {
                    return Err(QuantaError::invalid_param(
                        "SurfaceTarget::AndroidWindow a_native_window pointer is null",
                    ));
                }
                let info = ffi::VkAndroidSurfaceCreateInfoKHR {
                    s_type: ffi::VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
                    p_next: core::ptr::null(),
                    flags: 0,
                    window: *a_native_window,
                };
                let r = unsafe { create(self.instance, &info, core::ptr::null(), &mut surface) };
                if r != ffi::VK_SUCCESS {
                    return Err(QuantaError::internal("vkCreateAndroidSurfaceKHR failed"));
                }
            }
            SurfaceTarget::Win32 { hinstance, hwnd } => {
                let create = procs.create_win32.ok_or_else(|| {
                    QuantaError::not_supported(
                        "Win32 surfaces need VK_KHR_win32_surface (not offered here)",
                    )
                })?;
                if hinstance.is_null() {
                    return Err(QuantaError::invalid_param(
                        "SurfaceTarget::Win32 hinstance pointer is null",
                    ));
                }
                if hwnd.is_null() {
                    return Err(QuantaError::invalid_param(
                        "SurfaceTarget::Win32 hwnd pointer is null",
                    ));
                }
                let info = ffi::VkWin32SurfaceCreateInfoKHR {
                    s_type: ffi::VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                    p_next: core::ptr::null(),
                    flags: 0,
                    hinstance: *hinstance,
                    hwnd: *hwnd,
                };
                let r = unsafe { create(self.instance, &info, core::ptr::null(), &mut surface) };
                if r != ffi::VK_SUCCESS {
                    return Err(QuantaError::internal("vkCreateWin32SurfaceKHR failed"));
                }
            }
            _ => {
                return Err(QuantaError::not_supported(
                    "this surface target is not available on the Vulkan backend",
                ));
            }
        }

        // 2. The single graphics+compute queue must be able to present.
        let mut supported = 0u32;
        let r = unsafe {
            (procs.surface_support)(
                self.physical_device,
                self.queue_family,
                surface,
                &mut supported,
            )
        };
        if r != ffi::VK_SUCCESS || supported == 0 {
            unsafe { (procs.destroy_surface)(self.instance, surface, core::ptr::null()) };
            return Err(QuantaError::not_supported(
                "the device queue cannot present to this surface",
            ));
        }

        // 3. Build the swapchain + per-surface objects. The negotiated
        // format may differ from `config.format` (a preference on
        // Vulkan) — it is what acquired frames must report.
        let (swapchain, images, views, vk_format, format, extent) =
            self.build_swapchain(procs, surface, config, ffi::null_handle())?;

        let fence_info = ffi::VkFenceCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
        };
        let mut acquire_fence = ffi::null_handle();
        let signaled_info = ffi::VkFenceCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            p_next: core::ptr::null(),
            // Signaled so the FIRST present's bounded recycle-wait
            // passes immediately.
            flags: ffi::VK_FENCE_CREATE_SIGNALED_BIT,
        };
        let mut present_fence = ffi::null_handle();
        let sem_info = ffi::VkSemaphoreCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
        };
        let mut present_sem = ffi::null_handle();
        unsafe {
            if ffi::vkCreateFence(
                self.device,
                &fence_info,
                core::ptr::null(),
                &mut acquire_fence,
            ) != ffi::VK_SUCCESS
                || ffi::vkCreateFence(
                    self.device,
                    &signaled_info,
                    core::ptr::null(),
                    &mut present_fence,
                ) != ffi::VK_SUCCESS
                || ffi::vkCreateSemaphore(
                    self.device,
                    &sem_info,
                    core::ptr::null(),
                    &mut present_sem,
                ) != ffi::VK_SUCCESS
            {
                return Err(QuantaError::out_of_memory());
            }
        }
        let present_cb = self.alloc_command_buffer()?;

        let handle = self.alloc_handle();
        self.vk_surfaces
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                VkSurfaceEntry {
                    surface,
                    swapchain,
                    images,
                    views,
                    width: extent.0,
                    height: extent.1,
                    format,
                    vk_format,
                    present_mode: config.present_mode,
                    needs_rebuild: false,
                    acquire_fence,
                    present_sem,
                    present_cb,
                    present_fence,
                },
            );
        Ok(handle)
    }

    /// Create the swapchain and per-image views for `surface`.
    ///
    /// Returns the swapchain, its images and views, the negotiated
    /// `VkFormat`, the quanta [`Format`](crate::Format) that format maps
    /// to (the value acquired frames report), and the extent. The
    /// negotiated format is chosen by [`Self::negotiate_surface_format`]
    /// and is not necessarily `config.format` — see that method.
    #[allow(clippy::type_complexity)]
    fn build_swapchain(
        &self,
        procs: &SurfaceProcs,
        surface: ffi::VkSurfaceKHR,
        config: &SurfaceConfig,
        old_swapchain: ffi::VkSwapchainKHR,
    ) -> Result<
        (
            ffi::VkSwapchainKHR,
            Vec<ffi::VkImage>,
            Vec<ffi::VkImageView>,
            u32,
            crate::Format,
            (u32, u32),
        ),
        QuantaError,
    > {
        // Capabilities: extent, image counts, usage, transform, alpha.
        let mut caps = unsafe { core::mem::zeroed::<ffi::VkSurfaceCapabilitiesKHR>() };
        let r = unsafe { (procs.surface_caps)(self.physical_device, surface, &mut caps) };
        if r != ffi::VK_SUCCESS {
            return Err(QuantaError::internal(
                "vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed",
            ));
        }
        // A fixed current_extent (a real window) wins over the config;
        // the special 0xFFFFFFFF means the surface takes any size
        // (headless), so the config decides.
        let extent = if caps.current_extent.width == u32::MAX {
            (
                config
                    .width
                    .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                config
                    .height
                    .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            )
        } else {
            (caps.current_extent.width, caps.current_extent.height)
        };

        // Pixel format: negotiate against what the surface offers,
        // preferring the configured format but falling back so a surface
        // that offers a different color-renderable format (e.g. Android's
        // RGBA8 where the config asks for BGRA8) still works. All choices
        // use the SRGB-nonlinear colorspace.
        let (vk_format, format) = self.negotiate_surface_format(procs, surface, config)?;

        // Present mode: FIFO is always available per spec; others must
        // be advertised.
        let wanted_mode = match config.present_mode {
            PresentMode::Fifo => ffi::VK_PRESENT_MODE_FIFO_KHR,
            PresentMode::Immediate => ffi::VK_PRESENT_MODE_IMMEDIATE_KHR,
            PresentMode::Mailbox => ffi::VK_PRESENT_MODE_MAILBOX_KHR,
        };
        if wanted_mode != ffi::VK_PRESENT_MODE_FIFO_KHR {
            let mut n = 0u32;
            unsafe {
                (procs.present_modes)(self.physical_device, surface, &mut n, core::ptr::null_mut())
            };
            let mut modes = alloc::vec![0u32; n as usize];
            unsafe {
                (procs.present_modes)(self.physical_device, surface, &mut n, modes.as_mut_ptr())
            };
            if !modes.iter().take(n as usize).any(|&m| m == wanted_mode) {
                return Err(QuantaError::not_supported(format!(
                    "present mode {:?} not offered by this surface",
                    config.present_mode
                )));
            }
        }

        // Triple-buffer when allowed (matches the Metal drawable pool).
        let mut min_images = caps.min_image_count.max(3);
        if caps.max_image_count > 0 {
            min_images = min_images.min(caps.max_image_count);
        }
        let mut usage = ffi::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if caps.supported_usage_flags & ffi::VK_IMAGE_USAGE_TRANSFER_SRC_BIT != 0 {
            usage |= ffi::VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        }
        let composite_alpha =
            if caps.supported_composite_alpha & ffi::VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR != 0 {
                ffi::VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
            } else {
                // Lowest advertised bit — every surface offers at least one.
                caps.supported_composite_alpha & caps.supported_composite_alpha.wrapping_neg()
            };

        let create = ffi::VkSwapchainCreateInfoKHR {
            s_type: ffi::VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            p_next: core::ptr::null(),
            flags: 0,
            surface,
            min_image_count: min_images,
            image_format: vk_format,
            image_color_space: ffi::VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
            image_extent: ffi::VkExtent2D {
                width: extent.0,
                height: extent.1,
            },
            image_array_layers: 1,
            image_usage: usage,
            image_sharing_mode: ffi::VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: core::ptr::null(),
            pre_transform: caps.current_transform,
            composite_alpha,
            present_mode: wanted_mode,
            clipped: 1,
            old_swapchain,
        };
        let mut swapchain: ffi::VkSwapchainKHR = ffi::null_handle();
        let r = unsafe {
            (procs.create_swapchain)(self.device, &create, core::ptr::null(), &mut swapchain)
        };
        if r != ffi::VK_SUCCESS {
            return Err(QuantaError::internal("vkCreateSwapchainKHR failed"));
        }

        // Fetch images + create one view per image (the render path's
        // framebuffer needs the view).
        let mut n = 0u32;
        unsafe { (procs.get_images)(self.device, swapchain, &mut n, core::ptr::null_mut()) };
        let mut images = alloc::vec![ffi::null_handle(); n as usize];
        unsafe { (procs.get_images)(self.device, swapchain, &mut n, images.as_mut_ptr()) };

        let mut views = Vec::with_capacity(n as usize);
        for &image in images.iter().take(n as usize) {
            let view_info = ffi::VkImageViewCreateInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                p_next: core::ptr::null(),
                flags: 0,
                image,
                view_type: ffi::VK_IMAGE_VIEW_TYPE_2D,
                format: vk_format,
                components: ffi::VkComponentMapping::default(),
                subresource_range: ffi::VkImageSubresourceRange {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            let mut view = ffi::null_handle();
            let r = unsafe {
                ffi::vkCreateImageView(self.device, &view_info, core::ptr::null(), &mut view)
            };
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::out_of_memory());
            }
            views.push(view);
        }

        Ok((swapchain, images, views, vk_format, format, extent))
    }

    /// Choose the swapchain's pixel format from what the surface offers,
    /// all with `VK_COLOR_SPACE_SRGB_NONLINEAR_KHR`. First hit wins:
    ///
    /// 1. the configured format (`config.format`),
    /// 2. `BGRA8`,
    /// 3. `RGBA8` (skipped if it duplicates step 1),
    /// 4. otherwise the first offered pair whose `VkFormat` maps to a
    ///    quanta [`Format`](crate::Format).
    ///
    /// Returns the chosen `(VkFormat, Format)`. `NotSupported` — listing
    /// the offered formats — is returned only when a surface offers no
    /// SRGB-nonlinear format quanta can express, which is the sole
    /// unsatisfiable case left. The legacy single-entry
    /// `VK_FORMAT_UNDEFINED` case (the surface accepts any format) is
    /// treated as "step 1 is available".
    fn negotiate_surface_format(
        &self,
        procs: &SurfaceProcs,
        surface: ffi::VkSurfaceKHR,
        config: &SurfaceConfig,
    ) -> Result<(u32, crate::Format), QuantaError> {
        let mut count = 0u32;
        unsafe {
            (procs.surface_formats)(
                self.physical_device,
                surface,
                &mut count,
                core::ptr::null_mut(),
            )
        };
        let mut formats =
            alloc::vec![ffi::VkSurfaceFormatKHR { format: 0, color_space: 0 }; count as usize];
        unsafe {
            (procs.surface_formats)(
                self.physical_device,
                surface,
                &mut count,
                formats.as_mut_ptr(),
            )
        };
        formats.truncate(count as usize);

        // Legacy behavior: a single VK_FORMAT_UNDEFINED entry means the
        // surface imposes no format — step 1 (the configured format) is
        // free to use.
        if let [only] = formats.as_slice()
            && only.format == ffi::VK_FORMAT_UNDEFINED
        {
            return Ok((format_to_vulkan(config.format), config.format));
        }

        let srgb = ffi::VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        let offered = |vk: u32| -> bool {
            formats
                .iter()
                .any(|f| f.format == vk && f.color_space == srgb)
        };

        // Steps 1–3: the preference chain, deduped against the config.
        let requested_vk = format_to_vulkan(config.format);
        let mut preferred = alloc::vec![(requested_vk, config.format)];
        for cand in [crate::Format::BGRA8, crate::Format::RGBA8] {
            let vk = format_to_vulkan(cand);
            if vk != requested_vk {
                preferred.push((vk, cand));
            }
        }
        for (vk, fmt) in preferred {
            if offered(vk) {
                return Ok((vk, fmt));
            }
        }

        // Step 4: the first offered SRGB-nonlinear pair quanta can name.
        for f in &formats {
            if f.color_space == srgb
                && let Some(fmt) = vulkan_to_format(f.format)
            {
                return Ok((f.format, fmt));
            }
        }

        // Nothing expressible: report every offered (VkFormat,
        // VkColorSpace) pair so a caller on a surface with only exotic
        // formats can see exactly what it reported.
        let offered_list: Vec<(u32, u32)> =
            formats.iter().map(|f| (f.format, f.color_space)).collect();
        Err(QuantaError::not_supported(format!(
            "surface offers no SRGB-nonlinear format Quanta can express \
             (requested {:?}); offered (VkFormat, VkColorSpace) pairs: {:?}",
            config.format, offered_list
        )))
    }

    pub(crate) fn surface_configure_impl(
        &self,
        surface: u64,
        config: &SurfaceConfig,
    ) -> Result<(), QuantaError> {
        let procs = self.procs()?;
        let mut surfaces = self
            .vk_surfaces
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let entry = surfaces
            .get_mut(&surface)
            .ok_or_else(|| QuantaError::not_found("surface handle not found"))?;

        // Drain in-flight work referencing the old images, then rebuild.
        unsafe { ffi::vkQueueWaitIdle(self.queue) };
        for &view in &entry.views {
            unsafe { ffi::vkDestroyImageView(self.device, view, core::ptr::null()) };
        }
        let (swapchain, images, views, vk_format, format, extent) =
            self.build_swapchain(procs, entry.surface, config, entry.swapchain)?;
        unsafe { (procs.destroy_swapchain)(self.device, entry.swapchain, core::ptr::null()) };

        entry.swapchain = swapchain;
        entry.images = images;
        entry.views = views;
        entry.vk_format = vk_format;
        entry.width = extent.0;
        entry.height = extent.1;
        entry.format = format;
        entry.present_mode = config.present_mode;
        entry.needs_rebuild = false;
        Ok(())
    }

    pub(crate) fn surface_acquire_impl(&self, surface: u64) -> Result<(u64, Texture), QuantaError> {
        let procs = self.procs()?;

        // A discarded frame's image only returns through a swapchain
        // rebuild (Vulkan has no un-present) — do it lazily here so the
        // recycling contract holds; a skipped frame costs one rebuild.
        let rebuild_config = {
            let surfaces = self
                .vk_surfaces
                .read()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            let entry = surfaces
                .get(&surface)
                .ok_or_else(|| QuantaError::not_found("surface handle not found"))?;
            if entry.needs_rebuild {
                let mut config = SurfaceConfig::new(entry.width, entry.height);
                config.format = entry.format;
                config.present_mode = entry.present_mode;
                Some(config)
            } else {
                None
            }
        };
        if let Some(config) = rebuild_config {
            self.surface_configure_impl(surface, &config)?;
        }

        let surfaces = self
            .vk_surfaces
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let entry = surfaces
            .get(&surface)
            .ok_or_else(|| QuantaError::not_found("surface handle not found"))?;

        // Acquire with a 1 s deadline; the fence signals when the
        // presentation engine has released the image.
        unsafe { ffi::vkResetFences(self.device, 1, &entry.acquire_fence) };
        let mut index = 0u32;
        let r = unsafe {
            (procs.acquire_next)(
                self.device,
                entry.swapchain,
                1_000_000_000,
                ffi::null_handle(),
                entry.acquire_fence,
                &mut index,
            )
        };
        let mut heal_suboptimal = false;
        match r {
            ffi::VK_SUCCESS => {}
            ffi::VK_SUBOPTIMAL_KHR => {
                // The image is validly acquired — finish this frame,
                // but rebuild on the NEXT acquire so the swapchain
                // re-matches the surface (a resize the driver reports
                // as merely suboptimal heals without any error).
                heal_suboptimal = true;
            }
            ffi::VK_TIMEOUT => {
                return Err(QuantaError::timeout()
                    .with_context("surface_acquire: no image available within 1 s"));
            }
            ffi::VK_ERROR_OUT_OF_DATE_KHR => {
                return Err(QuantaError::surface_outdated(
                    "swapchain out of date — reconfigure the surface with the new extent",
                ));
            }
            other => {
                return Err(QuantaError::internal(format!(
                    "vkAcquireNextImageKHR failed (VkResult {other})"
                )));
            }
        }
        unsafe {
            ffi::vkWaitForFences(self.device, 1, &entry.acquire_fence, 1, u64::MAX);
        }

        // Register the swapchain image as an ordinary texture so the
        // render path can target it. Fresh handle per acquire; removed
        // at present/discard (same lifetime contract as Metal).
        let texture_handle = self.alloc_handle();
        self.textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                texture_handle,
                VkTexture {
                    image: entry.images[index as usize],
                    view: entry.views[index as usize],
                    memory: ffi::null_handle(),
                    width: entry.width,
                    height: entry.height,
                    format: entry.vk_format,
                    mip_levels: 1,
                    current_layout: core::sync::atomic::AtomicU32::new(
                        ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                    ),
                },
            );

        let frame = self.alloc_handle();
        self.vk_surface_frames
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                frame,
                VkSurfaceFrame {
                    surface,
                    image_index: index,
                    texture_handle,
                },
            );

        if heal_suboptimal {
            drop(surfaces);
            if let Ok(mut all) = self.vk_surfaces.write()
                && let Some(entry) = all.get_mut(&surface)
            {
                entry.needs_rebuild = true;
            }
            let surfaces = self
                .vk_surfaces
                .read()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            let entry = surfaces
                .get(&surface)
                .ok_or_else(|| QuantaError::not_found("surface handle not found"))?;
            return Ok((
                frame,
                Texture {
                    handle: texture_handle,
                    width: entry.width,
                    height: entry.height,
                    format: entry.format,
                    // Swapchain images are always single-sample.
                    sample_count: 1,
                    device: None,
                    live: false,
                },
            ));
        }

        Ok((
            frame,
            Texture {
                handle: texture_handle,
                width: entry.width,
                height: entry.height,
                format: entry.format,
                // Swapchain images are always single-sample.
                sample_count: 1,
                device: None,
                live: false,
            },
        ))
    }

    pub(crate) fn surface_format_impl(&self, surface: u64) -> Result<crate::Format, QuantaError> {
        let surfaces = self
            .vk_surfaces
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        surfaces
            .get(&surface)
            .map(|entry| entry.format)
            .ok_or_else(|| QuantaError::not_found("surface handle not found"))
    }

    pub(crate) fn surface_present_impl(&self, surface: u64, frame: u64) -> Result<(), QuantaError> {
        let procs = self.procs()?;
        let frame_entry = self
            .vk_surface_frames
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&frame)
            .ok_or_else(|| QuantaError::not_found("surface frame not found"))?;
        if frame_entry.surface != surface {
            return Err(QuantaError::invalid_param(
                "frame does not belong to this surface",
            ));
        }
        let surfaces = self
            .vk_surfaces
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let entry = surfaces
            .get(&surface)
            .ok_or_else(|| QuantaError::not_found("surface handle not found"))?;

        // Unregister the frame texture; its tracked layout tells the
        // transition below where the image currently is (the render
        // pass leaves COLOR_ATTACHMENT_OPTIMAL; UNDEFINED if the frame
        // was never rendered to — presenting it then shows garbage,
        // same as Metal).
        let old_layout = self
            .textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&frame_entry.texture_handle)
            .map(|t| t.current_layout.load(core::sync::atomic::Ordering::Relaxed))
            .unwrap_or(ffi::VK_IMAGE_LAYOUT_UNDEFINED);

        // Recycle the dedicated transition CB: bounded wait on the
        // PREVIOUS frame's transition only (depth-1 pipelining).
        unsafe {
            ffi::vkWaitForFences(self.device, 1, &entry.present_fence, 1, u64::MAX);
            ffi::vkResetFences(self.device, 1, &entry.present_fence);
        }

        // Record + submit the COLOR_ATTACHMENT → PRESENT_SRC transition,
        // signaling the present semaphore. Pipeline barriers order
        // against all earlier submissions on the queue, so this waits
        // (on the GPU timeline, not the CPU) for the frame's render.
        let begin = ffi::VkCommandBufferBeginInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: core::ptr::null(),
        };
        unsafe {
            if ffi::vkBeginCommandBuffer(entry.present_cb, &begin) != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            let barrier = ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: ffi::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dst_access_mask: 0,
                old_layout,
                new_layout: ffi::VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                image: entry.images[frame_entry.image_index as usize],
                subresource_range: ffi::VkImageSubresourceRange {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            ffi::vkCmdPipelineBarrier(
                entry.present_cb,
                ffi::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                ffi::VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &barrier,
            );
            if ffi::vkEndCommandBuffer(entry.present_cb) != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            let submit = ffi::VkSubmitInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_SUBMIT_INFO,
                p_next: core::ptr::null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: core::ptr::null(),
                p_wait_dst_stage_mask: core::ptr::null(),
                command_buffer_count: 1,
                p_command_buffers: &entry.present_cb,
                signal_semaphore_count: 1,
                p_signal_semaphores: &entry.present_sem,
            };
            if ffi::vkQueueSubmit(self.queue, 1, &submit, entry.present_fence) != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }

            let present = ffi::VkPresentInfoKHR {
                s_type: ffi::VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                p_next: core::ptr::null(),
                wait_semaphore_count: 1,
                p_wait_semaphores: &entry.present_sem,
                swapchain_count: 1,
                p_swapchains: &entry.swapchain,
                p_image_indices: &frame_entry.image_index,
                p_results: core::ptr::null_mut(),
            };
            let result = (procs.queue_present)(self.queue, &present);
            drop(surfaces);
            match result {
                ffi::VK_SUCCESS => Ok(()),
                ffi::VK_SUBOPTIMAL_KHR => {
                    // Presented fine, but the swapchain no longer
                    // matches the surface — heal on the next acquire.
                    if let Ok(mut all) = self.vk_surfaces.write()
                        && let Some(entry) = all.get_mut(&surface)
                    {
                        entry.needs_rebuild = true;
                    }
                    Ok(())
                }
                ffi::VK_ERROR_OUT_OF_DATE_KHR => Err(QuantaError::surface_outdated(
                    "swapchain out of date at present — reconfigure the surface",
                )),
                _ => Err(QuantaError::submit_failed()),
            }
        }
    }

    pub(crate) fn surface_discard_impl(
        &self,
        _surface: u64,
        frame: u64,
    ) -> Result<(), QuantaError> {
        // Vulkan has no un-present: the acquired image stays checked
        // out until a present or a swapchain rebuild. Mark the surface
        // so the NEXT acquire rebuilds and every image returns — the
        // recycling contract holds at the cost of one rebuild per
        // skipped frame.
        if let Some(frame_entry) = self
            .vk_surface_frames
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&frame)
        {
            self.textures
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?
                .remove(&frame_entry.texture_handle);
            if let Ok(mut surfaces) = self.vk_surfaces.write()
                && let Some(entry) = surfaces.get_mut(&frame_entry.surface)
            {
                entry.needs_rebuild = true;
            }
        }
        Ok(())
    }

    pub(crate) fn surface_destroy_impl(&self, surface: u64) -> Result<(), QuantaError> {
        let procs = self.procs()?;
        let Some(entry) = self
            .vk_surfaces
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&surface)
        else {
            return Ok(());
        };
        unsafe {
            ffi::vkQueueWaitIdle(self.queue);
            for &view in &entry.views {
                ffi::vkDestroyImageView(self.device, view, core::ptr::null());
            }
            (procs.destroy_swapchain)(self.device, entry.swapchain, core::ptr::null());
            (procs.destroy_surface)(self.instance, entry.surface, core::ptr::null());
            ffi::vkDestroyFence(self.device, entry.acquire_fence, core::ptr::null());
            ffi::vkDestroyFence(self.device, entry.present_fence, core::ptr::null());
            ffi::vkDestroySemaphore(self.device, entry.present_sem, core::ptr::null());
        }
        if let Ok(mut pool) = self.cmd_buffer_pool.lock() {
            pool.push(entry.present_cb);
        }
        Ok(())
    }
}
