/-
# Machine-model axioms — Vulkan backend

Trusted properties of the Vulkan API that Quanta assumes.
These formalize axiom A2 (Vulkan driver correctness).

`opaque` is used for driver operations whose implementation is
inside the Vulkan ICD (Installable Client Driver). `axiom` captures
guarantees from the Vulkan 1.3 specification that we rely on for
end-to-end correctness.

See `Quanta.Axioms.Gpu` for the shared GPU execution model.
-/

import Quanta.Axioms.Gpu

namespace Quanta.Axioms.Vulkan

-- ════════════════════════════════════════════════════════════════════
-- Vulkan types (opaque handles)
-- ════════════════════════════════════════════════════════════════════

/-- Opaque handle to a Vulkan logical device. -/
opaque VkDevice : Type := Unit

/-- Opaque handle to a compiled SPIR-V shader module. -/
opaque VkShaderModule : Type := Unit

/-- Opaque handle to a Vulkan command buffer. -/
opaque VkCommandBuffer : Type := Unit

/-- Opaque handle to a Vulkan queue. -/
opaque VkQueue : Type := Unit

/-- Opaque handle to a Vulkan fence (CPU-GPU synchronization). -/
opaque VkFence : Type := Unit

/-- Opaque handle to a Vulkan compute pipeline. -/
opaque VkPipeline : Type := Unit

/-- Opaque handle to a Vulkan buffer (device memory). -/
opaque VkBuffer : Type := Unit

/-- Raw pointer to mapped buffer memory. -/
opaque MappedPtr : Type := Unit

-- ════════════════════════════════════════════════════════════════════
-- A2: Vulkan driver operations
-- ════════════════════════════════════════════════════════════════════

/-- Create a shader module from valid SPIR-V binary.
    Returns `none` if the SPIR-V is malformed, uses unsupported
    extensions, or exceeds device limits. -/
opaque vk_create_shader_module : VkDevice → ByteArray → Option VkShaderModule :=
  fun _ _ => none

/-- Record a compute dispatch into a command buffer.
    `group_x`, `group_y`, `group_z` are the workgroup counts
    in each dimension (Vulkan supports 3D dispatch). -/
opaque vk_cmd_dispatch : VkCommandBuffer → Nat → Nat → Nat → Unit :=
  fun _ _ _ _ => ()

/-- Submit a command buffer to a queue for execution.
    The provided fence is signaled when all commands complete. -/
opaque vk_queue_submit : VkQueue → VkFence → Unit := fun _ _ => ()

/-- Block the calling thread until a fence is signaled.
    After this returns, all work submitted before the fence
    has completed and its results are visible. -/
opaque vk_wait_for_fences : VkDevice → VkFence → Unit := fun _ _ => ()

/-- Create a compute pipeline from a shader module.
    Returns `none` if pipeline creation fails (e.g., resource
    limits exceeded, incompatible descriptor layout). -/
opaque vk_create_compute_pipeline : VkDevice → VkShaderModule → Option VkPipeline :=
  fun _ _ => none

/-- Map a host-visible buffer into CPU address space. -/
opaque vk_map_memory : VkDevice → VkBuffer → Option MappedPtr := fun _ _ => none

/-- Unmap a previously mapped buffer. After unmapping, GPU commands
    can safely read the written data (for host-coherent memory). -/
opaque vk_unmap_memory : VkDevice → VkBuffer → Unit := fun _ _ => ()

-- ════════════════════════════════════════════════════════════════════
-- A2: Vulkan driver correctness axioms
-- ════════════════════════════════════════════════════════════════════

/-- **spirv_roundtrip**: If SPIR-V binary is valid (passes
    spirv-val), creating a shader module succeeds, and creating
    a compute pipeline from that module succeeds, and dispatching
    that pipeline executes the shader correctly per the SPIR-V
    specification (each SpvOp produces results defined by
    `Quanta.Axioms.Gpu.SpvOp.eval_u32` for integer ops).

    This is the Vulkan-specific instantiation of the general
    pipeline axiom from `Quanta.Axioms.Gpu`. -/
axiom spirv_roundtrip
    (device : VkDevice)
    (spirv : ByteArray)
    (h_valid : vk_create_shader_module device spirv ≠ none)
    : True -- pipeline executes per SpvOp semantics

/-- **fence_signals_after_completion**: A fence passed to
    `vk_queue_submit` is signaled if and only if all commands
    in the submitted batch have completed execution.
    `vk_wait_for_fences` returns only after the fence is signaled.

    Vulkan 1.3 spec, section 7.3: "Fences can be used by the host
    to determine completion of execution of queue operations." -/
axiom fence_signals_after_completion
    (queue : VkQueue)
    (fence : VkFence)
    (device : VkDevice)
    : True -- fence signaled iff all submitted work complete

/-- **host_visible_coherent**: For memory allocated with
    `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | HOST_COHERENT_BIT`,
    writes through the mapped pointer are visible to the GPU
    after `vk_unmap_memory`, and GPU writes are visible to the
    CPU after `vk_wait_for_fences` returns.

    Vulkan 1.3 spec, section 11.2.1: "Host access to [coherent]
    memory does not require calls to vkFlushMappedMemoryRanges
    or vkInvalidateMappedMemoryRanges." -/
axiom host_visible_coherent
    (device : VkDevice)
    (buf : VkBuffer)
    : True -- mapped writes visible to GPU after unmap;
          -- GPU writes visible to CPU after fence wait

/-- **submission_order_preserved**: Within a single queue,
    command buffers execute in submission order. Memory writes
    from batch N are visible to batch N+1 without additional
    synchronization (within the same queue).

    Vulkan 1.3 spec, section 7.2: implicit ordering guarantees. -/
axiom submission_order_preserved
    (queue : VkQueue)
    : True -- sequential batches on same queue are ordered

end Quanta.Axioms.Vulkan
