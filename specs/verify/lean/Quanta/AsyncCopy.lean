/-
Async memory copy (step 044).

An "async copy" is a buffer-to-buffer or texture-to-buffer transfer
issued on a dedicated DMA / transfer queue, running concurrently
with the compute / graphics queues. Backends:

- Vulkan: a queue from a family with `VK_QUEUE_TRANSFER_BIT` (often
  a dedicated DMA engine on discrete GPUs); `vkCmdCopyBuffer` /
  `vkCmdCopyImage`.
- Metal: `MTLCommandQueue` with `MTLBlitCommandEncoder`.
- WebGPU: `GPUQueue.writeBuffer` / `copyBufferToBuffer` are async.
- CPU: serial memcpy.

The proven contract: copies issued on the same async-copy queue
execute in submission order. The model captures this as a FIFO of
`(dst, src, size)` triples + a `live` flag.
-/

namespace Quanta.AsyncCopy

/-- A single recorded copy: destination handle, source handle,
    byte length. -/
abbrev CopyOp := Nat × Nat × Nat

/-- Async-copy queue state. -/
structure Queue where
  submitted : List CopyOp
  live      : Bool
  deriving Repr

/-- Create a fresh async-copy queue. -/
def Queue.create : Queue :=
  { submitted := [], live := true }

/-- Submit a `(dst, src, size)` copy. Fails when the queue is
    destroyed; otherwise appends to `submitted`. -/
def Queue.submitCopy (q : Queue) (dst src size : Nat) : Option Queue :=
  if q.live then
    some { q with submitted := q.submitted ++ [(dst, src, size)] }
  else
    none

/-- Mark the queue destroyed. -/
def Queue.destroy (q : Queue) : Queue :=
  { q with live := false }

/- ============================================================ -/
/-                          THEOREMS                              -/
/- ============================================================ -/

/-- T7800 — `create` produces a live queue with empty FIFO. -/
theorem t7800_create_shape :
    Queue.create.submitted = []
    ∧ Queue.create.live = true := by
  unfold Queue.create; exact ⟨rfl, rfl⟩

/-- T7801 — `submitCopy` appends `(dst, src, size)` to the FIFO,
    preserving earlier order. The contract every backend's
    transfer-queue submission path refines. -/
theorem t7801_submit_appends
    (q q' : Queue) (dst src size : Nat)
    (h_sub : q.submitCopy dst src size = some q')
    : q'.submitted = q.submitted ++ [(dst, src, size)] := by
  unfold Queue.submitCopy at h_sub
  by_cases h_live : q.live
  · rw [if_pos h_live] at h_sub
    have h_eq : q' = { q with submitted := q.submitted ++ [(dst, src, size)] } :=
      (Option.some.inj h_sub).symm
    rw [h_eq]
  · rw [if_neg h_live] at h_sub
    exact absurd h_sub (by simp)

/-- T7802 — `submitCopy` preserves the live flag. -/
theorem t7802_submit_preserves_live
    (q q' : Queue) (dst src size : Nat)
    (h_sub : q.submitCopy dst src size = some q')
    : q'.live = q.live := by
  unfold Queue.submitCopy at h_sub
  by_cases h_live : q.live
  · rw [if_pos h_live] at h_sub
    have h_eq : q' = { q with submitted := q.submitted ++ [(dst, src, size)] } :=
      (Option.some.inj h_sub).symm
    rw [h_eq]
  · rw [if_neg h_live] at h_sub
    exact absurd h_sub (by simp)

/-- T7803 — `destroy` blocks `submitCopy`. -/
theorem t7803_destroy_blocks_submit
    (q : Queue) (dst src size : Nat)
    : (q.destroy).submitCopy dst src size = none := by
  unfold Queue.destroy Queue.submitCopy
  simp

/-- T7804 — `destroy` is idempotent on the live flag. -/
theorem t7804_destroy_idempotent (q : Queue) :
    (q.destroy).destroy = q.destroy := by
  unfold Queue.destroy
  rfl

end Quanta.AsyncCopy
