/-
GPU printf (step 049).

GPU shaders emit debug messages from inside a kernel; the host
drains the message buffer after dispatch. Backends:

- Vulkan: `VK_EXT_debug_printf` (validation-layer feature) +
  `debug_printfEXT(...)` in shaders.
- Metal: `os_log` from MSL via the Metal Debugger; or a
  user-managed log buffer that the host polls.
- WebGPU: not exposed in W3C — software shim through a
  storage-buffer ring.
- CPU: software ring buffer.

Proven contract: a `PrintfBuffer` is a bounded FIFO of recorded
messages. Operations:

  - `create(cap)` — fresh empty buffer with capacity `cap`.
  - `record(buf, msg)` — append a message if `messages.length < cap`
    and `live`.
  - `drain(buf)` — return the messages, leave the buffer empty.
  - `destroy(buf)` — invalidate.
-/

namespace Quanta.Printf

/-- A recorded printf message, modeled as an opaque `Nat` (its hash
    or interned-id at the host). The IR doesn't care about the
    payload format. -/
abbrev Message := Nat

/-- Printf buffer state. -/
structure Buffer where
  cap      : Nat
  messages : List Message
  live     : Bool
  deriving Repr

/-- Create a fresh printf buffer with the given capacity. Returns
    `none` for `cap = 0`. -/
def Buffer.create (cap : Nat) : Option Buffer :=
  if 1 ≤ cap then some { cap := cap, messages := [], live := true }
  else none

/-- Record a message. Fails if the buffer is destroyed or full. -/
def Buffer.record (b : Buffer) (msg : Message) : Option Buffer :=
  if b.live ∧ b.messages.length < b.cap then
    some { b with messages := b.messages ++ [msg] }
  else
    none

/-- Drain the buffer: return its messages and leave it empty. The
    capacity and live flag are preserved. -/
def Buffer.drain (b : Buffer) : Option (Buffer × List Message) :=
  if b.live then
    some ({ b with messages := [] }, b.messages)
  else
    none

/-- Mark the buffer destroyed. -/
def Buffer.destroy (b : Buffer) : Buffer :=
  { b with live := false }

/- ============================================================ -/
/-                          THEOREMS                              -/
/- ============================================================ -/

/-- T7900 — `create cap` produces a fresh empty live buffer. -/
theorem t7900_create_shape (cap : Nat) (h : 1 ≤ cap)
    (b : Buffer) (h_c : Buffer.create cap = some b)
    : b.cap = cap ∧ b.messages = [] ∧ b.live = true := by
  unfold Buffer.create at h_c
  rw [if_pos h] at h_c
  have h_eq : b = { cap := cap, messages := [], live := true } :=
    (Option.some.inj h_c).symm
  rw [h_eq]; exact ⟨rfl, rfl, rfl⟩

/-- T7901 — `record` on a non-full live buffer appends in order. -/
theorem t7901_record_appends
    (b b' : Buffer) (msg : Message)
    (h_rec : b.record msg = some b')
    : b'.messages = b.messages ++ [msg] := by
  unfold Buffer.record at h_rec
  by_cases h_cond : b.live ∧ b.messages.length < b.cap
  · rw [if_pos h_cond] at h_rec
    have h_eq : b' = { b with messages := b.messages ++ [msg] } :=
      (Option.some.inj h_rec).symm
    rw [h_eq]
  · rw [if_neg h_cond] at h_rec
    exact absurd h_rec (by simp)

/-- T7902 — `record` preserves capacity + live flag. -/
theorem t7902_record_preserves
    (b b' : Buffer) (msg : Message)
    (h_rec : b.record msg = some b')
    : b'.cap = b.cap ∧ b'.live = b.live := by
  unfold Buffer.record at h_rec
  by_cases h_cond : b.live ∧ b.messages.length < b.cap
  · rw [if_pos h_cond] at h_rec
    have h_eq : b' = { b with messages := b.messages ++ [msg] } :=
      (Option.some.inj h_rec).symm
    rw [h_eq]; exact ⟨rfl, rfl⟩
  · rw [if_neg h_cond] at h_rec
    exact absurd h_rec (by simp)

/-- T7903 — when the buffer is full, `record` fails. -/
theorem t7903_record_full_fails
    (b : Buffer) (msg : Message)
    (h_full : b.messages.length ≥ b.cap)
    : b.record msg = none := by
  unfold Buffer.record
  have : ¬ (b.live ∧ b.messages.length < b.cap) := by
    intro ⟨_, h⟩; exact absurd h (Nat.not_lt_of_ge h_full)
  rw [if_neg this]

/-- T7904 — `drain` returns the recorded messages and empties the
    buffer. -/
theorem t7904_drain_returns_and_empties
    (b b' : Buffer) (msgs : List Message)
    (h_d : b.drain = some (b', msgs))
    : msgs = b.messages ∧ b'.messages = [] := by
  unfold Buffer.drain at h_d
  by_cases h_live : b.live
  · rw [if_pos h_live] at h_d
    have h_eq : (b', msgs) = ({ b with messages := [] }, b.messages) :=
      (Option.some.inj h_d).symm
    have hb' : b' = { b with messages := [] } := (Prod.mk.inj h_eq).1
    have hmsgs : msgs = b.messages := (Prod.mk.inj h_eq).2
    refine ⟨hmsgs, ?_⟩
    rw [hb']
  · rw [if_neg h_live] at h_d
    exact absurd h_d (by simp)

/-- T7905 — `destroy` blocks `record` and `drain`. -/
theorem t7905_destroy_blocks_record
    (b : Buffer) (msg : Message)
    : (b.destroy).record msg = none := by
  unfold Buffer.destroy Buffer.record
  simp

theorem t7905b_destroy_blocks_drain
    (b : Buffer)
    : (b.destroy).drain = none := by
  unfold Buffer.destroy Buffer.drain
  simp

end Quanta.Printf
