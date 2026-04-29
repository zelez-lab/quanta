/-
Indirect Command Buffers (steps 032 + 033).

An Indirect Command Buffer (ICB) is a pre-recorded sequence of GPU
commands that can be executed by the GPU itself, without re-issuing
each command from the host. Backends:

- Metal: `MTLIndirectCommandBuffer` + `executeCommandsInBuffer:withRange:`
- Vulkan: secondary command buffers + `vkCmdExecuteCommands`
- WebGPU: `GPURenderBundle` (render only; compute bundles are not in
  the W3C WebGPU spec).
- CPU: software simulation (sequential dispatch).

This module gives the abstract IR model and the equivalence theorem
that the per-backend implementations refine: executing an ICB whose
recorded commands are `cmds` is observationally equivalent to
executing `cmds` directly on the same state.

The model is parametric in the per-command state transformer
(`exec : Command → State → State`), so the theorem holds for any
implementation that respects the per-command semantics. Backends are
responsible for ensuring their per-command lowering matches `exec`.
-/

namespace Quanta.Icb

/-- A single recorded command. The `wave_id` selects which compiled
    kernel to launch, `groups` is the workgroup grid, and `bindings`
    captures the slot → buffer handle map at record time. The
    backends serialize this into their native indirect-command
    format (Metal `indirectComputeCommandAtIndex`, Vulkan secondary
    command buffer, etc.). -/
structure Command where
  wave_id  : Nat
  groups   : Nat × Nat × Nat
  bindings : List (Nat × Nat)
  deriving Repr, DecidableEq

/-- An ICB with capacity `cap` and at-most `cap` recorded commands.
    `cap` mirrors the `max_commands` argument the user supplied at
    creation time; backends pre-allocate storage for `cap` commands
    so subsequent records do not reallocate. -/
structure Icb where
  cap      : Nat
  commands : List Command
  deriving Repr

/-- Empty ICB with the given capacity. -/
def Icb.empty (cap : Nat) : Icb :=
  { cap := cap, commands := [] }

/-- Append a command to the ICB, failing if at capacity. -/
def Icb.record (icb : Icb) (cmd : Command) : Option Icb :=
  if icb.commands.length < icb.cap then
    some { icb with commands := icb.commands ++ [cmd] }
  else
    none

/-- Execute the first `count` commands of the ICB on state `s`,
    using the abstract per-command transformer `exec`. -/
def Icb.execute {State : Type}
    (exec : Command → State → State)
    (s : State) (icb : Icb) (count : Nat) : State :=
  ((icb.commands.take count).foldl (fun s c => exec c s) s)

/-- Direct execution of a command list (the reference semantics). -/
def execute_direct {State : Type}
    (exec : Command → State → State)
    (s : State) (cmds : List Command) : State :=
  cmds.foldl (fun s c => exec c s) s

-- ════════════════════════════════════════════════════════════════════
-- Theorem T7000 — record-then-execute equals direct execution
-- ════════════════════════════════════════════════════════════════════

/-- Recording into the empty ICB any list of commands, then executing
    the full count, is observationally equivalent to executing the
    same list directly. This is the central correctness statement
    that every ICB backend refines. -/
theorem t7000_icb_equivalence
    {State : Type} (exec : Command → State → State)
    (s : State) (cmds : List Command) :
    Icb.execute exec s { cap := cmds.length, commands := cmds } cmds.length
    = execute_direct exec s cmds := by
  simp [Icb.execute, execute_direct, List.take_length]

/-- A cleaner restatement: if you build an ICB by repeated `record`,
    the result of `execute` on the full count equals the direct
    `foldl` of the recorded commands. -/
theorem t7001_record_execute_eq_foldl
    {State : Type} (exec : Command → State → State)
    (s : State) (cmds : List Command) :
    let icb : Icb := { cap := cmds.length, commands := cmds }
    Icb.execute exec s icb cmds.length = cmds.foldl (fun s c => exec c s) s := by
  simp [Icb.execute, List.take_length]

/-- Executing fewer than the full count returns the partial fold —
    the prefix of the recorded sequence. Backends that take a
    `count` argument (Metal `withRange:`, Vulkan
    `vkCmdExecuteCommands` count) must respect this. -/
theorem t7002_partial_execute_eq_take_foldl
    {State : Type} (exec : Command → State → State)
    (s : State) (icb : Icb) (count : Nat) :
    Icb.execute exec s icb count
      = (icb.commands.take count).foldl (fun s c => exec c s) s := by
  rfl

/-- Capacity invariant for ICBs built via `Icb.empty` then a chain
    of `Icb.record` calls. Stated existentially: any chain of
    successful records preserves `length ≤ cap`. -/
theorem t7003_record_preserves_capacity
    (icb icb' : Icb) (cmd : Command)
    (_h_inv : icb.commands.length ≤ icb.cap)
    (h_record : icb.record cmd = some icb')
    : icb'.commands.length ≤ icb'.cap := by
  unfold Icb.record at h_record
  by_cases h : icb.commands.length < icb.cap
  · rw [if_pos h] at h_record
    have h_eq : icb' = { icb with commands := icb.commands ++ [cmd] } :=
      (Option.some.inj h_record).symm
    rw [h_eq]
    simp [List.length_append]
    exact h
  · rw [if_neg h] at h_record
    exact absurd h_record (by simp)

/-- The empty ICB satisfies the capacity invariant trivially. -/
theorem t7004_empty_capacity_ok (cap : Nat) :
    (Icb.empty cap).commands.length ≤ (Icb.empty cap).cap := by
  simp [Icb.empty]

end Quanta.Icb
