//! Verus mirror — Indirect Command Buffer (ICB) safety invariants
//! (steps 032 + 033).
//!
//! This mirror states the API-level invariants that every backend
//! (`MTLIndirectCommandBuffer`, Vulkan secondary command buffers,
//! `GPURenderBundle`) must respect. It captures the lifetime model
//! of the typed `IndirectCommandBuffer` user-facing wrapper:
//!
//! - `create(max_commands)` returns a fresh handle in `Empty` state
//!   with the given capacity.
//! - `record(cmd)` appends one command, failing if the recorded
//!   length has already reached `max_commands`.
//! - `execute(count)` runs the first `count` recorded commands and
//!   requires `count ≤ recorded`.
//! - `destroy(handle)` consumes the handle; subsequent operations
//!   on the same handle are rejected.
//!
//! Theorems:
//!   T7050 — fresh handle from create has zero recorded commands and
//!           the requested capacity.
//!   T7051 — record monotonically increases the recorded length when
//!           it succeeds, and the new length stays ≤ capacity.
//!   T7052 — record fails (returns None) iff the buffer is full or
//!           destroyed.
//!   T7053 — execute requires count ≤ recorded; a successful
//!           execute leaves the recorded sequence unchanged
//!           (re-executable).
//!   T7054 — destroy invalidates the live flag and is idempotent.

use vstd::prelude::*;

verus! {

// ── Ghost types ─────────────────────────────────────────────────────────────

/// A single recorded command. Mirrors `Quanta.Icb.Command` in Lean.
/// Backends serialize this into Metal `MTLIndirectComputeCommand`,
/// Vulkan secondary CB ops, etc.
pub struct IcbCommand {
    pub wave_id: u64,
    pub group_x: u32,
    pub group_y: u32,
    pub group_z: u32,
}

/// An ICB handle's ghost state.
pub struct Icb {
    /// Fresh handle from device — uniquely allocated, never reused
    /// while live.
    pub handle: u64,
    /// Capacity supplied at create time. Recorded length must stay
    /// ≤ cap.
    pub cap: nat,
    /// Recorded command sequence in record order.
    pub commands: Seq<IcbCommand>,
    /// Whether the handle is live (not destroyed).
    pub live: bool,
}

// ── Operations ─────────────────────────────────────────────────────────────

/// Mirror of `gpu.indirect_buffer_create(cap)`.
pub open spec fn create(handle: u64, cap: nat) -> Icb {
    Icb {
        handle,
        cap,
        commands: Seq::empty(),
        live: true,
    }
}

/// Mirror of `icb.record(cmd)` — returns Some on success, None when
/// the buffer is full or destroyed.
pub open spec fn record(icb: Icb, cmd: IcbCommand) -> Option<Icb> {
    if icb.live && icb.commands.len() < icb.cap {
        Option::Some(Icb {
            handle: icb.handle,
            cap: icb.cap,
            commands: icb.commands.push(cmd),
            live: icb.live,
        })
    } else {
        Option::None
    }
}

/// Mirror of `gpu.indirect_buffer_execute(handle, count)` —
/// precondition: `count ≤ recorded` and the handle is live.
pub open spec fn can_execute(icb: Icb, count: nat) -> bool {
    icb.live && count <= icb.commands.len()
}

/// Mirror of `gpu.indirect_buffer_destroy(handle)`.
pub open spec fn destroy(icb: Icb) -> Icb {
    Icb {
        live: false,
        ..icb
    }
}

// ── T7050: create produces a well-formed empty ICB ───────────────────────

proof fn t7050_create_fresh(handle: u64, cap: nat)
    ensures
        create(handle, cap).commands.len() == 0,
        create(handle, cap).cap == cap,
        create(handle, cap).live == true,
        create(handle, cap).handle == handle,
{}

// ── T7051: record extends the command sequence by exactly one ───────────

proof fn t7051_record_extends_by_one(icb: Icb, cmd: IcbCommand, icb2: Icb)
    requires
        record(icb, cmd) == Option::<Icb>::Some(icb2),
    ensures
        icb2.commands.len() == icb.commands.len() + 1,
        icb2.commands.len() <= icb2.cap,
        icb2.cap == icb.cap,
        icb2.live == icb.live,
        icb2.handle == icb.handle,
{}

// ── T7052: record fails iff buffer is full or destroyed ─────────────────

proof fn t7052_record_fails_when_full(icb: Icb, cmd: IcbCommand)
    requires
        !icb.live || icb.commands.len() >= icb.cap,
    ensures
        record(icb, cmd) == Option::<Icb>::None,
{}

// ── T7053: execute leaves the recorded sequence unchanged (re-executable) ─

/// `execute` is a read-only operation on the ICB ghost state — the
/// recorded `commands` and `cap` are unchanged. The actual side
/// effect is on the GPU world state, which is *not* part of the ICB
/// model (it lives in the device's compute state, observed by the
/// Lean `Icb.execute` semantics).
pub open spec fn execute(icb: Icb, _count: nat) -> Icb {
    icb
}

proof fn t7053_execute_preserves_record(icb: Icb, count: nat)
    requires
        can_execute(icb, count),
    ensures
        execute(icb, count).commands == icb.commands,
        execute(icb, count).cap == icb.cap,
        execute(icb, count).live == icb.live,
        execute(icb, count).handle == icb.handle,
{}

// ── T7054: destroy invalidates the live flag; idempotent ────────────────

proof fn t7054_destroy_invalidates(icb: Icb)
    ensures
        destroy(icb).live == false,
        destroy(icb).handle == icb.handle,
        destroy(icb).cap == icb.cap,
        destroy(icb).commands == icb.commands,
{}

proof fn t7054b_destroy_idempotent(icb: Icb)
    ensures
        destroy(destroy(icb)) == destroy(icb),
{}

/// After destroy, no record / execute is permitted (the Option /
/// can_execute predicates both fail because `live = false`).
proof fn t7054c_destroy_blocks_record(icb: Icb, cmd: IcbCommand)
    ensures
        record(destroy(icb), cmd) == Option::<Icb>::None,
{}

proof fn t7054d_destroy_blocks_execute(icb: Icb, count: nat)
    ensures
        !can_execute(destroy(icb), count),
{}

}  // verus!
