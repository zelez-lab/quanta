//! Minimal async executor for the WebGPU driver.
//!
//! Replaces `wasm-bindgen-futures::JsFuture` + `spawn_local`. The pieces:
//!
//! - [`Promise`] — a `Future` whose readiness is decided by the JS side
//!   via the `quanta_resolve` / `quanta_reject` exports.
//! - [`spawn_local`] — schedule a top-level future and run it to its
//!   first `Pending`. Top-level futures are kept alive in a thread-local
//!   table; resumption is driven by `quanta_resolve` / `quanta_reject`.
//!
//! Single-threaded by construction: wasm32-unknown-unknown has no
//! threads, so the entire executor lives behind a `RefCell` with no
//! locking. The [`RawWaker`] implementation is hand-rolled to avoid the
//! `Send + Sync` bound that `Waker::from(Arc<W>)` imposes.
//!
//! ## Why a callback-shaped ABI
//!
//! Wasm cannot block the browser event loop, so an `epoll`-style
//! `wait()` import has nowhere to land. The dual is a callback model:
//! every async import takes a `task: u32` argument; when JS resolves
//! the underlying Promise it calls back into wasm at
//! `quanta_resolve(task, handle)`. This file is the wasm side of that
//! contract — it tracks the wakers, drives the future graph, and turns
//! the JS callback into a `Waker::wake()`.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use core::cell::RefCell;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

// `thread_local!` lives in std. The `webgpu` feature pulls in `std`,
// and on wasm32 there is exactly one thread anyway — the macro reduces
// to a simple lazily-initialized cell per binary. Using std here keeps
// the executor short; a no_std re-implementation would buy nothing.
use std::thread_local;

// ── Executor state ──────────────────────────────────────────────────────────

struct PromiseSlot {
    state: PromiseState,
    waker: Option<Waker>,
}

enum PromiseState {
    Pending,
    Resolved(u32),
    Rejected,
}

struct Task {
    future: Pin<Box<dyn Future<Output = ()> + 'static>>,
}

struct Executor {
    next_task: u32,
    next_promise: u32,
    tasks: BTreeMap<u32, Task>,
    promises: BTreeMap<u32, PromiseSlot>,
    ready: VecDeque<u32>,
}

impl Executor {
    const fn new() -> Self {
        Self {
            next_task: 1,
            next_promise: 1,
            tasks: BTreeMap::new(),
            promises: BTreeMap::new(),
            ready: VecDeque::new(),
        }
    }
}

thread_local! {
    static EXECUTOR: RefCell<Executor> = const { RefCell::new(Executor::new()) };
}

// ── Promise future ──────────────────────────────────────────────────────────

/// A future that resolves once the JS side calls `quanta_resolve(id, h)`
/// with this Promise's id, where `h` is the handle returned by the
/// underlying WebGPU promise.
///
/// Async imports allocate the Promise via [`Promise::register`] before
/// passing the id into the FFI call.
pub struct Promise {
    id: u32,
}

impl Promise {
    /// Allocate a fresh promise id, then call `f` with it. The closure
    /// is expected to forward the id to a JS-side import that will
    /// eventually call `quanta_resolve(id, …)` or `quanta_reject(id)`.
    pub fn register<F: FnOnce(u32)>(f: F) -> Self {
        let id = EXECUTOR.with(|e| {
            let mut e = e.borrow_mut();
            let id = e.next_promise;
            e.next_promise += 1;
            e.promises.insert(
                id,
                PromiseSlot {
                    state: PromiseState::Pending,
                    waker: None,
                },
            );
            id
        });
        f(id);
        Promise { id }
    }
}

impl Drop for Promise {
    fn drop(&mut self) {
        // Make sure a Promise that was awaited (and so removed itself
        // from the table on Ready) doesn't double-free, and a Promise
        // that was dropped before resolution doesn't leak the slot.
        EXECUTOR.with(|e| {
            e.borrow_mut().promises.remove(&self.id);
        });
    }
}

impl Future for Promise {
    /// `Ok(handle)` — the JS Promise resolved; `handle` is 0 for unit
    /// promises (e.g. `mapAsync`, `onSubmittedWorkDone`).
    /// `Err(())` — the JS Promise rejected. We do not surface the
    /// rejection reason for B⁰; rejections always become a generic
    /// "WebGPU op failed" error on the Rust side. B″ tightens this
    /// later by piping the error string through a separate import.
    type Output = Result<u32, ()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        EXECUTOR.with(|e| {
            let mut e = e.borrow_mut();
            let slot = match e.promises.get_mut(&self.id) {
                Some(s) => s,
                // Slot is gone — only happens if `quanta_resolve` /
                // `quanta_reject` ran while the Promise was being
                // dropped, which is well-defined: we treat it as a
                // pending Promise that will never wake. Returning
                // `Pending` lets the executor keep the future parked.
                None => return Poll::Pending,
            };
            match slot.state {
                PromiseState::Pending => {
                    slot.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
                PromiseState::Resolved(h) => Poll::Ready(Ok(h)),
                PromiseState::Rejected => Poll::Ready(Err(())),
            }
        })
    }
}

// ── Hand-rolled Waker (no Send/Sync bound) ──────────────────────────────────
//
// The standard library's `Waker::from(Arc<W>)` requires `W: Send + Sync`.
// Our executor is single-threaded and lives behind a thread-local, so
// we hand-roll a `RawWaker` whose data pointer is the task id encoded as
// `usize`. No allocation per waker; no synchronization.

const VTABLE: &RawWakerVTable = &RawWakerVTable::new(
    raw_waker_clone,
    raw_waker_wake,
    raw_waker_wake_by_ref,
    raw_waker_drop,
);

fn raw_waker_clone(data: *const ()) -> RawWaker {
    RawWaker::new(data, VTABLE)
}

fn raw_waker_wake(data: *const ()) {
    let task_id = data as usize as u32;
    EXECUTOR.with(|e| e.borrow_mut().ready.push_back(task_id));
}

fn raw_waker_wake_by_ref(data: *const ()) {
    raw_waker_wake(data);
}

fn raw_waker_drop(_data: *const ()) {
    // Data is not heap-allocated; nothing to drop.
}

fn make_waker(task_id: u32) -> Waker {
    let raw = RawWaker::new(task_id as usize as *const (), VTABLE);
    // SAFETY: `VTABLE`'s functions are sound for `data = task_id as
    // *const ()`: they only ever cast it back to `u32` and never
    // dereference it as a real pointer.
    unsafe { Waker::from_raw(raw) }
}

// ── Task scheduling ─────────────────────────────────────────────────────────

/// Spawn a fire-and-forget future on the local executor and run it
/// until either its first `Pending` or completion. Re-driving happens
/// inside `quanta_resolve` / `quanta_reject` when JS feeds resolutions
/// back in.
pub fn spawn_local<F>(future: F)
where
    F: Future<Output = ()> + 'static,
{
    let task_id = EXECUTOR.with(|e| {
        let mut e = e.borrow_mut();
        let id = e.next_task;
        e.next_task += 1;
        e.tasks.insert(
            id,
            Task {
                future: Box::pin(future),
            },
        );
        e.ready.push_back(id);
        id
    });
    let _ = task_id;
    drive();
}

/// Drive the executor's ready queue to quiescence.
///
/// Called whenever a wake-up source (initial spawn, JS resolve, JS
/// reject) adds entries to the queue. Polls each ready task until it
/// reports `Pending` (in which case it stays in the table) or `Ready`
/// (in which case it's dropped).
fn drive() {
    loop {
        let task_id = match EXECUTOR.with(|e| e.borrow_mut().ready.pop_front()) {
            Some(id) => id,
            None => return,
        };
        let mut task = match EXECUTOR.with(|e| e.borrow_mut().tasks.remove(&task_id)) {
            Some(t) => t,
            None => continue,
        };
        let waker = make_waker(task_id);
        let mut cx = Context::from_waker(&waker);
        match task.future.as_mut().poll(&mut cx) {
            Poll::Pending => {
                EXECUTOR.with(|e| {
                    e.borrow_mut().tasks.insert(task_id, task);
                });
            }
            Poll::Ready(()) => {
                // Task drops here, releasing the boxed future.
            }
        }
    }
}

// ── JS-callable resume hooks (wasm exports) ────────────────────────────────

/// Resolve a pending Promise with a handle (or 0 for unit promises),
/// then drive the executor.
///
/// # Safety
/// Called by the JS-side glue when a WebGPU promise resolves. The
/// `task` argument must be one previously handed to JS by an
/// async-shaped FFI import; if it's stale or unknown, this function
/// silently no-ops.
#[unsafe(no_mangle)]
pub extern "C" fn quanta_resolve(task: u32, handle: u32) {
    let waker = EXECUTOR.with(|e| {
        let mut e = e.borrow_mut();
        let slot = e.promises.get_mut(&task)?;
        slot.state = PromiseState::Resolved(handle);
        slot.waker.take()
    });
    if let Some(w) = waker {
        w.wake();
    }
    drive();
}

/// Reject a pending Promise, then drive the executor.
///
/// # Safety
/// Same caveats as [`quanta_resolve`].
#[unsafe(no_mangle)]
pub extern "C" fn quanta_reject(task: u32) {
    let waker = EXECUTOR.with(|e| {
        let mut e = e.borrow_mut();
        let slot = e.promises.get_mut(&task)?;
        slot.state = PromiseState::Rejected;
        slot.waker.take()
    });
    if let Some(w) = waker {
        w.wake();
    }
    drive();
}
