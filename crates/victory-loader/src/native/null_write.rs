//! Writes to address 0 that a title makes on purpose (a stall watchdog's
//! deliberate crash) and that a hook recognises by their bytes: the thread
//! goes on where the hook says instead of the crash.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use super::seh::context::{EXCEPTION_ACCESS_VIOLATION, ExceptionPointers};
use super::seh::{Handler, add_vectored_handler};

/// How a faulting thread goes on: added to its instruction pointer, and a
/// register to copy first (hardware numbers: destination, source).
pub type Resume = (isize, Option<(u8, u8)>);
/// Given the bytes before a faulting instruction, from it on, and at each
/// image address near the top of the thread's stack (as many bytes as from
/// the instruction on), decides whether and how the thread goes on.
pub type NullWriteHook = fn(before: &[u8], at: &[u8], callers: &[&[u8]]) -> Option<Resume>;

/// Stack slots looked at for code addresses.
const STACK_SLOTS: usize = 32;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;

struct Hooked {
    hook: NullWriteHook,
    before: usize,
    at: usize,
    image: (usize, usize),
}

static HOOKED: OnceLock<Hooked> = OnceLock::new();

/// Lets `hook` decide about every write to address 0 by code of the image
/// at `base` (`size` bytes): it gets `before` bytes ahead of the faulting
/// instruction and `at` bytes from it on. Asked ahead of the title's own
/// handlers.
pub fn resume_null_writes(base: usize, size: usize, hook: NullWriteHook, before: usize, at: usize) {
    let hooked = Hooked {
        hook,
        before,
        at,
        image: (base, size),
    };
    if HOOKED.set(hooked).is_ok() {
        add_vectored_handler(true, Handler::Host(on_null_write));
    }
}

extern "C" fn on_null_write(info: *mut ExceptionPointers) -> i32 {
    let Some(h) = HOOKED.get() else {
        return EXCEPTION_CONTINUE_SEARCH;
    };
    // SAFETY: the dispatcher passes the exception being dispatched.
    let (record, context) = unsafe { (&*(*info).record, &mut *(*info).context) };
    if record.code != EXCEPTION_ACCESS_VIOLATION
        || record.number_parameters < 2
        || record.information[0] != 1
        || record.information[1] != 0
    {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let (base, size) = h.image;
    let image = |address: usize| (base..base + size).contains(&address);
    let rip = context.rip as usize;
    if rip < h.before || !image(rip - h.before) || !image(rip + h.at) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // SAFETY: mapped image code, checked above.
    let (b, a) = unsafe {
        (
            std::slice::from_raw_parts((rip - h.before) as *const u8, h.before),
            std::slice::from_raw_parts(rip as *const u8, h.at),
        )
    };
    // What the top of the stack holds that points into the image: the
    // thread's return addresses are among it.
    let mut callers: [&[u8]; STACK_SLOTS] = [&[]; STACK_SLOTS];
    let mut found = 0;
    let rsp = context.gpr[super::seh::context::RSP] as usize;
    for slot in (rsp..rsp + 8 * STACK_SLOTS).step_by(8) {
        // SAFETY: the faulting thread's stack just above its rsp, deep in
        // calls when the watchdog fires.
        let target = unsafe { (slot as *const usize).read() };
        if image(target) && image(target + h.at) {
            // SAFETY: mapped image code, checked just now.
            callers[found] = unsafe { std::slice::from_raw_parts(target as *const u8, h.at) };
            found += 1;
        }
    }
    let Some((delta, copy)) = (h.hook)(b, a, &callers[..found]) else {
        return EXCEPTION_CONTINUE_SEARCH;
    };
    static SEEN: AtomicU64 = AtomicU64::new(0);
    let n = SEEN.fetch_add(1, Relaxed) + 1;
    if n <= 4 || n.is_power_of_two() {
        tracing::warn!(
            "the title's deliberate crash at image+{:#x} is skipped (#{n})",
            rip - base
        );
    }
    if let Some((to, from)) = copy {
        context.gpr[usize::from(to)] = context.gpr[usize::from(from)];
    }
    context.rip = context.rip.wrapping_add_signed(delta as i64);
    EXCEPTION_CONTINUE_EXECUTION
}
