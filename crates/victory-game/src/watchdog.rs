//! Grand Theft Auto VI's stall watchdog. A thread of the title walks 64
//! heartbeat slots and, when one is older than its limit (about 30 s),
//! calls a hook that stores to address 0 and returns (which hook depends
//! on what the title is doing), then stores to address 0 itself: a
//! deliberate crash. A host that is slow for a while
//! (a first run compiling shaders, a CPU translator, a busy machine) trips
//! it although nothing hangs. Both stores are recognised by their bytes,
//! so the runtime can let the watchdog go on as if no slot were stale.

/// How the faulting thread goes on after one of the watchdog's stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resume {
    /// Added to the instruction pointer.
    pub rip: isize,
    /// A register to copy first (hardware numbers: destination, source).
    pub copy: Option<(u8, u8)>,
}

/// Bytes [`resume`] wants before the faulting instruction.
pub const BEFORE: usize = 0x42;
/// Bytes it wants from the faulting instruction on.
pub const AT: usize = 12;

/// The watchdog's own store, where its hook returns to: `mov dword ptr
/// [0], 0; int3`.
const OWN_STORE: [u8; AT] = [0xC7, 0x04, 0x25, 0, 0, 0, 0, 0, 0, 0, 0, 0xCC];

/// For a write to address 0 by the instruction at the start of `at`
/// ([`AT`] bytes), which `before` ([`BEFORE`] bytes) precedes: how to go
/// on if it is one of the watchdog's stores (build 1.0.2.1). `callers`
/// are the [`AT`] bytes at each address the faulting thread's stack holds
/// near its top (its return addresses among them).
pub fn resume(before: &[u8], at: &[u8], callers: &[[u8; AT]]) -> Option<Resume> {
    if before.len() != BEFORE || at.len() != AT {
        return None;
    }
    // mov dword ptr [0], imm32
    if at[..7] != [0xC7, 0x04, 0x25, 0, 0, 0, 0] {
        return None;
    }
    // A hook: `mov dword ptr [0], <its code>` in a function the watchdog
    // called, so that one of the return addresses is the watchdog's own
    // store. Step over the store; the hook returns. (The same store
    // anywhere else is the title's fatal error and stays one.)
    if at[7..11] != [0; 4] && callers.contains(&OWN_STORE) {
        return Some(Resume { rip: 11, copy: None });
    }
    // The watchdog itself: `mov dword ptr [0], 0; int3`, with `mov r14,
    // rsi; mov ebx, ebp` 0x29 bytes before and, 0x42 bytes before, the path
    // taken when no slot is stale (`mov r15, [rip+..]; mov r14, rsi; mov
    // ebp, 0x1e`), which expects RSI as R14 holds it. Go back there.
    let b = |back: usize, n: usize| &before[BEFORE - back..BEFORE - back + n];
    if at[7..] == [0, 0, 0, 0, 0xCC]
        && b(0x29, 5) == [0x49, 0x89, 0xF6, 0x89, 0xEB]
        && b(0x42, 3) == [0x4C, 0x8B, 0x3D]
        && b(0x3B, 8) == [0x49, 0x89, 0xF6, 0xBD, 0x1E, 0, 0, 0]
    {
        return Some(Resume {
            rip: -0x42,
            copy: Some((6, 14)),
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_both_stores_and_nothing_else() {
        let hook = [0xC7, 0x04, 0x25, 0, 0, 0, 0, 0x4B, 0xE9, 0xF1, 0x2E, 0xC3];
        let store = OWN_STORE;
        assert_eq!(
            resume(&[0x90; BEFORE], &hook, &[[0x90; AT], store]),
            Some(Resume { rip: 11, copy: None })
        );
        // The same store not called by the watchdog is a fatal error.
        assert_eq!(resume(&[0x90; BEFORE], &hook, &[[0x90; AT]]), None);
        assert_eq!(resume(&[0x90; BEFORE], &store, &[]), None);
        let mut before = [0x90; BEFORE];
        before[..3].copy_from_slice(&[0x4C, 0x8B, 0x3D]);
        before[7..15].copy_from_slice(&[0x49, 0x89, 0xF6, 0xBD, 0x1E, 0, 0, 0]);
        before[BEFORE - 0x29..][..5].copy_from_slice(&[0x49, 0x89, 0xF6, 0x89, 0xEB]);
        assert_eq!(
            resume(&before, &store, &[]),
            Some(Resume {
                rip: -0x42,
                copy: Some((6, 14))
            })
        );
        // Any other store to address 0 stays a crash.
        let other = [0xC7, 0x04, 0x25, 0, 0, 0, 0, 1, 0, 0, 0, 0x90];
        assert_eq!(resume(&before, &other, &[]), None);
    }
}
