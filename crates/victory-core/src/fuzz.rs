//! Mutation fuzzing that runs inside `cargo test`.
//!
//! The parsers read bytes nobody vouches for: the user's dump of the title,
//! its import libraries, its shader containers and their programs. A
//! malformed field must come back as an error, never as a panic, an
//! overflow or an out-of-bounds slice. These helpers take a valid input that
//! a test builds itself (synthetic: rule #1 holds), damage it in the ways a
//! bad file is damaged, and hand each case to the parser under test.
//!
//! Every case is reproducible: a failure names the seed and the case number,
//! and [`replay`] runs that one case again. No game data is read, so the
//! tests run anywhere `cargo test` does.

use std::panic::{AssertUnwindSafe, catch_unwind};

/// The cases each target runs by default: enough to cover the interesting
/// fields of a small input many times over, in well under a second.
pub const DEFAULT_CASES: u64 = 4096;

/// A small, fast, seedable generator (SplitMix64). Not for anything but
/// tests.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// A value in `0..n`; `n` must not be 0.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n != 0, "below(0)");
        (self.next_u64() % n as u64) as usize
    }

    pub fn chance(&mut self, one_in: usize) -> bool {
        self.below(one_in) == 0
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

/// Values that sit on the edges parsers get wrong: zero, one, signs,
/// powers of two and the ends of each width.
pub const EDGES: [u32; 16] = [
    0,
    1,
    2,
    4,
    7,
    8,
    0x7F,
    0x80,
    0xFF,
    0x100,
    0x7FFF,
    0xFFFF,
    0x1_0000,
    0x7FFF_FFFF,
    0x8000_0000,
    0xFFFF_FFFF,
];

/// Damages `data` with one to eight mutations. `fields` lists byte offsets
/// of header fields the caller knows matter (sizes, offsets, counts); about
/// half of the mutations land on them, the rest anywhere.
pub fn mutate(data: &mut Vec<u8>, fields: &[usize], rng: &mut Rng) {
    for _ in 0..1 + rng.below(8) {
        if data.is_empty() {
            data.push(rng.next_u32() as u8);
            continue;
        }
        let at = if !fields.is_empty() && rng.chance(2) {
            *rng.pick(fields) % data.len()
        } else {
            rng.below(data.len())
        };
        match rng.below(9) {
            // One byte, one bit.
            0 => data[at] = rng.next_u32() as u8,
            1 => data[at] ^= 1 << rng.below(8),
            // A 32-bit field set to an edge, or nudged past what it held.
            2 | 3 => put_u32(data, at, *rng.pick(&EDGES)),
            4 => {
                let old = get_u32(data, at);
                let delta = 1 + rng.below(16) as u32;
                let new = if rng.chance(2) {
                    old.wrapping_add(delta)
                } else {
                    old.wrapping_sub(delta)
                };
                put_u32(data, at, new);
            }
            // A 16-bit field set to an edge.
            5 => put_u16(data, at, *rng.pick(&EDGES) as u16),
            // The file cut short, anywhere.
            6 => data.truncate(rng.below(data.len())),
            // A run copied over another (offsets that point into their own
            // headers, tables that repeat).
            7 => {
                let from = rng.below(data.len());
                let len = 1 + rng.below(32.min(data.len() - from));
                let chunk = data[from..from + len].to_vec();
                let to = rng.below(data.len());
                let end = (to + len).min(data.len());
                data[to..end].copy_from_slice(&chunk[..end - to]);
            }
            // Bytes appended (trailing data the format did not ask for).
            _ => {
                for _ in 0..1 + rng.below(64) {
                    data.push(rng.next_u32() as u8);
                }
            }
        }
    }
}

fn get_u32(data: &[u8], at: usize) -> u32 {
    let mut b = [0u8; 4];
    for (i, byte) in b.iter_mut().enumerate() {
        *byte = data.get(at + i).copied().unwrap_or(0);
    }
    u32::from_le_bytes(b)
}

fn put_u32(data: &mut [u8], at: usize, v: u32) {
    for (i, byte) in v.to_le_bytes().into_iter().enumerate() {
        if let Some(slot) = data.get_mut(at + i) {
            *slot = byte;
        }
    }
}

fn put_u16(data: &mut [u8], at: usize, v: u16) {
    for (i, byte) in v.to_le_bytes().into_iter().enumerate() {
        if let Some(slot) = data.get_mut(at + i) {
            *slot = byte;
        }
    }
}

/// How many cases to run: [`DEFAULT_CASES`], or `VICTORY_FUZZ_CASES` for a
/// longer search on a developer's machine (for example 1000000; read by
/// every fuzz test, nothing else).
pub fn cases() -> u64 {
    std::env::var("VICTORY_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CASES)
}

/// A number mixed into every target's seed: 0, or `VICTORY_FUZZ_SEED` to
/// search cases other than the ones CI runs (read by every fuzz test,
/// nothing else). A failure's message names the seed it ran with.
pub fn extra_seed() -> u64 {
    std::env::var("VICTORY_FUZZ_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// The input of case `case` of a run seeded with `seed`: one of `seeds`,
/// mutated. The same three numbers always give the same bytes.
pub fn case_input(seeds: &[Vec<u8>], fields: &[usize], seed: u64, case: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed ^ case.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    let mut data = rng.pick(seeds).clone();
    mutate(&mut data, fields, &mut rng);
    data
}

/// Runs `check` on [`cases`] mutations of `seeds`. A panic inside `check`
/// (the parser's or the check's own assertion) fails the test with the
/// target, seed and case, and the input's length, so [`replay`] can run that
/// case alone.
pub fn run(target: &str, seed: u64, seeds: &[Vec<u8>], fields: &[usize], check: impl Fn(&[u8])) {
    assert!(!seeds.is_empty(), "{target}: no seed inputs");
    let seed = seed ^ extra_seed();
    for case in 0..cases() {
        let data = case_input(seeds, fields, seed, case);
        if let Err(panic) = catch_unwind(AssertUnwindSafe(|| check(&data))) {
            let why = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("(no message)");
            panic!(
                "{target}: case {case} of seed {seed:#x} panicked ({} bytes): {why}\n\
                 replay with victory_core::fuzz::replay(&seeds, &fields, {seed:#x}, {case})",
                data.len()
            );
        }
    }
}

/// The input of one failing case, for a regression test.
pub fn replay(seeds: &[Vec<u8>], fields: &[usize], seed: u64, case: u64) -> Vec<u8> {
    case_input(seeds, fields, seed, case)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_and_case_give_the_same_bytes() {
        let seeds = vec![vec![0u8; 64], (0..=255).collect()];
        for case in 0..64 {
            assert_eq!(
                case_input(&seeds, &[0, 8], 7, case),
                case_input(&seeds, &[0, 8], 7, case)
            );
        }
        assert_ne!(
            (0..64)
                .map(|c| case_input(&seeds, &[], 7, c))
                .collect::<Vec<_>>(),
            (0..64)
                .map(|c| case_input(&seeds, &[], 8, c))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn mutations_change_the_input() {
        let seed = vec![0xAAu8; 128];
        let changed = (0..256)
            .filter(|&c| case_input(std::slice::from_ref(&seed), &[], 1, c) != seed)
            .count();
        assert!(changed > 200, "only {changed} of 256 cases differ");
    }

    #[test]
    fn mutating_never_panics_on_tiny_inputs() {
        let mut rng = Rng::new(3);
        for len in 0..8 {
            for _ in 0..256 {
                let mut data = vec![0u8; len];
                mutate(&mut data, &[0, 1, 1000], &mut rng);
            }
        }
    }

    #[test]
    fn a_panicking_check_names_its_case() {
        let seeds = vec![vec![1u8; 16]];
        let err = catch_unwind(|| run("t", 5, &seeds, &[], |_| panic!("boom"))).unwrap_err();
        let msg = err.downcast_ref::<String>().unwrap();
        assert!(msg.contains("t: case 0 of seed 0x5"), "{msg}");
        assert!(msg.contains("boom"), "{msg}");
    }
}
