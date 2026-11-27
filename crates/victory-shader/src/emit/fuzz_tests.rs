//! Damaged console programs are refused, never a panic.
//!
//! A title's shader code is data the translator cannot trust: one wrong
//! word must make `translate` return an error, not take the run down. These
//! tests take small compute programs built here (no game data), damage
//! their words with [`victory_core::fuzz`], and run every stage of the
//! translation on each case: [`spill::build`], [`structure::structure`],
//! [`resource::track`], [`io::raise`] and [`emit`]. They need no GPU and no
//! `dxil.dll`.

use super::emit;
use crate::dxbc::{self, Container};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, resource, spill, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// `s_load_dwordx4 s[8:11], s[4:5], 0` after `s_mov_b32 s5, 0`: the raw
/// UAV's descriptor, read from the table whose address is in s4.
const PROLOGUE: [u32; 3] = [0xBE85_0380, 0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25];
/// `buffer_store_dword v0, off, s[8:11], 0`.
const STORE: [u32; 2] = [0x38 << 26 | 0x1C << 18, 0x80 << 24 | 2 << 16];

/// A compute program of `threads` threads with 4 root constants in s0..s3
/// and a raw UAV's descriptor table in s4: [`PROLOGUE`], `body`, the end.
fn native(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    let mut words = PROLOGUE.to_vec();
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// Straight-line arithmetic, then a store of v0.
fn straight() -> Vec<u32> {
    let mut w = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
        0x7E00_0207, // v_mov_b32 v0, s7
    ];
    w.extend_from_slice(&STORE);
    w
}

/// A per-lane branch: `if (v0 < s0) { store }`, EXEC saved and restored
/// around it.
fn branch() -> Vec<u32> {
    let mut w = vec![
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0003, // s_cbranch_execz past the three words below
        0x7E00_0280, // v_mov_b32 v0, 0
    ];
    w.extend_from_slice(&STORE);
    w.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
    w
}

/// Every stage of the translation; errors are fine, panics are not.
fn translate(native: &NativeProgram) {
    let container_bytes = dxbc::build(&[]);
    let c = Container::parse(&container_bytes).unwrap();
    let Ok(p) = spill::build(native) else {
        return;
    };
    let st = structure::structure(&p);
    let Ok(res) = resource::track(&p, &native.layout, native.stage, st.as_ref().ok()) else {
        return;
    };
    let Ok(io) = io::raise(&p, native) else {
        return;
    };
    let Ok(st) = st else {
        return;
    };
    let _ = emit(&c, native, &p, &st, &io, &res);
}

#[test]
fn the_seed_programs_translate() {
    for (body, wave64, threads) in [
        (straight(), true, 64),
        (branch(), true, 64),
        (straight(), false, 32),
    ] {
        let native = native(&body, wave64, threads);
        let p = spill::build(&native).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let c = dxbc::build(&[]);
        emit(&Container::parse(&c).unwrap(), &native, &p, &st, &io, &res).unwrap();
    }
}

/// Words changed, cut, repeated and appended after the prologue; the
/// program translated or refused, never a panic.
#[test]
fn damaged_programs_never_panic() {
    let bodies = [
        (straight(), true, 64),
        (branch(), true, 64),
        (branch(), false, 32),
    ];
    let seeds: Vec<Vec<u8>> = bodies
        .iter()
        .map(|(body, ..)| native(body, true, 64).code)
        .collect();
    // The words after the prologue, byte by byte: opcodes and operands.
    let fields: Vec<usize> = (4 * PROLOGUE.len()..64).collect();
    victory_core::fuzz::run("gfx10 compute", 0x6766_7831, &seeds, &fields, |code| {
        // A program is whole words; the shape of the program (which seed,
        // wave size) is taken from its length, so a case replays alike.
        let code = &code[..code.len() / 4 * 4];
        let (_, wave64, threads) = bodies[code.len() / 4 % bodies.len()];
        let mut program = native(&[], wave64, threads);
        program.code = code.to_vec();
        translate(&program);
    });
}

/// Found by [`damaged_programs_never_panic`]: a damaged word whose operand
/// names a scalar code past 127 (255 here). The search for parked scalars
/// (`ir.rs`) kept one flag per code below 128 and indexed it with any code
/// an operand named: index out of bounds, and the run taken down.
#[test]
fn an_operand_past_the_scalars_is_no_panic() {
    let words = [
        0xBE85_0380,
        0xF408_0202,
        0xFA00_0000,
        0x7E0A_0281,
        0xD40C_FFFF,
        0x340E_0B06,
        0x0080_0207,
        0xE070_8100,
        0x8002_0000,
        S_ENDPGM,
        0xBCDB_F9A6,
        0xA92A_002F,
        0xBF68_E1C6,
        0x39C1_2FC7,
    ];
    let mut program = native(&[], false, 32);
    program.code = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    translate(&program);
}
