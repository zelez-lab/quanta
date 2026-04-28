//! WebIDL parsing front-end.
//!
//! Wraps `weedle` to extract just the pieces B′ cares about today:
//! enum declarations and their string members. Methods/dictionaries
//! get parsed too — we hold onto the AST — but the public surface
//! starts with enums because that's where the lockstep hazard lives.
//!
//! "Project-relevant" enums are those Quanta's existing `ffi.rs` /
//! `web/src/codes.ts` reference. The list is hard-coded below; B″
//! will derive it instead by walking the methods we actually call.

use crate::Result;
use weedle::Definition;
use weedle::common::Identifier;

/// The set of WebIDL enum types Quanta currently uses.
/// Hard-coded list — when we wire a new descriptor, add the relevant
/// IDL enum here so `quanta codegen webgpu` picks it up. Future B″
/// will replace this with a transitive walk from method signatures.
pub const PROJECT_RELEVANT_ENUMS: &[&str] = &[
    "GPUTextureFormat",
    "GPUVertexFormat",
    "GPUPrimitiveTopology",
    "GPUCullMode",
    "GPUFrontFace",
    "GPUBlendFactor",
    "GPUBlendOperation",
    "GPUFilterMode",
    "GPUMipmapFilterMode",
    "GPUAddressMode",
    "GPUCompareFunction",
    "GPUVertexStepMode",
    "GPUIndexFormat",
    "GPULoadOp",
    "GPUStoreOp",
];

/// Subset of the parsed IDL we expose to emitters.
pub struct ParsedIdl {
    /// All enums in the IDL, in source order.
    pub enums: Vec<EnumDecl>,
    /// SHA-256 of the IDL source — used for stamping the generated
    /// files so a divergence is loud.
    pub source_hash: String,
}

#[derive(Clone)]
pub struct EnumDecl {
    pub name: String,
    /// String values in source order. e.g. `["r8unorm", "rg8unorm", …]`
    pub values: Vec<String>,
    /// True if this enum is in `PROJECT_RELEVANT_ENUMS`.
    pub project_relevant: bool,
}

pub struct Summary {
    pub enums_total: usize,
    pub enums_kept: usize,
}

pub fn parse(idl_text: &str) -> Result<ParsedIdl> {
    let definitions = weedle::parse(idl_text).map_err(|e| {
        format!(
            "weedle failed to parse webgpu.idl: {:?} — file may have shifted shape since 2026-04-28",
            e
        )
    })?;

    let mut enums = Vec::new();
    for def in definitions {
        if let Definition::Enum(e) = def {
            let name = identifier_to_string(&e.identifier);
            // Path: EnumDefinition.values: Braced<PunctuatedNonEmpty<StringLit, ',' >>.
            // Braced.body is the punctuated list; .list yields the
            // `StringLit`s; `.0` is the &str (no surrounding quotes).
            let values: Vec<String> = e.values.body.list.iter().map(|v| v.0.to_string()).collect();
            let project_relevant = PROJECT_RELEVANT_ENUMS.iter().any(|n| *n == name);
            enums.push(EnumDecl {
                name,
                values,
                project_relevant,
            });
        }
    }

    let source_hash = sha256_hex(idl_text);

    Ok(ParsedIdl { enums, source_hash })
}

pub fn summarize(parsed: &ParsedIdl) -> Summary {
    Summary {
        enums_total: parsed.enums.len(),
        enums_kept: parsed.enums.iter().filter(|e| e.project_relevant).count(),
    }
}

fn identifier_to_string(id: &Identifier<'_>) -> String {
    id.0.to_string()
}

/// Tiny pure-Rust SHA-256 (~80 LOC) so the codegen crate has zero
/// extra runtime deps. The output stamps generated files; if anyone
/// hand-edits the IDL or regenerates without updating headers, the
/// hash diff exposes the drift.
fn sha256_hex(input: &str) -> String {
    let bytes = input.as_bytes();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);

    // Padding: append 0x80, then zeros until length ≡ 56 (mod 64),
    // then 8-byte big-endian bit length.
    let mut padded = Vec::with_capacity(bytes.len() + 64);
    padded.extend_from_slice(bytes);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    // SHA-256 initial hash values (square roots of first 8 primes).
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // SHA-256 round constants (cube roots of first 64 primes).
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, b) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut s = String::with_capacity(64);
    for word in h.iter() {
        s.push_str(&format!("{word:08x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vector_empty() {
        assert_eq!(
            sha256_hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_known_vector_abc() {
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
