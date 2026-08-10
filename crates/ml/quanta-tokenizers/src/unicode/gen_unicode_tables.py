#!/usr/bin/env python3
"""Pinned generator for `src/unicode/tables.rs` (quanta-tokenizers).

Provenance contract (the `gen_fixtures.py` model): a maintainer runs this
script ONCE per deliberate UCD bump and commits the emitted `tables.rs`.
CI never runs Python — the committed Rust file is the truth. The emitted
output is a deterministic function of the pinned UCD files alone: no
timestamps, no environment-dependent content, stable ordering throughout.

UCD version pin: 16.0.0
Source files (all fetched from
https://www.unicode.org/Public/16.0.0/ucd/, SHA-256 verified against the
pins below, cached outside the repo):

  UnicodeData.txt                    general categories (incl. the
                                     First/Last range convention),
                                     canonical combining classes, raw
                                     canonical + compatibility
                                     decompositions, simple lowercase
  CompositionExclusions.txt          script-specific + post-composition
                                     exclusions (cross-check input)
  DerivedNormalizationProps.txt      Full_Composition_Exclusion — the
                                     authoritative filter for the
                                     canonical composition pair table
  PropList.txt                       White_Space
  SpecialCasing.txt                  unconditional full lowercase
                                     mappings (U+0130 and friends)
  Scripts.txt                        Script ranges (UnicodeScripts
                                     pre-tokenizer)
  auxiliary/GraphemeBreakProperty.txt  UAX #29 grapheme cluster classes
  emoji/emoji-data.txt               Extended_Pictographic (UAX #29 GB11)

Hangul syllables (U+AC00..U+D7A3) are deliberately absent from every
emitted table: (de)composition is algorithmic at runtime per UAX #15.

Self-check: when the running CPython's `unicodedata` is built from the
same UCD version, the script reimplements NFD/NFKD/NFC/NFKC over the
emitted data model and verifies every codepoint (and a deterministic set
of synthetic mark-heavy strings) against `unicodedata.normalize`, plus
full-sweep category / combining-class / lowercase comparisons. A version
mismatch skips the check with a warning; it never changes the output.

Usage:
  python3 gen_unicode_tables.py [--ucd-dir DIR] [--out FILE] [--no-selfcheck]
"""

import argparse
import hashlib
import os
import sys
import urllib.request

UCD_VERSION = "16.0.0"
UCD_BASE_URL = f"https://www.unicode.org/Public/{UCD_VERSION}/ucd"

# SHA-256 pins for every consumed UCD file.
UCD_FILES = {
    "UnicodeData.txt": "ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f",
    "CompositionExclusions.txt": "89e83cf9cc8bef6c1f8bf77e42cf6f0341dfa42e66261f4dbe9b492e7a23c8ee",
    "DerivedNormalizationProps.txt": "4d4c03892dea9146d674b686e495df2d55a28d071ac474041d73518f887abddc",
    "PropList.txt": "53d614508e2a0b2305a8aa21cd60d993de9326cdf65993660dfcce4503548583",
    "SpecialCasing.txt": "8d5de354eef79f2395a54c9c7dcebbaf3d30fc962d0f85611ea97aa973a0c451",
    "Scripts.txt": "9e88f0a677df47311106340be8ede2ecdacd9c1c931831218d2be6d5508e0039",
    "auxiliary/GraphemeBreakProperty.txt": "c29360bd6f7132811d701d29069541e827eb44bfc4c8fbde8c370d6982689dc1",
    "emoji/emoji-data.txt": "f1365a5173eee18e1f98b240cdc492e84a25f1ce7e0c9d1094eb29c41a22696a",
}

MAX_CP = 0x110000

# The 30 general categories, alphabetical. Index = position. The
# GeneralCategory enum in mod.rs mirrors this list exactly.
CATEGORIES = [
    "Cc", "Cf", "Cn", "Co", "Cs", "Ll", "Lm", "Lo", "Lt", "Lu",
    "Mc", "Me", "Mn", "Nd", "Nl", "No", "Pc", "Pd", "Pe", "Pf",
    "Pi", "Po", "Ps", "Sc", "Sk", "Sm", "So", "Zl", "Zp", "Zs",
]
CATEGORY_INDEX = {name: i for i, name in enumerate(CATEGORIES)}

# UAX #29 Grapheme_Cluster_Break classes as of UCD 16.0.0. Index =
# position; the GraphemeBreak enum in mod.rs mirrors this list exactly.
# An unlisted class name in the data file is a hard error (a future UCD
# bump must be a deliberate, reviewed change).
GCB_CLASSES = [
    "Other", "CR", "LF", "Control", "Extend", "ZWJ", "Regional_Indicator",
    "Prepend", "SpacingMark", "L", "V", "T", "LV", "LVT",
]
GCB_INDEX = {name: i for i, name in enumerate(GCB_CLASSES)}

HANGUL_S_BASE, HANGUL_L_BASE, HANGUL_V_BASE, HANGUL_T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
HANGUL_L_COUNT, HANGUL_V_COUNT, HANGUL_T_COUNT = 19, 21, 28
HANGUL_N_COUNT = HANGUL_V_COUNT * HANGUL_T_COUNT
HANGUL_S_COUNT = HANGUL_L_COUNT * HANGUL_N_COUNT


def fetch_files(ucd_dir):
    """Return {relpath: text}, verifying every SHA-256 pin."""
    if ucd_dir is None:
        cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        ucd_dir = os.path.join(cache_root, "quanta-ucd", UCD_VERSION)
    texts = {}
    for rel, want_sha in sorted(UCD_FILES.items()):
        path = os.path.join(ucd_dir, rel)
        if not os.path.exists(path):
            url = f"{UCD_BASE_URL}/{rel}"
            print(f"  downloading {url}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with urllib.request.urlopen(url, timeout=120) as resp:
                data = resp.read()
            with open(path, "wb") as f:
                f.write(data)
        with open(path, "rb") as f:
            data = f.read()
        got_sha = hashlib.sha256(data).hexdigest()
        if got_sha != want_sha:
            sys.exit(
                f"SHA-256 mismatch for {path}:\n  want {want_sha}\n  got  {got_sha}\n"
                f"Delete the file to re-download, or fix the pin deliberately."
            )
        texts[rel] = data.decode("utf-8")
    return texts


def data_lines(text):
    """UCD line iterator: strips comments and blanks."""
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            yield line


def parse_cp_range(field):
    field = field.strip()
    if ".." in field:
        lo, hi = field.split("..")
        return int(lo, 16), int(hi, 16)
    cp = int(field, 16)
    return cp, cp


def parse_prop_file(text, wanted_prop):
    """Yield (lo, hi) for every range carrying wanted_prop."""
    for line in data_lines(text):
        fields = [f.strip() for f in line.split(";")]
        if len(fields) >= 2 and fields[1] == wanted_prop:
            yield parse_cp_range(fields[0])


def parse_range_map(text):
    """Parse files of the `range ; Value` shape into {cp: value}."""
    out = {}
    for line in data_lines(text):
        fields = [f.strip() for f in line.split(";")]
        lo, hi = parse_cp_range(fields[0])
        for cp in range(lo, hi + 1):
            out[cp] = fields[1]
    return out


def parse_unicode_data(text):
    """Return (category[MAX_CP], ccc{cp}, canon{cp:[cp..]}, compat{cp:[cp..]}, lower{cp:cp})."""
    category = ["Cn"] * MAX_CP
    ccc = {}
    canon = {}
    compat = {}
    lower = {}
    lines = list(text.splitlines())
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip():
            continue
        f = line.split(";")
        cp, name, gc = int(f[0], 16), f[1], f[2]
        if name.endswith(", First>"):
            last = lines[i].split(";")
            i += 1
            assert last[1].endswith(", Last>"), f"unpaired First at {f[0]}"
            lo, hi = cp, int(last[0], 16)
            # Ranged entries carry no decomposition, no nonzero ccc and no
            # case mapping — assert rather than assume.
            assert f[3] == "0" and f[5] == "" and f[13] == "", f"ranged entry with data at {f[0]}"
            for c in range(lo, hi + 1):
                category[c] = gc
            continue
        category[cp] = gc
        klass = int(f[3])
        if klass != 0:
            ccc[cp] = klass
        decomp = f[5]
        if decomp:
            if decomp.startswith("<"):
                parts = decomp.split(">", 1)[1].split()
                compat[cp] = [int(p, 16) for p in parts]
            else:
                parts = [int(p, 16) for p in decomp.split()]
                assert 1 <= len(parts) <= 2, f"canonical decomposition of len {len(parts)} at {f[0]}"
                canon[cp] = parts
        if f[13]:
            lower[cp] = int(f[13], 16)
    return category, ccc, canon, compat, lower


def parse_special_casing(text):
    """Unconditional full lowercase mappings {cp: [cp..]} (len != 1 or != simple)."""
    out = {}
    for line in data_lines(text):
        fields = [f.strip() for f in line.split(";")]
        if fields and fields[-1] == "":
            fields.pop()
        if len(fields) != 4:
            continue  # conditional (language/context) mapping: not the Rust to_lowercase surface
        cp = int(fields[0], 16)
        lo = [int(p, 16) for p in fields[1].split()]
        if lo != [cp]:
            out[cp] = lo
    return out


def runs_from_array(values):
    """[(start, value)] partition runs over the whole codepoint space."""
    runs = []
    prev = None
    for cp in range(MAX_CP):
        v = values[cp]
        if v != prev:
            runs.append((cp, v))
            prev = v
    return runs


def sparse_runs(mapping):
    """[(lo, hi, value)] inclusive runs over a sparse {cp: value} map."""
    runs = []
    for cp in sorted(mapping):
        v = mapping[cp]
        if runs and runs[-1][1] == cp - 1 and runs[-1][2] == v:
            runs[-1] = (runs[-1][0], cp, v)
        else:
            runs.append((cp, cp, v))
    return runs


def merge_ranges(ranges):
    """Sort and coalesce [(lo, hi)] inclusive ranges."""
    out = []
    for lo, hi in sorted(ranges):
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


class Emitter:
    WIDTH = 116

    def __init__(self):
        self.chunks = []

    def raw(self, s):
        self.chunks.append(s)

    def _pack(self, values, fmt):
        lines = []
        line = "    "
        for v in values:
            tok = fmt(v) + ","
            if len(line) + len(tok) > self.WIDTH:
                lines.append(line.rstrip())
                line = "    "
            line += tok
        if line.strip():
            lines.append(line.rstrip())
        return lines

    def array(self, decl, values, fmt=str):
        body = [f"#[rustfmt::skip]\n{decl} = &["] + self._pack(values, fmt) + ["];\n"]
        self.chunks.append("\n".join(body))

    def cumsum_array(self, name, ty, values, vis="pub(super) "):
        """Sorted array stored as first-difference deltas, rebuilt by const eval."""
        deltas = [values[0]] + [b - a for a, b in zip(values, values[1:])]
        assert all(d >= 0 for d in deltas), f"{name} is not sorted"
        body = [f"#[rustfmt::skip]\n{vis}static {name}: [{ty}; {len(values)}] = cumsum_{ty}(["]
        body += self._pack(deltas, str)
        body.append("]);\n")
        self.chunks.append("\n".join(body))

    def text(self):
        return "\n".join(self.chunks)


def build_pool(sequences):
    """Pack sequences into one pool with exact- and sub-sequence reuse.

    Deterministic: sequences are laid out longest-first (ties by content)
    and each is first searched for as a subsequence of the pool built so
    far. Returns (pool, {tuple(seq): offset}).
    """
    pool = []
    offsets = {}

    def find_sub(seq):
        n, m = len(pool), len(seq)
        for i in range(n - m + 1):
            if pool[i : i + m] == seq:
                return i
        return None

    for seq in sorted(set(map(tuple, sequences)), key=lambda s: (-len(s), s)):
        seq_l = list(seq)
        at = find_sub(seq_l)
        if at is None:
            at = len(pool)
            pool.extend(seq_l)
        offsets[seq] = at
    return pool, offsets


def selfcheck(category, ccc, canon, compat, lowercase_full, compose_pairs, white_space):
    import unicodedata

    if unicodedata.unidata_version != UCD_VERSION:
        print(
            f"  WARNING: self-check skipped — running Python's unicodedata is UCD "
            f"{unicodedata.unidata_version}, tables are pinned to {UCD_VERSION}."
        )
        return

    def is_surrogate(cp):
        return 0xD800 <= cp <= 0xDFFF

    # -- category / ccc / lowercase sweeps ------------------------------
    for cp in range(MAX_CP):
        if is_surrogate(cp):
            continue
        c = chr(cp)
        assert unicodedata.category(c) == category[cp], f"category mismatch at {cp:04X}"
        assert unicodedata.combining(c) == ccc.get(cp, 0), f"ccc mismatch at {cp:04X}"
        mine = lowercase_full.get(cp, [cp])
        theirs = [ord(x) for x in c.lower()]
        assert mine == theirs, f"lowercase mismatch at {cp:04X}: {mine} vs {theirs}"

    # -- White_Space: property-derived, spot-checked --------------------
    ws = sorted(cp for lo, hi in white_space for cp in range(lo, hi + 1))
    assert 0x09 in ws and 0x20 in ws and 0xA0 in ws and 0x3000 in ws and 0x180E not in ws

    # -- normalization: reimplement over the emitted model --------------
    def decomp_char(cp, compat_mode, out):
        if HANGUL_S_BASE <= cp < HANGUL_S_BASE + HANGUL_S_COUNT:
            s = cp - HANGUL_S_BASE
            out.append(HANGUL_L_BASE + s // HANGUL_N_COUNT)
            out.append(HANGUL_V_BASE + (s % HANGUL_N_COUNT) // HANGUL_T_COUNT)
            if s % HANGUL_T_COUNT:
                out.append(HANGUL_T_BASE + s % HANGUL_T_COUNT)
            return
        if compat_mode and cp in compat:
            for c in compat[cp]:
                decomp_char(c, compat_mode, out)
            return
        if cp in canon:
            for c in canon[cp]:
                decomp_char(c, compat_mode, out)
            return
        k = ccc.get(cp, 0)
        i = len(out)
        if k != 0:
            while i > 0 and ccc.get(out[i - 1], 0) > k:
                i -= 1
        out.insert(i, cp)

    def decompose(s, compat_mode):
        out = []
        for ch in s:
            decomp_char(ord(ch), compat_mode, out)
        return out

    def primary_composite(a, b):
        if HANGUL_L_BASE <= a < HANGUL_L_BASE + HANGUL_L_COUNT and HANGUL_V_BASE <= b < HANGUL_V_BASE + HANGUL_V_COUNT:
            return HANGUL_S_BASE + ((a - HANGUL_L_BASE) * HANGUL_V_COUNT + (b - HANGUL_V_BASE)) * HANGUL_T_COUNT
        if (
            HANGUL_S_BASE <= a < HANGUL_S_BASE + HANGUL_S_COUNT
            and (a - HANGUL_S_BASE) % HANGUL_T_COUNT == 0
            and HANGUL_T_BASE < b < HANGUL_T_BASE + HANGUL_T_COUNT
        ):
            return a + (b - HANGUL_T_BASE)
        return compose_pairs.get((a, b))

    def compose(cps):
        out = []
        starter = None
        last_ccc = None  # None: nothing between the starter and the candidate
        for cp in cps:
            k = ccc.get(cp, 0)
            if starter is not None and (last_ccc is None or last_ccc < k):
                p = primary_composite(out[starter], cp)
                if p is not None:
                    out[starter] = p
                    continue
            out.append(cp)
            if k == 0:
                starter, last_ccc = len(out) - 1, None
            else:
                last_ccc = k

        return "".join(map(chr, out))

    def my_norm(form, s):
        cps = decompose(s, form in ("NFKD", "NFKC"))
        if form in ("NFC", "NFKC"):
            return compose(cps)
        return "".join(map(chr, cps))

    for cp in range(MAX_CP):
        if is_surrogate(cp):
            continue
        c = chr(cp)
        for form in ("NFD", "NFKD", "NFC", "NFKC"):
            assert my_norm(form, c) == unicodedata.normalize(form, c), f"{form} mismatch at {cp:04X}"

    import random

    rng = random.Random(16)
    interesting = (
        sorted(ccc)  # every nonzero-ccc char
        + sorted(canon)[::7]
        + sorted(compat)[::13]
        + list(range(0x1100, 0x1113))
        + list(range(0x1161, 0x1176))
        + list(range(0x11A8, 0x11C3))
        + [0xAC00, 0xAC01, 0xD7A3, 0x0041, 0x0301, 0x0958, 0x2126, 0x212B, 0xFB01, 0xFDFA]
    )
    for _ in range(4000):
        s = "".join(chr(rng.choice(interesting)) for _ in range(rng.randrange(1, 12)))
        for form in ("NFD", "NFKD", "NFC", "NFKC"):
            assert my_norm(form, s) == unicodedata.normalize(form, s), f"{form} mismatch on {s!r}"

    print("  self-check passed (full-sweep category/ccc/lowercase + all-codepoint 4-form")
    print("  normalization against unicodedata, + 4000 synthetic mark-heavy strings).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucd-dir", help="directory holding the pinned UCD files (default: XDG cache; missing files are downloaded)")
    ap.add_argument("--out", help="output path (default: tables.rs next to this script)")
    ap.add_argument("--no-selfcheck", action="store_true")
    args = ap.parse_args()

    print(f"UCD {UCD_VERSION}")
    texts = fetch_files(args.ucd_dir)

    category, ccc, canon, compat, simple_lower = parse_unicode_data(texts["UnicodeData.txt"])
    special_lower = parse_special_casing(texts["SpecialCasing.txt"])

    # Full unconditional lowercase = simple mappings overridden by
    # unconditional SpecialCasing entries (U+0130 → "i\u{307}" class).
    lowercase_full = {cp: [lo] for cp, lo in simple_lower.items()}
    lowercase_full.update(special_lower)

    # Full_Composition_Exclusion is authoritative for the compose table;
    # CompositionExclusions.txt + the derivable rules serve as a
    # cross-check that we understood both files.
    fce = set()
    for lo, hi in parse_prop_file(texts["DerivedNormalizationProps.txt"], "Full_Composition_Exclusion"):
        fce.update(range(lo, hi + 1))
    listed = set()
    for line in data_lines(texts["CompositionExclusions.txt"]):
        lo, hi = parse_cp_range(line)
        listed.update(range(lo, hi + 1))
    derived = set(listed)
    for cp, d in canon.items():
        if len(d) == 1 or ccc.get(cp, 0) != 0 or ccc.get(d[0], 0) != 0:
            derived.add(cp)
    assert derived == fce, "Full_Composition_Exclusion does not match the derivable rule set"

    compose_pairs = {}
    for cp, d in sorted(canon.items()):
        if len(d) == 2 and cp not in fce:
            key = (d[0], d[1])
            assert key not in compose_pairs, f"duplicate composition pair {key}"
            compose_pairs[key] = cp

    white_space = merge_ranges(parse_prop_file(texts["PropList.txt"], "White_Space"))

    script_map = parse_range_map(texts["Scripts.txt"])
    script_names = sorted(set(script_map.values())) + ["Unknown"]
    script_index = {name: i for i, name in enumerate(script_names)}
    script_arr = [script_index["Unknown"]] * MAX_CP
    for cp, name in script_map.items():
        script_arr[cp] = script_index[name]

    gcb_map = parse_range_map(texts["auxiliary/GraphemeBreakProperty.txt"])
    for name in set(gcb_map.values()):
        assert name in GCB_INDEX, f"unknown Grapheme_Cluster_Break class {name!r} — deliberate UCD bump required"
    gcb_arr = [0] * MAX_CP
    for cp, name in gcb_map.items():
        gcb_arr[cp] = GCB_INDEX[name]

    ext_pict = merge_ranges(parse_prop_file(texts["emoji/emoji-data.txt"], "Extended_Pictographic"))

    if not args.no_selfcheck:
        print("running self-check…")
        selfcheck(category, ccc, canon, compat, lowercase_full, compose_pairs, white_space)

    # ------------------------------------------------------------------
    # Build emitted representations.
    # ------------------------------------------------------------------
    gc_runs = runs_from_array([CATEGORY_INDEX[c] for c in category])
    ccc_runs = sparse_runs(ccc)
    script_runs = runs_from_array(script_arr)
    gcb_runs = runs_from_array(gcb_arr)

    canon_keys = sorted(canon)
    canon_firsts = [canon[cp][0] for cp in canon_keys]
    canon_seconds = [(canon[cp][1] if len(canon[cp]) == 2 else 0) for cp in canon_keys]

    compat_keys = sorted(compat)
    compat_pool, compat_offsets = build_pool(compat.values())
    compat_vals = []
    for cp in compat_keys:
        seq = tuple(compat[cp])
        off = compat_offsets[seq]
        assert len(seq) < 32 and off < (1 << 27)
        compat_vals.append((off << 5) | len(seq))

    compose_keys = sorted(compose_pairs)
    compose_vals = [compose_pairs[k] for k in compose_keys]

    lower_single_keys = sorted(cp for cp in lowercase_full if len(lowercase_full[cp]) == 1)
    lower_single_deltas = [lowercase_full[cp][0] - cp for cp in lower_single_keys]
    lower_multi_keys = sorted(cp for cp in lowercase_full if len(lowercase_full[cp]) > 1)
    lower_pool, lower_offsets = build_pool([tuple(lowercase_full[cp]) for cp in lower_multi_keys])
    lower_multi_vals = []
    for cp in lower_multi_keys:
        seq = tuple(lowercase_full[cp])
        off = lower_offsets[seq]
        assert len(seq) < 32 and off < (1 << 27)
        lower_multi_vals.append((off << 5) | len(seq))

    # ------------------------------------------------------------------
    # Emit.
    # ------------------------------------------------------------------
    e = Emitter()
    e.raw("//! GENERATED FILE — DO NOT EDIT BY HAND.")
    e.raw("//!")
    e.raw("//! Emitted by `gen_unicode_tables.py` (this directory) from the Unicode")
    e.raw(f"//! Character Database version {UCD_VERSION}. Regeneration is a maintainer")
    e.raw("//! action, never a CI step:")
    e.raw("//!")
    e.raw("//! ```sh")
    e.raw("//! python3 src/unicode/gen_unicode_tables.py")
    e.raw("//! ```")
    e.raw("//!")
    e.raw("//! Source files (SHA-256 pinned in the generator):")
    for rel in sorted(UCD_FILES):
        e.raw(f"//!   {rel}: {UCD_FILES[rel]}")
    e.raw("//!")
    e.raw("//! Table inventory:")
    e.raw(f"//!   general categories: {len(gc_runs)} partition runs over 30 categories")
    e.raw(f"//!   combining classes: {len(ccc_runs)} sparse runs")
    e.raw(f"//!   canonical decompositions (raw pairs): {len(canon_keys)}")
    e.raw(f"//!   compatibility decompositions: {len(compat_keys)} (pool: {len(compat_pool)} code points)")
    e.raw(f"//!   composition pairs (Full_Composition_Exclusion applied): {len(compose_keys)}")
    e.raw(f"//!   lowercase mappings: {len(lower_single_keys)} single + {len(lower_multi_keys)} multi-char")
    e.raw(f"//!   White_Space ranges: {len(white_space)}")
    e.raw(f"//!   scripts: {len(script_runs)} partition runs over {len(script_names)} scripts")
    e.raw(f"//!   grapheme break classes: {len(gcb_runs)} partition runs")
    e.raw(f"//!   Extended_Pictographic ranges: {len(ext_pict)}")
    e.raw("//!")
    e.raw("//! Every sorted key/start array is stored as first-difference deltas and")
    e.raw("//! rebuilt by const evaluation (`cumsum_*`), purely to keep this file")
    e.raw("//! small; the reconstructed arrays are the semantic content.")
    e.raw("")
    e.raw("/// Prefix-sum reconstruction for delta-stored sorted tables (const eval).")
    e.raw("const fn cumsum_u32<const N: usize>(mut d: [u32; N]) -> [u32; N] {")
    e.raw("    let mut i = 1;")
    e.raw("    while i < N {")
    e.raw("        d[i] += d[i - 1];")
    e.raw("        i += 1;")
    e.raw("    }")
    e.raw("    d")
    e.raw("}")
    e.raw("")
    e.raw("/// Prefix-sum reconstruction for delta-stored sorted tables (const eval).")
    e.raw("const fn cumsum_u64<const N: usize>(mut d: [u64; N]) -> [u64; N] {")
    e.raw("    let mut i = 1;")
    e.raw("    while i < N {")
    e.raw("        d[i] += d[i - 1];")
    e.raw("        i += 1;")
    e.raw("    }")
    e.raw("    d")
    e.raw("}")
    e.raw("")

    e.raw("// General categories: a full partition of 0..=0x10FFFF. Entry i covers")
    e.raw("// GENERAL_CATEGORY_STARTS[i]..GENERAL_CATEGORY_STARTS[i+1] and carries the")
    e.raw("// category index GENERAL_CATEGORY_VALUES[i] (see GeneralCategory in mod.rs;")
    e.raw(f"// index order: {' '.join(CATEGORIES)}).")
    e.cumsum_array("GENERAL_CATEGORY_STARTS", "u32", [r[0] for r in gc_runs])
    e.array("pub(super) const GENERAL_CATEGORY_VALUES: &[u8]", [r[1] for r in gc_runs])

    e.raw("// Canonical combining classes: sparse inclusive ranges; everything else is 0.")
    e.cumsum_array("CCC_STARTS", "u32", [r[0] for r in ccc_runs])
    e.cumsum_array("CCC_ENDS", "u32", [r[1] for r in ccc_runs])
    e.array("pub(super) const CCC_VALUES: &[u8]", [r[2] for r in ccc_runs])

    e.raw("// Raw (pairwise, unexpanded) canonical decompositions from UnicodeData.txt;")
    e.raw("// SECONDS[i] == 0 marks a singleton decomposition. Full expansion is")
    e.raw("// recursive at runtime. Hangul is algorithmic and absent.")
    e.cumsum_array("CANONICAL_DECOMP_KEYS", "u32", canon_keys)
    e.array("pub(super) const CANONICAL_DECOMP_FIRSTS: &[u32]", canon_firsts)
    e.array("pub(super) const CANONICAL_DECOMP_SECONDS: &[u32]", canon_seconds)

    e.raw("// Raw compatibility decompositions; value = pool_offset << 5 | len into")
    e.raw("// COMPAT_DECOMP_POOL. Chars whose full NFKD differs only via their")
    e.raw("// canonical decomposition are reached through CANONICAL_DECOMP_* instead.")
    e.cumsum_array("COMPAT_DECOMP_KEYS", "u32", compat_keys)
    e.array("pub(super) const COMPAT_DECOMP_VALUES: &[u32]", compat_vals)
    e.array("pub(super) const COMPAT_DECOMP_POOL: &[u32]", compat_pool)

    e.raw("// Canonical composition: sorted keys, key = starter << 21 | combining;")
    e.raw("// Full_Composition_Exclusion already applied. Hangul is algorithmic.")
    e.cumsum_array("COMPOSE_KEYS", "u64", [(a << 21) | b for a, b in compose_keys])
    e.array("pub(super) const COMPOSE_VALUES: &[u32]", compose_vals)

    e.raw("// Unconditional full lowercase mappings (UnicodeData simple mappings +")
    e.raw("// SpecialCasing unconditional entries). Single-char mappings store")
    e.raw("// `mapped - key` as i32; multi-char mappings live in the *_MULTI tables")
    e.raw("// (value = pool_offset << 5 | len into LOWERCASE_POOL).")
    e.cumsum_array("LOWERCASE_KEYS", "u32", lower_single_keys)
    e.array("pub(super) const LOWERCASE_DELTAS: &[i32]", lower_single_deltas)
    e.array("pub(super) const LOWERCASE_MULTI_KEYS: &[u32]", lower_multi_keys)
    e.array("pub(super) const LOWERCASE_MULTI_VALUES: &[u32]", lower_multi_vals)
    e.array("pub(super) const LOWERCASE_POOL: &[u32]", lower_pool)

    e.raw("// White_Space (PropList.txt), inclusive ranges.")
    e.array("pub(super) const WHITE_SPACE_STARTS: &[u32]", [r[0] for r in white_space])
    e.array("pub(super) const WHITE_SPACE_ENDS: &[u32]", [r[1] for r in white_space])

    e.raw("/// A Unicode script (Scripts.txt), plus `Unknown` for unassigned /")
    e.raw("/// unlisted code points.")
    e.raw("#[repr(u8)]")
    e.raw("#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]")
    e.raw("pub enum Script {")
    for name in script_names:
        e.raw(f"    {name.replace('_', '')},")
    e.raw("}\n")
    e.array(
        "pub(super) const SCRIPT_BY_INDEX: &[Script]",
        script_names,
        fmt=lambda n: f"Script::{n.replace('_', '')}",
    )
    e.raw("// Scripts: full partition of 0..=0x10FFFF, same layout as the categories.")
    e.cumsum_array("SCRIPT_STARTS", "u32", [r[0] for r in script_runs])
    e.array("pub(super) const SCRIPT_VALUES: &[u8]", [r[1] for r in script_runs])

    e.raw("// UAX #29 Grapheme_Cluster_Break: full partition; value indices follow the")
    e.raw(f"// GraphemeBreak enum in mod.rs ({' '.join(GCB_CLASSES)}).")
    e.cumsum_array("GRAPHEME_BREAK_STARTS", "u32", [r[0] for r in gcb_runs])
    e.array("pub(super) const GRAPHEME_BREAK_VALUES: &[u8]", [r[1] for r in gcb_runs])

    e.raw("// Extended_Pictographic (emoji/emoji-data.txt), inclusive ranges (GB11).")
    e.array("pub(super) const EXTENDED_PICTOGRAPHIC_STARTS: &[u32]", [r[0] for r in ext_pict])
    e.array("pub(super) const EXTENDED_PICTOGRAPHIC_ENDS: &[u32]", [r[1] for r in ext_pict])

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables.rs")
    text = e.text()
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {out_path} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
