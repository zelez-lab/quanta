#!/usr/bin/env python3
"""Generate the checked-in npy/npz interchange fixtures.

Provenance contract: this script is run ONCE by a maintainer with the
pinned numpy version below, and the resulting binaries are committed as
the interchange ground truth. CI never runs Python — the committed
bytes are the contract, this script documents where they came from.

PENDING: not yet run — numpy was absent on the authoring machine. Run
from this directory with the pinned numpy and commit the outputs; the
typed-layer test suite (dtype matrix slice) consumes them.

Pinned provenance:

    python3 -m pip install numpy==2.3.2
    python3 gen_fixtures.py

Fixture inventory (scope doc §8):
  - one small array per supported dtype: the ten element types plus
    f16 and bool
  - shape edges: 0-d scalar, (3,) 1-tuple, Fortran-order
  - v2.0 (long header), big-endian (the positive byteswap case)
  - stored.npz (np.savez, mixed dtypes — the stored container)
  - compressed.npz (np.savez_compressed — the positive inflate case;
    entries sized/patterned so zlib emits fixed-Huffman, dynamic-Huffman
    and window-copy activity, not just stored blocks)
  - object_error.npy (the |O pickle exclusion, an error-path fixture;
    written with allow_pickle semantics but never loaded by tests as
    anything but a loud Dtype error)

Every array is deterministic (arange-based) so regeneration with the
pinned version is byte-identical.
"""

import io
import os

import numpy as np

PINNED = "2.3.2"
assert np.__version__ == PINNED, (
    f"fixtures must be generated with numpy=={PINNED} (found {np.__version__}); "
    "the committed bytes are the provenance contract"
)

HERE = os.path.dirname(os.path.abspath(__file__))


def save(name, arr, **kwargs):
    path = os.path.join(HERE, name)
    with open(path, "wb") as f:
        np.save(f, arr, **kwargs)
    print(f"wrote {name}: dtype={arr.dtype.str} shape={arr.shape}")


# One per supported dtype (§4 of the scope doc).
save("f32.npy", np.arange(6, dtype=np.float32).reshape(2, 3) / 4)
save("f64.npy", np.arange(6, dtype=np.float64).reshape(2, 3) / 4)
save("i32.npy", np.arange(-3, 3, dtype=np.int32).reshape(2, 3))
save("u32.npy", np.arange(6, dtype=np.uint32).reshape(2, 3))
save("i64.npy", (np.arange(6, dtype=np.int64) - 3) * 2**33)
save("u64.npy", np.arange(6, dtype=np.uint64) * 2**33)
save("u8.npy", np.arange(6, dtype=np.uint8).reshape(2, 3))
save("i8.npy", np.arange(-3, 3, dtype=np.int8).reshape(2, 3))
save("u16.npy", np.arange(6, dtype=np.uint16) * 300)
save("i16.npy", (np.arange(6, dtype=np.int16) - 3) * 300)
save("f16.npy", np.arange(6, dtype=np.float16) / 4)  # loads upconverted
save("bool.npy", np.array([True, False, True, True], dtype=np.bool_))

# Shape edges.
save("scalar_0d.npy", np.float32(2.5))  # shape ()
save("vec3.npy", np.arange(3, dtype=np.float32))  # the (3,) 1-tuple form
save("fortran.npy", np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3)))

# v2.0: numpy caps ndarrays at 64 dims, so a simple-dtype header can
# never organically overflow the u16 length field — ask numpy's own
# low-level writer for the v2.0 framing directly instead. Still a
# numpy-written file; only the version is pinned.
with open("v2_long_header.npy", "wb") as f:
    np.lib.format.write_array(
        f, np.arange(6, dtype=np.float32).reshape(2, 3) / 4, version=(2, 0)
    )
print("wrote v2_long_header.npy (forced v2.0 framing)")

# Big-endian: the positive byteswap-on-load case.
save("big_endian.npy", (np.arange(6, dtype=np.float32) / 4).astype(">f4"))

# The |O pickle exclusion — an ERROR fixture (tests assert the loud
# Dtype error, never a load).
save("object_error.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)

# Containers: mixed dtypes, entry names exercised on the way back.
arrays = {
    "weights": np.arange(12, dtype=np.float32).reshape(3, 4) / 8,
    "labels": np.arange(3, dtype=np.uint64),
    "mask": np.array([1, 0, 1], dtype=np.uint8),
}
with open(os.path.join(HERE, "stored.npz"), "wb") as f:
    np.savez(f, **arrays)
print("wrote stored.npz:", ", ".join(arrays))

# Compressed: include a large repetitive array (window copies, dynamic
# Huffman) next to a tiny one (fixed Huffman) so the inflate paths are
# all exercised by real zlib output.
compressed = dict(arrays)
compressed["repetitive"] = np.tile(np.arange(64, dtype=np.float32), 256)
with open(os.path.join(HERE, "compressed.npz"), "wb") as f:
    np.savez_compressed(f, **compressed)
print("wrote compressed.npz:", ", ".join(compressed))

# Sanity: everything numpy just wrote reads back identically.
for name in os.listdir(HERE):
    if name.endswith(".npy") and name != "object_error.npy":
        np.load(os.path.join(HERE, name))
print("fixtures verified readable")
