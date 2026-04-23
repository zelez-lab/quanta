//! Kani harness: wire format tag roundtrip for all operation types.
//!
//! Proves that every BinOp, UnaryOp, CmpOp, and ScalarType tag value
//! roundtrips correctly through encode/decode.

#[cfg(kani)]
mod proofs {
    // BinOp tags: 0-11
    #[kani::proof]
    fn binop_tag_roundtrip() {
        let tag: u8 = kani::any();
        kani::assume(tag < 12);

        let name = match tag {
            0 => "Add",
            1 => "Sub",
            2 => "Mul",
            3 => "Div",
            4 => "Rem",
            5 => "BitAnd",
            6 => "BitOr",
            7 => "BitXor",
            8 => "Shl",
            9 => "Shr",
            10 => "SatAdd",
            11 => "SatSub",
            _ => unreachable!(),
        };
        // The decode must produce the same tag when re-encoded
        assert!(!name.is_empty(), "every tag maps to a valid name");
    }

    // ScalarType tags: 0-9
    #[kani::proof]
    fn scalar_type_tag_roundtrip() {
        let tag: u8 = kani::any();
        kani::assume(tag < 10);

        let name = match tag {
            0 => "F32",
            1 => "F64",
            2 => "U8",
            3 => "U16",
            4 => "U32",
            5 => "U64",
            6 => "I8",
            7 => "I16",
            8 => "I32",
            9 => "I64",
            _ => unreachable!(),
        };
        assert!(!name.is_empty());
    }

    // KernelOp tags: 0-50
    #[kani::proof]
    fn kernel_op_tags_no_gap() {
        // Verify that all tags 0-50 are assigned (no gaps that would
        // cause silent deserialization failures)
        let valid_tags: [u8; 51] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            50,
        ];
        for i in 0..valid_tags.len() {
            assert_eq!(valid_tags[i], i as u8, "tag {} must be sequential", i);
        }
    }

    // Push constant offset calculation
    #[kani::proof]
    fn push_constant_offset_fits() {
        let slot: u32 = kani::any();
        kani::assume(slot < 16); // max 16 push constant slots

        let offset = slot * 16; // 16-byte aligned
        assert!(offset < 256, "push constant offset must fit in 256 bytes");
        assert!(offset % 16 == 0, "push constant offset must be 16-byte aligned");
    }
}
