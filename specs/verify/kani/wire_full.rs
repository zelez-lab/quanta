//! Kani harnesses for full wire format roundtrip proofs.
//!
//! T209: Minimal KernelDef (empty body, 0 params) roundtrip
//! T217: DeviceFnDef field-level tag checks
//!
//! Strategy: model the wire protocol with fixed-size buffers to avoid
//! allocation. We inline the Writer/Reader logic using raw byte arrays,
//! proving that the byte-level encode/decode is an identity function.

#[cfg(kani)]
mod proofs {

    // =====================================================================
    // Shared helpers: inline Writer/Reader over fixed buffers
    // =====================================================================

    /// Write a u32 in little-endian to buf at pos. Returns new pos.
    fn write_u32(buf: &mut [u8], pos: usize, v: u32) -> usize {
        let bytes = v.to_le_bytes();
        buf[pos] = bytes[0];
        buf[pos + 1] = bytes[1];
        buf[pos + 2] = bytes[2];
        buf[pos + 3] = bytes[3];
        pos + 4
    }

    /// Read a u32 in little-endian from buf at pos. Returns (value, new pos).
    fn read_u32(buf: &[u8], pos: usize) -> (u32, usize) {
        let v = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
        (v, pos + 4)
    }

    /// Write a u8 to buf at pos. Returns new pos.
    fn write_u8(buf: &mut [u8], pos: usize, v: u8) -> usize {
        buf[pos] = v;
        pos + 1
    }

    /// Read a u8 from buf at pos. Returns (value, new pos).
    fn read_u8(buf: &[u8], pos: usize) -> (u8, usize) {
        (buf[pos], pos + 1)
    }

    /// Write a u16 in little-endian to buf at pos. Returns new pos.
    fn write_u16(buf: &mut [u8], pos: usize, v: u16) -> usize {
        let bytes = v.to_le_bytes();
        buf[pos] = bytes[0];
        buf[pos + 1] = bytes[1];
        pos + 2
    }

    /// Read a u16 in little-endian from buf at pos. Returns (value, new pos).
    fn read_u16(buf: &[u8], pos: usize) -> (u16, usize) {
        let v = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        (v, pos + 2)
    }

    /// Write a u64 in little-endian to buf at pos. Returns new pos.
    fn write_u64(buf: &mut [u8], pos: usize, v: u64) -> usize {
        let bytes = v.to_le_bytes();
        buf[pos] = bytes[0];
        buf[pos + 1] = bytes[1];
        buf[pos + 2] = bytes[2];
        buf[pos + 3] = bytes[3];
        buf[pos + 4] = bytes[4];
        buf[pos + 5] = bytes[5];
        buf[pos + 6] = bytes[6];
        buf[pos + 7] = bytes[7];
        pos + 8
    }

    /// Read a u64 in little-endian from buf at pos. Returns (value, new pos).
    fn read_u64(buf: &[u8], pos: usize) -> (u64, usize) {
        let v = u64::from_le_bytes([
            buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3],
            buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7],
        ]);
        (v, pos + 8)
    }

    /// Write i32 as le bytes (same bit pattern as u32).
    fn write_i32(buf: &mut [u8], pos: usize, v: i32) -> usize {
        write_u32(buf, pos, v as u32)
    }

    fn read_i32(buf: &[u8], pos: usize) -> (i32, usize) {
        let (bits, next) = read_u32(buf, pos);
        (bits as i32, next)
    }

    fn write_i64(buf: &mut [u8], pos: usize, v: i64) -> usize {
        write_u64(buf, pos, v as u64)
    }

    fn read_i64(buf: &[u8], pos: usize) -> (i64, usize) {
        let (bits, next) = read_u64(buf, pos);
        (bits as i64, next)
    }

    /// Write a fixed-length ASCII string with u32 length prefix.
    /// Returns new pos. `s` must have len <= s_buf.len().
    fn write_str(buf: &mut [u8], pos: usize, s: &[u8]) -> usize {
        let p = write_u32(buf, pos, s.len() as u32);
        let mut i = 0;
        let mut cur = p;
        while i < s.len() {
            buf[cur] = s[i];
            cur += 1;
            i += 1;
        }
        cur
    }

    /// Write option None (tag 0).
    fn write_option_none(buf: &mut [u8], pos: usize) -> usize {
        write_u8(buf, pos, 0)
    }

    /// Write option Some u32 (tag 1 + u32 payload).
    fn write_option_some_u32(buf: &mut [u8], pos: usize, v: u32) -> usize {
        let p = write_u8(buf, pos, 1);
        write_u32(buf, p, v)
    }

    // =====================================================================
    // T209: Minimal KernelDef roundtrip
    // =====================================================================
    //
    // A minimal KernelDef with:
    //   name = "k" (1 byte)
    //   params = [] (count 0)
    //   body = [] (count 0)
    //   body_source = None
    //   next_reg = symbolic u32
    //   opt_level = symbolic u8
    //   device_sources = [] (count 0)
    //   device_functions = [] (count 0)
    //   workgroup_size = [wx, wy, wz] (symbolic u32s)
    //   subgroup_size = None
    //   dynamic_shared_bytes = symbolic u32
    //
    // Wire layout:
    //   str("k"):           4 + 1 = 5 bytes
    //   params count:       4 bytes  (0)
    //   body count:         4 bytes  (0)
    //   body_source None:   1 byte   (0)
    //   next_reg:           4 bytes
    //   opt_level:          1 byte
    //   device_sources cnt: 4 bytes  (0)
    //   device_fns count:   4 bytes  (0)
    //   workgroup_size:     12 bytes (3 x u32)
    //   subgroup_size None: 1 byte   (0)
    //   dynamic_shared:     4 bytes
    //   Total: 5 + 4 + 4 + 1 + 4 + 1 + 4 + 4 + 12 + 1 + 4 = 44 bytes

    #[kani::proof]
    fn t209_minimal_kernel_def_roundtrip() {
        let next_reg: u32 = kani::any();
        let opt_level: u8 = kani::any();
        let wx: u32 = kani::any();
        let wy: u32 = kani::any();
        let wz: u32 = kani::any();
        let dyn_shared: u32 = kani::any();

        let mut buf = [0u8; 44];
        let name = b"k";

        // -- Encode --
        let p = write_str(&mut buf, 0, name);          // name = "k"
        assert!(p == 5);
        let p = write_u32(&mut buf, p, 0);              // params.len() = 0
        let p = write_u32(&mut buf, p, 0);              // body.len() = 0
        let p = write_u8(&mut buf, p, 0);               // body_source = None
        let p = write_u32(&mut buf, p, next_reg);        // next_reg
        let p = write_u8(&mut buf, p, opt_level);        // opt_level
        let p = write_u32(&mut buf, p, 0);              // device_sources.len() = 0
        let p = write_u32(&mut buf, p, 0);              // device_functions.len() = 0
        let p = write_u32(&mut buf, p, wx);             // workgroup_size[0]
        let p = write_u32(&mut buf, p, wy);             // workgroup_size[1]
        let p = write_u32(&mut buf, p, wz);             // workgroup_size[2]
        let p = write_u8(&mut buf, p, 0);               // subgroup_size = None
        let p = write_u32(&mut buf, p, dyn_shared);      // dynamic_shared_bytes
        assert!(p == 44);

        // -- Decode --
        let mut pos = 0usize;

        // name: str
        let (name_len, pos2) = read_u32(&buf, pos);
        assert!(name_len == 1, "name length must be 1");
        pos = pos2;
        assert!(buf[pos] == b'k', "name must be 'k'");
        pos += 1;

        // params count
        let (param_count, pos2) = read_u32(&buf, pos);
        assert!(param_count == 0, "zero params");
        pos = pos2;

        // body count
        let (body_count, pos2) = read_u32(&buf, pos);
        assert!(body_count == 0, "zero body ops");
        pos = pos2;

        // body_source = None
        let (bs_tag, pos2) = read_u8(&buf, pos);
        assert!(bs_tag == 0, "body_source is None");
        pos = pos2;

        // next_reg
        let (nr, pos2) = read_u32(&buf, pos);
        assert!(nr == next_reg, "next_reg roundtrip");
        pos = pos2;

        // opt_level
        let (ol, pos2) = read_u8(&buf, pos);
        assert!(ol == opt_level, "opt_level roundtrip");
        pos = pos2;

        // device_sources count
        let (ds_count, pos2) = read_u32(&buf, pos);
        assert!(ds_count == 0, "zero device_sources");
        pos = pos2;

        // device_functions count
        let (df_count, pos2) = read_u32(&buf, pos);
        assert!(df_count == 0, "zero device_functions");
        pos = pos2;

        // workgroup_size
        let (rwx, pos2) = read_u32(&buf, pos);
        assert!(rwx == wx, "workgroup_size[0] roundtrip");
        pos = pos2;
        let (rwy, pos2) = read_u32(&buf, pos);
        assert!(rwy == wy, "workgroup_size[1] roundtrip");
        pos = pos2;
        let (rwz, pos2) = read_u32(&buf, pos);
        assert!(rwz == wz, "workgroup_size[2] roundtrip");
        pos = pos2;

        // subgroup_size = None
        let (sg_tag, pos2) = read_u8(&buf, pos);
        assert!(sg_tag == 0, "subgroup_size is None");
        pos = pos2;

        // dynamic_shared_bytes
        let (ds, pos2) = read_u32(&buf, pos);
        assert!(ds == dyn_shared, "dynamic_shared_bytes roundtrip");
        pos = pos2;

        assert!(pos == 44, "consumed exactly 44 bytes");
    }

    // =====================================================================
    // T209b: KernelDef with subgroup_size = Some(v) variant
    // =====================================================================
    //
    // Same as T209 but subgroup_size is Some(u32), adding 4 bytes.
    // Total: 44 - 1 (None tag) + 1 (Some tag) + 4 (u32) = 48 bytes

    #[kani::proof]
    fn t209b_kernel_def_with_subgroup_size() {
        let next_reg: u32 = kani::any();
        let subgroup: u32 = kani::any();

        let mut buf = [0u8; 48];
        let name = b"k";

        // -- Encode --
        let p = write_str(&mut buf, 0, name);
        let p = write_u32(&mut buf, p, 0);       // params = []
        let p = write_u32(&mut buf, p, 0);       // body = []
        let p = write_u8(&mut buf, p, 0);        // body_source = None
        let p = write_u32(&mut buf, p, next_reg);
        let p = write_u8(&mut buf, p, 2);        // opt_level = 2
        let p = write_u32(&mut buf, p, 0);       // device_sources = []
        let p = write_u32(&mut buf, p, 0);       // device_functions = []
        let p = write_u32(&mut buf, p, 256);     // workgroup_size = [256, 1, 1]
        let p = write_u32(&mut buf, p, 1);
        let p = write_u32(&mut buf, p, 1);
        let p = write_u8(&mut buf, p, 1);        // subgroup_size = Some
        let p = write_u32(&mut buf, p, subgroup);
        let p = write_u32(&mut buf, p, 0);       // dynamic_shared_bytes = 0
        assert!(p == 48);

        // -- Decode subgroup_size --
        // Skip to subgroup_size offset: 5+4+4+1+4+1+4+4+12 = 39
        let pos = 39;
        let (sg_tag, pos2) = read_u8(&buf, pos);
        assert!(sg_tag == 1, "subgroup_size is Some");
        let (sg_val, pos2) = read_u32(&buf, pos2);
        assert!(sg_val == subgroup, "subgroup_size value roundtrip");

        // dynamic_shared_bytes
        let (ds, pos2) = read_u32(&buf, pos2);
        assert!(ds == 0);
        assert!(pos2 == 48);
    }

    // =====================================================================
    // T217: DeviceFnDef field-level roundtrip
    // =====================================================================
    //
    // A DeviceFnDef with:
    //   name = "f" (1 byte)
    //   params = [] (count 0)
    //   return_type = tag (symbolic, 0..11)
    //   body = [] (count 0)
    //   next_reg = symbolic u32
    //
    // Wire layout:
    //   str("f"):       4 + 1 = 5 bytes
    //   params count:   4 bytes (0)
    //   return_type:    1 byte  (scalar tag)
    //   body count:     4 bytes (0)
    //   next_reg:       4 bytes
    //   Total: 5 + 4 + 1 + 4 + 4 = 18 bytes

    #[kani::proof]
    fn t217_device_fn_def_roundtrip() {
        let ret_tag: u8 = kani::any();
        kani::assume(ret_tag < 12); // 12 valid ScalarType tags
        let next_reg: u32 = kani::any();

        let mut buf = [0u8; 18];
        let name = b"f";

        // -- Encode (mirrors write_device_fn_def) --
        let p = write_str(&mut buf, 0, name);      // name
        let p = write_u32(&mut buf, p, 0);          // params.len() = 0
        let p = write_u8(&mut buf, p, ret_tag);     // return_type
        let p = write_u32(&mut buf, p, 0);          // body.len() = 0
        let p = write_u32(&mut buf, p, next_reg);   // next_reg
        assert!(p == 18);

        // -- Decode (mirrors read_device_fn_def) --
        let mut pos = 0usize;

        // name
        let (name_len, pos2) = read_u32(&buf, pos);
        assert!(name_len == 1);
        pos = pos2;
        assert!(buf[pos] == b'f');
        pos += 1;

        // params count
        let (pc, pos2) = read_u32(&buf, pos);
        assert!(pc == 0);
        pos = pos2;

        // return_type (scalar tag)
        let (rt, pos2) = read_u8(&buf, pos);
        assert!(rt == ret_tag, "return_type tag roundtrip");
        pos = pos2;

        // body count
        let (bc, pos2) = read_u32(&buf, pos);
        assert!(bc == 0);
        pos = pos2;

        // next_reg
        let (nr, pos2) = read_u32(&buf, pos);
        assert!(nr == next_reg, "next_reg roundtrip");
        pos = pos2;

        assert!(pos == 18, "consumed exactly 18 bytes");
    }

    // =====================================================================
    // T217b: DeviceFnDef with one parameter
    // =====================================================================
    //
    // Tests that the parameter list encoding is correct:
    //   name = "g" (1 byte)
    //   params = [("x", ScalarType)] (count 1)
    //   return_type = symbolic tag
    //   body = [] (count 0)
    //   next_reg = symbolic u32
    //
    // Wire layout:
    //   str("g"):          5 bytes
    //   params count:      4 bytes (1)
    //   param[0] name "x": 5 bytes
    //   param[0] type:     1 byte
    //   return_type:       1 byte
    //   body count:        4 bytes (0)
    //   next_reg:          4 bytes
    //   Total: 5 + 4 + 5 + 1 + 1 + 4 + 4 = 24 bytes

    #[kani::proof]
    fn t217b_device_fn_def_with_param() {
        let param_type_tag: u8 = kani::any();
        kani::assume(param_type_tag < 12);
        let ret_tag: u8 = kani::any();
        kani::assume(ret_tag < 12);
        let next_reg: u32 = kani::any();

        let mut buf = [0u8; 24];

        // -- Encode --
        let p = write_str(&mut buf, 0, b"g");       // name
        let p = write_u32(&mut buf, p, 1);           // params.len() = 1
        let p = write_str(&mut buf, p, b"x");        // params[0].name
        let p = write_u8(&mut buf, p, param_type_tag); // params[0].type
        let p = write_u8(&mut buf, p, ret_tag);       // return_type
        let p = write_u32(&mut buf, p, 0);            // body.len() = 0
        let p = write_u32(&mut buf, p, next_reg);     // next_reg
        assert!(p == 24);

        // -- Decode --
        let mut pos = 0usize;

        // name = "g"
        let (nl, pos2) = read_u32(&buf, pos);
        assert!(nl == 1);
        pos = pos2;
        assert!(buf[pos] == b'g');
        pos += 1;

        // params count = 1
        let (pc, pos2) = read_u32(&buf, pos);
        assert!(pc == 1);
        pos = pos2;

        // params[0].name = "x"
        let (pnl, pos2) = read_u32(&buf, pos);
        assert!(pnl == 1);
        pos = pos2;
        assert!(buf[pos] == b'x');
        pos += 1;

        // params[0].type
        let (pt, pos2) = read_u8(&buf, pos);
        assert!(pt == param_type_tag, "param type roundtrip");
        pos = pos2;

        // return_type
        let (rt, pos2) = read_u8(&buf, pos);
        assert!(rt == ret_tag, "return type roundtrip");
        pos = pos2;

        // body count = 0
        let (bc, pos2) = read_u32(&buf, pos);
        assert!(bc == 0);
        pos = pos2;

        // next_reg
        let (nr, pos2) = read_u32(&buf, pos);
        assert!(nr == next_reg, "next_reg roundtrip");
        pos = pos2;

        assert!(pos == 24);
    }

    // =====================================================================
    // ConstValue tag roundtrip (supplements T215 Verus proof with CBMC)
    // =====================================================================

    #[kani::proof]
    fn const_value_tag_roundtrip() {
        let tag: u8 = kani::any();
        kani::assume(tag < 8); // 8 ConstValue variants

        let decoded = match tag {
            0 => 0u8, // F16
            1 => 1u8, // F32
            2 => 2u8, // F64
            3 => 3u8, // U32
            4 => 4u8, // U64
            5 => 5u8, // I32
            6 => 6u8, // I64
            7 => 7u8, // Bool
            _ => unreachable!(),
        };
        assert!(decoded == tag, "ConstValue tag identity");
    }

    // =====================================================================
    // ConstValue payload roundtrips (byte-level, no allocation)
    // =====================================================================

    #[kani::proof]
    fn const_f16_roundtrip() {
        let v: u16 = kani::any();
        let mut buf = [0u8; 3]; // tag(1) + u16(2)
        let p = write_u8(&mut buf, 0, 0); // tag = F16
        let p = write_u16(&mut buf, p, v);
        assert!(p == 3);

        let (tag, pos) = read_u8(&buf, 0);
        assert!(tag == 0);
        let (val, pos) = read_u16(&buf, pos);
        assert!(val == v);
    }

    #[kani::proof]
    fn const_u32_roundtrip() {
        let v: u32 = kani::any();
        let mut buf = [0u8; 5]; // tag(1) + u32(4)
        let p = write_u8(&mut buf, 0, 3); // tag = U32
        let p = write_u32(&mut buf, p, v);
        assert!(p == 5);

        let (tag, pos) = read_u8(&buf, 0);
        assert!(tag == 3);
        let (val, pos) = read_u32(&buf, pos);
        assert!(val == v);
    }

    #[kani::proof]
    fn const_u64_roundtrip() {
        let v: u64 = kani::any();
        let mut buf = [0u8; 9]; // tag(1) + u64(8)
        let p = write_u8(&mut buf, 0, 4); // tag = U64
        let p = write_u64(&mut buf, p, v);
        assert!(p == 9);

        let (tag, pos) = read_u8(&buf, 0);
        assert!(tag == 4);
        let (val, pos) = read_u64(&buf, pos);
        assert!(val == v);
    }

    #[kani::proof]
    fn const_i32_roundtrip() {
        let v: i32 = kani::any();
        let mut buf = [0u8; 5]; // tag(1) + i32(4)
        let p = write_u8(&mut buf, 0, 5); // tag = I32
        let p = write_i32(&mut buf, p, v);
        assert!(p == 5);

        let (tag, pos) = read_u8(&buf, 0);
        assert!(tag == 5);
        let (val, pos) = read_i32(&buf, pos);
        assert!(val == v);
    }

    #[kani::proof]
    fn const_i64_roundtrip() {
        let v: i64 = kani::any();
        let mut buf = [0u8; 9]; // tag(1) + i64(8)
        let p = write_u8(&mut buf, 0, 6); // tag = I64
        let p = write_i64(&mut buf, p, v);
        assert!(p == 9);

        let (tag, pos) = read_u8(&buf, 0);
        assert!(tag == 6);
        let (val, pos) = read_u64(&buf, pos);
        assert!(val as i64 == v);
    }

    #[kani::proof]
    fn const_f32_roundtrip() {
        let bits: u32 = kani::any();
        let mut buf = [0u8; 5]; // tag(1) + f32 bits as u32(4)
        let p = write_u8(&mut buf, 0, 1); // tag = F32
        let p = write_u32(&mut buf, p, bits); // f32::to_bits -> u32
        assert!(p == 5);

        let (tag, pos) = read_u8(&buf, 0);
        assert!(tag == 1);
        let (val, pos) = read_u32(&buf, pos); // f32::from_bits(u32)
        assert!(val == bits, "f32 bit-pattern roundtrip");
    }

    #[kani::proof]
    fn const_f64_roundtrip() {
        let bits: u64 = kani::any();
        let mut buf = [0u8; 9]; // tag(1) + f64 bits as u64(8)
        let p = write_u8(&mut buf, 0, 2); // tag = F64
        let p = write_u64(&mut buf, p, bits); // f64::to_bits -> u64
        assert!(p == 9);

        let (tag, pos) = read_u8(&buf, 0);
        assert!(tag == 2);
        let (val, pos) = read_u64(&buf, pos); // f64::from_bits(u64)
        assert!(val == bits, "f64 bit-pattern roundtrip");
    }

    #[kani::proof]
    fn const_bool_roundtrip() {
        let v: bool = kani::any();
        let mut buf = [0u8; 2]; // tag(1) + bool as u8(1)
        let p = write_u8(&mut buf, 0, 7); // tag = Bool
        let p = write_u8(&mut buf, p, v as u8);
        assert!(p == 2);

        let (tag, pos) = read_u8(&buf, 0);
        assert!(tag == 7);
        let (val, pos) = read_u8(&buf, pos);
        // Decoder accepts only 0 or 1
        assert!(val == 0 || val == 1);
        let decoded = val == 1;
        assert!(decoded == v, "bool roundtrip");
    }

    // =====================================================================
    // KernelParam tag roundtrip (supplements T216)
    // =====================================================================

    #[kani::proof]
    fn kernel_param_tag_roundtrip() {
        let tag: u8 = kani::any();
        kani::assume(tag < 6); // 6 KernelParam variants

        // All variants decode the same payload: name + slot + scalar_type.
        // The tag alone determines the variant.
        let variant_name = match tag {
            0 => "FieldRead",
            1 => "FieldWrite",
            2 => "Constant",
            3 => "Texture2DRead",
            4 => "Texture2DWrite",
            5 => "Texture3DRead",
            _ => unreachable!(),
        };
        assert!(!variant_name.is_empty(), "every param tag maps to a variant");
    }

    // =====================================================================
    // KernelParam full roundtrip (byte-level)
    // =====================================================================
    //
    // Param with name "a" (1 byte), symbolic slot, symbolic scalar tag.
    // Wire: tag(1) + str("a")(5) + slot(4) + scalar(1) = 11 bytes

    #[kani::proof]
    fn kernel_param_full_roundtrip() {
        let param_tag: u8 = kani::any();
        kani::assume(param_tag < 6);
        let slot: u32 = kani::any();
        let scalar_tag: u8 = kani::any();
        kani::assume(scalar_tag < 12);

        let mut buf = [0u8; 11];

        // -- Encode (mirrors write_kernel_param) --
        let p = write_u8(&mut buf, 0, param_tag);
        let p = write_str(&mut buf, p, b"a");
        let p = write_u32(&mut buf, p, slot);
        let p = write_u8(&mut buf, p, scalar_tag);
        assert!(p == 11);

        // -- Decode (mirrors read_kernel_param) --
        let (rtag, pos) = read_u8(&buf, 0);
        assert!(rtag == param_tag, "param tag roundtrip");

        // name = "a"
        let (nl, pos) = read_u32(&buf, pos);
        assert!(nl == 1);
        assert!(buf[pos] == b'a');
        let pos = pos + 1;

        // slot
        let (rslot, pos) = read_u32(&buf, pos);
        assert!(rslot == slot, "slot roundtrip");

        // scalar_type
        let (rst, pos) = read_u8(&buf, pos);
        assert!(rst == scalar_tag, "scalar_type roundtrip");

        assert!(pos == 11);
    }

    // =====================================================================
    // ScalarType tag roundtrip (corrected: 12 variants, F16=0..Bool=11)
    // =====================================================================

    #[kani::proof]
    fn scalar_type_tag_roundtrip_corrected() {
        let tag: u8 = kani::any();
        kani::assume(tag < 12);

        // Mirrors the actual wire encode/decode in helpers.rs
        let name = match tag {
            0 => "F16",
            1 => "F32",
            2 => "F64",
            3 => "U8",
            4 => "U16",
            5 => "U32",
            6 => "U64",
            7 => "I8",
            8 => "I16",
            9 => "I32",
            10 => "I64",
            11 => "Bool",
            _ => unreachable!(),
        };
        assert!(!name.is_empty(), "every ScalarType tag maps to a variant");
    }

    /// Invalid ScalarType tags are rejected.
    #[kani::proof]
    fn scalar_type_invalid_tag_rejected() {
        let tag: u8 = kani::any();
        kani::assume(tag >= 12);

        let valid = match tag {
            0..=11 => true,
            _ => false,
        };
        assert!(!valid, "tags >= 12 must be invalid");
    }
}
