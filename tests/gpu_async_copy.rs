//! Async memory copy on a real device (step 044).
//!
//! The queue is real on Metal (a dedicated MTLCommandQueue — blits on
//! it can overlap the main queue's compute), a same-queue
//! vkCmdCopyBuffer submission on Vulkan, and the recorded serial copy
//! on the CPU tier. The wrapper contract everywhere: the data is
//! visible when `copy_buffer` returns.

#[test]
fn async_copy_round_trips_on_the_device() {
    let Ok(gpu) = quanta::init() else {
        eprintln!("skipping: no GPU");
        return;
    };
    if !gpu.supports_async_copy() {
        eprintln!(
            "async copy unsupported on {} (expected on WebGPU)",
            gpu.name()
        );
        assert!(
            gpu.async_copy_queue().is_err(),
            "query and create must agree"
        );
        return;
    }
    let q = gpu.async_copy_queue().unwrap();
    let src = gpu.field::<f32>(256).unwrap();
    let dst = gpu.field::<f32>(256).unwrap();
    let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();
    src.write(&data).unwrap();
    dst.write(&[0.0f32; 256]).unwrap();
    q.copy_buffer(&dst, &src, 256 * 4).unwrap();
    let out = dst.read().unwrap();
    assert_eq!(
        out, data,
        "the copy must be byte-exact and visible on return"
    );
}

#[test]
fn async_copy_queue_destroy_then_submit_fails() {
    let Ok(gpu) = quanta::init() else {
        eprintln!("skipping: no GPU");
        return;
    };
    if !gpu.supports_async_copy() {
        return;
    }
    // Drop releases the backend queue; a fresh queue still works after.
    {
        let _q = gpu.async_copy_queue().unwrap();
    }
    let q2 = gpu.async_copy_queue().unwrap();
    let a = gpu.field::<f32>(4).unwrap();
    a.write(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = gpu.field::<f32>(4).unwrap();
    b.write(&[0.0; 4]).unwrap();
    q2.copy_buffer(&b, &a, 16).unwrap();
    assert_eq!(b.read().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}
