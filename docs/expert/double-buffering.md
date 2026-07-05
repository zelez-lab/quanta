# Expert: Double Buffering

Render to one target while presenting another, then swap. Eliminates
visible tearing and allows the GPU to pipeline work across frames.

## Basic double-buffer loop

```rust
use quanta::*;

fn render_loop(gpu: &Gpu, pipeline: &Pipeline, vb: &Field<f32>, vertex_count: u32) -> Result<(), QuantaError> {
    let target_a = gpu.render_target(1920, 1080, Format::RGBA8)?;
    let target_b = gpu.render_target(1920, 1080, Format::RGBA8)?;
    let mut front = &target_a;
    let mut back = &target_b;

    loop {
        // Render to the back buffer
        gpu.render(back)?
            .clear(Color::BLACK)
            .pipeline(pipeline)
            .vertices(0, vb)
            .draw(vertex_count)
            .pulse()?
            .wait()?;

        // Swap front and back
        core::mem::swap(&mut front, &mut back);

        // Present `front`: export it with `front.native_handle()` so a
        // compositor can consume it zero-copy — or, when rendering to a
        // window, skip manual double-buffering entirely and use a
        // `Surface` (`gpu.create_surface`): the swapchain owns the
        // buffers and `acquire`/`present` do the rotation for you.
    }
}
```

## Triple buffering

Add a third buffer to decouple GPU rendering from display refresh:

```rust
let targets = [
    gpu.render_target(w, h, Format::RGBA8)?,
    gpu.render_target(w, h, Format::RGBA8)?,
    gpu.render_target(w, h, Format::RGBA8)?,
];
let mut frame = 0usize;

loop {
    let target = &targets[frame % 3];

    gpu.render(target)?
        .clear(Color::BLACK)
        .pipeline(pipeline)
        .vertices(0, &vb)
        .draw(vertex_count)
        .pulse()?
        .wait()?;

    frame += 1;
}
```

## Ping-pong compute pattern

Process data back and forth between two fields without CPU roundtrips:

```rust
let mut src = gpu.field::<f32>(n)?;
let mut dst = gpu.field::<f32>(n)?;
src.write(&initial_data)?;

for _iteration in 0..num_iterations {
    let mut wave = process_kernel(&gpu)?;
    wave.bind(0, &src);
    wave.bind(1, &dst);
    gpu.dispatch(&wave, n as u32)?.wait()?;

    core::mem::swap(&mut src, &mut dst);
}

// Result is in `src` after an even number of iterations
let result = src.read()?;
```

## Timeline semaphores for frame pipelining

Avoid per-frame fences with monotonically increasing timeline values:

```rust
let timeline = gpu.timeline_create()?;

for frame in 0u64.. {
    // Wait for frame N-2 to finish before reusing its resources
    if frame >= 2 {
        gpu.timeline_wait(&timeline, frame - 2)?;
    }

    // Render frame N
    gpu.render(&targets[frame as usize % 3])?
        .pipeline(pipeline)
        .vertices(0, &vb)
        .draw(vertex_count)
        .pulse()?;

    // Signal that frame N is done
    gpu.timeline_signal(&timeline, frame)?;
}
```

## Per-frame uniform updates

For data that changes every frame, keep one uniform field alive and
rewrite it each iteration — no reallocation:

```rust
let mvp_buf = gpu.field_with_usage::<[f32; 16]>(1, FieldUsage::default_uniform())?;

loop {
    let mvp = compute_mvp_matrix(time);
    mvp_buf.write(&[mvp])?;

    gpu.render(target)?
        .pipeline(pipeline)
        .vertices(0, &vb)
        .uniform(0, &mvp_buf)
        .draw(vertex_count)
        .pulse()?
        .wait()?;
}
```

For compute-side data, `gpu.field_mapped` offers zero-copy CPU writes
(`as_mut_slice()` — no `write()` call at all); render binding slots take
regular `Field`s, so uniforms go through `write()` as above. On Apple
Silicon (unified memory) mapped writes are immediate; on discrete GPUs the
driver synchronizes on the next command buffer submission.
