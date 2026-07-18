# Particle System

Compute kernel updates particle positions; render kernel draws them as points.
Demonstrates the compute-to-render workflow.

## Data layout

```rust
#[quanta::gpu_type]
struct Particle {
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    life: f32,
    _pad: f32,
}
```

`#[quanta::gpu_type]` replaces the manual `#[repr(C)]` + `#[derive(Copy, Clone)]`.
It also generates MSL/WGSL struct declarations and implements `GpuType`, so the
struct can be used directly with `gpu.field::<Particle>(n)`.

## Compute kernel (update physics)

```rust
#[quanta::kernel]
fn update_particles(
    particles: &mut [f32],
    count: u32,
    dt: f32,
    gravity: f32,
) {
    let i = quark_id();
    if i >= count {
        return;
    }

    let base = i * 8u32; // 8 floats per particle

    // Load position and velocity
    let mut x = particles[base];
    let mut y = particles[base + 1u32];
    let mut z = particles[base + 2u32];
    let mut vx = particles[base + 3u32];
    let mut vy = particles[base + 4u32];
    let mut vz = particles[base + 5u32];
    let mut life = particles[base + 6u32];

    // Apply gravity
    vy += gravity * dt;

    // Integrate position
    x += vx * dt;
    y += vy * dt;
    z += vz * dt;

    // Floor bounce
    if y < 0.0f32 {
        y = 0.0f32;
        vy = -vy * 0.6f32; // Energy loss on bounce
    }

    // Decrease life
    life -= dt;

    // Store back
    particles[base] = x;
    particles[base + 1u32] = y;
    particles[base + 2u32] = z;
    particles[base + 3u32] = vx;
    particles[base + 4u32] = vy;
    particles[base + 5u32] = vz;
    particles[base + 6u32] = life;
}
```

## Vertex shader (draw particles as points)

Each particle's 8 floats bind as vertex attributes (locations 0–7); the shader
reads the three it needs. The view-projection matrix is a `&Mat4` uniform, and
the vertex forwards a fade `alpha` to the fragment through a `Varyings` struct.

```rust
use quanta::*;

// The vertex→fragment interface.
#[derive(quanta::Varyings)]
struct ParticleVarying {
    #[position] clip: Vec4, // gl_Position
    alpha: f32,             // Location 0
}

#[quanta::vertex]
fn particle_vertex(
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    _vx: f32,
    _vy: f32,
    _vz: f32,
    life: f32,
    _pad: f32,
    view_proj: &Mat4,
) -> ParticleVarying {
    let clip = view_proj * Vec4::new(pos_x, pos_y, pos_z, 1.0);
    let alpha = clamp(life / 2.0, 0.0, 1.0);
    ParticleVarying { clip, alpha }
}

#[quanta::fragment]
fn particle_fragment(s: ParticleVarying) -> Vec4 {
    // Fade from yellow to red as life decreases
    let r = 1.0;
    let g = s.alpha;
    let b = s.alpha * 0.3;
    Vec4::new(r, g, b, s.alpha)
}
```

## Host code

```rust
use quanta::{
    AttributeFormat, BlendState, Color, FieldUsage, Format, PipelineDesc, Primitive,
    RenderGpu, ResourceState, ShaderSource, StepMode, VertexAttribute, VertexLayout,
};

fn main() {
    let gpu = quanta::init().unwrap();

    let particle_count: u32 = 100_000;
    let floats_per_particle: usize = 8;

    // Initialize particles with random velocities
    let mut data = vec![0.0f32; particle_count as usize * floats_per_particle];
    for i in 0..particle_count as usize {
        let base = i * floats_per_particle;
        data[base] = 0.0;                                    // x
        data[base + 1] = 5.0;                                // y (start above ground)
        data[base + 2] = 0.0;                                // z
        data[base + 3] = (i as f32 * 0.1).sin() * 3.0;      // vx
        data[base + 4] = (i as f32 * 0.07).cos() * 8.0;     // vy
        data[base + 5] = (i as f32 * 0.13).sin() * 3.0;     // vz
        data[base + 6] = 3.0 + (i as f32 * 0.001);          // life
        data[base + 7] = 0.0;                                // pad
    }

    // Field used for both compute and vertex data
    let particles = gpu.field_with_usage::<f32>(
        particle_count as usize * floats_per_particle,
        FieldUsage::default_compute().union(FieldUsage::RENDER),
    ).unwrap();
    particles.write(&data).unwrap();

    // --- Compute wave (physics update) ---
    let mut update_wave = update_particles(&gpu).unwrap();
    update_wave.bind(0, &particles);
    update_wave.set_value(1, particle_count);
    update_wave.set_value(2, 0.016f32); // dt = 16ms
    update_wave.set_value(3, -9.8f32);  // gravity

    // --- Render pipeline (point sprites) ---
    // Render methods live on the RenderGpu extension trait (imported above).
    let render_target = gpu.render_target(1920, 1080, Format::BGRA8).unwrap();

    // View-projection matrix, bound to the vertex shader's `view_proj: &Mat4`.
    let view_proj: [f32; 16] = /* your camera matrix */ [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let mvp_buf = gpu
        .field_with_usage::<[f32; 16]>(1, FieldUsage::default_uniform())
        .unwrap();
    mvp_buf.write(&[view_proj]).unwrap();

    let layouts = [VertexLayout {
        stride: 32, // 8 floats x 4 bytes
        step: StepMode::Vertex,
        attributes: vec![
            VertexAttribute { location: 0, offset: 0,  format: AttributeFormat::Float },  // x
            VertexAttribute { location: 1, offset: 4,  format: AttributeFormat::Float },  // y
            VertexAttribute { location: 2, offset: 8,  format: AttributeFormat::Float },  // z
            VertexAttribute { location: 3, offset: 12, format: AttributeFormat::Float },  // vx
            VertexAttribute { location: 4, offset: 16, format: AttributeFormat::Float },  // vy
            VertexAttribute { location: 5, offset: 20, format: AttributeFormat::Float },  // vz
            VertexAttribute { location: 6, offset: 24, format: AttributeFormat::Float },  // life
            VertexAttribute { location: 7, offset: 28, format: AttributeFormat::Float },  // pad
        ],
    }];

    // The driver picks the right per-vendor payload from the embedded
    // multi-target binaries — no for_vendor() by hand.
    let pipeline = gpu.pipeline(
        &PipelineDesc::new(ShaderSource::Binaries {
            vertex: &PARTICLE_VERTEX_SHADER,
            fragment: &PARTICLE_FRAGMENT_SHADER,
        })
        .with_entries(
            PARTICLE_VERTEX_SHADER.entry_point,
            PARTICLE_FRAGMENT_SHADER.entry_point,
        )
        .with_vertex_layouts(&layouts)
        .with_color_formats(vec![Format::BGRA8])
        .with_primitive(Primitive::Point)
        .with_blend(BlendState::ADDITIVE),
    ).unwrap();

    // --- Simulation loop ---
    for _frame in 0..300 {
        // Step 1: Compute — update positions
        let mut pulse = gpu.dispatch(&update_wave, particle_count).unwrap();
        pulse.wait().unwrap();

        // Step 2: Barrier — transition from compute write to vertex read
        gpu.barrier_field(&particles, ResourceState::ComputeWrite, ResourceState::ShaderRead).unwrap();

        // Step 3: Render — draw particles as points
        let mut pulse = gpu.render(&render_target).unwrap()
            .clear(Color::BLACK)
            .pipeline(&pipeline)
            .vertices(0, &particles)
            .uniform(0, &mvp_buf)
            .draw(particle_count)
            .pulse().unwrap();
        pulse.wait().unwrap();
    }
}
```

## Compute-to-render barrier

The same field (`particles`) is written by the compute kernel and read by the
vertex shader. The `barrier_field` call between dispatch and render ensures:

1. All compute writes are visible
2. The GPU transitions the resource from compute to render usage

On Metal this is a no-op (automatic hazard tracking). On Vulkan it inserts
a pipeline barrier with the correct stage/access masks.
