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
```

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

```rust
#[repr(C)]
#[derive(Copy, Clone)]
struct CameraUniforms {
    view_proj: [f32; 16],
    screen_height: f32,
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
    camera: &CameraUniforms,
) -> ParticleVarying {
    let clip = mat4_mul_vec4(camera.view_proj, vec4(pos_x, pos_y, pos_z, 1.0));
    let alpha = clamp(life / 2.0, 0.0, 1.0);
    ParticleVarying { clip_pos: clip, alpha }
}

#[quanta::fragment]
fn particle_fragment(varying: ParticleVarying) -> Vec4 {
    // Fade from yellow to red as life decreases
    let r = 1.0;
    let g = varying.alpha;
    let b = varying.alpha * 0.3;
    vec4(r, g, b, varying.alpha)
}
```

## Host code

```rust
use quanta::{
    AttributeFormat, BlendState, Color, FieldUsage, Format, PipelineDesc, Primitive,
    ResourceState, StepMode, VertexAttribute, VertexLayout,
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
    let particles = gpu.field::<f32>(
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
    let render_target = gpu.render_target(1920, 1080, Format::BGRA8).unwrap();

    let pipeline = gpu.pipeline(&PipelineDesc {
        vertex: particle_vertex().for_vendor(gpu.caps().vendor).unwrap(),
        fragment: particle_fragment().for_vendor(gpu.caps().vendor).unwrap(),
        vertex_entry: "particle_vertex",
        fragment_entry: "particle_fragment",
        vertex_layouts: &[VertexLayout {
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
        }],
        primitive: Primitive::Point,
        blend: BlendState::ADDITIVE,
        ..PipelineDesc::default()
    }).unwrap();

    // --- Simulation loop ---
    for _frame in 0..300 {
        // Step 1: Compute — update positions
        let mut pulse = gpu.dispatch(&update_wave, particle_count).unwrap();
        pulse.wait().unwrap();

        // Step 2: Barrier — transition from compute write to vertex read
        gpu.barrier_buffer(&particles, ResourceState::ComputeWrite, ResourceState::ShaderRead).unwrap();

        // Step 3: Render — draw particles as points
        let mut pass = gpu.render_begin(&render_target).unwrap();
        pass.clear(Color::BLACK);
        pass.set_pipeline(&pipeline);
        pass.bind_vertices(0, &particles);
        pass.draw(particle_count);
        let mut pulse = gpu.render_end(pass).unwrap();
        pulse.wait().unwrap();
    }
}
```

## Compute-to-render barrier

The same field (`particles`) is written by the compute kernel and read by the
vertex shader. The `barrier_buffer` call between dispatch and render ensures:

1. All compute writes are visible
2. The GPU transitions the resource from compute to render usage

On Metal this is a no-op (automatic hazard tracking). On Vulkan it inserts
a pipeline barrier with the correct stage/access masks.
