/**
 * Playwright config — headless WebGPU smoke harness (B‴).
 *
 * Each smoke test loads one of the staged `examples/web_<name>/`
 * pages, waits for the page-side PASS/FAIL banner, and asserts it
 * shows PASS. The Rust + wasm + JS build is the responsibility of
 * `quanta build web` (run before this harness in CI). The harness
 * launches its own static HTTP server on a per-test fixture
 * basis — one server per test rather than one shared instance, so
 * `cwd` is set per-test rather than per-suite.
 *
 * We pin to Chromium with the WebGPU runtime feature enabled.
 * Firefox WebGPU support is shipping behind a flag at the time of
 * writing; once it stabilises a sibling project entry can be
 * added without restructuring the harness.
 */

import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  testMatch: /.*\.spec\.ts$/,
  // One worker keeps the static-server fixtures simple; smoke tests
  // are I/O-bound on `mapAsync` so parallelism would not help.
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    // Trace on first retry only — full traces on every test would
    // bloat CI artifacts; first-retry trace is enough to debug
    // intermittent failures.
    trace: "on-first-retry",
    // Most smoke tests load a wasm + run a render pass; 30s is
    // generous but lets a slow CI runner pass without re-tries.
    actionTimeout: 30_000,
  },
  projects: [
    {
      // The offscreen lanes (compute + render-to-texture readback).
      // These stay on the default headless shell: its rasterization is
      // what the vendored golden SHAs were recorded against, and the
      // full build drifts them by a rounding step.
      name: "chromium-webgpu",
      grepInvert: /web_canvas/,
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          // WebGPU is on by default in Chrome 113+; this flag is a
          // belt-and-braces guard for older Chromium revisions.
          args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
        },
      },
    },
    {
      // The presentation lane (step 096) needs a real compositor: the
      // headless shell renders WebGPU offscreen fine but never
      // composites a presented canvas frame — screenshots show white.
      // The full Chromium build in new-headless mode does composite.
      name: "chromium-webgpu-present",
      grep: /web_canvas/,
      use: {
        ...devices["Desktop Chrome"],
        channel: "chromium",
        launchOptions: {
          args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
        },
      },
    },
  ],
});
