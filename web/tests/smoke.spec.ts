/**
 * Headless WebGPU smoke tests for B‴.
 *
 * Each test spins up a tiny static HTTP server rooted at the
 * staged `examples/web_<name>/` directory, opens the index page,
 * waits for the page-side PASS banner, and asserts the banner
 * matches. PASS/FAIL is decided by the page itself (validating
 * either the compute output bytes or the rendered framebuffer);
 * this harness only checks the banner — same contract a human
 * eyeballs when running `quanta serve` locally.
 *
 * The test runner expects `quanta build web` to have already
 * staged wasm + quanta.js + generated/codes.js into each
 * example dir. CI runs the build step before invoking Playwright;
 * locally, `cargo run -p quanta-cli -- build web` does the same.
 */

import { test, expect } from "@playwright/test";
import { createServer, type Server } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = resolve(HERE, "..", "..");

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".idl": "text/plain; charset=utf-8",
};

/**
 * Static-file server fixture rooted at `rootDir`. Returns the
 * server handle + the URL it bound to. Caller closes the server
 * in a `finally` block.
 */
async function startStaticServer(rootDir: string): Promise<{ server: Server; url: string }> {
  const server = createServer(async (req, res) => {
    if (!req.url) {
      res.writeHead(400);
      res.end();
      return;
    }
    const path = req.url === "/" ? "/index.html" : req.url.split("?")[0];
    const fsPath = join(rootDir, path);
    try {
      const s = await stat(fsPath);
      if (!s.isFile()) {
        res.writeHead(404);
        res.end();
        return;
      }
      const data = await readFile(fsPath);
      const mime = MIME[extname(fsPath)] ?? "application/octet-stream";
      res.writeHead(200, { "content-type": mime });
      res.end(data);
    } catch {
      res.writeHead(404);
      res.end();
    }
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const addr = server.address();
  if (!addr || typeof addr === "string") {
    throw new Error("server.address() returned unexpected shape");
  }
  return { server, url: `http://127.0.0.1:${addr.port}/` };
}

test("web_add_one — compute path returns [1..64]", async ({ page }) => {
  const { server, url } = await startStaticServer(join(REPO_ROOT, "examples", "web_add_one"));
  try {
    await page.goto(url);
    const status = page.locator("#status");
    await expect(status).toContainText("PASS", { timeout: 30_000 });
    await expect(status).toContainText("buffer = [1, 2");
  } finally {
    server.close();
  }
});

test("web_triangle — render path produces triangle blue", async ({ page }) => {
  const { server, url } = await startStaticServer(join(REPO_ROOT, "examples", "web_triangle"));
  try {
    await page.goto(url);
    const status = page.locator("#status");
    await expect(status).toContainText("PASS", { timeout: 30_000 });
    await expect(status).toContainText("center pixel");
    // Golden-image SHA assertion — catches sub-pixel drift the
    // rgb-tolerance check would miss.
    await expect(status).toContainText("sha matches golden");
  } finally {
    server.close();
  }
});

test("web_textured — SetTexture+SetSampler wiring (step C)", async ({ page }) => {
  const { server, url } = await startStaticServer(join(REPO_ROOT, "examples", "web_textured"));
  try {
    await page.goto(url);
    const status = page.locator("#status");
    await expect(status).toContainText("PASS", { timeout: 30_000 });
    await expect(status).toContainText("all 16 pixels match");
    await expect(status).toContainText("sha matches golden");
  } finally {
    server.close();
  }
});

test("web_canvas — surface presentation: triangle visible on a live canvas (step 096)", async ({ page }) => {
  const consoleMsgs: string[] = [];
  page.on("console", (msg) => consoleMsgs.push(`[${msg.type()}] ${msg.text()}`));
  page.on("pageerror", (e) => consoleMsgs.push(`[pageerror] ${e.message}`));
  const { server, url } = await startStaticServer(join(REPO_ROOT, "examples", "web_canvas"));
  try {
    await page.goto(url);
    const status = page.locator("#status");
    try {
      await expect(status).toContainText("PASS", { timeout: 30_000 });
      await expect(status).toContainText("presented via SurfaceTarget::Canvas");
      await expect(status).toContainText("4x MSAA and resolved");
    } catch (e) {
      console.error("---console output---\n" + consoleMsgs.join("\n"));
      throw e;
    }

    // The pixels: compositor ground truth via an element screenshot
    // (in-page canvas readback of presented WebGPU frames is not
    // reliable in Chromium — this is why the page banner does not
    // assert pixels itself). Decode the PNG in-page with an Image +
    // 2d canvas and sample inside the triangle and in a corner.
    const png = await page.locator("#canvas").screenshot();
    const pixels = await page.evaluate(async (b64) => {
      const img = new Image();
      img.src = "data:image/png;base64," + b64;
      await img.decode();
      const c = document.createElement("canvas");
      c.width = img.width;
      c.height = img.height;
      const ctx = c.getContext("2d")!;
      ctx.drawImage(img, 0, 0);
      const at = (fx: number, fy: number) =>
        Array.from(
          ctx.getImageData(Math.floor(img.width * fx), Math.floor(img.height * fy), 1, 1).data,
        );
      return { center: at(0.5, 0.56), corner: at(0.03, 0.03) };
    }, png.toString("base64"));
    // Center: triangle blue (0.2, 0.4, 0.9). Corner: clear red.
    expect(pixels.center[2]!).toBeGreaterThan(200);
    expect(pixels.center[0]!).toBeLessThan(80);
    expect(pixels.center[1]!).toBeLessThan(130);
    expect(pixels.corner[0]!).toBeGreaterThan(200);
    expect(pixels.corner[1]!).toBeLessThan(60);
    expect(pixels.corner[2]!).toBeLessThan(60);
  } finally {
    server.close();
  }
});

test("web_diff — WGSL lane: saxpy + reduce_sum + counter + race + op-matrix", async ({ page }) => {
  const consoleMsgs: string[] = [];
  page.on("console", (msg) => consoleMsgs.push(`[${msg.type()}] ${msg.text()}`));
  page.on("pageerror", (e) => consoleMsgs.push(`[pageerror] ${e.message}`));
  const { server, url } = await startStaticServer(join(REPO_ROOT, "examples", "web_diff"));
  try {
    await page.goto(url);
    const status = page.locator("#status");
    try {
      await expect(status).toContainText("PASS", { timeout: 30_000 });
      await expect(status).toContainText("5 / 5 checks match");
    } catch (e) {
      console.error("---console output---\n" + consoleMsgs.join("\n"));
      throw e;
    }
  } finally {
    server.close();
  }
});
