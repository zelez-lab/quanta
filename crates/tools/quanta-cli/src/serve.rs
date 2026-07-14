//! `quanta serve` — minimal embedded HTTP server.
//!
//! Replaces `python3 -m http.server` for the smoke-test workflow. Lives
//! inside `quanta-cli` so the user never has to think about Python /
//! Node availability or which port-binding command their OS prefers.
//!
//! Scope is deliberately tiny: HTTP/1.1, GET only, sync TCP, file-system
//! roots. ~150 LOC of `std::net`. Security boundary: every request path
//! is normalized and checked against the configured serve root before
//! a file is opened — `..` traversal is rejected, not silently
//! handled.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};

use crate::Result;
use crate::workspace;

pub fn run(example: &str, port: u16) -> Result<()> {
    let root = workspace::root()?;
    // `serve` only takes a single example name (not "all"); validate.
    if example == "all" {
        return Err("`quanta serve` needs a single example name, not 'all'".into());
    }
    let examples = workspace::resolve_examples(example)?;
    let name = examples[0];
    let serve_root = root.join("examples").join(name);
    if !serve_root.is_dir() {
        return Err(format!("serve root missing: {}", serve_root.display()).into());
    }

    let bind = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&bind).map_err(|e| format!("failed to bind {bind}: {e}"))?;
    eprintln!(
        "[quanta serve] http://{bind}/  → {} (Ctrl+C to stop)",
        serve_root.display()
    );

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let root = serve_root.clone();
                std::thread::spawn(move || {
                    if let Err(e) = handle(s, &root) {
                        eprintln!("[quanta serve] connection error: {e}");
                    }
                });
            }
            Err(e) => eprintln!("[quanta serve] accept error: {e}"),
        }
    }
    Ok(())
}

fn handle(mut stream: TcpStream, root: &Path) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);

    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let raw_path = parts.next().unwrap_or("/");

    // Drain headers — we don't use any of them but the connection
    // contract is that we read the full request before responding.
    loop {
        let mut hdr = String::new();
        let n = reader.read_line(&mut hdr)?;
        if n == 0 || hdr == "\r\n" || hdr == "\n" {
            break;
        }
    }

    if method != "GET" && method != "HEAD" {
        return write_status(&mut stream, 405, "Method Not Allowed", b"");
    }

    let url_path = raw_path.split('?').next().unwrap_or("/");
    let resolved = match resolve(root, url_path) {
        Some(p) => p,
        None => {
            return write_status(
                &mut stream,
                400,
                "Bad Request",
                b"path traversal rejected\n",
            );
        }
    };

    let final_path = if resolved.is_dir() {
        resolved.join("index.html")
    } else {
        resolved
    };

    if !final_path.is_file() {
        return write_status(&mut stream, 404, "Not Found", b"not found\n");
    }

    let mut body = Vec::new();
    std::fs::File::open(&final_path)?.read_to_end(&mut body)?;
    let ctype = content_type(&final_path);

    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    if method == "GET" {
        stream.write_all(&body)?;
    }
    stream.flush()?;
    Ok(())
}

fn write_status(stream: &mut TcpStream, code: u16, reason: &str, body: &[u8]) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 {code} {reason}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}

/// Normalize the URL path against `root`, rejecting any traversal
/// ("..", absolute paths, drive letters, NUL bytes). Returns `None` if
/// the request would escape `root`.
fn resolve(root: &Path, url_path: &str) -> Option<PathBuf> {
    if url_path.contains('\0') {
        return None;
    }
    let mut acc = PathBuf::from(root);
    for raw in url_path.split('/') {
        if raw.is_empty() || raw == "." {
            continue;
        }
        if raw == ".." {
            return None;
        }
        let decoded = percent_decode(raw)?;
        if decoded.contains('/') || decoded.contains('\\') || decoded == ".." {
            return None;
        }
        acc.push(decoded);
    }
    // Final containment check: canonicalize each side and verify the
    // resolved path is under root. We tolerate the case where the
    // file does not yet exist on disk by walking the parent chain.
    let canonical_root = std::fs::canonicalize(root).ok()?;
    let mut probe = acc.clone();
    while !probe.exists() {
        probe.pop();
        if probe.as_os_str().is_empty() {
            return None;
        }
    }
    let canonical = std::fs::canonicalize(&probe).ok()?;
    if canonical.starts_with(&canonical_root) {
        Some(acc)
    } else {
        None
    }
}

fn percent_decode(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hi = hex(bytes[i + 1])?;
                let lo = hex(bytes[i + 2])?;
                out.push(char::from((hi << 4) | lo));
                i += 3;
            }
            b'+' => {
                // Some clients use '+' for space; accept it even though
                // we only serve files where spaces are unlikely.
                out.push(' ');
                i += 1;
            }
            c => {
                out.push(char::from(c));
                i += 1;
            }
        }
    }
    Some(out)
}

fn hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|s| s.to_str()) {
        Some("html") | Some("htm") => "text/html; charset=utf-8",
        Some("js") | Some("mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("wasm") => "application/wasm",
        Some("json") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("ico") => "image/x-icon",
        Some("map") => "application/json; charset=utf-8",
        _ => "application/octet-stream",
    }
}
