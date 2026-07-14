//! Minimal hand-rolled JSON for the report. No serde dep — keeps the bench
//! crate's footprint small enough that bench builds don't drag the Quanta
//! build into a serde-cycle. Format is fixed; we control both ends.

use crate::result::{BenchResult, Report};

pub fn encode_report(r: &Report) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"platform\": {},\n", quote(&r.platform)));
    s.push_str(&format!("  \"gpu\": {},\n", quote(&r.gpu_name)));
    s.push_str("  \"results\": [\n");
    for (i, b) in r.results.iter().enumerate() {
        s.push_str("    {");
        s.push_str(&format!("\"name\": {}, ", quote(&b.name)));
        s.push_str(&format!("\"workload\": {}, ", quote(&b.workload)));
        s.push_str(&format!("\"elements\": {}, ", b.elements));
        s.push_str(&format!("\"gpu_ms\": {}", num(b.gpu_ms)));
        if let Some(c) = b.cpu_ms {
            s.push_str(&format!(", \"cpu_ms\": {}", num(c)));
        }
        s.push('}');
        if i + 1 < r.results.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n");
    s.push('}');
    s.push('\n');
    s
}

pub fn decode_report(input: &str) -> Result<Report, String> {
    let mut p = Parser::new(input);
    p.skip_ws();
    p.expect('{')?;
    let mut platform = String::new();
    let mut gpu = String::new();
    let mut results = Vec::new();
    loop {
        p.skip_ws();
        if p.peek() == Some('}') {
            p.bump();
            break;
        }
        let key = p.parse_string()?;
        p.skip_ws();
        p.expect(':')?;
        p.skip_ws();
        match key.as_str() {
            "platform" => platform = p.parse_string()?,
            "gpu" => gpu = p.parse_string()?,
            "results" => results = p.parse_results()?,
            _ => p.skip_value()?,
        }
        p.skip_ws();
        if p.peek() == Some(',') {
            p.bump();
        }
    }
    Ok(Report {
        platform,
        gpu_name: gpu,
        results,
    })
}

fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn num(v: f64) -> String {
    if v.is_finite() {
        format!("{:.4}", v)
    } else {
        "null".to_string()
    }
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            src: s.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.src.get(self.pos).map(|b| *b as char)
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, c: char) -> Result<(), String> {
        match self.bump() {
            Some(g) if g == c => Ok(()),
            Some(g) => Err(format!("expected {}, got {} at {}", c, g, self.pos)),
            None => Err(format!("expected {}, got EOF", c)),
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.skip_ws();
        self.expect('"')?;
        let mut out = String::new();
        loop {
            match self.bump() {
                None => return Err("unterminated string".into()),
                Some('"') => return Ok(out),
                Some('\\') => match self.bump() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some(c) => out.push(c),
                    None => return Err("bad escape".into()),
                },
                Some(c) => out.push(c),
            }
        }
    }

    fn parse_number(&mut self) -> Result<f64, String> {
        self.skip_ws();
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| "bad utf8 in number".to_string())?;
        if s == "null" {
            return Ok(f64::NAN);
        }
        s.parse::<f64>().map_err(|e| e.to_string())
    }

    fn parse_int(&mut self) -> Result<u64, String> {
        let n = self.parse_number()?;
        Ok(n as u64)
    }

    fn parse_results(&mut self) -> Result<Vec<BenchResult>, String> {
        self.skip_ws();
        self.expect('[')?;
        let mut out = Vec::new();
        loop {
            self.skip_ws();
            if self.peek() == Some(']') {
                self.bump();
                return Ok(out);
            }
            self.expect('{')?;
            let mut name = String::new();
            let mut workload = String::new();
            let mut elements = 0u64;
            let mut gpu_ms = 0.0f64;
            let mut cpu_ms: Option<f64> = None;
            loop {
                self.skip_ws();
                if self.peek() == Some('}') {
                    self.bump();
                    break;
                }
                let key = self.parse_string()?;
                self.skip_ws();
                self.expect(':')?;
                self.skip_ws();
                match key.as_str() {
                    "name" => name = self.parse_string()?,
                    "workload" => workload = self.parse_string()?,
                    "elements" => elements = self.parse_int()?,
                    "gpu_ms" => gpu_ms = self.parse_number()?,
                    "cpu_ms" => {
                        let n = self.parse_number()?;
                        cpu_ms = if n.is_finite() { Some(n) } else { None };
                    }
                    _ => self.skip_value()?,
                }
                self.skip_ws();
                if self.peek() == Some(',') {
                    self.bump();
                }
            }
            out.push(BenchResult {
                name,
                workload,
                elements,
                gpu_ms,
                cpu_ms,
            });
            self.skip_ws();
            if self.peek() == Some(',') {
                self.bump();
            }
        }
    }

    fn skip_value(&mut self) -> Result<(), String> {
        self.skip_ws();
        match self.peek() {
            Some('"') => {
                self.parse_string()?;
                Ok(())
            }
            Some('[') => {
                self.bump();
                let mut depth = 1;
                while let Some(c) = self.bump() {
                    match c {
                        '[' => depth += 1,
                        ']' => {
                            depth -= 1;
                            if depth == 0 {
                                return Ok(());
                            }
                        }
                        '"' => {
                            self.pos -= 1;
                            self.parse_string()?;
                        }
                        _ => {}
                    }
                }
                Err("unterminated array".into())
            }
            Some('{') => {
                self.bump();
                let mut depth = 1;
                while let Some(c) = self.bump() {
                    match c {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                return Ok(());
                            }
                        }
                        '"' => {
                            self.pos -= 1;
                            self.parse_string()?;
                        }
                        _ => {}
                    }
                }
                Err("unterminated object".into())
            }
            _ => {
                self.parse_number()?;
                Ok(())
            }
        }
    }
}
