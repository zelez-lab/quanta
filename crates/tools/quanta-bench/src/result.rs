//! Bench result data model.

#[derive(Clone, Debug)]
pub struct BenchResult {
    pub name: String,
    pub workload: String,
    pub elements: u64,
    pub gpu_ms: f64,
    pub cpu_ms: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub platform: String,
    pub gpu_name: String,
    pub results: Vec<BenchResult>,
}

impl Report {
    pub fn new(platform: String, gpu_name: String) -> Self {
        Self {
            platform,
            gpu_name,
            results: Vec::new(),
        }
    }
}

/// The file-name slug of a device name: ASCII-lowercased, every run of
/// non-alphanumerics collapsed to one `-`, trimmed. `"Intel(R) Iris(R) Xe
/// Graphics"` → `intel-r-iris-r-xe-graphics`, `"llvmpipe (LLVM 20.1.2, 256
/// bits)"` → `llvmpipe-llvm-20-1-2-256-bits`. The LLVM version is part of
/// llvmpipe's name on purpose: a different llvmpipe is a different device
/// for baseline purposes, exactly as the compare gate already treats it.
pub fn device_slug(gpu_name: &str) -> String {
    let mut out = String::with_capacity(gpu_name.len());
    let mut pending_dash = false;
    for ch in gpu_name.chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_dash && !out.is_empty() {
                out.push('-');
            }
            pending_dash = false;
            out.push(ch.to_ascii_lowercase());
        } else {
            pending_dash = true;
        }
    }
    out
}

impl Report {
    /// The baseline file this report belongs to, relative to the baselines
    /// directory: `<platform>-<device-slug>.json`. One file per device per
    /// OS/arch, so an integrated GPU and a discrete card on the same host
    /// each keep their own baseline.
    pub fn baseline_file_name(&self) -> String {
        format!("{}-{}.json", self.platform, device_slug(&self.gpu_name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_collapses_punctuation_and_lowercases() {
        assert_eq!(device_slug("Apple M1 Pro"), "apple-m1-pro");
        assert_eq!(
            device_slug("Intel(R) Iris(R) Xe Graphics"),
            "intel-r-iris-r-xe-graphics"
        );
        assert_eq!(
            device_slug("llvmpipe (LLVM 20.1.2, 256 bits)"),
            "llvmpipe-llvm-20-1-2-256-bits"
        );
        assert_eq!(
            device_slug("  AMD Radeon RX 9060 XT  "),
            "amd-radeon-rx-9060-xt"
        );
    }

    #[test]
    fn baseline_file_name_is_platform_then_device() {
        let r = Report::new(
            "windows-x86_64".into(),
            "Intel(R) Iris(R) Xe Graphics".into(),
        );
        assert_eq!(
            r.baseline_file_name(),
            "windows-x86_64-intel-r-iris-r-xe-graphics.json"
        );
    }
}
