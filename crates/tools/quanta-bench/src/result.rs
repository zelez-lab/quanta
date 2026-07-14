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
