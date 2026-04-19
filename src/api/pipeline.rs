/// Compiled render pipeline (vertex + fragment shaders).
///
/// Created via [`GpuDevice::pipeline`]. Used with render passes.
pub struct Pipeline {
    pub(crate) handle: u64,
    pub(crate) drop_fn: Option<Box<dyn FnOnce(u64)>>,
}

impl Pipeline {
    pub fn handle(&self) -> u64 {
        self.handle
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if let Some(f) = self.drop_fn.take() {
            f(self.handle);
        }
    }
}
