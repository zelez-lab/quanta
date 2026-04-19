use crate::Format;

/// GPU-resident 2D image. Used for rendering targets and image data.
pub struct Texture {
    pub(crate) handle: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: Format,
    pub(crate) drop_fn: Option<Box<dyn FnOnce(u64)>>,
}

impl Texture {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn format(&self) -> Format {
        self.format
    }

    pub fn handle(&self) -> u64 {
        self.handle
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        if let Some(f) = self.drop_fn.take() {
            f(self.handle);
        }
    }
}
