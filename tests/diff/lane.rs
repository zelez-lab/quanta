//! Backend lanes the differential runner can target.
//!
//! `Reference` is the pure-Rust oracle. Every other lane is compared
//! against it. New lanes are gated by their backend feature flag so
//! the harness compiles on any host configuration.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lane {
    Reference,
    #[cfg(feature = "software")]
    Software,
    #[cfg(feature = "metal")]
    Metal,
}

impl Lane {
    pub fn name(self) -> &'static str {
        match self {
            Lane::Reference => "reference",
            #[cfg(feature = "software")]
            Lane::Software => "software",
            #[cfg(feature = "metal")]
            Lane::Metal => "metal",
        }
    }
}
