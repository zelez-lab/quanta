fn main() {
    // Ensure the LLVM lib directory is in the link search path.
    if let Ok(prefix) = std::env::var("LLVM_SYS_221_PREFIX") {
        println!("cargo:rustc-link-search=native={}/lib", prefix);
    }
}
