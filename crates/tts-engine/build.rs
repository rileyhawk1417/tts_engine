//! espeak-rs vendored-build link patch (T011).
//!
//! `espeak-rs-sys` 0.2.0 vendors espeak-ng 1.52.0.1 and builds it with
//! CMake. On this host the build links the static `libespeak-ng.a` against
//! the system `libpcaudio.so` + `libsonic.so` (CMakeCache:
//! `USE_LIBPCAUDIO=ON`, `PCAUDIO_LIB=/usr/lib/libpcaudio.so`,
//! `SONIC_LIB=/usr/lib/libsonic.so`), but the crate's build script never
//! forwards `-lpcaudio`/`-lsonic` to the final binary link — leaving
//! `audio_object_*` (and sonic) undefined for any binary that links the
//! feature. Emitted from here so every binary in the workspace inherits
//! the directives whenever the `piper` feature is on.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "piper")]
    {
        for dir in ["/usr/lib", "/usr/lib64"] {
            if std::path::Path::new(dir).join("libpcaudio.so").exists() {
                println!("cargo:rustc-link-search=native={dir}");
                println!("cargo:rustc-link-lib=dylib=pcaudio");
                if std::path::Path::new(dir).join("libsonic.so").exists() {
                    println!("cargo:rustc-link-lib=dylib=sonic");
                }
                return;
            }
        }
        println!(
            "cargo:warning=libpcaudio.so not found in /usr/lib or /usr/lib64 — the piper feature link will fail until the espeak-ng audio deps are available"
        );
    }
}
