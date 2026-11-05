//! Workspace-side link glue. Two jobs:
//!   1. T011 (piper): espeak-rs vendored-build link patch — `espeak-rs-sys`
//!      0.2.0 vendors espeak-ng 1.52.0.1 and links the static archive
//!      against the system `libpcaudio.so` + `libsonic.so`, but never
//!      forwards those to the final binary link.
//!   2. T010 (ggml-ffi): compile the C++ shim over llama.cpp's native
//!      Chatterbox TTS path and emit the static-archive link lines
//!      (order: dependents first, mirroring tools/tts link deps).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    piper_link_patch();
    #[cfg(feature = "ggml-ffi")]
    ggml_ffi_shim();
}

fn piper_link_patch() {
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

/// T010: build + link the chatterbox C++ shim against the pinned llama.cpp
/// static archives (release build at `/teranux/third_party/llama.cpp`,
/// commit `186eab1cb83899282bec44b837270959b56ee8f3`). Override the root
/// with `LLAMA_CPP_DIR`.
#[cfg(feature = "ggml-ffi")]
fn ggml_ffi_shim() {
    let root = std::env::var("LLAMA_CPP_DIR")
        .unwrap_or_else(|_| "/teranux/third_party/llama.cpp".to_string());
    let root = std::path::PathBuf::from(root);
    let build = root.join("build");
    for lib in [
        "tools/mtmd/libmtmd.a",
        "common/libllama-common.a",
        "common/libllama-common-base.a", // build-info.cpp.o (llama_build_target…)
        "src/libllama.a",
        "ggml/src/libggml.a",
        "ggml/src/libggml-cpu.a",
        "ggml/src/libggml-base.a",
        "vendor/hash/libvendor-hash.a",
    ] {
        if !build.join(lib).is_file() {
            panic!(
                "llama.cpp archive missing: {} (build the pinned checkout \
                 with cmake --build {}/build --target llama-common, or set \
                 LLAMA_CPP_DIR to a checkout carrying static archives)",
                build.join(lib).display(),
                root.display()
            );
        }
    }
    if !root.join("tools/mtmd/mtmd-helper.h").is_file() {
        panic!(
            "llama.cpp headers missing under {} (the pinned sparse checkout \
             must include tools/, common/, include/, ggml/, vendor/)",
            root.display()
        );
    }

    // --- compile the shim (same flags as llama-tts's tts.cpp.o) ---
    let mut b = cc::Build::new();
    b.cpp(true)
        .std("c++17")
        .file("src/shim/chatterbox_shim.cpp")
        .flag("-DGGML_USE_CPU")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-Wno-unused-function")
        .include(root.join("include"))
        .include(root.join("ggml/include"))
        .include(root.join("common"))
        .include(root.join("tools/mtmd"))
        .include(root.join("vendor"))
        .include(root.join("vendor/nlohmann"))
        .include(root.join("vendor/sheredom"));
    b.compile("tts_cbx_shim");

    // --- static archives, dependents first ---
    for dir in ["tools/mtmd", "common", "src", "ggml/src", "vendor/hash"] {
        println!(
            "cargo:rustc-link-search=native={}",
            build.join(dir).display()
        );
    }
    for lib in [
        "mtmd",
        "llama-common",
        "llama-common-base",
        "llama",
        "ggml",
        "ggml-cpu",
        "ggml-base",
        "vendor-hash",
    ] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    // System libs the ggml-cpu/common static code pulls in (measured link
    // failures → this list; mirrors llama-tts's transitive link closure).
    for sys in ["gomp", "numa", "m", "dl", "pthread"] {
        println!("cargo:rustc-link-lib=dylib={sys}");
    }
    // Re-run when the pin's archives change (checkout upgrades).
    println!("cargo:rerun-if-changed={}", root.join("build").display());
}
