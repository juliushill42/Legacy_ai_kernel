use cc::Build;

fn main() {
    Build::new()
        .file("src/zig_quantization.c")
        .flag("-O3")
        .flag("-march=native")
        .flag("-ffast-math")
        .compile("zig_quantization");
    println!("cargo:rerun-if-changed=src/zig_quantization.c");
}
