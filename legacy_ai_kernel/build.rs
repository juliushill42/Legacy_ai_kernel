fn main() {
    println!("cargo:rerun-if-changed=zig_bindings/src/zig_quantization.c");
}
