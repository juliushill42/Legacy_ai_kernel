pub mod quantization;
pub mod memory;
pub mod gpu;
pub mod cpu;

pub use quantization::QuantizedModel;
pub use memory::KVCache;
pub use cpu::CPUMatMul;
pub use gpu::GPUKernelLauncher;

pub const VERSION: &str = "0.1.0";

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_version() { assert_eq!(VERSION, "0.1.0"); }
    #[test]
    fn test_quantize() {
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let m = QuantizedModel::new(&w);
        assert!(m.memory_footprint() < 16);
    }
}
