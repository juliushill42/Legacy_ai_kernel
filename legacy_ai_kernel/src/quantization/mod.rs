pub mod gguf;
pub mod int4;

use crate::zig_bindings::{quantize_int4_array, dequantize_int4_array};

pub struct QuantizedModel {
    pub weights_int4: Vec<u8>,
    pub scale: f32,
    pub min_val: f32,
    pub n: usize,
}

impl QuantizedModel {
    pub fn new(weights: &[f32]) -> Self {
        let n = weights.len();
        let weights_int4 = quantize_int4_array(weights);
        let mut scale = 0.0f32;
        let mut min_val = weights[0];
        for w in weights {
            scale += w * w;
            if w < min_val { min_val = *w; }
        }
        Self { weights_int4, scale: (scale / n as f32) * 0.5, min_val, n }
    }
    
    pub fn dequantize(&self) -> Vec<f32> {
        dequantize_int4_array(&self.weights_int4, self.n)
    }
    
    pub fn memory_footprint(&self) -> usize {
        self.weights_int4.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_quantize_dequantize_int4() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let model = QuantizedModel::new(&weights);
        let dequant = model.dequantize();
        assert_eq!(dequant.len(), weights.len());
        for (i, w) in weights.iter().enumerate() {
            assert!((dequant[i] - w).abs() < 0.5);
        }
    }
    #[test]
    fn test_memory_reduction() {
        let weights = vec![1.0f32; 1000];
        let model = QuantizedModel::new(&weights);
        assert!(model.memory_footprint() < 1000 * 4 / 4);
    }
}
