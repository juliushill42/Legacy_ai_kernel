pub struct INT4Encoder { scale: f32, min_val: f32 }

impl INT4Encoder {
    pub fn new(weights: &[f32]) -> Self {
        let mut min_val = weights[0];
        let mut max_val = weights[0];
        for w in weights {
            if w < min_val { min_val = *w; }
            if w > max_val { max_val = *w; }
        }
        Self { scale: (max_val - min_val) / 7.0, min_val }
    }
    
    pub fn encode(&self, weights: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(weights.len() / 8);
        for chunk in weights.chunks(8) {
            let mut packed = 0u8;
            for (i, w) in chunk.iter().enumerate() {
                let q = ((w - self.min_val) / self.scale).round() as i8;
                packed |= (((q.clamp(-8, 7) & 0xF) << (i * 4)) as u8);
            }
            out.push(packed);
        }
        out
    }
    
    pub fn decode(&self, encoded: &[u8]) -> Vec<f32> {
        let mut dec = Vec::with_capacity(encoded.len() * 8);
        for p in encoded {
            for i in 0..8 {
                let bits = ((p >> (i * 4)) & 0xF) as i8;
                dec.push(bits * self.scale + self.min_val);
            }
        }
        dec
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_roundtrip() {
        let w = vec![-8.0, 0.0, 8.0, 16.0];
        let enc = INT4Encoder::new(&w);
        let dec = enc.decode(&enc.encode(&w));
        assert_eq!(dec.len(), 4);
    }
}
