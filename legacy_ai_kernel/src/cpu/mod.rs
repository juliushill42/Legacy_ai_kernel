pub mod matmul;

use rayon::prelude::*;

pub struct CPUMatMul { m: usize, k: usize, n: usize }

impl CPUMatMul {
    pub fn new(m: usize, k: usize, n: usize) -> Self { Self { m, k, n } }
    
    pub fn execute_serial(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; self.m * self.n];
        for i in 0..self.m {
            for j in 0..self.n {
                let mut sum = 0.0f32;
                for l in 0..self.k { sum += a[i * self.k + l] * b[l * self.n + j]; }
                c[i * self.n + j] = sum;
            }
        }
        c
    }
    
    pub fn execute_parallel(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; self.m * self.n];
        c.par_iter_mut().enumerate().for_each(|(i_j, val)| {
            let i = i_j / self.n;
            let j = i_j % self.n;
            let mut sum = 0.0f32;
            for l in 0..self.k { sum += a[i * self.k + l] * b[l * self.n + j]; }
            *val = sum;
        });
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_matmul() {
        let m = CPUMatMul::new(2, 2, 2);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        assert_eq!(m.execute_serial(&a, &b), vec![19.0, 22.0, 43.0, 50.0]);
    }
}
