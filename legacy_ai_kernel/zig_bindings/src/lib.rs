extern "C" {
    pub fn quantize_int4(input: *const f32, output: *mut u8, n: usize);
    pub fn dequantize_int4(input: *const u8, output: *mut f32, n: usize);
    pub fn matmul_legacy(a: *const f32, b: *const f32, c: *mut f32, m: usize, k: usize, n: usize);
}

pub fn quantize_int4_array(input: &[f32]) -> Vec<u8> {
    let n = input.len();
    let mut output = vec![0u8; n / 8 + 1];
    quantize_int4(input.as_ptr(), output.as_mut_ptr(), n);
    output
}

pub fn dequantize_int4_array(input: &[u8], n: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; n];
    dequantize_int4(input.as_ptr(), output.as_mut_ptr(), n);
    output
}

pub fn matmul_legacy_array(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    matmul_legacy(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, k, n);
    c
}
