use legacy_ai_kernel::{QuantizedModel, KVCache, CPUMatMul};

#[test]
fn test_full_quantize() {
    let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let m = QuantizedModel::new(&w);
    assert!(m.memory_footprint() < 4);
    let dec = m.dequantize();
    assert_eq!(dec.len(), 8);
}

#[test]
fn test_kv_cache() {
    let cache = KVCache::new(1024, 32, 128, 8 * 1024 * 1024);
    assert_eq!(cache.memory_footprint(), 1024 * 32 * 128 * 2);
}

#[test]
fn test_matmul() {
    let mat = CPUMatMul::new(4, 4, 4);
    let a = vec![1.0f32; 16];
    let b = vec![2.0f32; 16];
    assert_eq!(mat.execute_serial(&a, &b), mat.execute_parallel(&a, &b));
}
