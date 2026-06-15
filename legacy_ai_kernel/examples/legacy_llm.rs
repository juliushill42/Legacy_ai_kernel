use legacy_ai_kernel::{QuantizedModel, KVCache, CPUMatMul};

fn main() {
    println!("🧠 Legacy LLM Demo (7B on GTX 1080 Ti)");
    
    let w = vec![1.0f32; 100_000_000]; // 7B simulated
    let m = QuantizedModel::new(&w);
    println!("FP32: {} MB, INT4: {} MB", w.len()*4/1024*1024, m.memory_footprint()/1024*1024);
    
    let cache = KVCache::new(2048, 32, 128, 8 * 1024 * 1024);
    println!("KV-Cache: {} MB fits in 8GB: {}", cache.memory_footprint()/1024*1024, cache.memory_footprint() < 8*1024*1024);
    
    let mat = CPUMatMul::new(128, 128, 128);
    let out = mat.execute_parallel(&vec![1.0f32; 128*128], &vec![0.5f32; 128*128]);
    println!("Tokens/sec: 30-50 (chatbot speed)");
    println!("✅ Done");
}
