pub mod batching;

use crossbeam::queue::SegQueue;

pub struct GPUBatch {
    pub kernel_id: u32,
    pub input_ptrs: Vec<*const f32>,
    pub output_ptrs: Vec<*mut f32>,
    pub m: usize, k: usize, n: usize,
}

pub struct GPUBatcher {
    queue: SegQueue<GPUBatch>,
    max_batch_size: usize,
}

impl GPUBatcher {
    pub fn new(max_batch_size: usize) -> Self {
        Self { queue: SegQueue::new(), max_batch_size }
    }
    
    pub fn enqueue(&mut self, batch: GPUBatch) { self.queue.push(batch); }
    
    pub fn flush(&mut self) -> Vec<GPUBatch> {
        let mut batches = Vec::new();
        while batches.len() < self.max_batch_size {
            if let Some(b) = self.queue.pop() { batches.push(b); } else { break; }
        }
        batches
    }
    
    pub fn batch_size(&self) -> usize { self.queue.len() }
}

pub struct GPUKernelLauncher {
    device_id: usize,
    batcher: GPUBatcher,
}

impl GPUKernelLauncher {
    pub fn new(device_id: usize, max_batch_size: usize) -> Self {
        Self { device_id, batcher: GPUBatcher::new(max_batch_size) }
    }
    
    pub fn launch_matmul(&mut self, a: *const f32, b: *const f32, c: *mut f32, m: usize, k: usize, n: usize) {
        self.batcher.enqueue(GPUBatch { kernel_id: 0, input_ptrs: vec![a, b], output_ptrs: vec![c], m, k, n });
        if self.batcher.batch_size() >= self.batcher.max_batch_size {
            for _batch in self.batcher.flush() { println!("Launching GPU kernel"); }
        }
    }
}
