use crate::gpu::GPUBatch;
use crossbeam::queue::SegQueue;

pub struct BatchedKernel {
    pub batches: Vec<GPUBatch>,
    pub max_batch_norm: usize,
}

impl BatchedKernel {
    pub fn new(max_batch_norm: usize) -> Self {
        Self { batches: Vec::new(), max_batch_norm }
    }
    
    pub fn add_batch(&mut self, batch: GPUBatch) -> bool {
        let current = self.batches.iter().map(|b| b.m * b.k * b.n).sum::<usize>();
        if current + batch.m * batch.k * batch.n > self.max_batch_norm { return false; }
        self.batches.push(batch);
        true
    }
}

pub struct KernelBatcher {
    queue: SegQueue<BatchedKernel>,
    current: BatchedKernel,
}

impl KernelBatcher {
    pub fn new(max_norm: usize) -> Self {
        Self { queue: SegQueue::new(), current: BatchedKernel::new(max_norm) }
    }
    
    pub fn enqueue(&mut self, batch: GPUBatch) {
        if !self.current.add_batch(batch) {
            self.queue.push(self.current);
            self.current = BatchedKernel::new(self.current.max_batch_norm);
            self.current.add_batch(batch);
        }
    }
    
    pub fn flush(&mut self) -> Option<BatchedKernel> {
        if self.current.batches.is_empty() { return None; }
        self.queue.push(self.current);
        self.current = BatchedKernel::new(self.current.max_batch_norm);
        self.queue.pop()
    }
}
