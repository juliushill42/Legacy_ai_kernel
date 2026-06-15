pub mod paged_kv;

pub struct PagedAllocator {
    page_size: usize,
    pages: Vec<Vec<u8>>,
    free_pages: Vec<usize>,
}

impl PagedAllocator {
    pub fn new(page_size: usize, num_pages: usize) -> Self {
        let pages = vec![vec![0u8; page_size]; num_pages];
        Self { page_size, pages, free_pages: (0..num_pages).collect() }
    }
    
    pub fn allocate(&mut self, size: usize) -> Result<*mut u8, std::alloc::AllocError> {
        let needed = (size / self.page_size) + 1;
        if needed > self.free_pages.len() { return Err(std::alloc::AllocError); }
        let mut ptr = None;
        for _ in 0..needed {
            let idx = self.free_pages.pop().ok_or(std::alloc::AllocError)?;
            if ptr.is_none() { ptr = Some(self.pages[idx].as_mut_ptr()); }
        }
        Ok(ptr.unwrap())
    }
    
    pub fn memory_usage(&self) -> usize {
        (self.pages.len() - self.free_pages.len()) * self.page_size
    }
}

pub struct KVCache {
    allocator: PagedAllocator,
    pub max_seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    cache_size: usize,
}

impl KVCache {
    pub fn new(max_seq_len: usize, num_heads: usize, head_dim: usize, vram_size: usize) -> Self {
        let page_size = 4096;
        let num_pages = vram_size / page_size;
        let allocator = PagedAllocator::new(page_size, num_pages);
        let cache_size = max_seq_len * num_heads * head_dim * 2;
        Self { allocator, max_seq_len, num_heads, head_dim, cache_size }
    }
    
    pub fn memory_footprint(&self) -> usize { self.cache_size }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_kv_cache() {
        let cache = KVCache::new(1024, 32, 128, 8 * 1024 * 1024);
        assert_eq!(cache.memory_footprint(), 1024 * 32 * 128 * 2);
    }
}
