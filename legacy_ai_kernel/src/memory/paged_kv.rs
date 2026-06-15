use std::sync::atomic::{AtomicUsize, Ordering};

pub struct PagedKVPage {
    pub data: Vec<f32>,
    pub page_id: usize,
    pub is_evicted: AtomicUsize,
}

impl PagedKVPage {
    pub fn new(page_id: usize, size: usize) -> Self {
        Self { data: vec![0.0f32; size], page_id, is_evicted: AtomicUsize::new(0) }
    }
}

pub struct PagedKVCache {
    pages: Vec<PagedKVPage>,
    page_table: Vec<usize>,
    next_id: AtomicUsize,
}

impl PagedKVCache {
    pub fn new(num_heads: usize, head_dim: usize, max_seq: usize) -> Self {
        Self { pages: Vec::new(), page_table: vec![0; max_seq], next_id: AtomicUsize::new(0) }
    }
    
    pub fn allocate_page(&mut self, pos: usize) -> *mut f32 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let page = PagedKVPage::new(id, 4096);
        self.pages.push(page);
        self.page_table[pos] = id;
        self.pages.last_mut().unwrap().data.as_mut_ptr()
    }
    
    pub fn get_page(&self, pos: usize) -> Option<&PagedKVPage> {
        let id = self.page_table[pos];
        self.pages.iter().find(|p| p.page_id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_page_alloc() {
        let mut cache = PagedKVCache::new(32, 128, 1024);
        let ptr = cache.allocate_page(0);
        assert!(ptr != std::ptr::null_mut());
    }
}
