use std::io::Write;
use std::fs::File;

pub struct GGUFHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
}

impl GGUFHeader {
    pub const MAGIC: u32 = 0x67677566;
}

pub struct GGUFWriter { file: File }

impl GGUFWriter {
    pub fn new(path: &str) -> std::io::Result<Self> {
        Ok(Self { file: File::create(path)? })
    }
    
    pub fn write_header(&mut self, tensor_count: u64, metadata_count: u64) -> std::io::Result<()> {
        let h = GGUFHeader { magic: GGUFHeader::MAGIC, version: 2, tensor_count, metadata_count };
        self.file.write_all(&h.magic.to_le_bytes())?;
        self.file.write_all(&h.version.to_le_bytes())?;
        self.file.write_all(&h.tensor_count.to_le_bytes())?;
        self.file.write_all(&h.metadata_count.to_le_bytes())?;
        Ok(())
    }
    
    pub fn write_tensor_int4(&mut self, name: &str, data: &[u8], shape: &[u64]) -> std::io::Result<()> {
        let nb = name.as_bytes();
        self.file.write_all(&(nb.len() as u64).to_le_bytes())?;
        self.file.write_all(nb)?;
        self.file.write_all(&(shape.len() as u64).to_le_bytes())?;
        for d in shape { self.file.write_all(&d.to_le_bytes())?; }
        self.file.write_all(&20u32.to_le_bytes())?;
        self.file.write_all(&(data.len() as u64).to_le_bytes())?;
        self.file.write_all(data)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_magic() { assert_eq!(GGUFHeader::MAGIC, 0x67677566); }
}
