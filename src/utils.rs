use std::fs::File;
use std::io::Read;
use std::io::Result;

pub trait ReadBytes {
    fn read_into<T: Copy>(&mut self, dst: &mut [T]) -> Result<()>;
}

impl ReadBytes for File {
    fn read_into<T: Copy>(&mut self, dst: &mut [T]) -> Result<()> {
        let len = std::mem::size_of::<T>() * dst.len();
        {
            let mut buf =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, len) };
            self.read_exact(&mut buf)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    #[test]
    fn test_fopen_check() {
        let path = "/Users/long/projects/open-source-projects/llm.c/gpt2_124M.bin";
        let file = File::open(path).expect("Failed to open file!");
        let metadata = file.metadata().expect("Failed to get file metadata!");
        println!("{}", metadata.size());
        assert_ne!(metadata.size(), 0, "File is Empty!");
    }

    #[test]
    fn test_fread_check() {
        let path = "/Users/long/projects/open-source-projects/llm.c/gpt2_124M.bin";
        let mut file = File::open(path).expect("Failed to open file!");
        let mut model_header = vec![0; 256];
        file.read_into::<i32>(&mut model_header).unwrap();
        println!("{:?}", model_header);
        assert_eq!(model_header[0], 20240326);
        assert_eq!(model_header[1], 3);
        assert_eq!(model_header[4], 12);
    }
}
