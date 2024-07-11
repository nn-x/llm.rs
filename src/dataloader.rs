use crate::utils::ReadBytes;
use std::fs::File;
use std::io::{Seek, SeekFrom};

pub struct DataLoader {
    B: usize,
    T: usize,
    file: File,
    cursor: usize,
    file_size: usize,
    num_batches: usize,
    batch: Vec<i32>,
}

impl DataLoader {
    pub fn new(path: &str, B: usize, T: usize) -> Self {
        let file = File::open(path).expect("Failed to open file!");
        let metadata = file.metadata().expect("Failed to get file metadata!");
        let file_size = metadata.len() as usize;
        let num_batches = file_size / (B * T * std::mem::size_of::<i32>());

        Self {
            B: B,
            T: T,
            file: file,
            cursor: 0,
            file_size: file_size,
            num_batches: num_batches,
            batch: vec![0; B * T + 1],
        }
    }

    pub fn num_batches(&self) -> usize {
        self.num_batches
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    pub fn next_batch(&mut self) -> (&[i32], &[i32]) {
        let B = self.B;
        let T = self.T;

        if self.cursor + ((B * T + 1) * std::mem::size_of::<i32>()) >= self.file_size {
            self.reset();
        }

        self.file.seek(SeekFrom::Start(self.cursor as u64)).unwrap();

        let mut buffer = vec![0_i32; B * T + 1];
        self.file.read_into::<i32>(&mut buffer).unwrap();

        self.cursor += (B * T) * std::mem::size_of::<i32>();
        self.batch = buffer;

        let inputs = &self.batch[..B * T];
        let targets = &self.batch[1..self.B * T + 1];
        (inputs, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next() {
        let path = "data/tiny_shakespeare_val.bin";
        let mut dataloader = DataLoader::new(path, 16, 16);
        println!("{}", dataloader.num_batches());
        let (inputs, targets) = dataloader.next_batch();

        println!("{:?}", inputs);
        println!("{:?}", targets);

        assert!(inputs.len() == 256)
    }
}
