use crate::utils::*;
use std::fs::File;
use std::io::{self, Read};

pub struct Tokenizer {
    pub vocab_size: u32,
    pub vacab_table: Vec<String>,
    pub init_ok: bool,
    pub eot_token: u32, // <|endoftext|> token id
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            vocab_size: 0,
            vacab_table: Vec::new(),
            init_ok: false,
            eot_token: 50256,
        }
    }

    pub fn safe_print(piece: &str) {
        if piece
            .chars()
            .all(|c| c.is_ascii() && (c.is_ascii_graphic() || c.is_whitespace()))
        {
            print!("{}", piece);
        }
    }

    pub fn init(&mut self, path: &str) -> io::Result<()>{
        let mut file = File::open(path).expect("Failed to open file!");
        let mut model_header = vec![0; 256];
        file.read_into::<u32>(&mut model_header).unwrap();
        assert_eq!(model_header[0], 20240328);

        self.vocab_size = model_header[2];
        let version = model_header[1];

        match version {
            1 => {
                assert_eq!(self.vocab_size, 50257);
                self.eot_token = 50256;
            },
            2 => self.eot_token = model_header[3],
            _ => panic!("Unsupported model version: {}!", version),
        }

        self.vacab_table = Vec::with_capacity(self.vocab_size as usize);
        for i in 0..self.vocab_size {
            let mut length = [0; 1];
            file.read_into::<u8>(&mut length)?;
            let length = length[0] as usize;
            assert!(length > 0);
            let mut token_bytes = vec![0; length];
            file.read_exact(&mut token_bytes)?;
            let token_str = String::from_utf8_lossy(&token_bytes).into_owned();
            self.vacab_table.push(token_str);
        }

        self.init_ok = true;
        Ok(())
    }

    pub fn decode(&self, token_id: u32) -> Option<&str> {
        if !self.init_ok {
            return None;
        }

        if token_id < self.vocab_size {
            Some(&self.vacab_table[token_id as usize])
        } else {
            eprintln!("Invalid token id: {}!", token_id);
            None
        }
    }
}

