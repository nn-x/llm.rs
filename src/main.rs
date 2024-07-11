#![allow(non_snake_case)]

mod dataloader;
mod transformers;
mod utils;
mod tokenizer;

use dataloader::DataLoader;
use std::io::{Result, Write};
use transformers::gpt2::GPT2;
use tokenizer::Tokenizer;

const TRAIN_DATA: &str = "data/tiny_shakespeare_train.bin";
const VALID_DATA: &str = "data/tiny_shakespeare_val.bin";
const B: usize = 4;
const T: usize = 64;
const GEN_T: usize = 64;

fn main() -> Result<()> {
    let mut model = GPT2::from_checkpoint("checkpoints/gpt2_124M.bin");
    let mut train_loader = DataLoader::new(TRAIN_DATA, B, T);
    let mut valid_loader = DataLoader::new(VALID_DATA, B, T);

    let mut tokenizer = Tokenizer::new();
    tokenizer.init("gpt2_tokenizer.bin")?;

    let mut rng_state = 1337u64;

    for step in 0..=40 {
        if step % 10 == 0 {
            let mut val_loss = 0_f32;
            for _ in 0..5 {
                let (inputs, targets) = valid_loader.next_batch();
                model.forward(inputs, Some(targets), B, T);
                val_loss += model.mean_loss().unwrap();
            }
            val_loss /= 5 as f32;
            println!("Validation loss: {}", val_loss);
        }

        if step > 0 && step % 20 == 0 {
            let mut gen_tokens = vec![tokenizer.eot_token as i32; B * T];
            for t in 1..GEN_T {
                model.forward(&gen_tokens, None, B, T);
                if let Some(acts) = &model.acts {
                    let probs = &acts.probs[(t-1)*model.config.padded_vocab_size..];
                    let coin = random_f32(&mut rng_state);
                    let next_token = sample_mult(probs, model.config.vocab_size as i32, coin);
                    gen_tokens[t] = next_token;
                    if tokenizer.init_ok {
                        if let Some(token_str) = tokenizer.decode(next_token as u32) {
                            Tokenizer::safe_print(token_str);
                        } else {
                            print!("<unk>");
                        }

                    } else {
                        print!("<{}>", next_token);
                    }
                    std::io::stdout().flush()?;
                }
            }
            println!("\n---");

        }

        let start = std::time::Instant::now();
        let (inputs, targets) = train_loader.next_batch();
        model.forward(inputs, Some(targets), B, T);
        model.zero_grad();
        model.backward();
        model.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step + 1);
        let elapsed = start.elapsed();
        println!(
            "step {}: train loss {} (took {} ms)",
            step,
            model.mean_loss().unwrap(),
            elapsed.as_millis()
        );
    }

    Ok(())
}

fn random_u32(state: &mut u64) -> u32 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ((*state).wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32
}

fn random_f32(state: &mut u64) -> f32 {
    (random_u32(state) >> 8) as f32 / 16777216.0
} 

fn sample_mult(probs: &[f32], n: i32, coin: f32) -> i32 {
    let mut cdf =0.0;
    for i in 0..n {
        cdf += probs[i as usize];
        if cdf > coin {
            return i;
        }
    }
    n-1
}

