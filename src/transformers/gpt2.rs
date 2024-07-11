use super::*;
use crate::utils::ReadBytes;
use std::fs::File;
use std::path::Path;

#[derive(Copy, Clone)]
pub struct GPT2Config {
    pub max_seq_len: usize,       // max sequence length, e.g. 1024
    pub vocab_size: usize,        // vocab size, e.g. 50257
    pub padded_vocab_size: usize, // padded to e.g. %128==0, 50304
    pub num_layers: usize,        // number of layers, e.g. 12
    pub num_heads: usize,         // number of heads in attention, e.g. 12
    pub channels: usize,          // number of channels, e.g. 768
}

macro_rules! zeros {
    ($field: ident, $shape: expr) => {
        vec![0.0; $shape]
    };
}

#[derive(Default)]
struct ParameterTensors {
    wte: Vec<f32>,
    wpe: Vec<f32>,
    ln1w: Vec<f32>,
    ln1b: Vec<f32>,
    qkvw: Vec<f32>,
    qkvb: Vec<f32>,
    attprojw: Vec<f32>,
    attprojb: Vec<f32>,
    ln2w: Vec<f32>,
    ln2b: Vec<f32>,
    fcw: Vec<f32>,
    fcb: Vec<f32>,
    fcprojw: Vec<f32>,
    fcprojb: Vec<f32>,
    lnfw: Vec<f32>,
    lnfb: Vec<f32>,
}

impl ParameterTensors {
    fn zeros(config: &GPT2Config) -> Self {
        let l = config.num_layers;
        let c = config.channels;
        let max_t = config.max_seq_len;
        let vp = config.padded_vocab_size;

        Self {
            wte: zeros!(wte, vp * c),
            wpe: zeros!(wpe, max_t * c),
            ln1w: zeros!(ln1w, l * c),
            ln1b: zeros!(ln1b, l * c),
            qkvw: zeros!(qkvw, l * (3 * c) * c),
            qkvb: zeros!(qkvb, l * (3 * c)),
            attprojw: zeros!(attprojw, l * c * c),
            attprojb: zeros!(attprojb, l * c),
            ln2w: zeros!(ln2w, l * c),
            ln2b: zeros!(ln2b, l * c),
            fcw: zeros!(fcw, l * (4 * c) * c),
            fcb: zeros!(fcb, l * (4 * c)),
            fcprojw: zeros!(fcprojw, l * c * (4 * c)),
            fcprojb: zeros!(fcprojb, l * c),
            lnfw: zeros!(lnfw, c),
            lnfb: zeros!(lnfb, c),
        }
    }

    fn reset(&mut self) {
        macro_rules! reset {
            ($field: ident) => {
                for e in &mut self.$field {
                    *e = 0.0;
                }
            };
        } 
        reset!(wte);
        reset!(wpe);
        reset!(ln1w);
        reset!(ln1b);
        reset!(qkvw);
        reset!(qkvb);
        reset!(attprojw);
        reset!(attprojb);
        reset!(ln2w);
        reset!(ln2b);
        reset!(fcw);
        reset!(fcb);
        reset!(fcprojw);
        reset!(fcprojb);
        reset!(lnfw);
        reset!(lnfb);
    }

    fn num_parameters(&self) -> usize {
        self.wte.len()
            + self.wpe.len()
            + self.ln1w.len()
            + self.ln1b.len()
            + self.qkvw.len()
            + self.qkvb.len()
            + self.attprojw.len()
            + self.attprojb.len()
            + self.ln2w.len()
            + self.ln2b.len()
            + self.fcw.len()
            + self.fcb.len()
            + self.fcprojw.len()
            + self.fcprojb.len()
            + self.lnfw.len()
            + self.lnfb.len()
    }

    fn iter(&self) -> impl Iterator<Item = &f32> {
        self.wte
            .iter()
            .chain(self.wpe.iter())
            .chain(self.ln1w.iter())
            .chain(self.ln1b.iter())
            .chain(self.qkvw.iter())
            .chain(self.qkvb.iter())
            .chain(self.attprojw.iter())
            .chain(self.attprojb.iter())
            .chain(self.ln2w.iter())
            .chain(self.ln2b.iter())
            .chain(self.fcw.iter())
            .chain(self.fcb.iter())
            .chain(self.fcprojw.iter())
            .chain(self.fcprojb.iter())
            .chain(self.lnfw.iter())
            .chain(self.lnfb.iter())
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.wte
            .iter_mut()
            .chain(self.wpe.iter_mut())
            .chain(self.ln1w.iter_mut())
            .chain(self.ln1b.iter_mut())
            .chain(self.qkvw.iter_mut())
            .chain(self.qkvb.iter_mut())
            .chain(self.attprojw.iter_mut())
            .chain(self.attprojb.iter_mut())
            .chain(self.ln2w.iter_mut())
            .chain(self.ln2b.iter_mut())
            .chain(self.fcw.iter_mut())
            .chain(self.fcb.iter_mut())
            .chain(self.fcprojw.iter_mut())
            .chain(self.fcprojb.iter_mut())
            .chain(self.lnfw.iter_mut())
            .chain(self.lnfb.iter_mut())
    }

    fn from_buffer(config: &GPT2Config, buffer: &mut File) -> Self {
        macro_rules! read_tensors {
            ($field: expr) => {
                buffer.read_into::<f32>(&mut $field).unwrap();
            };
        }
        let mut params = Self::zeros(config);
        read_tensors!(params.wte);
        read_tensors!(params.wpe);
        read_tensors!(params.ln1w);
        read_tensors!(params.ln1b);
        read_tensors!(params.qkvw);
        read_tensors!(params.qkvb);
        read_tensors!(params.attprojw);
        read_tensors!(params.attprojb);
        read_tensors!(params.ln2w);
        read_tensors!(params.ln2b);
        read_tensors!(params.fcw);
        read_tensors!(params.fcb);
        read_tensors!(params.fcprojw);
        read_tensors!(params.fcprojb);
        read_tensors!(params.lnfw);
        read_tensors!(params.lnfb);
        params
    }
}

#[derive(Default)]
pub struct ActivationTensors {
    encoded: Vec<f32>,
    ln1: Vec<f32>,
    ln1mean: Vec<f32>,
    ln1rstd: Vec<f32>,
    qkv: Vec<f32>,
    atty: Vec<f32>,
    preatt: Vec<f32>,
    att: Vec<f32>,
    attproj: Vec<f32>,
    residual2: Vec<f32>,
    ln2: Vec<f32>,
    ln2mean: Vec<f32>,
    ln2rstd: Vec<f32>,
    fch: Vec<f32>,
    fchgelu: Vec<f32>,
    fchproj: Vec<f32>,
    residual3: Vec<f32>,
    lnf: Vec<f32>,
    lnfmean: Vec<f32>,
    lnfrstd: Vec<f32>,
    logits: Vec<f32>,
    pub probs: Vec<f32>,
    losses: Vec<f32>,
}

impl ActivationTensors {
    fn zeros(config: &GPT2Config, batch_size: usize, seq_len: usize) -> Self {
        let b = batch_size;
        let t = seq_len;
        let c = config.channels;
        let l = config.num_layers;
        let vp = config.padded_vocab_size;
        let nh = config.num_heads;

        Self {
            encoded: zeros!(encoded, b * t * c),
            ln1: zeros!(ln1, l * b * t * c),
            ln1mean: zeros!(ln1mean, l * b * t),
            ln1rstd: zeros!(ln1rstd, l * b * t),
            qkv: zeros!(qkv, l * b * t * (3 * c)),
            atty: zeros!(atty, l * b * t * c),
            preatt: zeros!(preatt, l * b * nh * t * t),
            att: zeros!(att, l * b * nh * t * t),
            attproj: zeros!(attproj, l * b * t * c),
            residual2: zeros!(residual2, l * b * t * c),
            ln2: zeros!(ln2, l * b * t * c),
            ln2mean: zeros!(ln2mean, l * b * t),
            ln2rstd: zeros!(ln2rstd, l * b * t),
            fch: zeros!(fch, l * b * t * (4 * c)),
            fchgelu: zeros!(fchgelu, l * b * t * (4 * c)),
            fchproj: zeros!(fchproj, l * b * t * c),
            residual3: zeros!(residual3, l * b * t * c),
            lnf: zeros!(lnf, b * t * c),
            lnfmean: zeros!(lnfmean, b * t),
            lnfrstd: zeros!(lnfrstd, b * t),
            logits: zeros!(logits, b * t * vp),
            probs: zeros!(probs, b * t * vp),
            losses: zeros!(losses, b * t),
        }
    }

    fn reset(&mut self) {
        macro_rules! reset {
            ($field: ident) => {
                for e in &mut self.$field {
                    *e = 0.0;
                }
            };
        } 
        reset!(encoded);
        reset!(ln1);
        reset!(ln1mean);
        reset!(ln1rstd);
        reset!(qkv);
        reset!(atty);
        reset!(preatt);
        reset!(att);
        reset!(attproj);
        reset!(residual2);
        reset!(ln2);
        reset!(ln2mean);
        reset!(ln2rstd);
        reset!(fch);
        reset!(fchgelu);
        reset!(fchproj);
        reset!(residual3);
        reset!(lnf);
        reset!(lnfmean);
        reset!(lnfrstd);
        reset!(logits);
        reset!(probs);
        reset!(losses);
    }

    fn num_parameters(&self) -> usize {
        self.encoded.len()
            + self.ln1.len()
            + self.ln1mean.len()
            + self.ln1rstd.len()
            + self.qkv.len()
            + self.atty.len()
            + self.preatt.len()
            + self.att.len()
            + self.attproj.len()
            + self.residual2.len()
            + self.ln2.len()
            + self.ln2mean.len()
            + self.ln2rstd.len()
            + self.fch.len()
            + self.fchgelu.len()
            + self.fchproj.len()
            + self.residual3.len()
            + self.lnf.len()
            + self.lnfmean.len()
            + self.lnfrstd.len()
            + self.logits.len()
            + self.probs.len()
            + self.losses.len()
    }
}

pub struct GPT2 {
    pub config: GPT2Config,
    // the parameters of the model
    params: ParameterTensors,
    // the gradients of the parameters
    grads: Option<ParameterTensors>,
    // the activations of forward pass
    pub acts: Option<ActivationTensors>,
    // the gradients of activations
    grads_acts: Option<ActivationTensors>,
    // AdamW m memory
    m_memory: Option<ParameterTensors>,
    // AdamW v memory
    v_memory: Option<ParameterTensors>,
    // the input tokens for the current forward pass
    inputs: Vec<i32>,
    //the target tokens for the current forward pass
    targets: Option<Vec<i32>>,
    // the batch size (B) of current forward pass
    batch_size: usize,
    // the sequence length (T) of current forward pass
    seq_len: usize,
    // after a forward pass with targets, will be populated with the mean loss
    mean_loss: Option<f32>,
}

impl GPT2 {
    pub fn from_checkpoint<P: AsRef<Path>>(path: P) -> Self {
        let mut file = File::open(path).expect("Failed to open the file!");

        let mut model_header = vec![0; 256];
        file.read_into::<i32>(&mut model_header)
            .expect("Failed to read the file!");

        assert_eq!(model_header[0], 20240326, "Bad magic model file!");
        assert_eq!(
            model_header[1], 3,
            "Bad version in model file!\n try to re-run `python train_gpt2.py\n"
        );

        let config = GPT2Config {
            max_seq_len: model_header[2] as usize,
            vocab_size: model_header[3] as usize,
            num_layers: model_header[4] as usize,
            num_heads: model_header[5] as usize,
            channels: model_header[6] as usize,
            padded_vocab_size: model_header[7] as usize,
        };
        println!("[GPT-2]");
        println!("max_seq_len: {}", config.max_seq_len);
        println!("vocab_size: {}", config.vocab_size);
        println!("padded_vocab_size: {}", config.padded_vocab_size);
        println!("num_layers: {}", config.num_layers);
        println!("num_heads: {}", config.num_heads);
        println!("channels: {}", config.channels);

        let params = ParameterTensors::from_buffer(&config, &mut file);

        Self {
            config: config,
            params: params,
            grads: None,
            acts: None,
            grads_acts: None,
            inputs: Vec::new(),
            targets: None,
            m_memory: None,
            v_memory: None,
            batch_size: 0,
            seq_len: 0,
            mean_loss: None,
        }
    }

    pub fn forward(
        &mut self,
        inputs: &[i32],
        targets: Option<&[i32]>,
        batch_size: usize,
        seq_len: usize,
    ) {
        let v = self.config.vocab_size;
        let vp = self.config.padded_vocab_size;
        let c = self.config.channels;
        let L = self.config.num_layers;
        let nh = self.config.num_heads;
        let b = batch_size;
        let t = seq_len;

        for i in 0..b * t {
            assert!(inputs[i] >= 0 && inputs[i] < v as i32);
            if let Some(targets) = targets {
                assert!(targets[i] >= 0 && targets[i] < v as i32);
            }
        }

        if self.acts.is_some() {
            assert_eq!(b, self.batch_size, "Batch size mismatch!");
            assert_eq!(t, self.seq_len, "Sequence length mismatch!");
        }

        if self.acts.is_none() {
            self.acts = Some(ActivationTensors::zeros(&self.config, batch_size, seq_len));
            self.batch_size = b;
            self.seq_len = t;
        }

        // todo
        self.inputs = inputs.to_vec();
        if let Some(targets) = targets {
            self.targets = Some(targets.to_vec());
        }

 
        let params = &self.params;
        let acts = self.acts.as_mut().unwrap();

        encoded_forward(
            &mut acts.encoded,
            inputs,
            &params.wte,
            &params.wpe,
            b,
            t,
            c,
        );

        for l in 0..L {
            let residual = if l == 0 {
                &acts.encoded
            } else {
                &acts.residual3[(l - 1) * b * t * c..l * b * t * c]
            };

            let l_ln1 = &mut acts.ln1[l * b * t * c..(l + 1) * b * t * c];
            let l_ln1mean = &mut acts.ln1mean[l * b * t..(l + 1) * b * t];
            let l_ln1rstd = &mut acts.ln1rstd[l * b * t..(l + 1) * b * t];
            let l_ln1w = &params.ln1w[l * c..(l + 1) * c];
            let l_ln1b = &params.ln1b[l * c..(l + 1) * c];
            layernorm_forward(
                l_ln1,
                l_ln1mean,
                l_ln1rstd,
                residual,
                l_ln1w,
                l_ln1b,
                b,
                t,
                c,
            );

            let l_qkv = &mut acts.qkv[l * b * t * 3 * c..(l + 1) * b * t * 3 * c];
            let l_qkvw = &params.qkvw[l * 3 * c * c..(l + 1) * 3 * c * c];
            let l_qkvb = &params.qkvb[l * 3 * c..(l + 1) * 3 * c];
            matmul_forward(l_qkv, l_ln1, l_qkvw, Some(l_qkvb), b, t, c, 3 * c);

            let l_atty = &mut acts.atty[l * b * t * c..(l + 1) * b * t * c];
            let l_preatt = &mut acts.preatt[l * b * nh * t * t..(l + 1) * b * nh * t * t];
            let l_att = &mut acts.att[l * b * nh * t * t..(l + 1) * b * nh * t * t];
            attention_forward(l_atty, l_preatt, l_att, l_qkv, b, t, c, nh);

            let l_attproj = &mut acts.attproj[l * b * t * c..(l + 1) * b * t * c];
            let l_attprojw = &params.attprojw[l * c * c..(l + 1) * c * c];
            let l_attprojb = &params.attprojb[l * c..(l + 1) * c];
            matmul_forward(l_attproj, l_atty, l_attprojw, Some(l_attprojb), b, t, c, c);

            let l_residual2 = &mut acts.residual2[l * b * t * c..(l + 1) * b * t * c];
            residual_forward(l_residual2, residual, l_attproj, b * t * c);

            let l_ln2 = &mut acts.ln2[l * b * t * c..(l + 1) * b * t * c];
            let l_ln2mean = &mut acts.ln2mean[l * b * t..(l + 1) * b * t];
            let l_ln2rstd = &mut acts.ln2rstd[l * b * t..(l + 1) * b * t];
            let l_ln2w = &params.ln2w[l * c..(l + 1) * c];
            let l_ln2b = &params.ln2b[l * c..(l + 1) * c];
            layernorm_forward(
                l_ln2,
                l_ln2mean,
                l_ln2rstd,
                l_residual2,
                l_ln2w,
                l_ln2b,
                b,
                t,
                c,
            );

            let l_fch = &mut acts.fch[l * b * t * 4 * c..(l + 1) * b * t * 4 * c];
            let l_fcw = &params.fcw[l * 4 * c * c..(l + 1) * 4 * c * c];
            let l_fcb = &params.fcb[l * 4 * c..(l + 1) * 4 * c];
            matmul_forward(l_fch, l_ln2, l_fcw, Some(l_fcb), b, t, c, 4 * c);

            let l_fchgelu = &mut acts.fchgelu[l * b * t * 4 * c..(l + 1) * b * t * 4 * c];
            gelu_forward(l_fchgelu, l_fch, b * t * 4 * c);

            let l_fcproj = &mut acts.fchproj[l * b * t * c..(l + 1) * b * t * c];
            let l_fcprojw = &params.fcprojw[l * c * 4 * c..(l + 1) * c * 4 * c];
            let l_fcprojb = &params.fcprojb[l * c..(l + 1) * c];
            matmul_forward(
                l_fcproj,
                l_fchgelu,
                l_fcprojw,
                Some(l_fcprojb),
                b,
                t,
                4 * c,
                c,
            );

            let l_residual3 = &mut acts.residual3[l * b * t * c..(l+1) * b * t * c];
            residual_forward(l_residual3, l_residual2, l_fcproj, b * t * c);
        }

        let residual = &acts.residual3[(L - 1) * b * t * c..];
        layernorm_forward(
            &mut acts.lnf,
            &mut acts.lnfmean,
            &mut acts.lnfrstd,
            residual,
            &params.lnfw,
            &params.lnfb,
            b,
            t,
            c,
        );
        matmul_forward(&mut acts.logits, &acts.lnf, &params.wte, None, b, t, c, vp);
        softmax_forward(&mut acts.probs, &acts.logits, b, t, v, vp);

        if let Some(targets) = targets {
            crossentropy_forward(&mut acts.losses, &acts.probs, targets, b, t, vp);
            self.mean_loss = Some(acts.losses.iter().sum::<f32>() / (b * t) as f32)
        } else {
            self.mean_loss = None
        }
    }

    pub fn zero_grad(&mut self) {
        if let Some(grads) = &mut self.grads {
            grads.reset();
        }
        if let Some(grads_acts) = &mut self.grads_acts {
            grads_acts.reset();
        }
    }

    pub fn backward(&mut self) {
        assert!(self.mean_loss.is_some(), "no loss to backpropagate!");

        if self.grads.is_none() {
            self.grads = Some(ParameterTensors::zeros(&self.config));
            self.grads_acts = Some(ActivationTensors::zeros(&self.config, self.batch_size, self.seq_len));
        }

        let b = self.batch_size;
        let t = self.seq_len;
        let v = self.config.vocab_size;
        let vp = self.config.padded_vocab_size;
        let L = self.config.num_layers;
        let nh = self.config.num_heads;
        let c = self.config.channels;

        let params = &self.params;
        let acts = self.acts.as_ref().unwrap();
        let grads = self.grads.as_mut().unwrap();
        let grads_acts = self.grads_acts.as_mut().unwrap();

        let dloss_mean = 1.0 / (b * t) as f32;
        grads_acts.losses = vec![dloss_mean; b * t];

        let targets = self.targets.as_ref().unwrap();

        crossentropy_softmax_backward(
            &mut grads_acts.logits,
            &grads_acts.losses,
            &acts.probs,
            targets,
            b,
            t,
            v,
            vp,
        );
        matmul_backward(
            &mut grads_acts.lnf,
            &mut grads.wte,
            None,
            &grads_acts.logits,
            &acts.lnf,
            &params.wte,
            b,
            t,
            c,
            vp,
        );
        let residual = &acts.residual3[(L - 1) * b * t * c..];
        let dresidual = &mut grads_acts.residual3[(L - 1) * b * t * c..];
        layernorm_backward(
            dresidual,
            &mut grads.lnfw,
            &mut grads.lnfb,
            &grads_acts.lnf,
            residual,
            &params.lnfw,
            &acts.lnfmean,
            &acts.lnfrstd,
            b,
            t,
            c,
        );
        for l in (0..L).rev() {
            let dl_residual2 = &mut grads_acts.residual2[l * b * t * c..(l + 1) * b * t * c];
            let dl_fchproj = &mut grads_acts.fchproj[l * b * t * c..(l + 1) * b * t * c];
            let dl_residual3 = &mut grads_acts.residual3[l * b * t * c..(l + 1) * b * t * c];
            residual_backward(dl_residual2, dl_fchproj, dl_residual3, b * t * c);

            let dl_fchgelu = &mut grads_acts.fchgelu[l * b * t * 4 * c..(l + 1) * b * t * 4 * c];
            let dl_fcprojw = &mut grads.fcprojw[l * c * 4 * c..(l + 1) * c * 4 * c];
            let dl_fcprojb = &mut grads.fcprojb[l * c..(l + 1) * c];
            let l_fchgelu = &acts.fchgelu[l * b * t * 4 * c..(l + 1) * b * t * 4 * c];
            let l_fcprojw = &params.fcprojw[l * c * 4 * c..(l + 1) * c * 4 * c];
            matmul_backward(
                dl_fchgelu,
                dl_fcprojw,
                Some(dl_fcprojb),
                dl_fchproj,
                l_fchgelu,
                l_fcprojw,
                b,
                t,
                4 * c,
                c,
            );

            let dl_fch = &mut grads_acts.fch[l * b * t * 4 * c..(l + 1) * b * t * 4 * c];
            let l_fch = &acts.fch[l * b * t * 4 * c..(l + 1) * b * t * 4 * c];
            gelu_backward(dl_fch, dl_fchgelu, l_fch, b * t * 4 * c);
            

            let dl_ln2 = &mut grads_acts.ln2[l * b * t * c..(l + 1) * b * t * c];
            let dl_fcw = &mut grads.fcw[l * 4 * c * c..(l + 1) * 4 * c * c];
            let dl_fcb = &mut grads.fcb[l * 4 * c..(l + 1) * 4 * c];
            let l_ln2 = &acts.ln2[l * b * t * c..(l + 1) * b * t * c];
            let l_fcw = &params.fcw[l * 4 * c * c..(l + 1) * 4 * c * c];
            matmul_backward(
                dl_ln2,
                dl_fcw,
                Some(dl_fcb),
                dl_fch,
                l_ln2,
                l_fcw,
                b,
                t,
                c,
                4 * c,
            );


            let dl_ln2w = &mut grads.ln2w[l * c..(l + 1) * c];
            let dl_ln2b = &mut grads.ln2b[l * c..(l + 1) * c];
            let l_residual2 = &acts.residual2[l * b * t * c..(l + 1) * b * t * c];
            let l_ln2w = &params.ln2w[l * c..(l + 1) * c];
            let l_ln2mean = &acts.ln2mean[l * b * t..(l + 1) * b * t];
            let l_ln2rstd = &acts.ln2rstd[l * b * t..(l + 1) * b * t];
            layernorm_backward(
                dl_residual2,
                dl_ln2w,
                dl_ln2b,
                dl_ln2,
                l_residual2,
                l_ln2w,
                l_ln2mean,
                l_ln2rstd,
                b,
                t,
                c,
            );

            let dresidual = if l == 0 {
                &mut grads_acts.encoded
            } else {
                &mut grads_acts.residual3[(l - 1) * b * t * c..]
            };
            let dl_attproj = &mut grads_acts.attproj[l * b * t * c..(l + 1) * b * t * c];
            residual_backward(dresidual, dl_attproj, dl_residual2, b * t * c);
            

            let dl_atty = &mut grads_acts.atty[l * b * t * c..(l + 1) * b * t * c];
            let dl_attprojw = &mut grads.attprojw[l * c * c..(l + 1) * c * c];
            let dl_attprojb = &mut grads.attprojb[l * c..(l + 1) * c];
            let l_atty = &acts.atty[l * b * t * c..(l + 1) * b * t * c];
            let l_attprojw = &params.attprojw[l * c * c..(l + 1) * c * c];
            matmul_backward(
                dl_atty,
                dl_attprojw,
                Some(dl_attprojb),
                dl_attproj,
                l_atty,
                l_attprojw,
                b,
                t,
                c,
                c,
            );

            let dl_qkv = &mut grads_acts.qkv[l * b * t * 3 * c..(l + 1) * b * t * 3 * c];
            let dl_preatt = &mut grads_acts.preatt[l * b * nh * t * t..(l + 1) * b * nh * t * t];
            let dl_att = &mut grads_acts.att[l * b * nh * t * t..(l + 1) * b * nh * t * t];
            let l_qkv = &acts.qkv[l * b * t * 3 * c..(l + 1) * b * t * 3 * c];
            let l_att = &acts.att[l * b * nh * t * t..(l + 1) * b * nh * t * t];
            attention_backward(
                dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, b, t, c, nh,
            );


            let dl_ln1 = &mut grads_acts.ln1[l * b * t * c..(l + 1) * b * t * c];
            let dl_qkvw = &mut grads.qkvw[l * 3 * c * c..(l + 1) * 3 * c * c];
            let dl_qkvb = &mut grads.qkvb[l * 3 * c..(l + 1) * 3 * c];
            let l_ln1 = &acts.ln1[l * b * t * c..(l + 1) * b * t * c];
            let l_qkvw = &params.qkvw[l * 3 * c * c..(l + 1) * 3 * c * c];
            matmul_backward(
                dl_ln1,
                dl_qkvw,
                Some(dl_qkvb),
                dl_qkv,
                l_ln1,
                l_qkvw,
                b,
                t,
                c,
                3 * c,
            );

            let residual = if l == 0 {
                &acts.encoded
            } else {
                &acts.residual3[(l - 1) * b * t * c..l * b * t * c]
            };
            let dl_ln1w = &mut grads.ln1w[l * c..(l + 1) * c];
            let dl_ln1b = &mut grads.ln1b[l * c..(l + 1) * c];
            let l_ln1w = &params.ln1w[l * c..(l + 1) * c];
            let l_ln1mean = &acts.ln1mean[l * b * t..(l + 1) * b * t];
            let l_ln1rstd = &acts.ln1rstd[l * b * t..(l + 1) * b * t];
            layernorm_backward(
                dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1mean, l_ln1rstd, b, t,
                c,
            );
        }
        encoded_backward(
            &mut grads.wte,
            &mut grads.wpe,
            &grads_acts.encoded,
            &self.inputs,
            b,
            t,
            c,
        );
    }

    pub fn update(
        &mut self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        t: i32,
    ) {
        if self.m_memory.is_none() {
            self.m_memory = Some(ParameterTensors::zeros(&self.config));
            self.v_memory = Some(ParameterTensors::zeros(&self.config));
        }

        let params = &mut self.params;
        let grads = self.grads.as_ref().unwrap();
        let m_memory = self.m_memory.as_mut().unwrap();
        let v_memory = self.v_memory.as_mut().unwrap();

        params
            .iter_mut()
            .zip(grads.iter())
            .zip(m_memory.iter_mut())
            .zip(v_memory.iter_mut())
            .for_each(|(((param, grad), m), v)| {
                let m_new = beta1 * (*m) + (1.0 - beta1) * grad;
                let v_new = beta2 * (*v) + (1.0 - beta2) * grad * grad;
                let m_hat = m_new / (1.0 - beta1.powi(t));
                let v_hat = v_new / (1.0 - beta2.powi(t));

                *m = m_new;
                *v = v_new;
                *param -= learning_rate * (m_hat / (v_hat.sqrt() + eps) + weight_decay * (*param));
            });
    }

    pub fn mean_loss(&self) -> Option<f32> {
        self.mean_loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_get_wte() {
        let mut model = GPT2::from_checkpoint("checkpoints/gpt2_124M.bin");
        let inputs = vec![1_i32; 1024];
        let targets = None;

        model.forward(&inputs, targets, 1, 1024);
    }
}
