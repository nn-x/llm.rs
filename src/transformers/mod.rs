use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};
use std::f32::consts::PI;

fn encoded_forward(
    out: &mut [f32],
    inp: &[i32],
    wte: &[f32],
    wpe: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    // out: (B, T, C)
    // inputs: (B, T)
    // wte: (Vp, C)
    // wpe: (T, C)
    for b in 0..B {
        for t in 0..T {
            let token_idx = inp[b * T + t] as usize;
            let wte_ix = &wte[token_idx * C..(token_idx+1) * C];
            let wpe_t = &wpe[t * C..(t+1) * C];
            let out_bt = &mut out[b * T * C + t * C..b * T * C + t * C + C];

            for c in 0..C {
                out_bt[c] = wte_ix[c] + wpe_t[c];
            }
        }
    }
}

fn encoded_backward(
    dwte: &mut [f32],
    dwpe: &mut [f32],
    dout: &[f32],
    inp: &[i32],
    B: usize,
    T: usize,
    C: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let dout_bt= &dout[b * T * C + t * C..b * T * C + t * C + C];
            let token_idx = inp[b * T + t] as usize;
            let dwte_ix = &mut dwte[token_idx * C..(token_idx+1) * C];
            let dwpe_t = &mut dwpe[t * C..(t+1) * C];

            for c in 0..C {
                let d = dout_bt[c];
                dwte_ix[c] += d;
                dwpe_t[c] += d;
            }
        }
    }
}

fn layernorm_forward(
    out: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    // out: (B, T, C)
    // inputs: (B, T, C)
    // mean: (B, T)
    // rstd: (B, T)
    // weight: (C,)
    // bias: (C,)
    let eps: f32 = 1e-5;
    for b in 0..B {
        for t in 0..T {
            let inp_bt = &inp[b * T * C + t * C..b * T * C + t * C + C];
            let mut m = 0_f32;
            for c in 0..C {
                m += inp_bt[c];
            }
            m /= C as f32;
            let mut v = 0_f32;
            for c in 0..C {
                let diff = inp_bt[c] - m;
                v += diff * diff;
            }
            v /= C as f32;
            let s = 1_f32 / (v + eps).sqrt();

            let out_bt = &mut out[b * T * C + t * C..b * T * C + t * C + C];
            for c in 0..C {
                let n = (inp_bt[c] - m) * s; // normalize
                out_bt[c] = n * weight[c] + bias[c]; // scale and shift
            }
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

fn layernorm_backward(
    dinps: &mut [f32],
    dweight: &mut [f32],
    dbias: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    mean: &[f32],
    rstd: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    // dinps: (B, T, C)
    // dweight: (C,)
    // dbias: (C,)
    // dout: (B, T, C)
    // inputs: (B, T, C)
    // mean: (B, T)
    // rstd: (B, T)
    for b in 0..B {
        for t in 0..T {
            let dinp_bt = &mut dinps[b * T * C + t * C..b * T * C + t * C + C];
            let dout_bt = &dout[b * T * C + t * C..b * T * C + t * C + C];
            let inp_bt = &inp[b * T * C + t * C..b * T * C + t * C + C];
            let mean_bt = mean[b * T + t];
            let rstd_bt = rstd[b * T + t];

            let mut dnorm_mean = 0.0;
            let mut dnorm_norm_mean = 0.0;
            for c in 0..C {
                let norm_btc = (inp_bt[c] - mean_bt) * rstd_bt;
                let dnorm_c = weight[c] * dout_bt[c];
                dnorm_mean += dnorm_c;
                dnorm_norm_mean += norm_btc * dnorm_c;
            }
            dnorm_mean /= C as f32;
            dnorm_norm_mean /= C as f32;
            for c in 0..C {
                let norm_btc = (inp_bt[c] - mean_bt) * rstd_bt;
                let dnorm_c = weight[c] * dout_bt[c];
                dbias[c] += dout_bt[c];
                dweight[c] += norm_btc * dout_bt[c];
                let mut dval = 0.0;
                dval += dnorm_c;
                dval -= dnorm_mean;
                dval -= norm_btc * dnorm_norm_mean;
                dval *= rstd_bt;
                dinp_bt[c] += dval;
            }
        }
    }
}

fn matmul_forward(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    // out: (B, T, OC)
    // inp: (B, T, C)
    // weight: (OC, C)
    // bias: (OC,)

    out.par_chunks_mut(OC)
        .zip(inp.par_chunks(C))
        .take(B * T)
        .for_each(|(out_bt, inp_bt)| {
            out_bt.iter_mut().enumerate().for_each(|(o, val)| {
                let mut sum = if let Some(bias) = bias {
                    bias[o]
                } else {
                    0.0
                };
                let wrow = &weight[o * C..(o + 1) * C];
                for c in 0..C {
                    sum += inp_bt[c] * wrow[c];
                }
                *val = sum;
            })
        })
}

fn matmul_backward(
    dinps: &mut [f32],
    dweight: &mut [f32],
    dbias: Option<&mut [f32]>,
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    // backward into inp first
    dinps
        .par_chunks_mut(C)
        .zip(dout.par_chunks(OC))
        .take(B * T)
        .for_each(|(dinp_bt, dout_bt)| {
            dout_bt.iter().enumerate().for_each(|(o, d)| {
                let wrow = &weight[o * C..(o + 1) * C];
                for c in 0..C {
                    dinp_bt[c] += d * wrow[c];
                }
            })
        });

    // backward into weight and bias
    let update_chunk = |dwrow: &mut [f32], dbias_o: Option<&mut f32>, o: usize| {
        for b in 0..B {
            for t in 0..T {
                let d = dout[b * T * OC + t * OC + o];
                if let Some(&mut ref mut dbias_o) = dbias_o {
                    *dbias_o += d;
                }
                for c in 0..C {
                    let inp_btc = inp[b * T * C + t * C + c];
                    dwrow[c] += d * inp_btc;
                }
            }
        }
    };

    if let Some(dbias) = dbias {
        dweight
            .par_chunks_mut(C)
            .zip(dbias.par_iter_mut())
            .take(OC)
            .enumerate()
            .for_each(|(o, (dwrow, dbias_o))| {
                update_chunk(dwrow, Some(dbias_o), o);
            })
    } else {
        dweight
            .par_chunks_mut(C)
            .take(OC)
            .enumerate()
            .for_each(|(o, dwrow)| {
                update_chunk(dwrow, None, o);
            })
    }
}

fn attention_forward(
    out: &mut [f32],
    preatt: &mut [f32],
    att: &mut [f32],
    inp: &[f32],
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    // inp: (B, T, 3C)
    // out: (B, T, C)
    // preatt: (B, NH, T, T)
    // att: (B, NH, T, T)
    let C3 = C * 3;
    let hs = C / NH;
    let scale = 1.0 / (hs as f32).sqrt();

    out.par_chunks_mut(T * C)
        .zip(preatt.par_chunks_mut(NH * T * T))
        .zip(att.par_chunks_mut(NH * T * T))
        .zip(inp.par_chunks(T * C3))
        .take(B)
        .for_each(|(((out_b, preatt_b), att_b), inp_b)| {
            for t in 0..T {
                for h in 0..NH {
                    let query_t = &inp_b[t * C3 + h * hs..t * C3 + (h + 1) * hs];
                    let preatt_bth = &mut preatt_b[h * T * T + t * T..h * T * T + (t + 1) * T];
                    let att_bth = &mut att_b[h * T * T + t * T..h * T * T + (t + 1) * T];
                    // pass 1: calculate query dot key and maxval
                    let mut maxval = -10000_f32;
                    for t2 in 0..=t {
                        let key_t = &inp_b[t2 * C3 + h * hs + C..t2 * C3 + h * hs + C + hs];

                        let mut val = 0_f32;
                        for i in 0..hs {
                            val += query_t[i] * key_t[i];
                        }
                        val *= scale;
                        if val > maxval {
                            maxval = val;
                        }
                        preatt_bth[t2] = val;
                    }

                    // pass 2: calculate the exp and keep track of sum
                    // maxval is being calculated and subtracted only for numerical stability
                    let mut expsum = 0_f32;
                    for t2 in 0..=t {
                        let expv = (preatt_bth[t2] - maxval).exp();
                        att_bth[t2] = expv;
                        expsum += expv;
                    }
                    let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                    // pass 3: normalize to get the softmax
                    for t2 in 0..T {
                        if t2 <= t {
                            att_bth[t2] *= expsum_inv;
                        } else {
                            // causal attention mask. not strictly necessary to set to zero here
                            // only doing this explicitly for debugging and checking to PyTorch
                            att_bth[t2] = 0.0;
                        }
                    }

                    // pass 4: accumulate weighted values into the output of attention
                    let out_bth = &mut out_b[t * C + h * hs..t * C + (h + 1) * hs];
                    out_bth.iter_mut().for_each(|x| *x = 0.0);
                    for t2 in 0..=t {
                        let values_t =
                            &inp_b[t2 * C3 + h * hs + 2 * C..t2 * C3 + h * hs + 2 * C + hs];
                        let score = att_bth[t2];
                        for c in 0..hs {
                            out_bth[c] += score * values_t[c];
                        }
                    }
                }
            }
        })
}

fn attention_backward(
    dinp: &mut [f32],
    dpreatt: &mut [f32],
    datt: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    att: &[f32],
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    // dinp: (B, T, 3C)
    // dpreatt: (B, NH, T, T)
    // datt: (B, NH, T, T)
    // dout: (B, T, C)
    // inp: (B, T, 3C)
    // att: (B, NH, T, T)
    let C3 = C * 3;
    let hs = C / NH;
    let scale = 1.0 / (hs as f32).sqrt();
    for b in 0..B {
        for t in 0..T {
            for h in 0..NH {
                let att_offset = b * NH * T * T + h * T * T + t * T;

                let att_bth = &att[att_offset..att_offset + T];
                let datt_bth = &mut datt[att_offset..att_offset + T];
                let dpreatt_bth = &mut dpreatt[att_offset..att_offset + T];

                // backward pass 4, through the value accumulation
                let dout_offset = b * T * C + t * C + h * hs;
                let dout_bth = &dout[dout_offset..dout_offset + hs];
                for t2 in 0..=t {
                    let value_offset = b * T * C3 + t2 * C3 + h * hs + 2 * C;
                    let values = &inp[value_offset..value_offset + hs];
                    for c in 0..hs {
                        let dvalues_index = value_offset + c;
                        datt_bth[t2] += dout_bth[c] * values[c];
                        dinp[dvalues_index] += dout_bth[c] * att_bth[t2];
                    }
                }
                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for t2 in 0..=t {
                    for t3 in 0..=t {
                        let indicator = if t2 == t3 { 1.0 } else { 0.0 };
                        let local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                let query_offset = b * T * C3 + t * C3 + h * hs;
                let query = &inp[query_offset..query_offset + hs];
                for t2 in 0..=t {
                    let key_offset = b * T * C3 + t2 * C3 + h * hs + C;
                    let key = &inp[key_offset..key_offset + hs];
                    for c in 0..hs {
                        let dquery_index = query_offset + c;
                        let dkey_index = key_offset + c;
                        dinp[dquery_index] += dpreatt_bth[t2] * key[c] * scale;
                        dinp[dkey_index] += dpreatt_bth[t2] * query[c] * scale;
                    }
                }
            }
        }
    }
}

fn gelu_forward(out: &mut [f32], inp: &[f32], N: usize) {
    let GELU_SCALING_FACTOR: f32 = (2.0 / PI).sqrt();
    for i in 0..N {
        let x = inp[i];
        let cube = 0.044715 * x * x * x;
        out[i] = x * 0.5 * (1.0 + (GELU_SCALING_FACTOR * (x + cube)).tanh());
    }
}

fn gelu_backward(dinp: &mut [f32], dout: &[f32], inp: &[f32], N: usize) {
    let GELU_SCALING_FACTOR: f32 = (2.0 / PI).sqrt();
    for i in 0..N {
        let x = inp[i];
        let cube = 0.044715 * x * x * x;
        let tanh_x = GELU_SCALING_FACTOR * (x + cube);
        let tanh_out = tanh_x.tanh();
        let cosh_out = tanh_x.cosh();
        let sech_out = 1.0 / (cosh_out * cosh_out);
        let local_grad =
            0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * GELU_SCALING_FACTOR * (1.0 + 3.0 * 0.044715 * x * x);
        dinp[i] += dout[i] * local_grad;
    }
}

fn residual_forward(out: &mut [f32], inp1: &[f32], inp2: &[f32], N: usize) {
    for i in 0..N {
        out[i] = inp1[i] + inp2[i];
    }
}

fn residual_backward(dinp1: &mut [f32], dinp2: &mut [f32], dout: &[f32], N: usize) {
    for i in 0..N {
        dinp1[i] += dout[i]; 
        dinp2[i] += dout[i];
    }
}

fn softmax_forward(probs: &mut [f32], logits: &[f32], B: usize, T: usize, V: usize, Vp: usize) {
    // logits: [B, T, Vp]
    // probs: [B, T, Vp]

    probs
        .par_chunks_mut(Vp)
        .zip(logits.par_chunks(Vp))
        .take(B * T)
        .for_each(|(probs_bt, logits_bt)| {
            let maxval = logits_bt.iter().take(V).fold(-10000_f32, |a, b| a.max(*b));
            let mut sum = 0.0;
            for v in 0..V {
                probs_bt[v] = (logits_bt[v] - maxval).exp();
                sum += probs_bt[v];
            }

            for v in 0..V {
                probs_bt[v] /= sum;
            }

            for v in V..Vp {
                probs_bt[v] = 0.0;
            }
        })
}

fn crossentropy_forward(
    loss: &mut [f32],
    probs: &[f32],
    targets: &[i32],
    B: usize,
    T: usize,
    Vp: usize,
) {
    // loss: [B, T]
    // probs: [B, T, Vp]
    // targets: [B, T]
    for b in 0..B {
        for t in 0..T {
            let offset = b * T * Vp + t * Vp;
            let probs_bt = &probs[offset..offset + Vp];
            let target_idx = &targets[b * T + t];
            loss[b * T + t] = -probs_bt[*target_idx as usize].ln();
        }
    }
}

fn crossentropy_softmax_backward(
    dlogits: &mut [f32],
    dlosses: &[f32],
    probs: &[f32],
    targets: &[i32],
    B: usize,
    T: usize,
    V: usize,
    Vp: usize,
) {
    // dlogits: [B, T, Vp]
    // dloss: [B, T]
    // probs: [B, T, Vp]
    // targets: [B, T]
    for b in 0..B {
        for t in 0..T {
            let offset = b * T * Vp + t * Vp;
            let dlogits_bt = &mut dlogits[offset..offset + Vp];
            let probs_bt = &probs[offset..offset + Vp];
            let dloss = &dlosses[b * T + t];
            let target_idx = &targets[b * T + t];
            for v in 0..V {
                let p = probs_bt[v];
                let indicator = if v as i32 == *target_idx { 1.0 } else { 0.0 };
                dlogits_bt[v] += (p - indicator) * dloss;
            }
        }
    }
}

pub mod gpt2;
