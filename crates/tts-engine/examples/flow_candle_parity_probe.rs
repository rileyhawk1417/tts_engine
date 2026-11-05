//! T008 session-3 — parity harness: drive the ORACLE (ptts `TTSModel`, f32,
//! xn backend) and the CANDLE PORT (`candle_ptts::TTSModel`) through the same
//! generation steps on the same inputs — text tokens from the oracle
//! `plan::Chunk` (one tokenization for both sides), default voice, default
//! conditions, and a fixed `ReplayRng` vector — then diff per-step latents,
//! eos decisions, and audio.
//!
//! This is the oracle's own cross-implementation mechanism: `flow_lm.rs`
//! `ReplayRng` replays a recorded noise sequence so two implementations can be
//! compared step for step.
//!
//! Run:
//!   cargo run --release -q -p tts-engine \
//!     --features flow-ptts,flow-candle \
//!     --example flow_candle_parity_probe -- models/phonon-31m-en [steps] [text]
#![cfg(all(feature = "flow-ptts", feature = "flow-candle"))]

use std::collections::HashMap;

use candle_core::Tensor;
use ptts::checkpoint::Checkpoint;
use ptts::flow_lm::{ReplayRng as OracleReplay, StepInput as OracleStep};
use ptts::plan::Chunk as OracleChunk;
use ptts::tts_model::TTSModel as OracleModel;
use xn::{CpuDevice, Unquantized};

type Q = Unquantized<f32, CpuDevice>;

const REPLAY: usize = 64;
const FRAME_RATE: f64 = 12.5;

fn replay_values() -> Vec<f32> {
    (0..REPLAY)
        .map(|i| ((i as f64 * 0.37).cos() * 0.42) as f32)
        .collect()
}

fn fmt8(vals: &[f32]) -> String {
    vals.iter()
        .take(8)
        .map(|v| format!("{v:+.4}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// L2 across the short side, plus max |a-b|, for two host vectors.
fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    let n = a.len().min(b.len());
    if n == 0 {
        return (0.0, 0.0);
    }
    let l2 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt();
    let max = a
        .iter()
        .zip(b)
        .take(n)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    (l2, max)
}

fn host_vec(t: &Tensor) -> candle_core::Result<Vec<f32>> {
    // drop the (always 1) batch row
    let t = if t.dims().len() >= 2 {
        t.narrow(0, 0, 1)?.squeeze(0)?
    } else {
        t.clone()
    };
    match t.dims().len() {
        1 => t.to_vec1(),
        2 => Ok(t.to_vec2()?.into_iter().flatten().collect()),
        _ => t.to_vec1(),
    }
}

/// Compare individual stage tensors on identical inputs.
/// A = conditions sum; B = voice embedding; C = text token embeddings;
/// D = transformer forward on two identical synthetic inputs, fresh states.
fn stages_probe(cp: &Checkpoint, chunk: &OracleChunk) {
    use candle_core::{DType, Device};

    // ---- A: conditions ----
    let weights = ptts::loader::load_weights::<Q>(&cp.weights, &xn::CPU).expect("oracle weights");
    let tok_path = cp.tokenizer.clone().expect("tokenizer.json");

    let oracle = OracleModel::<Q>::load(
        &weights,
        Box::new(ptts::tok::Tok::open(&tok_path).expect("oracle tokenizer")),
        &cp.config,
        &HashMap::new(),
    )
    .expect("oracle TTSModel");
    let oracle_cond =
        ptts::loader::load_conditions::<f32, CpuDevice>(&weights, &cp.config, &HashMap::new())
            .expect("oracle conditions")
            .expect("oracle cond present");

    let vw = tts_engine::candle_ptts::loader::Weights::from_safetensors(cp.weights.as_path())
        .expect("candle weights");
    let vb = vw.var_builder();
    let ccfg = tts_engine::candle_ptts::cfg::from_checkpoint(
        &std::path::Path::new(cp.weights.as_path())
            .parent()
            .expect("dir")
            .join("config.json"),
    )
    .expect("candle cfg");
    let cmodel = tts_engine::candle_ptts::model::TTSModel::load(
        &vb,
        Box::new(
            tts_engine::candle_ptts::tok::Tok::open(std::path::Path::new(&tok_path))
                .expect("candle tokenizer"),
        ),
        &ccfg,
        &HashMap::new(),
    )
    .expect("candle TTSModel");
    // default conditions map: empty (defaults) — must match oracle load path
    let candle_cond = tts_engine::candle_ptts::model::Conditioners::load(
        &vb.pp("flow_lm").pp("condition_provider.conditioners"),
        &ccfg,
    )
    .expect("candle conditioners")
    .sum(&HashMap::new())
    .expect("candle cond sum")
    .expect("candle cond present");

    let oh: Vec<f32> = oracle_cond.to_vec().expect("oracle cond host");
    let ch = host_vec(&candle_cond).expect("candle cond host");
    println!(
        "A conditions   : dims {} vs {} | max={:.8} L2={:.8}",
        fmt_dims(oracle_cond.dims()),
        fmt_dims(candle_cond.dims()),
        diff_stats(&oh, &ch).1,
        diff_stats(&oh, &ch).0
    );

    // ---- B: voice embedding ----
    let voice_path = cp.voices.first().map(|(_, p)| p.clone()).expect("voice");
    let o_emb = ptts::loader::load_voice_emb(
        &voice_path,
        cp.config.embeddings_ext.as_deref(),
        oracle.speaker_proj(),
        &xn::CPU,
    )
    .expect("oracle voice emb");
    let c_emb = tts_engine::candle_ptts::loader::load_voice_emb(
        &voice_path,
        cp.config.embeddings_ext.as_deref(),
        cmodel.speaker_proj(),
    )
    .expect("candle voice emb");
    let oh: Vec<f32> = o_emb.to_vec().expect("oracle emb host");
    let ch = host_vec(&c_emb).expect("candle emb host");
    let (l2, mx) = diff_stats(&oh, &ch);
    println!(
        "B voice emb    : dims {} vs {} | max={mx:.8} L2={l2:.8} n={}/{}",
        fmt_dims(o_emb.dims()),
        fmt_dims(c_emb.dims()),
        oh.len(),
        ch.len()
    );

    // ---- C: text token embeddings ----
    let o_text = oracle
        .flow_lm
        .conditioner
        .embed_tokens(&chunk.tokens)
        .expect("oracle text emb");
    let c_text = cmodel
        .flow_lm
        .conditioner
        .embed_tokens(&chunk.tokens)
        .expect("candle text emb");
    let oh: Vec<f32> = o_text.to_vec().expect("oracle text host");
    let ch = host_vec(&c_text).expect("candle text host");
    let (l2, mx) = diff_stats(&oh, &ch);
    println!(
        "C text emb     : dims {} vs {} | max={mx:.8} L2={l2:.8}",
        fmt_dims(o_text.dims()),
        fmt_dims(c_text.dims())
    );

    // ---- D: transformer forward on identical synthetic inputs, fresh states ----
    let budget = ptts::plan::seq_budget(chunk.tokens.len(), chunk.frame_budget);
    for (label, input) in [("zeros", None), ("random", Some(replay_like_input()))] {
        let (ox, cx) = match input {
            Some(v) => (
                xn::Tensor::<f32, CpuDevice>::from_vec(v.clone(), (1, 3, 512), &xn::CPU)
                    .expect("oracle input"),
                Tensor::from_vec(v, (1, 3, 512), &Device::Cpu).expect("candle input"),
            ),
            None => (
                xn::Tensor::<f32, CpuDevice>::from_vec(vec![0f32; 3 * 512], (1, 3, 512), &xn::CPU)
                    .expect("oracle zeros"),
                Tensor::zeros((1, 3, 512), DType::F32, &Device::Cpu).expect("candle zeros"),
            ),
        };
        let mut os = oracle
            .flow_lm
            .transformer
            .init_state(1, budget)
            .expect("o state");
        let o = oracle
            .flow_lm
            .transformer
            .forward(&ox, &mut os)
            .expect("oracle transformer");
        let mut cs = cmodel
            .flow_lm
            .transformer
            .init_state(1, budget, &Device::Cpu)
            .expect("c state");
        let c = cmodel
            .flow_lm
            .transformer
            .forward(&cx, &mut cs)
            .expect("candle transformer");
        let oh: Vec<f32> = o.to_vec().expect("oracle tf host");
        let ch = host_vec(&c).expect("candle tf host");
        let (l2, mx) = diff_stats(&oh, &ch);
        println!(
            "D tf[{label:6}]   : dims {} vs {} | max={mx:.8} L2={l2:.8}",
            fmt_dims(o.dims()),
            fmt_dims(c.dims())
        );
    }

    // ---- F: chunk plan ----
    for text in ["The quick brown fox jumps over the lazy dog."] {
        let norm = ptts::preprocess::Normalize::for_lang(ptts::preprocess::Lang::En);
        let otok = ptts::tok::Tok::open(std::path::Path::new(&tok_path)).expect("otok");
        let och = ptts::plan::chunks(&otok, text, norm, 50, 12.5).expect("oplan");
        let ctok =
            tts_engine::candle_ptts::tok::Tok::open(std::path::Path::new(&tok_path)).expect("ctok");
        let cnorm = ptts::preprocess::Normalize::for_lang(ptts::preprocess::Lang::En);
        let cch =
            tts_engine::candle_ptts::plan::chunks(&ctok, text, cnorm, 50, 12.5).expect("cplan");
        println!(
            "F plan[{text}] : oracle {} chunks vs candle {} chunks",
            och.len(),
            cch.len()
        );
        for (i, (o, c)) in och.iter().zip(cch.iter()).enumerate() {
            println!(
                "   chunk {i}: oracle tok={} fb={} | candle tok={} fb={} | same_tokens={}",
                o.tokens.len(),
                o.frame_budget,
                c.tokens.len(),
                c.frame_budget,
                o.tokens == c.tokens
            );
            if o.tokens != c.tokens {
                println!("     oracle tokens: {:?}", o.tokens);
                println!("     candle tokens: {:?}", c.tokens);
            }
        }
        if och.len() == cch.len() {
            println!(
                "   all token lists equal: {}",
                och.iter()
                    .zip(cch.iter())
                    .all(|(o, c)| o.tokens == c.tokens)
            );
        }
    }
}

/// Phonon-31m-en pins eos_threshold = 4.0 (from config.json); there is no
/// getter on the oracle TTSModel, so the probe pins the config value.
fn model_eos_threshold(_cp: &Checkpoint) -> f32 {
    4.0
}

/// Deterministic [1,3,512] input with structure (not zeros).
fn replay_like_input() -> Vec<f32> {
    (0..3 * 512)
        .map(|i| {
            let t = (i % 512) as f64;
            let p = (i / 512) as f64;
            ((t * 0.031 + p * 1.7).sin() * (0.4 - 0.1 * p)) as f32
        })
        .collect()
}

fn fmt_dims(d: &[usize]) -> String {
    d.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

fn main() {
    let dir = std::env::args().nth(1).expect("checkpoint dir");
    let arg2 = std::env::args().nth(2).unwrap_or_default();
    let stages_mode = arg2 == "stages";
    let steps: usize = arg2
        .parse()
        .ok()
        .or({
            let _ = &arg2;
            Some(48usize)
        })
        .unwrap_or(48);
    let text = std::env::args().nth(3).unwrap_or_else(|| "a.".to_string());

    let cp = Checkpoint::open(&dir).expect("checkpoint");
    println!(
        "# weights = {} (expect model.safetensors = f32)",
        cp.weights.display()
    );

    // One shared chunk from the ORACLE tokenizer: tokenization is identical on
    // both sides by construction (Normalize is delegated to ptts in the port).
    let tok_path = cp
        .tokenizer
        .clone()
        .expect("checkpoint ships tokenizer.json");
    let chunk = {
        let tok = ptts::tok::Tok::open(&tok_path).expect("chunk tokenizer");
        OracleChunk::new(text.clone(), &tok, FRAME_RATE).expect("chunk")
    };
    println!(
        "# chunk: tokens={:?} frame_budget={} frames_after_eos={}",
        chunk.tokens, chunk.frame_budget, chunk.frames_after_eos
    );

    if stages_mode {
        stages_probe(&cp, &chunk);
        return;
    }

    let values = replay_values();
    let fresh_replay =
        || -> Box<OracleReplay> { Box::new(OracleReplay::new(values.clone()).expect("replay")) };

    // ================= oracle side =================
    let oracle_latents: Vec<(Vec<f32>, f32, bool)> = {
        let weights = ptts::loader::load_weights::<Q>(&cp.weights, &xn::CPU).expect("weights");
        let model = OracleModel::<Q>::load(
            &weights,
            Box::new(ptts::tok::Tok::open(&tok_path).expect("tokenizer")),
            &cp.config,
            &HashMap::new(),
        )
        .expect("oracle TTSModel");

        let voice_path = cp.voices.first().map(|(_, p)| p.clone()).expect("voice");
        let emb = ptts::loader::load_voice_emb(
            &voice_path,
            cp.config.embeddings_ext.as_deref(),
            model.speaker_proj(),
            &xn::CPU,
        )
        .expect("voice emb");
        let cond =
            ptts::loader::load_conditions::<f32, CpuDevice>(&weights, &cp.config, &HashMap::new())
                .expect("conditions")
                .expect("cond present");

        let budget = ptts::plan::seq_budget(chunk.tokens.len(), chunk.frame_budget);
        let mut state = model.init_flow_lm_state(1, budget).expect("state");
        state.flow_lm_state.conditions = Some(cond);
        model.prompt_audio(&mut state, &emb).expect("prompt_audio");
        model
            .prompt_text(&mut state, &chunk.tokens)
            .expect("prompt_text");

        let mut mimi = model.init_mimi_state(1).expect("mimi state");
        let mut rng = fresh_replay();
        let mut prev = None;
        let mut out = Vec::new();
        for i in 0..steps {
            let input = match prev.as_ref() {
                Some(t) => OracleStep::Latent(t),
                None => OracleStep::Bos { batch: 1 },
            };
            let (next, is_eos, eosv) = {
                let (_l, _e) = model
                    .generate_step_parts(&mut state, input, rng.as_mut())
                    .expect("step parts");
                let is_eos = ptts::flow_lm::FlowLM::<Q>::eos_from_logit(
                    &_e.to_vec().expect("eos host"),
                    model_eos_threshold(&cp),
                );
                let eosv: Vec<f32> = _e.to_vec().expect("eos host");
                (_l, is_eos, eosv)
            };
            let audio = model.decode_latent(&next, &mut mimi).expect("decode");
            let host: Vec<f32> = next.to_vec().expect("latent host");
            let norm: f32 = host.iter().map(|v| v * v).sum::<f32>().sqrt();
            let audio_vec: Vec<f32> = audio.to_vec().expect("audio host");
            let a_rms = (audio_vec.iter().map(|v| v * v).sum::<f32>()
                / audio_vec.len().max(1) as f32)
                .sqrt();
            println!(
                "oracle step {i:2}: latent[..8] = {} | L2={norm:.5} eos={is_eos} eos_l={:.4} | audio rms={a_rms:.5} n={}",
                fmt8(&host),
                eosv.first().copied().unwrap_or(f32::NAN),
                audio_vec.len()
            );
            out.push((host, a_rms, is_eos));
            prev = Some(next);
        }
        out
    };

    // ================= candle side =================
    let candle_latents: Vec<(Vec<f32>, f32, bool)> = {
        use tts_engine::candle_ptts::loader::{Weights, load_voice_emb};
        use tts_engine::candle_ptts::model::{StepInput as CandleStep, TTSModel as CandleModel};
        use tts_engine::candle_ptts::rng::ReplayRng as CandleReplay;

        let weights =
            Weights::from_safetensors(std::path::Path::new(&cp.weights)).expect("candle weights");
        let vb = weights.var_builder();
        let tok_path = cp.tokenizer.clone().expect("tokenizer.json");
        let tokenizer: Box<dyn tts_engine::candle_ptts::tok::Tokenizer> = Box::new(
            tts_engine::candle_ptts::tok::Tok::open(std::path::Path::new(&tok_path))
                .expect("candle tok"),
        );
        let ccfg = tts_engine::candle_ptts::cfg::from_checkpoint(
            &std::path::Path::new(&dir).join("config.json"),
        )
        .map_err(|e| panic!("candle config: {e}"))
        .unwrap();
        let model = CandleModel::load(&vb, tokenizer, &ccfg, &HashMap::new())
            .map_err(|e| panic!("candle TTSModel: {e}"))
            .unwrap();

        let voice_path = cp.voices.first().map(|(_, p)| p.clone()).expect("voice");
        let emb = load_voice_emb(
            &voice_path,
            cp.config.embeddings_ext.as_deref(),
            model.speaker_proj(),
        )
        .expect("candle voice emb");

        let cond = tts_engine::candle_ptts::model::Conditioners::load(
            &vb.pp("flow_lm").pp("condition_provider.conditioners"),
            &ccfg,
        )
        .map_err(|e| panic!("candle conditioners: {e}"))
        .unwrap()
        .sum(&HashMap::new())
        .map_err(|e| panic!("candle conditions: {e}"))
        .unwrap();

        let budget = ptts::plan::seq_budget(chunk.tokens.len(), chunk.frame_budget);
        let mut state = model
            .init_flow_lm_state(1, budget)
            .map_err(|e| panic!("candle state: {e}"))
            .unwrap();
        model.set_conditions(&mut state, cond);
        model
            .prompt_audio(&mut state, &emb)
            .map_err(|e| panic!("candle prompt_audio: {e}"))
            .unwrap();
        model
            .prompt_text(&mut state, &chunk.tokens)
            .map_err(|e| panic!("candle prompt_text: {e}"))
            .unwrap();

        let mut mimi = model
            .init_mimi_state(1)
            .map_err(|e| panic!("candle mimi state: {e}"))
            .unwrap();
        let mut rng = Box::new(CandleReplay::new(values.clone()).expect("candle replay"));
        let mut prev: Option<Tensor> = None;
        let mut out = Vec::new();
        for i in 0..steps {
            let input = match prev.as_ref() {
                Some(t) => CandleStep::Latent(t),
                None => CandleStep::Bos { batch: 1 },
            };
            let (next_val, eos_logit) = model
                .generate_step_parts(&mut state, input, rng.as_mut())
                .map_err(|e| panic!("candle step {i}: {e}"))
                .unwrap();
            let eos_host = host_vec(&eos_logit).expect("eos host");
            let is_eos = model.eos_from_logit(&eos_host);
            let audio = model
                .decode_latent(&next_val, &mut mimi)
                .map_err(|e| panic!("candle decode {i}: {e}"))
                .unwrap();
            let host = host_vec(&next_val).expect("latent host");
            let norm: f32 = host.iter().map(|v| v * v).sum::<f32>().sqrt();
            let audio_vec = host_vec(&audio).expect("audio host");
            let a_rms = (audio_vec.iter().map(|v| v * v).sum::<f32>()
                / audio_vec.len().max(1) as f32)
                .sqrt();
            println!(
                "candle step {i:2}: latent[..8] = {} | L2={norm:.5} eos={is_eos} eos_l={:.4} | audio rms={a_rms:.5} n={}",
                fmt8(&host),
                eos_host.first().copied().unwrap_or(f32::NAN),
                audio_vec.len()
            );
            out.push((host, a_rms, is_eos));
            prev = Some(next_val);
        }
        out
    };

    // ================= diff =================
    println!();
    println!(
        "# steps compared: oracle={} candle={}",
        oracle_latents.len(),
        candle_latents.len()
    );
    let n = oracle_latents.len().min(candle_latents.len());
    let mut first_div = None;
    for i in 0..n {
        let (ol, orms, oe) = &oracle_latents[i];
        let (cl, crms, ce) = &candle_latents[i];
        let (l2, max) = diff_stats(ol, cl);
        let flag = if max > 1e-4 && first_div.is_none() {
            first_div = Some(i);
            "  <-- FIRST DIVERGENCE"
        } else {
            ""
        };
        println!(
            "step {i:2}: |diff| max={max:.6} L2={l2:.6} | eos {oe}/{ce} | audio rms {orms:.5}/{crms:.5}{flag}"
        );
    }
    match first_div {
        Some(i) => println!("# FIRST DIVERGENCE at step {i}"),
        None => println!("# latents comparable across all {n} steps (max |diff| <= 1e-4)"),
    }
}
