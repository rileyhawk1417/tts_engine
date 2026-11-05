//! Session-1 probe for the T008 candle port: parse a phonon checkpoint's
//! config.json through the ported cfg module and inventory the mmap'd
//! weights. No synthesis — the transformer port is later sessions. Run via:
//!   cargo run --release -p tts-engine --features flow-candle \
//!     --example flow_candle_probe -- models/phonon-31m-en
//!
//! Checks (hardwired-hyperparameter audit, T008 done-condition seed):
//!   - every decode-relevant knob comes from config.json (printed, not assumed)
//!   - the remap table classifies all tensors; used counts sane
//!   - frame_samples derivation vs 24 kHz / 12.5 Hz contract

use std::path::PathBuf;

fn main() {
    let dir = match std::env::args().nth(1) {
        Some(d) => PathBuf::from(d),
        None => {
            eprintln!("usage: flow_candle_probe <checkpoint-dir>");
            std::process::exit(2);
        }
    };
    let cfg_path = dir.join("config.json");
    let cfg = tts_engine::candle_ptts::cfg::from_checkpoint(&cfg_path)
        .unwrap_or_else(|e| {
            eprintln!("cfg: {e}");
            std::process::exit(3);
        });

    // --- decode-relevant knobs, all from the checkpoint --------------------
    let fl = &cfg.flow_lm;
    let mi = &cfg.mimi;
    println!("flow_lm: d_model={} heads={} kv_heads={:?} layers={} ffn={} rope_period={}",
        fl.d_model, fl.num_heads, fl.num_kv_heads, fl.num_layers, fl.dim_feedforward, fl.max_period);
    println!("flow_lm: n_bins={} lut_dim={} flow_dim={} flow_depth={} ldim={} gelu={:?} (default {:?}) time_rms={:?} flow_type={:?}",
        fl.n_bins, fl.lut_dim, fl.flow_dim, fl.flow_depth, fl.ldim,
        fl.gelu, cfg.default_gelu(), fl.time_rms_norm, fl.flow_type);
    println!("mimi: sr={} frame_rate={} ratios={:?} dim={} qdim={} frame_samples={} (decoder_dim={:?})",
        mi.sample_rate, mi.frame_rate, mi.ratios, mi.dimension, mi.quantizer_dimension,
        mi.frame_samples(), mi.decoder_dimension);
    println!("decode: temp={:?} lsd_steps={} eos_threshold={}",
        cfg.temp, cfg.lsd_decode_steps, cfg.eos_threshold);
    println!("conditioners: {:?}",
        cfg.conditioners.iter().map(|c| (c.name(), match c {
            tts_engine::candle_ptts::cfg::ConditionerConfig::Lut { .. } => "lut",
            tts_engine::candle_ptts::cfg::ConditionerConfig::Continuous { .. } => "continuous",
        })).collect::<Vec<_>>());
    println!("languages={:?} embeddings_ext={:?} weights={} tokenizer={:?} mimi_cfg_null_audio={} cfg_null_text={}",
        cfg.languages, cfg.embeddings_ext, cfg.weights_file(),
        cfg.tokenizer_name, cfg.cfg_null_audio_empty, cfg.cfg_null_text_empty);
    println!("model_id={:?}", cfg.model_id);
    assert_eq!(mi.sample_rate, 24000, "phonon contract: 24 kHz mono");
    assert_eq!(mi.frame_samples(), 1920, "12.5 Hz frames => 1920 samples");

    // --- weight inventory (mmap headers only) -------------------------------
    let weights = dir.join(cfg.weights_file());
    let inv = tts_engine::candle_ptts::loader::inventory(&weights)
        .unwrap_or_else(|e| {
            eprintln!("inventory: {e}");
            std::process::exit(3);
        });
    println!(
        "inventory: {} tensors, {} used by remap, {} MiB on disk",
        inv.tensors.len(),
        inv.used_count,
        inv.total_bytes / (1 << 20)
    );
    for (d, n) in inv.bytes_by_dtype() {
        println!("  {:?}: {} MiB", d, n / (1 << 20));
    }
    let mut biggest = inv.tensors.clone();
    biggest.sort_by_key(|t| std::cmp::Reverse(t.data_len));
    for t in biggest.iter().take(5) {
        println!("  top: {} {:?} {:?} ({} KiB)", t.name, t.dtype, t.shape, t.data_len / 1024);
    }
    // Remap-table sanity on the names the later port modules will expect.
    let remapped = inv
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("flow_lm.transformer.") || t.name.starts_with("mimi.decoder"))
        .count();
    println!("sanity: remapped backbone/decoder keys present: {remapped}");
}
