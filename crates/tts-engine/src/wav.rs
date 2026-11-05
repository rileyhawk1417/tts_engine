//! Canonical WAV writer — mono 16-bit PCM little-endian (docs/seam.md §3).

use std::io::{self, Write};
use std::path::Path;

/// Write `samples` as a 16-bit mono PCM WAV file at `sample_rate`.
pub fn write_wav(path: impl AsRef<Path>, sample_rate: u32, samples: &[i16]) -> io::Result<()> {
    let data_len = samples.len() * 2;
    let mut f = std::fs::File::create(path)?;
    let mut w: Vec<u8> = Vec::with_capacity(44 + data_len);

    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    w.extend_from_slice(b"WAVE");
    // fmt chunk
    w.extend_from_slice(b"fmt ");
    w.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    w.extend_from_slice(&1u16.to_le_bytes()); // PCM
    w.extend_from_slice(&1u16.to_le_bytes()); // mono
    w.extend_from_slice(&sample_rate.to_le_bytes());
    w.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    w.extend_from_slice(&2u16.to_le_bytes()); // block align
    w.extend_from_slice(&16u16.to_le_bytes()); // bits
    // data chunk
    w.extend_from_slice(b"data");
    w.extend_from_slice(&(data_len as u32).to_le_bytes());
    for s in samples {
        w.extend_from_slice(&s.to_le_bytes());
    }
    f.write_all(&w)?;
    f.flush()
}

/// Parse a 16/24/32-bit PCM or float32 WAV **byte buffer** (any sample
/// rate), downmix to mono i16. No resampling — callers that need a rate
/// conversion use the file variant below. Added for T010 ggml-ffi: the
/// shim hands back a WAV buffer owned by llama.cpp's pipeline.
/// Returns `(sample_rate, samples)`.
pub fn parse_wav_buffer_mono_i16(bytes: &[u8]) -> io::Result<(u32, Vec<i16>)> {
    let (rate, f32s) = wav_to_mono_f32(bytes)?;
    Ok((rate, f32s.iter().map(|&v| clamp_i16(v)).collect()))
}

fn wav_to_mono_f32(bytes: &[u8]) -> io::Result<(u32, Vec<f32>)> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a RIFF/WAVE file",
        ));
    }
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (bits_per_sample, channels, rate, format)
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body = &bytes[pos + 8..(pos + 8 + size).min(bytes.len())];
        match id {
            b"fmt " => {
                if body.len() < 16 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "short fmt chunk",
                    ));
                }
                let format = u16::from_le_bytes([body[0], body[1]]);
                let channels = u16::from_le_bytes([body[2], body[3]]);
                let rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
                let bits = u16::from_le_bytes([body[14], body[15]]);
                fmt = Some((bits, channels, rate, format));
            }
            b"data" => data = Some(body),
            _ => {}
        }
        pos += 8 + size + (size & 1); // chunks are word-aligned
    }
    let Some((bits, channels, rate, format)) = fmt else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing fmt chunk",
        ));
    };
    let Some(data) = data else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing data chunk",
        ));
    };
    if format != 1 && format != 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported format {format}"),
        ));
    }

    // Decode frames to f32 in [-1, 1)
    let bytes_per_sample = (bits / 8) as usize;
    let channels = channels as usize;
    let frame = channels * bytes_per_sample;
    let mut f32s: Vec<f32> = Vec::with_capacity(data.len() / frame.max(1) * channels);
    for chunk in data.chunks_exact(frame) {
        let mut acc = 0f32;
        for ch in 0..channels {
            let s = &chunk[ch * bytes_per_sample..(ch + 1) * bytes_per_sample];
            let v = match bits {
                16 => i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
                24 => {
                    let raw = ((s[2] as i32) << 16 | (s[1] as i32) << 8 | s[0] as i32) as f32;
                    raw / 8388608.0
                }
                32 => {
                    if format == 3 {
                        f32::from_le_bytes([s[0], s[1], s[2], s[3]])
                    } else {
                        i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2147483648.0
                    }
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unsupported bits {bits}"),
                    ));
                }
            };
            acc += v;
        }
        f32s.push(acc / channels as f32);
    }
    Ok((rate, f32s))
}

/// Parse a 16/24/32-bit PCM or float32 WAV file (any sample rate), downmix to
/// mono, resample (naive linear) to `target_rate`, and return i16 samples.
/// Used by the eval harness to sanity-check our own outputs before judging.
pub fn read_wav_as_mono_i16(path: impl AsRef<Path>, target_rate: u32) -> io::Result<Vec<i16>> {
    let bytes = std::fs::read(path)?;
    let (rate, f32s) = wav_to_mono_f32(&bytes)?;

    // Resample to target_rate (linear; evaluation-side sanity only)
    if rate == target_rate || f32s.is_empty() {
        return Ok(f32s.iter().map(|&v| clamp_i16(v)).collect());
    }
    let ratio = rate as f64 / target_rate as f64;
    let out_len = ((f32s.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * ratio;
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(f32s.len() - 1);
        let t = src - i0 as f64;
        out.push(clamp_i16(f32s[i0] * (1.0 - t as f32) + f32s[i1] * t as f32));
    }
    Ok(out)
}

fn clamp_i16(v: f32) -> i16 {
    v.clamp(-1.0, 0.99997).mul_add(32767.0, 0.0) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_roundtrip() {
        let dir = std::env::temp_dir().join("tts-engine-wav-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("rt.wav");
        let samples = (0..2400i64)
            .map(|i| ((i * 17) % 1000) as i16 - 500)
            .collect::<Vec<i16>>();
        write_wav(&p, 24_000, &samples).unwrap();
        let back = read_wav_as_mono_i16(&p, 24_000).unwrap();
        assert_eq!(back.len(), samples.len());
        for (a, b) in back.iter().zip(&samples) {
            assert!((a - b).abs() <= 1, "{a} vs {b}");
        }
    }

    #[test]
    fn wav_resample_len() {
        let dir = std::env::temp_dir().join("tts-engine-wav-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("rs.wav");
        write_wav(&p, 48_000, &vec![100i16; 48_000]).unwrap();
        let back = read_wav_as_mono_i16(&p, 16_000).unwrap();
        assert!((back.len() as i64 - 16_000).abs() <= 2, "{}", back.len());
    }
}
