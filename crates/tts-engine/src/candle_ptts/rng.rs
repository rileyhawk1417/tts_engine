//! Noise sources for the flow-matching sampler — faithful port of the
//! ptts-0.6.0 oracle's `Rng` trait in `flow_lm.rs`.
//!
//! NormalRng: rand 0.9 StdRng + Normal(0, sqrt(temperature)) — the same crate
//! versions as the oracle (rand 0.9.0, rand_distr 0.5.0), so seeded streams
//! match. ReplayRng: cycles a fixed sequence — how parity across
//! implementations is proven (`--replay` flows).

use rand::SeedableRng;

pub trait Rng {
    fn sample(&mut self) -> f32;
}

impl Rng for Box<dyn Rng + Send> {
    fn sample(&mut self) -> f32 {
        (**self).sample()
    }
}

pub struct NormalRng {
    inner: rand::rngs::StdRng,
    distr: rand_distr::Normal<f32>,
}

impl NormalRng {
    pub fn new(temperature: f32, seed: u64) -> Result<Self, String> {
        let distr = rand_distr::Normal::new(0f32, temperature.sqrt())
            .map_err(|e| format!("invalid normal distribution: {e}"))?;
        Ok(Self {
            inner: rand::rngs::StdRng::seed_from_u64(seed),
            distr,
        })
    }
}

impl Rng for NormalRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng as _;
        self.inner.sample(self.distr)
    }
}

/// Replays a fixed sequence of values, cycling when exhausted.
pub struct ReplayRng {
    values: Vec<f32>,
    index: usize,
}

impl ReplayRng {
    pub fn new(values: Vec<f32>) -> Result<Self, String> {
        if values.is_empty() {
            return Err("ReplayRng needs at least one value".to_string());
        }
        Ok(Self { values, index: 0 })
    }
}

impl Rng for ReplayRng {
    fn sample(&mut self) -> f32 {
        let value = self.values[self.index % self.values.len()];
        self.index += 1;
        value
    }
}
