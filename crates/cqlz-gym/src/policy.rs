//! The policy network: a small Transformer encoder over the candidate tables of one design problem.

use anyhow::{Context, Result};
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use std::path::Path;

/// Number of per-candidate features (see `rl::Env::features`).
pub const N_FEAT: usize = 8;

#[derive(Config, Debug)]
pub struct PolicyConfig {
    #[config(default = 64)]
    pub d_model: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 128)]
    pub d_ff: usize,
    #[config(default = 2)]
    pub n_layers: usize,
}

#[derive(Module, Debug)]
pub struct PolicyNet {
    inp: Linear,
    enc: TransformerEncoder,
    hid: Linear,
    out: Linear,
}

impl PolicyConfig {
    pub fn init(&self, device: &Device) -> PolicyNet {
        PolicyNet {
            inp: LinearConfig::new(N_FEAT, self.d_model).init(device),
            enc: TransformerEncoderConfig::new(
                self.d_model,
                self.d_ff,
                self.n_heads,
                self.n_layers,
            )
            .with_dropout(0.0)
            .with_norm_first(true)
            .init(device),
            hid: LinearConfig::new(self.d_model, self.d_model).init(device),
            out: LinearConfig::new(self.d_model, 1).init(device),
        }
    }
}

impl PolicyNet {
    /// `feats`: `[batch, candidates + 1, N_FEAT]` (last slot is STOP), `pad`: true where padded.
    /// Returns unmasked logits `[batch, candidates + 1]`.
    pub fn forward(&self, feats: Tensor<3>, pad: Tensor<2, Bool>) -> Tensor<2> {
        let x = self.inp.forward(feats);
        let x = self
            .enc
            .forward(TransformerEncoderInput::new(x).mask_pad(pad));
        let x = burn::tensor::activation::gelu(self.hid.forward(x));
        self.out.forward(x).squeeze_dim::<2>(2)
    }
}

pub const WEIGHTS_FILE: &str = "model.safetensors";
pub const CONFIG_FILE: &str = "config.json";

pub fn save(model: &PolicyNet, config: &PolicyConfig, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    config
        .save(dir.join(CONFIG_FILE))
        .context("writing config.json")?;
    let mut store = SafetensorsStore::from_file(dir.join(WEIGHTS_FILE))
        .metadata("format", "cqlz-gym/policy")
        .overwrite(true);
    model
        .save_into(&mut store)
        .map_err(|e| anyhow::anyhow!("saving safetensors: {e:?}"))?;
    Ok(())
}

pub fn load(dir: &Path, device: &Device) -> Result<PolicyNet> {
    let config = PolicyConfig::load(dir.join(CONFIG_FILE)).context("reading config.json")?;
    let mut model = config.init(device);
    let mut store = SafetensorsStore::from_file(dir.join(WEIGHTS_FILE));
    let r = model
        .load_from(&mut store)
        .map_err(|e| anyhow::anyhow!("loading safetensors: {e:?}"))?;
    if !r.errors.is_empty() || !r.missing.is_empty() {
        anyhow::bail!("weights do not match the model: {r}");
    }
    Ok(model)
}
