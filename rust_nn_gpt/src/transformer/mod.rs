use better_default::Default;
use std::{error::Error, fs, path::Path};

use tch::{
    nn::{
        self, Embedding, Init, LayerNorm, LayerNormConfig, Linear, LinearConfig, Module, Optimizer,
        OptimizerConfig,
    },
    Device, Kind, Tensor,
};

use crate::{dataset::Dataset, transformer::block::BlockConfig};

mod block;
mod mla_attention;
mod moe;

use block::Block;

// use mla_attention::MLAAttention;
// use moe::MixtureOfExperts;

pub fn xavier_uniform_init(input_dim: i64, output_dim: i64) -> Init {
    let limit = (6.0_f64 / (input_dim + output_dim) as f64).sqrt();

    return Init::Uniform {
        lo: -limit,
        up: limit,
    };
}

#[derive(Default)]
pub struct MLAConfig {
    #[default(4)]
    pub heads: i64,
    #[default(0.5)]
    pub mla_compression: f64,
}

#[derive(Default)]
pub struct MoEConfig {
    #[default(8)]
    pub num_experts: i64,
    #[default(2)]
    pub experts_per_token: i64,
    #[default(true)]
    pub use_load_balancing: bool,
    #[default(0.01)]
    pub load_balance_weight: f64,
}

pub struct Config<'d> {
    pub path: &'d str,
    pub device: &'d Device,
    pub vocab_size: i64,
    pub height: i64,
    pub layers: i64,
    pub context_window: i64,
    pub learning_rate: f64,
    pub mla_config: MLAConfig,
    pub moe_config: MoEConfig,
}

pub struct Transformer<'c> {
    pub model_save_path: String,
    token_embedding_table: Embedding,
    position_embedding_table: Embedding,
    transformer_blocks: Vec<Block>,
    final_layer_norm: LayerNorm,
    language_model_head: Linear,
    optimizer: Optimizer,
    variable_store: nn::VarStore,
    embedding_dim: i64,
    num_layers: i64,
    num_attention_heads: i64,
    num_kv_heads: i64,
    pub config: Config<'c>,
}

impl<'c> Transformer<'c> {
    pub fn new(config: Config<'c>) -> Result<Self, Box<dyn Error>> {
        let dropout = 0.01;
        let weight_decay = 1e-5;

        let var_store = nn::VarStore::new(*config.device);
        let root = var_store.root();

        let token_embedding_table = nn::embedding(
            &root / "token_emb",
            config.vocab_size,
            config.height,
            Default::default(),
        );

        let position_embedding_table = nn::embedding(
            &root / "pos_emb",
            config.context_window,
            config.height,
            Default::default(),
        );

        let blocks: Vec<Block> = (0..config.layers)
            .map(|layer_idx| {
                let block_config = BlockConfig {
                    var_store_path: &(&root / format!("block_{}", layer_idx)),
                    embedding_dim: config.height,
                    num_attention_heads: config.mla_config.heads,
                    num_kv_heads: config.mla_config.heads / 4, // For GQA (e.g., heads/4)
                    context_window: config.context_window,
                    dropout,
                    num_experts: config.moe_config.num_experts,
                    experts_per_token: config.moe_config.experts_per_token,
                    mla_compression: config.mla_config.mla_compression,
                    use_load_balancing: config.moe_config.use_load_balancing,
                };

                return Block::new(block_config);
            })
            .collect();

        let final_layer_norm = nn::layer_norm(
            &root / "ln_f",
            vec![config.height],
            LayerNormConfig::default(),
        );

        let language_model_head = nn::linear(
            &root / "lm_head",
            config.height,
            config.vocab_size,
            LinearConfig {
                ws_init: xavier_uniform_init(config.height, config.vocab_size),
                ..Default::default()
            },
        );

        let optimizer = nn::AdamW {
            wd: weight_decay,
            ..Default::default()
        }
        .build(&var_store, config.learning_rate)?;

        let total_param_count: i64 = var_store
            .variables()
            .values()
            .map(|tensor| return tensor.numel() as i64)
            .sum();

        println!("{:.6} M parameters", total_param_count as f64 / 1e6);

        return Ok(Self {
            model_save_path: config.path.to_string(),
            token_embedding_table,
            position_embedding_table,
            transformer_blocks: blocks,
            final_layer_norm,
            language_model_head,
            optimizer,
            variable_store: var_store,
            embedding_dim: config.height,
            num_layers: config.layers,
            num_attention_heads: config.mla_config.heads,
            num_kv_heads: config.mla_config.heads / 4,
            config,
        });
    }

    pub fn forward(
        &self,
        token_indices: &Tensor,
        targets: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let tensor_size = token_indices.size();
        let (batch_size, sequence_length) = (tensor_size[0], tensor_size[1]);

        let token_embeddings = self.token_embedding_table.forward(token_indices);
        let position_indices = Tensor::arange(sequence_length, (Kind::Int64, *self.config.device));
        let position_embeddings = self.position_embedding_table.forward(&position_indices);

        let mut hidden_states = token_embeddings + position_embeddings;
        let mut total_load_balance_loss = None;

        for block in &self.transformer_blocks {
            let (block_output, load_balance_loss) =
                block.forward_with_aux_loss(&hidden_states, train);
            hidden_states = block_output;

            if let Some(current_lb_loss) = load_balance_loss {
                total_load_balance_loss = Some(match total_load_balance_loss {
                    Some(accumulated_loss) => accumulated_loss + current_lb_loss,
                    None => current_lb_loss,
                });
            }
        }

        hidden_states = self.final_layer_norm.forward(&hidden_states);
        let logits = self.language_model_head.forward(&hidden_states);

        let loss = if let Some(target_tokens) = targets {
            let logits_reshaped = logits.view([batch_size * sequence_length, logits.size()[2]]);
            let targets_reshaped = target_tokens.view([batch_size * sequence_length]);
            let cross_entropy_loss = logits_reshaped.cross_entropy_for_logits(&targets_reshaped);

            if let Some(lb_loss) = total_load_balance_loss {
                Some(cross_entropy_loss + lb_loss * self.config.moe_config.load_balance_weight)
            } else {
                Some(cross_entropy_loss)
            }
        } else {
            None
        };

        return (logits, loss);
    }

    pub fn train(&mut self, data: &Dataset, min_loss: f64) -> Result<(), Box<dyn Error>> {
        let mut iteration = 0;
        let mut best_validation_loss = 10.0;

        loop {
            if iteration % 100 == 0 {
                let (train_loss, val_loss) = self.estimate_loss(data);
                if val_loss < best_validation_loss {
                    self.save(&self.model_save_path)?;
                    best_validation_loss = val_loss;
                }

                if val_loss < min_loss {
                    break;
                }

                println!(
                    "Epoch {}: train loss {:.4}, val loss {:.4}",
                    iteration, train_loss, val_loss
                );
            }

            let (batch_inputs, batch_targets) = data.get_batch_from(&data.training);
            let (_logits, loss) = self.forward(&batch_inputs, Some(&batch_targets), true);
            if let Some(loss) = loss {
                self.optimizer.zero_grad();
                loss.backward();
                self.optimizer.step();
            }

            iteration += 1;
        }

        return Ok(());
    }

    fn estimate_loss(&self, data: &Dataset) -> (f64, f64) {
        let training_losses = tch::no_grad(|| {
            let mut losses = Vec::new();
            let (batch_inputs, batch_targets) = data.get_batch_from(&data.training);
            let (_logits, loss) = self.forward(&batch_inputs, Some(&batch_targets), false);
            if let Some(loss) = loss {
                losses.push(loss.double_value(&[]));
            }
            return losses;
        });

        let validation_losses = tch::no_grad(|| {
            let mut losses = Vec::new();
            let (batch_inputs, batch_targets) = data.get_batch_from(&data.validation);
            let (_logits, loss) = self.forward(&batch_inputs, Some(&batch_targets), false);
            if let Some(loss) = loss {
                losses.push(loss.double_value(&[]));
            }
            return losses;
        });

        let train_mean_loss = training_losses.iter().sum::<f64>() / training_losses.len() as f64;
        let val_mean_loss = validation_losses.iter().sum::<f64>() / validation_losses.len() as f64;

        return (train_mean_loss, val_mean_loss);
    }

    pub fn generate(&self, token_indices: &Tensor, max_new_tokens: i64) -> Tensor {
        return tch::no_grad(|| {
            let mut current_indices = token_indices.shallow_clone();
            for _ in 0..max_new_tokens {
                let conditioned_indices = if current_indices.size()[1] > self.config.context_window
                {
                    current_indices.narrow(
                        1,
                        current_indices.size()[1] - self.config.context_window,
                        self.config.context_window,
                    )
                } else {
                    current_indices.shallow_clone()
                };

                let (logits, _loss) = self.forward(&conditioned_indices, None, false);
                let last_logits = logits.select(1, logits.size()[1] - 1);
                let probabilities = last_logits.softmax(-1, Kind::Float);
                let next_token_index = probabilities.multinomial(1, false);
                current_indices = Tensor::cat(&[current_indices, next_token_index], 1);
            }
            return current_indices;
        });
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        use std::fs;
        use std::path::Path;

        let dir_path = Path::new(path);
        fs::create_dir_all(dir_path)?;

        // Save model weights using tch's native format (PyTorch compatible)
        let model_path = dir_path.join("model.safetensors");
        self.variable_store.save(&model_path)?;

        // Save config as JSON (Hugging Face compatible)
        let config = serde_json::json!({
            "model_type": "rust_nn_gpt",
            "vocab_size": self.config.vocab_size,
            "hidden_size": self.embedding_dim,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_kv_heads,
            "max_position_embeddings": self.config.context_window,
            "num_experts": self.config.moe_config.num_experts,
            "num_experts_per_tok": self.config.moe_config.experts_per_token,
            "mla_compression": self.config.mla_config.mla_compression,
            "use_load_balancing": self.config.moe_config.use_load_balancing,
            "load_balance_weight": self.config.moe_config.load_balance_weight,
            "architectures": ["RustNNGPT"]
        });

        let config_path = dir_path.join("config.json");
        fs::write(config_path, serde_json::to_string_pretty(&config)?)?;

        println!("Model saved to {}", path);
        return Ok(());
    }

    pub fn load(path: &'c str, device: &'c Device) -> Result<Self, Box<dyn Error>> {
        let dir_path = Path::new(path);

        // Load config
        let config_path = dir_path.join("config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_str)?;

        // Extract config values
        let vocab_size = config["vocab_size"]
            .as_i64()
            .ok_or("Missing or invalid vocab_size in config")?;
        let height = config["hidden_size"]
            .as_i64()
            .ok_or("Missing or invalid hidden_size in config")?;
        let layers = config["num_hidden_layers"]
            .as_i64()
            .ok_or("Missing or invalid num_hidden_layers in config")?;
        let heads = config["num_attention_heads"]
            .as_i64()
            .ok_or("Missing or invalid num_attention_heads in config")?;
        let _kv_heads = config["num_key_value_heads"]
            .as_i64()
            .ok_or("Missing or invalid num_key_value_heads in config")?;
        let block_size = config["max_position_embeddings"]
            .as_i64()
            .ok_or("Missing or invalid max_position_embeddings in config")?;
        let num_experts = config["num_experts"]
            .as_i64()
            .ok_or("Missing or invalid num_experts in config")?;
        let experts_per_token = config["num_experts_per_tok"]
            .as_i64()
            .ok_or("Missing or invalid num_experts_per_tok in config")?;
        let mla_compression = config["mla_compression"]
            .as_f64()
            .ok_or("Missing or invalid mla_compression in config")?;
        let use_load_balancing = config["use_load_balancing"]
            .as_bool()
            .ok_or("Missing or invalid use_load_balancing in config")?;
        let load_balance_weight = config["load_balance_weight"]
            .as_f64()
            .ok_or("Missing or invalid load_balance_weight in config")?;

        // Create model with config
        let model_config = Config {
            path,
            device,
            vocab_size,
            height,
            layers,
            context_window: block_size,
            learning_rate: 1e-3,
            mla_config: MLAConfig {
                heads,
                mla_compression,
            },
            moe_config: MoEConfig {
                num_experts,
                experts_per_token,
                use_load_balancing,
                load_balance_weight,
            },
        };

        let mut transformer_model = Self::new(model_config)?;

        let model_path = dir_path.join("model.safetensors");
        transformer_model.variable_store.load(&model_path)?;

        println!("Model loaded from {}", path);
        return Ok(transformer_model);
    }
}
