use tch::{
    nn::{layer_norm, LayerNorm, LayerNormConfig, Module, Path},
    Tensor,
};

use super::{mla_attention::MLAAttention, moe::MixtureOfExperts};

pub struct Block {
    self_attention: MLAAttention,
    mixture_of_experts: MixtureOfExperts,
    attention_layer_norm: LayerNorm,
    moe_layer_norm: LayerNorm,
    use_load_balancing: bool,
}

pub struct BlockConfig<'p> {
    pub var_store_path: &'p Path<'p>,
    pub embedding_dim: i64,
    pub num_attention_heads: i64,
    pub num_kv_heads: i64, // For GQA
    pub context_window: i64,
    pub dropout: f64,
    pub num_experts: i64,
    pub experts_per_token: i64,
    pub mla_compression: f64,
    pub use_load_balancing: bool,
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    pub fn new(config: BlockConfig) -> Self {
        let self_attention = MLAAttention::new(
            &(config.var_store_path / "sa"),
            config.embedding_dim,
            config.num_attention_heads,
            config.num_kv_heads,
            config.context_window,
            config.dropout,
            config.mla_compression,
        );

        let expert_hidden_dim = 4 * config.embedding_dim;
        let mixture_of_experts = MixtureOfExperts::new(
            &(config.var_store_path / "moe"),
            config.embedding_dim,
            config.num_experts,
            config.experts_per_token,
            expert_hidden_dim,
            config.dropout,
        );

        let attention_layer_norm = layer_norm(
            config.var_store_path / "ln1",
            vec![config.embedding_dim],
            LayerNormConfig::default(),
        );
        let moe_layer_norm = layer_norm(
            config.var_store_path / "ln2",
            vec![config.embedding_dim],
            LayerNormConfig::default(),
        );

        return Self {
            self_attention,
            mixture_of_experts,
            attention_layer_norm,
            moe_layer_norm,
            use_load_balancing: config.use_load_balancing,
        };
    }

    pub fn forward_with_aux_loss(
        &self,
        hidden_states: &Tensor,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let normalized_states_1 = self.attention_layer_norm.forward(hidden_states);
        let attention_output =
            hidden_states + self.self_attention.forward(&normalized_states_1, train);
        let normalized_states_2 = self.moe_layer_norm.forward(&attention_output);

        if train && self.use_load_balancing {
            let (moe_output, load_balance_loss) = self
                .mixture_of_experts
                .forward_with_load_balancing(&normalized_states_2, train);

            return (attention_output + moe_output, Some(load_balance_loss));
        } else {
            return (
                attention_output + self.mixture_of_experts.forward(&normalized_states_2, train),
                None,
            );
        }
    }
}
