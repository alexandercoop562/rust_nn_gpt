use tch::{
    nn::{linear, Linear, LinearConfig, Module, Path},
    Kind, Tensor,
};

use super::xavier_uniform_init;

/// Multi-Latent Attention (MLA) with Grouped Query Attention (GQA)
/// Compresses KV cache through low-rank projections while using GQA to share KV across query groups
pub struct MLAAttention {
    // Low-rank compression for efficient KV cache
    compression_down: Linear, // embedding_dim -> latent_dim (compress)
    compression_up_keys: Linear, // latent_dim -> kv_dim (decompress for keys)
    compression_up_values: Linear, // latent_dim -> kv_dim (decompress for values)

    // Query projection (full rank)
    query_projection: Linear, // embedding_dim -> embedding_dim

    // Output projection
    output_projection: Linear, // embedding_dim -> embedding_dim

    // Attention mask
    causal_mask: Tensor,

    // Configuration
    num_attention_heads: i64,
    num_kv_heads: i64, // Number of KV heads (for GQA)
    head_dim: i64,
    dropout: f64,
}

impl MLAAttention {
    pub fn new(
        var_store_path: &Path,
        embedding_dim: i64,
        num_attention_heads: i64,
        num_kv_heads: i64, // For GQA: typically num_attention_heads/4 or num_attention_heads/8
        block_size: i64,
        dropout: f64,
        compression_ratio: f64, // e.g., 0.5 means latent_dim = embedding_dim/2
    ) -> Self {
        assert_eq!(embedding_dim % num_attention_heads, 0, "embedding_dim must be divisible by num_attention_heads");
        assert_eq!(
            num_attention_heads % num_kv_heads,
            0,
            "num_attention_heads must be divisible by num_kv_heads"
        );

        let head_dim = embedding_dim / num_attention_heads;
        let kv_dim = num_kv_heads * head_dim;
        let latent_dim = (embedding_dim as f64 * compression_ratio) as i64;

        // Low-rank compression for KV
        let compression_down = linear(
            var_store_path / "c_down",
            embedding_dim,
            latent_dim,
            LinearConfig {
                ws_init: xavier_uniform_init(embedding_dim, latent_dim),
                bias: false,
                ..Default::default()
            },
        );

        let compression_up_keys = linear(
            var_store_path / "c_up_k",
            latent_dim,
            kv_dim,
            LinearConfig {
                ws_init: xavier_uniform_init(latent_dim, kv_dim),
                bias: false,
                ..Default::default()
            },
        );

        let compression_up_values = linear(
            var_store_path / "c_up_v",
            latent_dim,
            kv_dim,
            LinearConfig {
                ws_init: xavier_uniform_init(latent_dim, kv_dim),
                bias: false,
                ..Default::default()
            },
        );

        // Query projection (full rank)
        let query_projection = linear(
            var_store_path / "q_proj",
            embedding_dim,
            embedding_dim,
            LinearConfig {
                ws_init: xavier_uniform_init(embedding_dim, embedding_dim),
                bias: false,
                ..Default::default()
            },
        );

        // Output projection
        let output_projection = linear(
            var_store_path / "o_proj",
            embedding_dim,
            embedding_dim,
            LinearConfig {
                ws_init: xavier_uniform_init(embedding_dim, embedding_dim),
                ..Default::default()
            },
        );

        let causal_mask = Tensor::ones([block_size, block_size], (Kind::Float, var_store_path.device())).tril(0);

        return Self {
            compression_down,
            compression_up_keys,
            compression_up_values,
            query_projection,
            output_projection,
            causal_mask,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            dropout,
        };
    }

    pub fn forward(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        let tensor_size = hidden_states.size();
        let (batch_size, sequence_length, _embedding_dim) = (tensor_size[0], tensor_size[1], tensor_size[2]);

        // Compress input to latent representation (efficient KV cache)
        let latent_representation = self.compression_down.forward(hidden_states); // (B, T, latent_dim)

        // Decompress to keys and values
        let keys = self.compression_up_keys.forward(&latent_representation); // (B, T, kv_dim)
        let values = self.compression_up_values.forward(&latent_representation); // (B, T, kv_dim)

        // Full query projection
        let queries = self.query_projection.forward(hidden_states); // (B, T, embedding_dim)

        // Reshape for multi-head attention
        // Q: (B, T, num_attention_heads, head_dim)
        let queries = queries.view([batch_size, sequence_length, self.num_attention_heads, self.head_dim]).transpose(1, 2); // (B, num_attention_heads, T, head_dim)

        // K, V: (B, T, num_kv_heads, head_dim)
        let keys = keys
            .view([batch_size, sequence_length, self.num_kv_heads, self.head_dim])
            .transpose(1, 2); // (B, num_kv_heads, T, head_dim)
        let values = values
            .view([batch_size, sequence_length, self.num_kv_heads, self.head_dim])
            .transpose(1, 2); // (B, num_kv_heads, T, head_dim)

        // Repeat KV heads for GQA
        let head_repetitions = self.num_attention_heads / self.num_kv_heads;
        let keys = if head_repetitions > 1 {
            keys.repeat([1, head_repetitions, 1, 1])
        } else {
            keys
        };
        let values = if head_repetitions > 1 {
            values.repeat([1, head_repetitions, 1, 1])
        } else {
            values
        };

        // Attention scores: (B, num_attention_heads, T, T)
        let mut attention_scores = queries.matmul(&keys.transpose(-2, -1)) * (self.head_dim as f64).powf(-0.5);

        // Apply causal mask
        attention_scores = attention_scores.masked_fill(
            &self.causal_mask.narrow(0, 0, sequence_length).narrow(1, 0, sequence_length).eq(0.0),
            f64::NEG_INFINITY,
        );

        attention_scores = attention_scores.softmax(-1, Kind::Float);

        if train {
            attention_scores = attention_scores.dropout(self.dropout, train);
        }

        // Apply attention to values: (B, num_attention_heads, T, head_dim)
        let attention_output = attention_scores.matmul(&values);

        // Reshape back: (B, T, embedding_dim)
        let attention_output = attention_output
            .transpose(1, 2)
            .contiguous()
            .view([batch_size, sequence_length, self.num_attention_heads * self.head_dim]);

        // Output projection
        let output = self.output_projection.forward(&attention_output);

        if train {
            return output.dropout(self.dropout, train);
        } else {
            return output;
        }
    }
}
