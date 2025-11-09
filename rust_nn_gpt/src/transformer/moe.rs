use tch::{
    nn::{linear, Linear, LinearConfig, Module, Path},
    Kind, Tensor,
};

use super::xavier_uniform_init;

/// Single Expert in the Mixture of Experts
struct Expert {
    first_layer: Linear,
    second_layer: Linear,
}

impl Expert {
    fn new(var_store_path: &Path, embedding_dim: i64, hidden_dim: i64) -> Self {
        let first_layer = linear(
            var_store_path / "fc1",
            embedding_dim,
            hidden_dim,
            LinearConfig {
                ws_init: xavier_uniform_init(embedding_dim, hidden_dim),
                ..Default::default()
            },
        );

        let second_layer = linear(
            var_store_path / "fc2",
            hidden_dim,
            embedding_dim,
            LinearConfig {
                ws_init: xavier_uniform_init(hidden_dim, embedding_dim),
                ..Default::default()
            },
        );

        return Self { first_layer, second_layer };
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let hidden = self.first_layer.forward(input);
        let activated = hidden.relu();
        return self.second_layer.forward(&activated);
    }
}

/// Mixture of Experts (MoE) layer
/// Uses top-k routing where each token is processed by k experts
pub struct MixtureOfExperts {
    expert_networks: Vec<Expert>,
    router_network: Linear,
    num_experts: i64,
    num_active_experts: i64,
    dropout: f64,
}

impl MixtureOfExperts {
    pub fn new(
        var_store_path: &Path,
        embedding_dim: i64,
        num_experts: i64,
        top_k: i64,
        expert_hidden_dim: i64, // Hidden dimension per expert (e.g., 4 * embedding_dim)
        dropout: f64,
    ) -> Self {
        assert!(top_k <= num_experts, "top_k must be <= num_experts");

        // Create experts
        let experts: Vec<Expert> = (0..num_experts)
            .map(|expert_idx| return Expert::new(&(var_store_path / format!("expert_{}", expert_idx)), embedding_dim, expert_hidden_dim))
            .collect();

        // Router network to select experts
        let router = linear(
            var_store_path / "router",
            embedding_dim,
            num_experts,
            LinearConfig {
                ws_init: xavier_uniform_init(embedding_dim, num_experts),
                ..Default::default()
            },
        );

        return Self {
            expert_networks: experts,
            router_network: router,
            num_experts,
            num_active_experts: top_k,
            dropout,
        };
    }

    pub fn forward(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        let tensor_size = hidden_states.size();
        let (batch_size, sequence_length, embedding_dim) = (tensor_size[0], tensor_size[1], tensor_size[2]);

        // Flatten batch and sequence dimensions for routing
        let flattened_states = hidden_states.view([batch_size * sequence_length, embedding_dim]); // (B*T, C)

        // Compute router logits
        let router_logits = self.router_network.forward(&flattened_states); // (B*T, num_experts)

        // Get top-k experts and their weights
        let (routing_weights, selected_expert_indices) = router_logits.topk(self.num_active_experts, -1, true, true);
        let routing_weights = routing_weights.softmax(-1, Kind::Float); // (B*T, top_k)

        // Initialize output
        let mut output = Tensor::zeros([batch_size * sequence_length, embedding_dim], (hidden_states.kind(), hidden_states.device()));

        // Process each token through its selected experts
        for k_idx in 0..self.num_active_experts {
            let expert_index = selected_expert_indices.select(1, k_idx); // (B*T,)
            let weight = routing_weights.select(1, k_idx).unsqueeze(-1); // (B*T, 1)

            // Group tokens by expert for efficient batch processing
            for expert_id in 0..self.num_experts {
                let token_mask = expert_index.eq(expert_id); // (B*T,)
                let num_selected_tokens = token_mask.sum(Kind::Int64).int64_value(&[]);

                if num_selected_tokens > 0 {
                    // Get tokens for this expert
                    let token_indices = token_mask.nonzero().squeeze_dim(1);
                    let expert_input = flattened_states.index_select(0, &token_indices); // (num_selected_tokens, C)

                    // Process through expert
                    let expert_output = self.expert_networks[expert_id as usize].forward(&expert_input);

                    // Get weights for these tokens
                    let expert_weight = weight.index_select(0, &token_indices); // (num_selected_tokens, 1)

                    // Weighted contribution
                    let weighted_output = expert_output * expert_weight;

                    // Scatter back to output
                    output = output.index_add(0, &token_indices, &weighted_output);
                }
            }
        }

        // Reshape back to original dimensions
        let output = output.view([batch_size, sequence_length, embedding_dim]);

        if train {
            return output.dropout(self.dropout, train);
        } else {
            return output;
        }
    }

    /// Alternative forward with load balancing loss
    /// Returns (output, load_balancing_loss)
    pub fn forward_with_load_balancing(&self, hidden_states: &Tensor, train: bool) -> (Tensor, Tensor) {
        let tensor_size = hidden_states.size();
        let (batch_size, sequence_length, embedding_dim) = (tensor_size[0], tensor_size[1], tensor_size[2]);

        let flattened_states = hidden_states.view([batch_size * sequence_length, embedding_dim]);
        let router_logits = self.router_network.forward(&flattened_states);

        // Compute load balancing loss to encourage uniform expert usage
        let router_probabilities = router_logits.softmax(-1, Kind::Float);
        let average_expert_usage = router_probabilities.mean_dim(&[0i64][..], false, Kind::Float); // Mean usage per expert

        // Load balancing loss: variance of expert usage (we want it to be uniform)
        let target_usage = 1.0 / self.num_experts as f64;
        let load_balance_loss = ((average_expert_usage - target_usage).pow_tensor_scalar(2))
            .sum(Kind::Float)
            * self.num_experts as f64;

        // Get top-k experts and their weights
        let (routing_weights, selected_expert_indices) = router_logits.topk(self.num_active_experts, -1, true, true);
        let routing_weights = routing_weights.softmax(-1, Kind::Float);

        let mut output = Tensor::zeros([batch_size * sequence_length, embedding_dim], (hidden_states.kind(), hidden_states.device()));

        for k_idx in 0..self.num_active_experts {
            let expert_index = selected_expert_indices.select(1, k_idx);
            let weight = routing_weights.select(1, k_idx).unsqueeze(-1);

            for expert_id in 0..self.num_experts {
                let token_mask = expert_index.eq(expert_id);
                let num_selected_tokens = token_mask.sum(Kind::Int64).int64_value(&[]);

                if num_selected_tokens > 0 {
                    let token_indices = token_mask.nonzero().squeeze_dim(1);
                    let expert_input = flattened_states.index_select(0, &token_indices);
                    let expert_output = self.expert_networks[expert_id as usize].forward(&expert_input);
                    let expert_weight = weight.index_select(0, &token_indices);
                    let weighted_output = expert_output * expert_weight;
                    output = output.index_add(0, &token_indices, &weighted_output);
                }
            }
        }

        let output = output.view([batch_size, sequence_length, embedding_dim]);

        let output = if train {
            output.dropout(self.dropout, train)
        } else {
            output
        };

        return (output, load_balance_loss);
    }
}
