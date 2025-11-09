# Rust NN GPT

A General Purpose Transformer library built in rust.

Implements a Mixture of Experts (MoE) transformer with Multi-Latent Attention (MLA) and Grouped Query Attention (GQA).

![Demo](demo.gif)

# Usage

To use the library, you will import it from GitHub:

```toml
[dependencies]
rust_nn_gpt = { git = "https://github.com/alexandercoop562/rust_nn_gpt.git" }
```

An example of how to use the library is here: [main.rs](tiny_shakespeare/src/main.rs)

# Test it out!

You can run this when you clone the repo for a infinite scroll of shakespearean text.

```bash
cargo run -p tiny_shakespeare
```
