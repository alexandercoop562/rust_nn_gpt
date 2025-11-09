#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::implicit_return)]
#![allow(clippy::needless_return)]
#![deny(clippy::assigning_clones)]
#![deny(clippy::implicit_clone)]
#![deny(unused_must_use)]

use std::thread::sleep;
use std::time::Duration;
use std::{error::Error, fs::read_to_string};

use rust_nn_gpt::tokenizer::LLMTokenizer;
use rust_nn_gpt::{
    dataset::Dataset,
    // tokenizer::Tokenizer,
    transformer::{Config, MLAConfig, MoEConfig, Transformer},
    Device,
    Tensor,
};

fn main() -> Result<(), Box<dyn Error>> {
    let batch_size = 16;
    let dataset_text = read_to_string("dataset/input.txt")?;

    // let tokenizer = Tokenizer::new(&dataset_text);
    let tokenizer = LLMTokenizer::new()?;

    let encoded_tokens = tokenizer.encode(&dataset_text);

    let device = Device::cuda_if_available();

    let model_config = Config {
        path: "model/",
        device: &device,
        vocab_size: tokenizer.vocab_size,
        height: 64,
        layers: 8,
        context_window: 128,
        learning_rate: 1e-3,
        mla_config: MLAConfig::default(),
        moe_config: MoEConfig::default(),
    };

    let training_data = Dataset::default().llm_dataset(
        Tensor::from_slice(&encoded_tokens),
        model_config.context_window,
        batch_size,
        device,
    );

    let mut transformer_model = match Transformer::load("./model", &device) {
        Ok(loaded_model) => loaded_model,
        Err(load_error) => {
            println!("Could not load model, starting fresh: {}", load_error);
            Transformer::new(model_config)?
        }
    };

    transformer_model.train(&training_data, 5.0)?;

    let mut generation_context = Tensor::from_slice(&tokenizer.encode("KING COOP:\n"))
        .to(device)
        .unsqueeze(0);

    loop {
        generation_context = transformer_model.generate(&generation_context, 1);

        if generation_context.size()[1] >= 1000 {
            let max_context_length = 1000;
            let context_start_position = generation_context.size()[1] - max_context_length;
            generation_context =
                generation_context.narrow(1, context_start_position, max_context_length);
        }

        let generated_token_ids: Vec<i64> = generation_context.get(0).iter::<i64>()?.collect();

        print!("\x1B[2J\x1B[1;1H");
        // println!("\n{}", tokenizer.decode(&generated_token_ids));
        println!("\n{}", tokenizer.decode(&generated_token_ids)?);
        sleep(Duration::from_millis(50));
    }

    #[allow(unreachable_code)]
    return Ok(());
}
