use std::error::Error;
use std::fs::{self, create_dir_all};
use std::path::Path;

use tch::{Device, Kind, Tensor};

use better_default::Default;

#[derive(Default)]
pub struct Dataset {
    pub training: (Tensor, Tensor),
    pub validation: (Tensor, Tensor),
    block_size: i64,
    batch_size: i64,
    #[default(Device::cuda_if_available())]
    device: Device,
}

impl Dataset {
    pub fn new(
        training: (Tensor, Tensor),
        validation: (Tensor, Tensor),
        block_size: i64,
        batch_size: i64,
        device: Device,
    ) -> Self {
        return Self {
            training: (training.0.to(device), training.1.to(device)),
            validation: (validation.0.to(device), validation.1.to(device)),
            block_size,
            batch_size,
            device,
        };
    }

    pub fn llm_dataset(
        &self,
        encoded_data: Tensor,
        block_size: i64,
        batch_size: i64,
        device: Device,
    ) -> Self {
        let train_split_index = (0.9 * encoded_data.size()[0] as f64) as i64;
        let training_inputs = encoded_data.narrow(0, 0, train_split_index).to(device);
        let training_targets = encoded_data.narrow(0, 1, train_split_index).to(device);
        let validation_inputs = encoded_data.narrow(0, train_split_index, encoded_data.size()[0] - train_split_index).to(device);
        let validation_targets = encoded_data.narrow(0, train_split_index + 1, encoded_data.size()[0] - train_split_index - 1).to(device);

        return Self {
            training: (training_inputs, training_targets),
            validation: (validation_inputs, validation_targets),
            block_size,
            batch_size,
            device,
        };
    }

    pub fn get_batch_from(&self, data: &(Tensor, Tensor)) -> (Tensor, Tensor) {
        let max_start_index = data.0.size()[0] - self.block_size;
        let random_indices = Tensor::randint(max_start_index, [self.batch_size], (Kind::Int64, self.device));

        let inputs = Tensor::stack(
            &(0..self.batch_size)
                .map(|batch_idx| {
                    let start_index = random_indices.int64_value(&[batch_idx]);
                    return data.0.narrow(0, start_index, self.block_size);
                })
                .collect::<Vec<_>>(),
            0,
        );

        let targets = Tensor::stack(
            &(0..self.batch_size)
                .map(|batch_idx| {
                    let start_index = random_indices.int64_value(&[batch_idx]);
                    return data.1.narrow(0, start_index, self.block_size);
                })
                .collect::<Vec<_>>(),
            0,
        );

        return (inputs, targets);
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let dir_path = Path::new(path);
        create_dir_all(dir_path)?;

        Tensor::save(&self.training.0, dir_path.join("training_inputs.pt"))?;
        Tensor::save(&self.training.1, dir_path.join("training_targets.pt"))?;
        Tensor::save(&self.validation.0, dir_path.join("validation_inputs.pt"))?;
        Tensor::save(&self.validation.1, dir_path.join("validation_targets.pt"))?;

        let config = serde_json::json!({
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "training_size": self.training.0.size()[0],
            "validation_size": self.validation.0.size()[0],
        });

        let config_path = dir_path.join("dataset_config.json");
        fs::write(config_path, serde_json::to_string_pretty(&config)?)?;

        println!("Dataset saved to {}", path);
        return Ok(());
    }

    pub fn load(path: &str, device: Device) -> Result<Self, Box<dyn std::error::Error>> {
        let dir_path = Path::new(path);

        let config_path = dir_path.join("dataset_config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_str)?;

        let block_size = config["block_size"]
            .as_i64()
            .ok_or("Missing or invalid block_size in config")?;
        let batch_size = config["batch_size"]
            .as_i64()
            .ok_or("Missing or invalid batch_size in config")?;

        let training_inputs = Tensor::load(dir_path.join("training_inputs.pt"))?.to(device);
        let training_targets = Tensor::load(dir_path.join("training_targets.pt"))?.to(device);
        let validation_inputs = Tensor::load(dir_path.join("validation_inputs.pt"))?.to(device);
        let validation_targets = Tensor::load(dir_path.join("validation_targets.pt"))?.to(device);

        println!("Dataset loaded from {}", path);

        return Ok(Self {
            training: (training_inputs, training_targets),
            validation: (validation_inputs, validation_targets),
            block_size,
            batch_size,
            device,
        });
    }
}
