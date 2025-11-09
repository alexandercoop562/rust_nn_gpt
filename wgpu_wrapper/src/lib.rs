#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::implicit_return)]
#![allow(clippy::needless_return)]
#![deny(clippy::assigning_clones)]
#![deny(clippy::implicit_clone)]
#![deny(unused_must_use)]

use std::{collections::HashMap, error::Error};

use wgpu::{
    Backends, BindGroup, BindGroupEntry, Buffer, BufferDescriptor, BufferUsages,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor,
    ExperimentalFeatures, Features, Instance, InstanceDescriptor, InstanceFlags, Limits, MapMode,
    MemoryHints, PipelineCompilationOptions, PowerPreference, Queue, RequestAdapterOptions,
    ShaderModuleDescriptor, Trace,
};

pub struct GpuInstance {
    pub instance: Instance,
    pub device: Device,
    pub queue: Queue,
}

impl GpuInstance {
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            flags: InstanceFlags::from_build_config(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: None,
                required_features: Features::empty(),
                required_limits: Limits::default(),
                trace: Trace::Off,
                memory_hints: MemoryHints::Performance,
                experimental_features: ExperimentalFeatures::disabled(),
            })
            .await?;

        return Ok(Self {
            instance,
            device,
            queue,
        });
    }
}

#[derive(Eq, PartialEq, Hash)]
pub enum BufferTypes {
    Input,
    Staging,
    Output,
    Other,
}

fn to_memory_size<T>(len: usize) -> u64 {
    return (len * std::mem::size_of::<T>()) as u64;
}

pub struct Shader<'i> {
    gpu: &'i GpuInstance,
    label: String,
    compute_pipeline: ComputePipeline,
    bind_group: Option<BindGroup>,
    buffers: Vec<Buffer>,
    buffer_map: HashMap<BufferTypes, usize>,
    output_length: usize,
}

impl<'i> Shader<'i> {
    pub fn new(shader_descriptor: ShaderModuleDescriptor, gpu: &'i GpuInstance) -> Self {
        let label = shader_descriptor.label.unwrap_or("Shader").to_string();
        let shader_module = gpu.device.create_shader_module(shader_descriptor);

        let compute_pipeline = gpu
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(&format!("({}) Compute Pipeline", label)),
                layout: None,
                module: &shader_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
            });

        return Self {
            gpu,
            label,
            bind_group: None,
            compute_pipeline,
            buffers: vec![],
            buffer_map: HashMap::new(),
            output_length: 0,
        };
    }

    pub fn add_buffer<T>(
        &mut self,
        buffer_label: &str,
        buffer_type: BufferTypes,
        buffer_size: usize,
    ) -> &mut Self {
        let usage = match &buffer_type {
            BufferTypes::Input => BufferUsages::STORAGE | BufferUsages::COPY_DST,
            BufferTypes::Staging => BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            BufferTypes::Output => BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferTypes::Other => BufferUsages::STORAGE,
        };

        if buffer_type == BufferTypes::Staging {
            self.output_length = buffer_size
        }

        self.buffer_map
            .entry(buffer_type)
            .or_insert(self.buffers.len());

        self.buffers
            .push(self.gpu.device.create_buffer(&BufferDescriptor {
                label: Some(buffer_label),
                size: to_memory_size::<T>(buffer_size),
                usage,
                mapped_at_creation: false,
            }));

        return self;
    }

    pub fn add_bind_group(&mut self) -> Result<&mut Self, Box<dyn Error>> {
        let bind_group_layout = self.compute_pipeline.get_bind_group_layout(0);

        let mut bind_group_entries: Vec<BindGroupEntry> = vec![];

        let [Some(staging_buffer_index)] = self.buffer_map.get_disjoint_mut([&BufferTypes::Staging])
        else {
            return Err("Failed to gex index of staging buffer".into());
        };

        for (buffer_index, buffer) in self.buffers.iter().enumerate() {
            if buffer_index == *staging_buffer_index {
                continue;
            }

            bind_group_entries.push(BindGroupEntry {
                binding: buffer_index as u32,
                resource: buffer.as_entire_binding(),
            });
        }

        self.bind_group = Some(
            self.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("({}) Bind Group", self.label)),
                    layout: &bind_group_layout,
                    entries: &bind_group_entries,
                }),
        );

        return Ok(self);
    }

    pub fn set_input<T: bytemuck::Pod>(&mut self, input_data: &[T]) -> Result<(), Box<dyn Error>> {
        let [Some(input_buffer_index)] = self.buffer_map.get_disjoint_mut([&BufferTypes::Input]) else {
            return Err("Failed to gex index of input buffer".into());
        };

        self.gpu
            .queue
            .write_buffer(&self.buffers[*input_buffer_index], 0, bytemuck::cast_slice(input_data));

        return Ok(());
    }

    pub async fn run<T: bytemuck::Pod>(&mut self) -> Result<Vec<T>, Box<dyn Error>> {
        let bind_group = match &self.bind_group {
            Some(bg) => bg,
            None => {
                return Err("Bind Group was not created.".into());
            }
        };

        let mut command_encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("({}) Command Encoder", self.label)),
                });

        {
            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(&format!("({}) Compute Pass", self.label)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(self.output_length as u32, 1, 1);
        }

        let [Some(output_buffer_index), Some(staging_buffer_index)] = self
            .buffer_map
            .get_disjoint_mut([&BufferTypes::Output, &BufferTypes::Staging])
        else {
            return Err("Failed to gex index of output and or staging".into());
        };

        command_encoder.copy_buffer_to_buffer(
            &self.buffers[*output_buffer_index],
            0,
            &self.buffers[*staging_buffer_index],
            0,
            (self.output_length * std::mem::size_of::<u32>()) as u64,
        );

        self.gpu.queue.submit(Some(command_encoder.finish()));

        let buffer_slice = self.buffers[*staging_buffer_index].slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(MapMode::Read, move |map_result| {
            let _ = sender.send(map_result);
        });

        self.gpu.instance.poll_all(true);

        receiver
            .receive()
            .await
            .ok_or("Failed to receive buffer mapping result")??;

        let mapped_data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&mapped_data).to_vec();

        drop(mapped_data);
        self.buffers[*staging_buffer_index].unmap();

        return Ok(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::include_wgsl;

    #[tokio::test]
    #[cfg_attr(ci, ignore)]
    async fn test_shader() -> Result<(), Box<dyn Error>> {
        let gpu_instance = GpuInstance::new().await?;
        let input_data: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let mut test_shader = Shader::new(include_wgsl!("shaders/test.wgsl"), &gpu_instance);
        test_shader
            .add_buffer::<u32>("Input Buffer", BufferTypes::Input, input_data.len())
            .add_buffer::<u32>("Output Buffer", BufferTypes::Output, input_data.len())
            .add_buffer::<u32>("Staging Buffer", BufferTypes::Staging, input_data.len())
            .add_bind_group()?;

        test_shader.set_input(&input_data)?;
        let result = test_shader.run::<u32>().await?;
        assert_eq!(result, vec![2, 4, 6, 8, 10, 12, 14, 16]);
        return Ok(());
    }
}
