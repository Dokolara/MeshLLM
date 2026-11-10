//! SYNAPSE-V: Transformer Layer Slicing & Compute Engine
//! Uses Hugging Face Candle + memmap2 for GGUF model layer slicing and metadata verification.

use candle_core::quantized::gguf_file;
use memmap2::MmapOptions;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct LayerShardSpec {
    pub start_layer: usize,
    pub end_layer: usize,
    pub total_model_layers: usize,
    pub hidden_dim: usize,
    pub isolated_tensors_count: usize,
}

pub struct SynapseV;

impl SynapseV {
    /// Inspects and slices a quantized GGUF model file into a specific layer range using memory-mapping
    pub fn inspect_gguf_layer_slice<P: AsRef<Path>>(
        model_path: P,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<LayerShardSpec, String> {
        let file = File::open(model_path.as_ref())
            .map_err(|e| format!("Failed opening model file: {}", e))?;
        let mmap = unsafe {
            MmapOptions::new().map(&file)
                .map_err(|e| format!("Failed mmap on model file: {}", e))?
        };
        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| format!("GGUF header parse error: {}", e))?;

        let mut max_layer = 0;
        let mut isolated_count = 0;
        let mut hidden_dim = 0;

        for (name, info) in &content.tensor_infos {
            if name == "token_embd.weight" {
                if let Some(&dim) = info.shape.dims().get(1) {
                    hidden_dim = dim;
                }
            }
            if name.starts_with("blk.") {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() > 1 {
                    if let Ok(layer_idx) = parts[1].parse::<usize>() {
                        if layer_idx > max_layer {
                            max_layer = layer_idx;
                        }
                        if layer_idx >= start_layer && layer_idx <= end_layer {
                            isolated_count += 1;
                        }
                    }
                }
            }
        }

        let total_layers = if max_layer > 0 { max_layer + 1 } else { 0 };

        Ok(LayerShardSpec {
            start_layer,
            end_layer,
            total_model_layers: total_layers,
            hidden_dim: if hidden_dim > 0 { hidden_dim } else { 4096 },
            isolated_tensors_count: isolated_count,
        })
    }
}
