use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use memmap2::MmapOptions;
use std::fs::File;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 2.2 Prototype Option A: Candle GGUF Layer-Slicing & Memory Mapping ===");

    // Look for GGUF model files
    let test_model_paths = [
        "/home/foor/MeshLLM/stories15M.gguf",
        "/tmp/stories15M.gguf",
    ];

    let mut found_path = None;
    for path in &test_model_paths {
        if std::path::Path::new(path).exists() {
            found_path = Some(*path);
            break;
        }
    }

    let model_path = match found_path {
        Some(p) => p,
        None => {
            println!("Note: GGUF model not found in standard paths. Creating synthetic tensor slice test to benchmark Candle CPU forward pass.");
            run_synthetic_candle_slice_bench()?;
            return Ok(());
        }
    };

    println!("Opening GGUF file: {}", model_path);
    let file = File::open(model_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let mut cursor = std::io::Cursor::new(&mmap[..]);

    let start_t = Instant::now();
    let content = gguf_file::Content::read(&mut cursor)?;
    println!("GGUF header parsed in {:?}. Total tensors: {}", start_t.elapsed(), content.tensor_infos.len());

    let start_layer = 1;
    let end_layer = 3;
    println!("Selecting layers: {}..={}", start_layer, end_layer);

    let mut filtered_tensors = Vec::new();
    for (name, info) in &content.tensor_infos {
        if name.starts_with("blk.") {
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() > 1 {
                if let Ok(layer_idx) = parts[1].parse::<usize>() {
                    if layer_idx >= start_layer && layer_idx <= end_layer {
                        filtered_tensors.push((name.clone(), info.clone()));
                    }
                }
            }
        }
    }

    println!("Matched {} layer tensors (isolated to L{}..L{})", filtered_tensors.len(), start_layer, end_layer);
    for (name, info) in filtered_tensors.iter().take(6) {
        println!("  - Tensor: {} | Shape: {:?} | GGML Type: {:?}", name, info.shape, info.ggml_dtype);
    }

    run_synthetic_candle_slice_bench()?;
    Ok(())
}

fn run_synthetic_candle_slice_bench() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    println!("\n--- Benchmarking Candle CPU Layer Forward Pass ---");
    let hidden_dim = 2048; // Standard 1B-2B model dim
    let intermediate_dim = 5632;
    let num_layers = 4;

    // Create 4 synthetic linear projection weights representing 4 transformer MLP layers
    let mut weights = Vec::new();
    for _ in 0..num_layers {
        let w_up = Tensor::randn(0.0f32, 0.02f32, (intermediate_dim, hidden_dim), &device)?;
        let w_down = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, intermediate_dim), &device)?;
        weights.push((w_up, w_down));
    }

    let input = Tensor::randn(0.0f32, 1.0f32, (1, hidden_dim), &device)?;

    // Warmup
    let mut x = input.clone();
    for (w_up, w_down) in &weights {
        let h = x.matmul(&w_up.t()?)?;
        x = h.matmul(&w_down.t()?)?;
    }

    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        let mut cur = input.clone();
        for (w_up, w_down) in &weights {
            let h = cur.matmul(&w_up.t()?)?;
            cur = h.matmul(&w_down.t()?)?;
        }
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("Executed 4 transformer layers across {} iters: {:6.2} ms/token", iters, elapsed);
    println!("Per-layer computation time: {:6.3} ms", elapsed / num_layers as f64);
    Ok(())
}
