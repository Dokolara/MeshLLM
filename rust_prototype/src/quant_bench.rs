use std::time::Instant;

/// Dynamic Activation Quantization (FP32 -> INT8 symmetric with scale)
pub fn quantize_fp32_to_int8(input: &[f32], output: &mut [i8]) -> f32 {
    let mut max_abs = 0.0f32;
    for &val in input {
        let abs = val.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    if max_abs == 0.0 {
        output.fill(0);
        return 1.0;
    }
    let scale = max_abs / 127.0;
    let inv_scale = 127.0 / max_abs;
    for (i, &val) in input.iter().enumerate() {
        let q = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;
        output[i] = q;
    }
    scale
}

/// Dequantization (INT8 -> FP32)
pub fn dequantize_int8_to_fp32(input: &[i8], scale: f32, output: &mut [f32]) {
    for (i, &val) in input.iter().enumerate() {
        output[i] = (val as f32) * scale;
    }
}

fn main() {
    println!("=== 3.3 Dynamic Activation Quantization Benchmark ===");
    // Test for typical hidden dimensions: Mistral 7B (dim=4096), Llama 3.1 8B (dim=4096), Qwen 14B (dim=5120)
    let dims = [288, 2048, 4096, 5120];

    for &dim in &dims {
        // Generate pseudo-random hidden states
        let mut input = vec![0.0f32; dim];
        for i in 0..dim {
            input[i] = ((i as f32 * 0.1337).sin()) * 2.5;
        }
        let mut int8_buf = vec![0i8; dim];
        let mut recon_buf = vec![0.0f32; dim];

        let iters = 10_000;
        let start = Instant::now();
        let mut scale = 0.0;
        for _ in 0..iters {
            scale = quantize_fp32_to_int8(&input, &mut int8_buf);
        }
        let quant_time = start.elapsed().as_nanos() as f64 / iters as f64;

        let start_deq = Instant::now();
        for _ in 0..iters {
            dequantize_int8_to_fp32(&int8_buf, scale, &mut recon_buf);
        }
        let dequant_time = start_deq.elapsed().as_nanos() as f64 / iters as f64;

        // Measure reconstruction error (MAE and Max Error)
        let mut sum_err = 0.0;
        let mut max_err = 0.0f32;
        for i in 0..dim {
            let err = (input[i] - recon_buf[i]).abs();
            sum_err += err;
            if err > max_err {
                max_err = err;
            }
        }
        let mae = sum_err / dim as f32;

        let original_bytes = dim * 4;
        let quant_bytes = dim * 1 + 4; // int8 array + 4-byte scale
        let ratio = original_bytes as f64 / quant_bytes as f64;

        println!("Dim: {:5} | Orig: {:5} B | INT8: {:5} B ({:.2}x smaller)", dim, original_bytes, quant_bytes, ratio);
        println!("       Quant: {:6.2} µs | Dequant: {:6.2} µs | Total: {:6.2} µs | MAE: {:.6} | MaxErr: {:.6}",
                 quant_time / 1000.0, dequant_time / 1000.0, (quant_time + dequant_time) / 1000.0, mae, max_err);
    }
}
