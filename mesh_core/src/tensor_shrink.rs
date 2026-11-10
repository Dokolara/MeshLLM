//! TENSOR-SHRINK: Dynamic Activation Quantization Subsystem
//! High-speed symmetric INT8 compression & dequantization for intermediate hidden states.

pub struct TensorShrink;

impl TensorShrink {
    /// Quantize outgoing FP32 hidden state to symmetric INT8 with optimal scale factor
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

    /// Dequantize incoming INT8 activations back to FP32 for downstream transformer execution
    pub fn dequantize_int8_to_fp32(input: &[i8], scale: f32, output: &mut [f32]) {
        for (i, &val) in input.iter().enumerate() {
            output[i] = (val as f32) * scale;
        }
    }
}
