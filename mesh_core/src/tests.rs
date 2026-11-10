//! Silicon-Grade Subsystem Automated Unit Tests

use crate::tensor_shrink::TensorShrink;
use crate::aether_link::{WireHeader, MESH_MAGIC, WIRE_HEADER_LEN};
use crate::lexicon_core::LexiconCore;
use crate::nexus_air::{NexusAir, NodeCapacity};

#[test]
fn test_tensor_shrink_int8_quantization() {
    let dim = 4096;
    let mut input = vec![0.0f32; dim];
    for i in 0..dim {
        input[i] = ((i as f32 * 0.05).sin()) * 3.5;
    }

    let mut int8_buf = vec![0i8; dim];
    let scale = TensorShrink::quantize_fp32_to_int8(&input, &mut int8_buf);
    assert!(scale > 0.0);

    let mut recon_buf = vec![0.0f32; dim];
    TensorShrink::dequantize_int8_to_fp32(&int8_buf, scale, &mut recon_buf);

    let mut sum_err = 0.0;
    for i in 0..dim {
        sum_err += (input[i] - recon_buf[i]).abs();
    }
    let mae = sum_err / dim as f32;
    println!("TENSOR-SHRINK MAE for 4096-dim: {:.6}", mae);
    assert!(mae < 0.015, "Quantization error exceeds threshold: {}", mae);
}

#[test]
fn test_aether_link_wire_header_binary_framing() {
    assert_eq!(WIRE_HEADER_LEN, 32, "WireHeader must be 32 bytes aligned");

    let header = WireHeader::new_int8(123456789, 42, 108, 16, 4096, 0.0275);
    let bytes = header.to_bytes();
    assert_eq!(bytes.len(), 32);

    let parsed = WireHeader::from_bytes(&bytes).expect("Valid WireHeader parsing");
    assert_eq!(parsed.magic, MESH_MAGIC);
    assert_eq!(parsed.request_id, 123456789);
    assert_eq!(parsed.token_idx, 42);
    assert_eq!(parsed.seq_pos, 108);
    assert_eq!(parsed.layer_idx, 16);
    assert_eq!(parsed.dtype, 2);
    assert_eq!(parsed.num_elements, 4096);
    assert_eq!(parsed.scale, 0.0275);
    assert_eq!(parsed.payload_byte_size(), 4096);
}

#[test]
fn test_lexicon_core_eog_and_sanitization() {
    let eog_list = vec![2, 128001, 128009];
    assert!(LexiconCore::is_eog(2, &eog_list));
    assert!(LexiconCore::is_eog(128009, &eog_list));
    // Verify token 0 (<unk>) does NOT halt generation unless explicitly in eog list!
    assert!(!LexiconCore::is_eog(0, &eog_list));

    // Tekken control token sanitization
    assert_eq!(LexiconCore::sanitize_piece("[control_13]"), "");
    assert_eq!(LexiconCore::sanitize_piece("[control_19]"), "");
    assert_eq!(LexiconCore::sanitize_piece(" hello"), " hello");
}

#[test]
fn test_nexus_air_dynamic_layer_allotment() {
    let nodes = vec![
        NodeCapacity {
            node_id: "master_0".to_string(),
            benchmark_score: 100.0,
            sensor_health_factor: 1.0,
        },
        NodeCapacity {
            node_id: "slave_1".to_string(),
            benchmark_score: 100.0,
            sensor_health_factor: 1.0,
        },
    ];

    // Split 32 layers (Mistral 7B) across 2 identical devices
    let blocks = NexusAir::compute_layer_allocations(&nodes, 32);
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].start_layer, 0);
    assert_eq!(blocks[0].end_layer, 15);
    assert_eq!(blocks[0].layer_count, 16);
    assert!(blocks[0].is_input_block);
    assert!(!blocks[0].is_output_block);

    assert_eq!(blocks[1].start_layer, 16);
    assert_eq!(blocks[1].end_layer, 31);
    assert_eq!(blocks[1].layer_count, 16);
    assert!(!blocks[1].is_input_block);
    assert!(blocks[1].is_output_block);
}
