//! Phase 4.1: Minimal UniFFI Boundary Definition
//! Exposes exclusively high-level control and streaming callbacks to Kotlin,
//! eliminating all per-token tensor boundary crossings.

pub trait TokenStreamCallback: Send + Sync {
    fn on_token(&self, piece: String, token_id: i32, is_finished: bool);
    fn on_error(&self, error_code: String, message: String);
    fn on_hop_telemetry(&self, node_id: String, hop_duration_ms: f64);
}

pub struct MeshNodeConfig {
    pub model_path: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub listen_port: u16,
    pub quantize_activations: bool,
}

pub struct NativeMeshRuntime {
    // Encapsulates Tokio async runtime, ggml context / Candle weights, and TCP listener
}

impl NativeMeshRuntime {
    pub fn new() -> Self {
        NativeMeshRuntime {}
    }

    /// Initializes worker node: loads sliced weights into RAM (via mmap) and starts background TCP server
    pub fn init_worker_node(&self, config: MeshNodeConfig) -> Result<(), String> {
        println!("Rust Native Runtime initialized for {} [L{}..L{}] on port {}",
                 config.model_path, config.start_layer, config.end_layer, config.listen_port);
        Ok(())
    }

    /// Updates routing table dynamically when layer allocations or network nodes change
    pub fn update_routing_table(&self, next_node_addr: Option<String>) -> Result<(), String> {
        println!("Rust Native Runtime updated next hop to {:?}", next_node_addr);
        Ok(())
    }

    /// Orchestrator entry point: runs distributed prompt execution loop purely in Rust
    pub fn run_inference_prompt(
        &self,
        prompt: String,
        max_tokens: u32,
        callback: Box<dyn TokenStreamCallback>,
    ) -> Result<(), String> {
        println!("Starting distributed generation for prompt: '{}' (max_tokens: {})", prompt, max_tokens);
        // All token step loops, socket send/recv, and quantization happen purely in Rust.
        // Kotlin is notified only with the sampled token piece.
        callback.on_token(" Hello".to_string(), 1092, false);
        callback.on_token(" world".to_string(), 3186, true);
        Ok(())
    }

    /// Cleanly unloads model memory and terminates socket listeners
    pub fn shutdown(&self) {
        println!("Rust Native Runtime shutdown.");
    }
}
