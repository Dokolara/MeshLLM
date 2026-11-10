//! NEXUS-AIR: Cluster Topology & Sequential Dynamic Layer Routing
//! Computes contiguous, zero-gap layer allotments across active heterogeneous nodes.

#[derive(Debug, Clone, PartialEq)]
pub struct NodeCapacity {
    pub node_id: String,
    pub benchmark_score: f64,
    pub sensor_health_factor: f64, // 0.0 to 1.0 (from VitalSense)
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssignedBlock {
    pub node_id: String,
    pub start_layer: usize,
    pub end_layer: usize,
    pub layer_count: usize,
    pub is_input_block: bool,
    pub is_output_block: bool,
}

pub struct NexusAir;

impl NexusAir {
    /// Sequential layer rebalancing without gaps (Exo/Petals architecture).
    /// Assigns [0..total_layers-1] proportionally based on dynamic node capacity.
    pub fn compute_layer_allocations(
        nodes: &[NodeCapacity],
        total_layers: usize,
    ) -> Vec<AssignedBlock> {
        if nodes.is_empty() || total_layers == 0 {
            return Vec::new();
        }

        let effective_capacities: Vec<f64> = nodes
            .iter()
            .map(|n| (n.benchmark_score * n.sensor_health_factor).max(1.0))
            .collect();

        let sum_capacity: f64 = effective_capacities.iter().sum();
        let n = nodes.len();
        let mut allocations = Vec::with_capacity(n);
        let mut current_layer = 0;

        for (i, node) in nodes.iter().enumerate() {
            let count = if i == n - 1 {
                total_layers.saturating_sub(current_layer)
            } else {
                let share = (effective_capacities[i] / sum_capacity) * total_layers as f64;
                let count = share.round() as usize;
                count.min(total_layers.saturating_sub(current_layer))
            };

            let end_layer = if count > 0 { current_layer + count - 1 } else { current_layer };
            allocations.push(AssignedBlock {
                node_id: node.node_id.clone(),
                start_layer: current_layer,
                end_layer,
                layer_count: count,
                is_input_block: current_layer == 0,
                is_output_block: end_layer == total_layers - 1,
            });

            current_layer += count;
        }

        allocations
    }
}
