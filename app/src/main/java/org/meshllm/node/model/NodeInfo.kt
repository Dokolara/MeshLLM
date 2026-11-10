package org.meshllm.node.model

data class NodeInfo(
    val nodeId: String,
    val deviceName: String,
    val ipAddress: String,
    val port: Int,
    var role: NodeRole,
    val joinOrder: Int, // 0 = original master, 1 = first joined slave, 2 = second joined slave, etc.
    val joinTimestamp: Long = System.currentTimeMillis(),
    var lastHeartbeat: Long = System.currentTimeMillis(),
    var isAlive: Boolean = true,
    var metrics: SensorMetrics = SensorMetrics(),
    var benchmark: BenchmarkProfile = BenchmarkProfile(),
    var assignedBlocks: List<LLMBlock> = emptyList()
) {
    val assignedLayersDescription: String
        get() {
            if (assignedBlocks.isEmpty()) return "No layers assigned (Idle)"
            return assignedBlocks.joinToString(", ") { "L[${it.startLayer}..${it.endLayer}]" }
        }

    val totalAssignedLayers: Int
        get() = assignedBlocks.sumOf { it.layerCount }
}
