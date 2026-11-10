package org.meshllm.node.model

data class ClusterState(
    val sessionToken: String,
    val modelSpec: ModelSpec = ModelCatalog.getDefaultModel(),
    val modelName: String = modelSpec.name,
    val totalLayers: Int = modelSpec.totalLayers,
    val masterNodeId: String,
    val masterAddress: String,
    val nodes: List<NodeInfo> = emptyList(),
    val isRunningInference: Boolean = false,
    val activePrompt: String = "",
    val generatedOutput: String = "",
    val tokensGenerated: Int = 0,
    val tokensPerSecond: Double = 0.0,
    val generationSequence: Long = 0,
    val isModelAvailableOnMaster: Boolean = false,
    val masterDownloadedModelIds: List<String> = emptyList(),
    val layerAllotmentMode: String = ALLOTMENT_MODE_AUTO
) {
    companion object {
        const val ALLOTMENT_MODE_AUTO = "AUTO"
        const val ALLOTMENT_MODE_MANUAL = "MANUAL"
    }
}
