package org.meshllm.node.model

/**
 * Metadata and architecture specification for an open-source LLM,
 * including dynamic Hugging Face download endpoints and suitability classification.
 */
data class ModelSpec(
    val id: String,
    val name: String,
    val provider: String,
    val category: String,
    val totalLayers: Int,
    val hiddenDim: Int,
    val intermediateDim: Int,
    val numHeads: Int,
    val contextLength: Int,
    val paramCountStr: String,
    val memoryRequirementStr: String,
    val description: String,
    val samplePrompt: String,
    val vocabTokens: List<String> = emptyList(),
    val huggingFaceRepo: String? = null,
    val downloadUrl: String = "",
    val fileName: String = "",
    val fileSizeBytes: Long = 0L,
    val fileFormat: String = "GGUF", // "BIN_LLAMA2", "GGUF"
    val recommendedMinDevices: Int = 1,
    val isStarterTestModel: Boolean = false,
    val singleDeviceOomWarning: Boolean = false,
    val quantization: String = "GGUF",
    val isSuitableForMobile: Boolean = true,
    val downloadsCount: Long = 0L,
    val likesCount: Long = 0L
) {
    val architectureSummary: String
        get() = "$totalLayers Layers • Hidden: $hiddenDim • Heads: $numHeads • Intermediate: $intermediateDim"

    val memoryFootprintSummary: String
        get() = "$paramCountStr params • $memoryRequirementStr"

    val fileSizeMb: Double
        get() = Math.round((fileSizeBytes.toDouble() / (1024.0 * 1024.0)) * 10.0) / 10.0

    val fileSizeDisplay: String
        get() = if (fileSizeBytes > 0) {
            if (fileSizeMb >= 1024.0) {
                String.format("%.2f GB", fileSizeMb / 1024.0)
            } else {
                "${fileSizeMb} MB"
            }
        } else memoryRequirementStr
}
