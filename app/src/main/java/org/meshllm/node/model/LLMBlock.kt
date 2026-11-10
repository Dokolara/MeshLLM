package org.meshllm.node.model

enum class BlockType {
    EMBEDDING_INPUT,    // Token embedding + RoPE + Layers [0..k]
    TRANSFORMER_BLOCK,  // Middle sequential transformer decoder layers [k..m]
    OUTPUT_LM_HEAD      // Final layers [m..L-1] + RMSNorm + LM Head Logits + Sampler
}

data class LLMBlock(
    val blockId: String,
    val blockType: BlockType,
    val startLayer: Int,
    val endLayer: Int,
    val totalModelLayers: Int = 6,
    val inputDim: Int = 288,
    val hiddenDim: Int = 288,
    val intermediateDim: Int = 768,
    val numHeads: Int = 6,
    val quantization: String = "INT8",
    val weightChecksum: String = "",
    val modelId: String = "tinystories-15m",
    val modelName: String = "TinyStories-15M"
) {
    val layerCount: Int
        get() = (endLayer - startLayer + 1).coerceAtLeast(0)

    val isInputBlock: Boolean
        get() = (startLayer == 0 || blockType == BlockType.EMBEDDING_INPUT)

    val isOutputBlock: Boolean
        get() = (endLayer >= totalModelLayers - 1 || blockType == BlockType.OUTPUT_LM_HEAD)

    val displayName: String
        get() = when (blockType) {
            BlockType.EMBEDDING_INPUT -> "Embedding + Layers [$startLayer..$endLayer]"
            BlockType.TRANSFORMER_BLOCK -> "Transformer Layers [$startLayer..$endLayer]"
            BlockType.OUTPUT_LM_HEAD -> "Layers [$startLayer..$endLayer] + LM Head"
        }

    companion object {
        fun fromModelSpec(
            blockId: String,
            blockType: BlockType,
            startLayer: Int,
            endLayer: Int,
            model: ModelSpec
        ): LLMBlock {
            return LLMBlock(
                blockId = blockId,
                blockType = blockType,
                startLayer = startLayer,
                endLayer = endLayer,
                totalModelLayers = model.totalLayers,
                inputDim = model.hiddenDim,
                hiddenDim = model.hiddenDim,
                intermediateDim = model.intermediateDim,
                numHeads = model.numHeads,
                modelId = model.id,
                modelName = model.name
            )
        }
    }
}
