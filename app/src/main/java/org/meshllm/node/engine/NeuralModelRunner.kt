package org.meshllm.node.engine

import android.util.Log
import org.meshllm.node.model.BlockType
import org.meshllm.node.model.LLMBlock
import org.meshllm.node.model.LlmErrorCode
import org.meshllm.node.model.LlmInferenceException
import org.meshllm.node.model.ModelSpec
import java.io.File

/**
 * Standard inference engine for distributed transformer layers.
 * Uses native llama.cpp (via LlamaNativeBridge) and pure Kotlin LlamaNeuralEngine.
 *
 * Implements:
 * 1. Real weights loaded into Node RAM with exact RAM tracking.
 * 2. 10-minute inactivity auto-unload to protect device memory.
 * 3. Strict error reporting using LlmInferenceException (no hardcoded fallbacks).
 * 4. Pipeline activation support across distributed mesh devices.
 */
class NeuralModelRunner {

    companion object {
        private const val TAG = "NeuralModelRunner"
        const val INACTIVITY_UNLOAD_MS = 10 * 60 * 1000L // 10 minutes

        /**
         * Singleton runner shared across ViewModel, Orchestrator, and SlaveWorker.
         * Ensures preloaded model in Node RAM is instantly accessible for zero TTFT latency.
         */
        val shared by lazy { NeuralModelRunner() }
    }

    val nativeBridge = LlamaNativeBridge
    val llamaNeuralEngine = LlamaNeuralEngine()
    val spTokenizer = SentencePieceTokenizer()

    var loadedFile: File? = null
        private set
    var loadedBlock: LLMBlock? = null
        private set
    var isRealWeightsActive: Boolean = false
        private set
    var weightFormatDescription: String = "No weights loaded"
        private set

    var lastInferenceTimestamp: Long = System.currentTimeMillis()
        private set

    val allocatedRamMb: Double
        get() = when {
            nativeBridge.isModelLoaded() -> {
                Math.round((nativeBridge.getAllocatedRamBytes().toDouble() / (1024.0 * 1024.0)) * 10.0) / 10.0
            }
            llamaNeuralEngine.isLoadedInRam -> llamaNeuralEngine.allocatedMemoryMb
            else -> 0.0
        }

    val isModelLoadedInRam: Boolean
        get() = nativeBridge.isModelLoaded() || llamaNeuralEngine.isLoadedInRam

    /**
     * Unloads model weights from Node RAM to free memory.
     */
    fun unloadModelWeights(): Boolean {
        var unloaded = false
        if (nativeBridge.isModelLoaded()) {
            nativeBridge.unloadModel()
            unloaded = true
        }
        if (llamaNeuralEngine.isLoadedInRam) {
            llamaNeuralEngine.unloadWeights()
            unloaded = true
        }
        loadedFile = null
        loadedBlock = null
        isRealWeightsActive = false
        weightFormatDescription = "Unloaded (Memory Freed)"
        Log.i(TAG, "Model weights unloaded from Node RAM")
        return unloaded
    }

    /**
     * Checks if 10 minutes of inactivity have elapsed; if so, unloads the model from RAM.
     */
    fun checkInactivityUnload(): Boolean {
        if (!isModelLoadedInRam) return false
        val elapsed = System.currentTimeMillis() - lastInferenceTimestamp
        if (elapsed > INACTIVITY_UNLOAD_MS) {
            Log.i(TAG, "Inactivity timeout reached (${elapsed / 1000}s > ${INACTIVITY_UNLOAD_MS / 1000}s). Unloading model...")
            unloadModelWeights()
            return true
        }
        return false
    }

    /**
     * Loads model weights from a disk file into RAM.
     * Uses native llama.cpp if available; falls back to JVM LlamaNeuralEngine.
     */
    @Synchronized
    fun loadModelWeightsIfAvailable(file: File?, block: LLMBlock? = null): Boolean {
        if (file == null || !file.exists() || file.length() < 1024L) {
            isRealWeightsActive = false
            weightFormatDescription = "Model file not found on disk"
            return false
        }

        val sameFile = isModelLoadedInRam && loadedFile?.absolutePath == file.absolutePath
        val sameBlock = if (block == null) {
            loadedBlock == null
        } else {
            loadedBlock?.startLayer == block.startLayer &&
            loadedBlock?.endLayer == block.endLayer &&
            loadedBlock?.totalModelLayers == block.totalModelLayers
        }

        if (sameFile && sameBlock) {
            lastInferenceTimestamp = System.currentTimeMillis()
            return true
        }

        // Unload any previously loaded model from memory before allocating the new model
        if (isModelLoadedInRam) {
            unloadModelWeights()
        }

        val startLayer = block?.startLayer ?: 0
        val endLayer = block?.endLayer ?: -1

        // 1. Try Native llama.cpp if supported (GGUF format)
        if (nativeBridge.isNativeLoaded && file.name.endsWith(".gguf", ignoreCase = true)) {
            try {
                // nThreads = 0 enables optimal big-core thread detection
                val status = nativeBridge.loadModel(file, startLayer, endLayer, nThreads = 0)
                if (status == 0) {
                    loadedFile = file
                    loadedBlock = block
                    isRealWeightsActive = true
                    val layerInfo = if (block != null) " • L[${block.startLayer}..${block.endLayer}]" else ""
                    val gpuStr = if (nativeBridge.isGpuAccelerated()) " [GPU: Vulkan]" else " [CPU]"
                    weightFormatDescription = "Native llama.cpp$gpuStr • ${file.name}$layerInfo • $allocatedRamMb MB in RAM"
                    lastInferenceTimestamp = System.currentTimeMillis()
                    loadTokenizerForModel(file, 32000)
                    Log.i(TAG, "Loaded model via native llama.cpp: $weightFormatDescription")
                    return true
                } else {
                    Log.w(TAG, "Native llama.cpp failed loading ${file.name} (code $status)")
                    isRealWeightsActive = false
                    weightFormatDescription = "Failed loading ${file.name} (Error $status)"
                    return false // Never pass .gguf to LlamaNeuralEngine
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Exception in nativeBridge.loadModel: ${t.message}", t)
                isRealWeightsActive = false
                weightFormatDescription = "Failed loading ${file.name} (${t.message})"
                return false
            }
        }

        // 2. Load via LlamaNeuralEngine (only for .bin files in llama2.c format)
        if (file.name.endsWith(".bin", ignoreCase = true)) {
            try {
                val loaded = llamaNeuralEngine.loadWeights(file, startLayer, endLayer)
                if (loaded) {
                    loadedFile = file
                    loadedBlock = block
                    isRealWeightsActive = true
                    val layerInfo = if (block != null) " • L[${block.startLayer}..${block.endLayer}]" else ""
                    weightFormatDescription = "Neural Engine • ${llamaNeuralEngine.config?.nLayers}L$layerInfo • Dim ${llamaNeuralEngine.config?.dim} • ${llamaNeuralEngine.allocatedMemoryMb} MB in RAM"
                    lastInferenceTimestamp = System.currentTimeMillis()

                    // Load tokenizer
                    loadTokenizerForModel(file, llamaNeuralEngine.config?.vocabSize ?: 32000)
                    Log.i(TAG, "Loaded model via neural engine: $weightFormatDescription")
                    return true
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Exception loading binary weights via LlamaNeuralEngine: ${t.message}", t)
            }
        }

        isRealWeightsActive = false
        weightFormatDescription = "Unsupported or invalid model format: ${file.name}"
        return false
    }

    private fun loadTokenizerForModel(modelFile: File, vocabSize: Int) {
        // 1. Direct extraction from GGUF metadata (supports 128k Llama-3, 152k Qwen-2.5, DeepSeek)
        if (modelFile.name.endsWith(".gguf", ignoreCase = true)) {
            try {
                val tokens = GGUFParser().extractVocabulary(modelFile)
                if (tokens.isNotEmpty() && spTokenizer.loadFromTokens(tokens)) {
                    Log.i(TAG, "Loaded GGUF tokenizer directly from ${modelFile.name} (Vocab: ${spTokenizer.vocabSize})")
                    return
                }
            } catch (e: Exception) {
                Log.w(TAG, "Could not extract vocabulary directly from ${modelFile.name}: ${e.message}")
            }
        }

        val tokName = if (vocabSize <= 512) "tok512.bin" else "tokenizer.bin"
        val candidates = listOf(
            File(modelFile.parentFile, tokName),
            File(modelFile.parentFile?.parentFile, tokName),
            File(modelFile.parentFile?.parentFile, "models/$tokName"),
            File("src/main/assets/$tokName"),
            File("app/src/main/assets/$tokName"),
            File("../$tokName")
        )

        for (cand in candidates) {
            if (cand.exists() && spTokenizer.loadFromFile(cand)) {
                Log.i(TAG, "Loaded tokenizer from ${cand.absolutePath} (Vocab: ${spTokenizer.vocabSize})")
                return
            }
        }

        // Try Android assets
        try {
            val context = org.meshllm.node.MeshLLMApp.instance
            context.assets.open(tokName).use { input ->
                if (spTokenizer.loadFromStream(input)) {
                    Log.i(TAG, "Loaded tokenizer from assets/$tokName (Vocab: ${spTokenizer.vocabSize})")
                    return
                }
            }
        } catch (_: Throwable) {}
    }

    private var activePrompt: String? = null
    private var activePromptTokens: List<Int> = emptyList()
    private val recentTokens = mutableListOf<Int>()
    private var currentSeqPos: Int = 0
    private var lastSampledTokenId: Int = 1

    /**
     * Executes transformer forward pass for assigned block on prompt context.
     * Throws LlmInferenceException with clear error codes if weights are missing or broken.
     */
    @Synchronized
    fun executeBlock(
        block: LLMBlock,
        prompt: String,
        tokenIndex: Int,
        incomingHidden: FloatArray?,
        activeModel: ModelSpec,
        previousTokenId: Int? = null,
        seqPos: Int = 0,
        promptTokensCount: Int = 0
    ): BlockExecutionResult {
        // Enforce activity timestamp
        lastInferenceTimestamp = System.currentTimeMillis()

        if (!isModelLoadedInRam) {
            throw LlmInferenceException(
                LlmErrorCode.ERR_MODEL_NOT_DOWNLOADED,
                "Model '${activeModel.name}' is not loaded in Node RAM. Download or load the model file before inference."
            )
        }

        val cleanPrompt = prompt.trim()
        val isInputBlock = block.isInputBlock
        val isOutputBlock = block.isOutputBlock

        // Branch 1: Native llama.cpp execution
        if (nativeBridge.isModelLoaded()) {
            return executeNativeBlock(
                block = block,
                prompt = cleanPrompt,
                tokenIndex = tokenIndex,
                incomingHidden = incomingHidden,
                isInputBlock = isInputBlock,
                isOutputBlock = isOutputBlock,
                previousTokenId = previousTokenId,
                seqPos = seqPos,
                incomingPromptCount = promptTokensCount
            )
        }

        // Branch 2: Neural Engine execution
        if (llamaNeuralEngine.isLoadedInRam) {
            return executeNeuralBlock(block, cleanPrompt, tokenIndex, incomingHidden, isInputBlock, isOutputBlock, previousTokenId)
        }

        throw LlmInferenceException(
            LlmErrorCode.ERR_WEIGHTS_CORRUPTED,
            "Engine failed to execute transformer block. Weights are invalid or corrupted."
        )
    }

    private fun executeNativeBlock(
        block: LLMBlock,
        prompt: String,
        tokenIndex: Int,
        incomingHidden: FloatArray?,
        isInputBlock: Boolean,
        isOutputBlock: Boolean,
        previousTokenId: Int?,
        seqPos: Int,
        incomingPromptCount: Int
    ): BlockExecutionResult {
        if (tokenIndex == 0) {
            nativeBridge.resetContext()
            activePrompt = prompt

            var pCount = incomingPromptCount
            val res = if (isInputBlock) {
                val tokens = nativeBridge.tokenize(prompt)
                if (tokens.isEmpty()) {
                    throw LlmInferenceException(
                        LlmErrorCode.ERR_TOKENIZER_MISSING,
                        "Native tokenizer failed to tokenize prompt: '$prompt'"
                    )
                }
                activePromptTokens = tokens.toList()
                pCount = tokens.size
                nativeBridge.prefillPipeline(tokens, null, isInputBlock = true, isOutputBlock = isOutputBlock)
            } else {
                nativeBridge.prefillPipeline(null, incomingHidden, isInputBlock = false, isOutputBlock = isOutputBlock)
            }

            if (res == null) {
                throw LlmInferenceException(
                    LlmErrorCode.ERR_WEIGHTS_CORRUPTED,
                    "Native prefill pipeline failed for block ${block.blockId}"
                )
            }

            if (isOutputBlock) {
                lastSampledTokenId = res.sampledToken
            }

            return BlockExecutionResult(
                outputHidden = res.outputHidden ?: FloatArray(0),
                generatedTokenId = if (isOutputBlock && res.sampledToken >= 0) res.sampledToken else null,
                generatedPiece = if (isOutputBlock) res.piece else null,
                isFinished = if (isOutputBlock) res.isFinished else false,
                promptTokensCount = pCount
            )
        }

        // Autoregressive generation step (tokenIndex > 0)
        val pos = if (seqPos > 0) seqPos else ((if (incomingPromptCount > 0) incomingPromptCount else activePromptTokens.size) + tokenIndex - 1)
        val inputToken = previousTokenId ?: lastSampledTokenId
        if (previousTokenId != null) {
            lastSampledTokenId = previousTokenId
        }

        val res = if (isInputBlock) {
            nativeBridge.forwardPipeline(inputToken, pos, null, isInputBlock = true, isOutputBlock = isOutputBlock)
        } else {
            nativeBridge.forwardPipeline(0, pos, incomingHidden, isInputBlock = false, isOutputBlock = isOutputBlock)
        }

        if (res == null) {
            throw LlmInferenceException(
                LlmErrorCode.ERR_WEIGHTS_CORRUPTED,
                "Native forward pipeline failed at pos $pos for block ${block.blockId}"
            )
        }

        if (isOutputBlock) {
            lastSampledTokenId = res.sampledToken
        }

        return BlockExecutionResult(
            outputHidden = res.outputHidden ?: FloatArray(0),
            generatedTokenId = if (isOutputBlock && res.sampledToken >= 0) res.sampledToken else null,
            generatedPiece = if (isOutputBlock) res.piece else null,
            isFinished = if (isOutputBlock) res.isFinished else false,
            promptTokensCount = incomingPromptCount
        )
    }

    private fun executeNeuralBlock(
        block: LLMBlock,
        prompt: String,
        tokenIndex: Int,
        incomingHidden: FloatArray?,
        isInputBlock: Boolean,
        isOutputBlock: Boolean,
        previousTokenId: Int?
    ): BlockExecutionResult {
        if (!spTokenizer.isLoaded) {
            throw LlmInferenceException(
                LlmErrorCode.ERR_TOKENIZER_MISSING,
                "SentencePiece tokenizer not loaded for model '${block.modelId}'"
            )
        }

        if (tokenIndex == 0 || activePrompt != prompt || activePromptTokens.isEmpty()) {
            activePrompt = prompt
            activePromptTokens = spTokenizer.encode(prompt)
            if (activePromptTokens.isEmpty()) {
                throw LlmInferenceException(
                    LlmErrorCode.ERR_TOKENIZER_MISSING,
                    "Tokenizer produced 0 tokens for prompt: '$prompt'"
                )
            }
            llamaNeuralEngine.resetKvCache()
            recentTokens.clear()
            recentTokens.addAll(activePromptTokens)
            currentSeqPos = 0

            // Prefill prompt tokens up to last token only on input block
            if (isInputBlock) {
                for (i in 0 until activePromptTokens.size - 1) {
                    llamaNeuralEngine.forward(
                        token = activePromptTokens[i],
                        pos = currentSeqPos,
                        incomingHidden = null,
                        startLayer = block.startLayer,
                        endLayer = block.endLayer,
                        isInput = true,
                        isOutput = false,
                        recentTokens = recentTokens
                    )
                    currentSeqPos++
                }
            }
            lastSampledTokenId = activePromptTokens.last()
        }

        val actualPos = (activePromptTokens.size - 1) + tokenIndex
        val inputToken = if (tokenIndex == 0) lastSampledTokenId else (previousTokenId ?: lastSampledTokenId)
        val fwdResult = llamaNeuralEngine.forward(
            token = inputToken,
            pos = actualPos,
            incomingHidden = incomingHidden,
            startLayer = block.startLayer,
            endLayer = block.endLayer,
            isInput = isInputBlock,
            isOutput = isOutputBlock,
            recentTokens = recentTokens
        )

        var generatedTokenId: Int? = null
        var generatedPiece: String? = null
        var isFinished = false

        if (isOutputBlock) {
            val sampled = fwdResult.sampledTokenId
            if (sampled == null) {
                throw LlmInferenceException(
                    LlmErrorCode.ERR_WEIGHTS_CORRUPTED,
                    "Forward pass output head produced null token at position $actualPos"
                )
            }
            generatedTokenId = sampled
            lastSampledTokenId = sampled
            recentTokens.add(sampled)
            generatedPiece = spTokenizer.decode(sampled)
            isFinished = fwdResult.isFinished
        }

        return BlockExecutionResult(
            outputHidden = fwdResult.hiddenState,
            generatedTokenId = generatedTokenId,
            generatedPiece = generatedPiece,
            isFinished = isFinished
        )
    }

    data class BlockExecutionResult(
        val outputHidden: FloatArray,
        val generatedTokenId: Int?,
        val generatedPiece: String?,
        val isFinished: Boolean,
        val promptTokensCount: Int = 0
    )
}

