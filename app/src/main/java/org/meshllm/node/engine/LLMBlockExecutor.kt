package org.meshllm.node.engine

import android.content.Context
import android.os.Process
import android.util.Log
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import org.meshllm.node.model.LLMBlock
import org.meshllm.node.model.ModelCatalog
import org.meshllm.node.model.PipelineActivation
import org.meshllm.node.model.SensorMetrics
import org.meshllm.node.telemetry.PerformanceGovernor
import org.meshllm.node.telemetry.SensorMonitor
import java.io.File
import java.util.concurrent.Executors

class LLMBlockExecutor(
    private val sensorMonitor: SensorMonitor? = null,
    private val performanceGovernor: PerformanceGovernor = PerformanceGovernor(),
    private val context: Context? = null,
    runnerInstance: NeuralModelRunner? = null
) {
    companion object {
        private const val TAG = "LLMBlockExecutor"

        // Dedicated background inference thread pool:
        // 1. Thread priority set to THREAD_PRIORITY_BACKGROUND (nice 10) so Android UI and Choreographer never lag.
        // 2. Capped at 2-4 worker threads so CPU cores are never oversubscribed while GGML manages internal compute threads.
        private val backgroundInferenceExecutor = Executors.newFixedThreadPool(
            maxOf(1, minOf(4, Runtime.getRuntime().availableProcessors() - 2))
        ) { runnable ->
            Thread {
                try {
                    Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
                } catch (_: Exception) {}
                runnable.run()
            }.apply {
                name = "MeshLLM-InferenceThread"
                isDaemon = true
            }
        }
        val inferenceDispatcher = backgroundInferenceExecutor.asCoroutineDispatcher()

        val sharedRunner: NeuralModelRunner get() = NeuralModelRunner.shared
    }

    private val neuralRunner = runnerInstance ?: NeuralModelRunner.shared
    val runner: NeuralModelRunner get() = neuralRunner

    private var cachedModelId: String? = null
    private var cachedWeightFile: File? = null

    private fun findModelWeightFile(modelId: String): File? {
        val cached = cachedWeightFile
        if (modelId == cachedModelId && cached != null && cached.exists() && cached.length() > 1024L) {
            return cached
        }
        val ctx = context ?: try { org.meshllm.node.MeshLLMApp.instance } catch (_: Throwable) { null }
        if (ctx == null) return null

        val modelDir = File(ctx.filesDir, "models/$modelId")
        var weightFile = modelDir.listFiles()?.firstOrNull { it.isFile && it.length() > 1024L && !it.name.endsWith(".part") }
        if (weightFile == null || !weightFile.exists()) {
            val spec = ModelCatalog.getModelById(modelId)
            if (spec != null) {
                val candidate = File(modelDir, spec.fileName)
                if (candidate.exists() && candidate.length() > 1024L) {
                    weightFile = candidate
                } else {
                    val rootModelsDir = File(ctx.filesDir, "models")
                    val candidateRoot = File(rootModelsDir, spec.fileName)
                    if (candidateRoot.exists() && candidateRoot.length() > 1024L) {
                        weightFile = candidateRoot
                    }
                }
            }
        }
        if (weightFile != null && weightFile.exists()) {
            cachedModelId = modelId
            cachedWeightFile = weightFile
        }
        return weightFile
    }

    /**
     * Execute forward pass for assigned block on the given input activation.
     * Guaranteed to run on background worker threads with THREAD_PRIORITY_BACKGROUND.
     */
    suspend fun executeForward(
        block: LLMBlock,
        inputActivation: PipelineActivation
    ): PipelineActivation = withContext(inferenceDispatcher) {
        val startNs = System.nanoTime()

        // 1. Apply thermal / battery governor and background thread priority
        performanceGovernor.applyBackgroundPriority()
        val currentMetrics = sensorMonitor?.currentMetrics?.value ?: SensorMetrics()
        performanceGovernor.throttleIfOverheating(currentMetrics)

        // 2. Load model weights file if downloaded and not yet active in RAM for this specific block
        val weightFile = findModelWeightFile(block.modelId)
        val needsLoad = weightFile != null && (!neuralRunner.isModelLoadedInRam ||
                neuralRunner.loadedFile?.absolutePath != weightFile.absolutePath ||
                neuralRunner.loadedBlock?.modelId != block.modelId ||
                neuralRunner.loadedBlock?.startLayer != block.startLayer ||
                neuralRunner.loadedBlock?.endLayer != block.endLayer)
        if (needsLoad && weightFile != null) {
            neuralRunner.loadModelWeightsIfAvailable(weightFile, block)
            sensorMonitor?.updateModelRam(neuralRunner.allocatedRamMb)
            sensorMonitor?.updateMetrics()
        }

        val activeModel = ModelCatalog.getModelById(block.modelId) ?: ModelCatalog.getDefaultModel()
        val incomingHidden = inputActivation.extractHiddenState().let {
            if (it.isNotEmpty()) it else null
        }

        // 3. Execute block through the neural model runner conditioned on prompt
        val result = neuralRunner.executeBlock(
            block = block,
            prompt = inputActivation.promptId,
            tokenIndex = inputActivation.tokenIndex,
            incomingHidden = incomingHidden,
            activeModel = activeModel,
            previousTokenId = inputActivation.generatedTokenId,
            seqPos = inputActivation.seqPos,
            promptTokensCount = inputActivation.promptTokensCount
        )

        val elapsedMs = (System.nanoTime() - startNs) / 1_000_000

        val encodedHiddenBase64 = if (result.outputHidden.isNotEmpty()) {
            PipelineActivation.encodeFloatArrayToBase64(result.outputHidden)
        } else null

        PipelineActivation(
            sessionToken = inputActivation.sessionToken,
            requestId = inputActivation.requestId,
            promptId = inputActivation.promptId,
            tokenIndex = inputActivation.tokenIndex,
            blockId = block.blockId,
            senderNodeId = block.blockId,
            receiverNodeId = "",
            hiddenState = emptyList(), // Use fast compact Base64 to eliminate GSON boxing overhead
            hiddenStateBase64 = encodedHiddenBase64,
            generatedTokenId = result.generatedTokenId ?: inputActivation.generatedTokenId,
            generatedTextPiece = result.generatedPiece,
            isFinished = result.isFinished,
            executionTimeMs = elapsedMs,
            timestamp = System.currentTimeMillis(),
            seqPos = inputActivation.seqPos,
            promptTokensCount = if (result.promptTokensCount > 0) result.promptTokensCount else inputActivation.promptTokensCount
        )
    }
}
