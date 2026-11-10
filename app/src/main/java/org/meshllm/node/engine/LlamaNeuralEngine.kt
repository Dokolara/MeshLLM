package org.meshllm.node.engine

import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Pure Kotlin/JVM execution engine for LLaMA transformer models (llama2.c .bin format).
 * Loads real weight matrices directly into Node RAM (FloatArray memory) and executes
 * the full autoregressive forward pass:
 * Token Embedding -> RMSNorm -> QKV Matmul -> RoPE -> Scaled Dot-Product MHA ->
 * Wo Matmul -> Residual Add -> FFN RMSNorm -> SwiGLU (SiLU(w1) * w3) -> W2 Matmul ->
 * Final RMSNorm -> Classifier Matmul -> Token Sampling.
 *
 * Supports distributed slicing: each mesh node loads only its assigned layers into RAM.
 */
class LlamaNeuralEngine {

    companion object {
        private const val TAG = "LlamaNeuralEngine"
    }

    data class Config(
        val dim: Int,
        val hiddenDim: Int,
        val nLayers: Int,
        val nHeads: Int,
        val nKvHeads: Int,
        val vocabSize: Int,
        val seqLen: Int,
        val sharedWeights: Boolean
    ) {
        val headSize: Int get() = dim / nHeads
        val kvDim: Int get() = (dim * nKvHeads) / nHeads
        val kvMul: Int get() = nHeads / nKvHeads
    }

    var config: Config? = null
        private set

    // Real weights loaded into Node RAM
    var tokenEmbeddingTable: FloatArray? = null
        private set
    var rmsAttWeight: FloatArray? = null
        private set
    var wq: FloatArray? = null
        private set
    var wk: FloatArray? = null
        private set
    var wv: FloatArray? = null
        private set
    var wo: FloatArray? = null
        private set
    var rmsFfnWeight: FloatArray? = null
        private set
    var w1: FloatArray? = null
        private set
    var w2: FloatArray? = null
        private set
    var w3: FloatArray? = null
        private set
    var rmsFinalWeight: FloatArray? = null
        private set
    var wcls: FloatArray? = null
        private set

    // RAM allocation telemetry
    var allocatedMemoryBytes: Long = 0L
        private set
    val allocatedMemoryMb: Double
        get() = Math.round((allocatedMemoryBytes.toDouble() / (1024.0 * 1024.0)) * 10.0) / 10.0

    var isLoadedInRam: Boolean = false
        private set
    var loadedFilePath: String = ""
        private set
    var loadedStartLayer: Int = 0
        private set
    var loadedEndLayer: Int = -1
        private set

    // Run buffers & KV cache allocated in RAM
    private var keyCache: FloatArray? = null
    private var valCache: FloatArray? = null
    private var xb: FloatArray? = null
    private var xb2: FloatArray? = null
    private var hb: FloatArray? = null
    private var hb2: FloatArray? = null
    private var q: FloatArray? = null
    private var k: FloatArray? = null
    private var v: FloatArray? = null
    private var att: FloatArray? = null
    private var logits: FloatArray? = null

    /**
     * Unloads all model tensors and activation buffers from RAM and triggers GC.
     */
    fun unloadWeights() {
        tokenEmbeddingTable = null
        rmsAttWeight = null
        wq = null
        wk = null
        wv = null
        wo = null
        rmsFfnWeight = null
        w1 = null
        w2 = null
        w3 = null
        rmsFinalWeight = null
        wcls = null
        keyCache = null
        valCache = null
        xb = null
        xb2 = null
        hb = null
        hb2 = null
        q = null
        k = null
        v = null
        att = null
        logits = null
        allocatedMemoryBytes = 0L
        isLoadedInRam = false
        loadedFilePath = ""
        loadedStartLayer = 0
        loadedEndLayer = -1
        config = null
        System.gc()
        Log.i(TAG, "LlamaNeuralEngine: weights completely unloaded and memory freed")
    }

    /**
     * Loads model weights from disk file into RAM.
     * When [startLayer] and [endLayer] are specified, only weights for those layers
     * are loaded to conserve RAM in distributed edge execution.
     */
    fun loadWeights(
        file: File,
        startLayer: Int = 0,
        endLayer: Int = -1
    ): Boolean {
        if (!file.exists() || file.length() < 28) {
            Log.e(TAG, "File does not exist or invalid: ${file.absolutePath}")
            return false
        }

        try {
            RandomAccessFile(file, "r").use { raf ->
                val channel = raf.channel
                val headerBuf = channel.map(FileChannel.MapMode.READ_ONLY, 0, 28)
                headerBuf.order(ByteOrder.LITTLE_ENDIAN)

                val dim = headerBuf.int
                val hiddenDim = headerBuf.int
                val nLayers = headerBuf.int
                val nHeads = headerBuf.int
                val nKvHeads = headerBuf.int
                val rawVocab = headerBuf.int
                val seqLen = headerBuf.int

                val shared = rawVocab > 0
                val vocabSize = kotlin.math.abs(rawVocab)

                if (dim <= 0 || nLayers <= 0 || vocabSize <= 0) {
                    Log.e(TAG, "Invalid config: dim=$dim, layers=$nLayers, vocab=$vocabSize")
                    return false
                }

                val cfg = Config(dim, hiddenDim, nLayers, nHeads, nKvHeads, vocabSize, seqLen, shared)
                this.config = cfg

                val maxL = if (endLayer >= 0) min(endLayer, nLayers - 1) else nLayers - 1
                val minL = max(0, startLayer)
                val numAssignedLayers = max(0, maxL - minL + 1)
                val isInputNode = (minL == 0)
                val isOutputNode = (maxL == nLayers - 1)
                loadedStartLayer = minL
                loadedEndLayer = maxL

                Log.i(TAG, "Loading transformer blocks [$minL..$maxL] ($numAssignedLayers layers, input=$isInputNode, output=$isOutputNode) of $nLayers from ${file.name} into Node RAM")

                // Map float weights
                val fileSize = channel.size()
                val dataBuf = channel.map(FileChannel.MapMode.READ_ONLY, 28, fileSize - 28)
                dataBuf.order(ByteOrder.LITTLE_ENDIAN)
                val floatBuffer = dataBuf.asFloatBuffer()

                var offset = 0
                var totalBytesAllocated = 0L

                // 1. Token Embeddings: (vocab_size, dim) - only input node needs embeddings
                val embSize = cfg.vocabSize * cfg.dim
                if (isInputNode) {
                    val emb = FloatArray(embSize)
                    floatBuffer.position(offset)
                    floatBuffer.get(emb)
                    tokenEmbeddingTable = emb
                    totalBytesAllocated += embSize * 4L
                } else {
                    tokenEmbeddingTable = null
                }
                offset += embSize

                // Helper to load sliced layer matrix for assigned transformer blocks
                fun loadLayerSlice(perLayerFloats: Int): FloatArray {
                    val skipBefore = minL * perLayerFloats
                    val sliceFloats = numAssignedLayers * perLayerFloats
                    val arr = FloatArray(sliceFloats)
                    floatBuffer.position(offset + skipBefore)
                    floatBuffer.get(arr)
                    offset += cfg.nLayers * perLayerFloats
                    totalBytesAllocated += sliceFloats * 4L
                    return arr
                }

                // 2. RMS Attention weights: (dim per layer)
                rmsAttWeight = loadLayerSlice(cfg.dim)

                // 3. WQ: (dim * dim per layer)
                wq = loadLayerSlice(cfg.dim * cfg.dim)

                // 4. WK: (kvDim * dim per layer)
                wk = loadLayerSlice(cfg.kvDim * cfg.dim)

                // 5. WV: (kvDim * dim per layer)
                wv = loadLayerSlice(cfg.kvDim * cfg.dim)

                // 6. WO: (dim * dim per layer)
                wo = loadLayerSlice(cfg.dim * cfg.dim)

                // 7. RMS FFN: (dim per layer)
                rmsFfnWeight = loadLayerSlice(cfg.dim)

                // 8. W1 (gate_proj): (hiddenDim * dim per layer)
                w1 = loadLayerSlice(cfg.hiddenDim * cfg.dim)

                // 9. W2 (down_proj): (dim * hiddenDim per layer)
                w2 = loadLayerSlice(cfg.dim * cfg.hiddenDim)

                // 10. W3 (up_proj): (hiddenDim * dim per layer)
                w3 = loadLayerSlice(cfg.hiddenDim * cfg.dim)

                // 11. Final RMSNorm: (dim) - only output node needs final norm
                val rmsFinalSize = cfg.dim
                if (isOutputNode) {
                    val rmsFinal = FloatArray(rmsFinalSize)
                    floatBuffer.position(offset)
                    floatBuffer.get(rmsFinal)
                    rmsFinalWeight = rmsFinal
                    totalBytesAllocated += rmsFinalSize * 4L
                } else {
                    rmsFinalWeight = null
                }
                offset += rmsFinalSize

                // Skip RoPE tables if present
                offset += (cfg.seqLen * cfg.headSize / 2) // freq_cis_real
                offset += (cfg.seqLen * cfg.headSize / 2) // freq_cis_imag

                // 12. Classifier (wcls): (vocab_size, dim) - only output node needs LM head
                if (isOutputNode) {
                    if (cfg.sharedWeights) {
                        if (tokenEmbeddingTable != null) {
                            wcls = tokenEmbeddingTable
                        } else {
                            val curPos = floatBuffer.position()
                            floatBuffer.position(0) // Read embedding table at offset 0 for tied classifier
                            val clsArr = FloatArray(cfg.vocabSize * cfg.dim)
                            floatBuffer.get(clsArr)
                            wcls = clsArr
                            totalBytesAllocated += clsArr.size * 4L
                            floatBuffer.position(curPos)
                        }
                    } else {
                        val wclsSize = cfg.vocabSize * cfg.dim
                        val wclsArr = FloatArray(wclsSize)
                        floatBuffer.position(offset)
                        floatBuffer.get(wclsArr)
                        wcls = wclsArr
                        totalBytesAllocated += wclsSize * 4L
                        offset += wclsSize
                    }
                } else {
                    wcls = null
                    if (!cfg.sharedWeights) {
                        offset += cfg.vocabSize * cfg.dim
                    }
                }

                // Allocate runtime activation buffers & KV Cache in RAM for assigned layers only
                allocateRunBuffers(cfg, numAssignedLayers, isOutputNode)
                totalBytesAllocated += (keyCache?.size ?: 0) * 4L + (valCache?.size ?: 0) * 4L

                allocatedMemoryBytes = totalBytesAllocated
                isLoadedInRam = true
                loadedFilePath = file.absolutePath

                Log.i(TAG, "Successfully loaded ${file.name} into Node RAM! Transformer blocks [$minL..$maxL] ($numAssignedLayers layers), Allocated: $allocatedMemoryMb MB")
                return true
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed loading model into RAM: ${e.message}", e)
            isLoadedInRam = false
            return false
        }
    }

    private fun allocateRunBuffers(cfg: Config, numLayers: Int, isOutput: Boolean) {
        val maxTokens = cfg.seqLen
        val kvDim = cfg.kvDim

        keyCache = FloatArray(numLayers * maxTokens * kvDim)
        valCache = FloatArray(numLayers * maxTokens * kvDim)

        xb = FloatArray(cfg.dim)
        xb2 = FloatArray(cfg.dim)
        hb = FloatArray(cfg.hiddenDim)
        hb2 = FloatArray(cfg.hiddenDim)
        q = FloatArray(cfg.dim)
        k = FloatArray(cfg.kvDim)
        v = FloatArray(cfg.kvDim)
        att = FloatArray(cfg.nHeads * maxTokens)
        logits = if (isOutput) FloatArray(cfg.vocabSize) else null
    }

    /**
     * Resets the KV cache across all layers for a fresh prompt.
     */
    fun resetKvCache() {
        keyCache?.fill(0f)
        valCache?.fill(0f)
    }

    /**
     * Executes the forward pass for assigned layer slice from startLayer to endLayer.
     *
     * @param token Context token ID (used if isInput=true to look up embedding).
     * @param pos Current sequence position for RoPE and KV-cache.
     * @param incomingHidden Activation vector from previous mesh node (if not input block).
     * @param startLayer Index of first layer assigned to this node.
     * @param endLayer Index of last layer assigned to this node.
     * @param isInput True if this node executes embedding lookup.
     * @param isOutput True if this node executes final RMSNorm and LM Head projection.
     */
    fun forward(
        token: Int,
        pos: Int,
        incomingHidden: FloatArray?,
        startLayer: Int,
        endLayer: Int,
        isInput: Boolean,
        isOutput: Boolean,
        recentTokens: List<Int> = emptyList()
    ): ForwardResult {
        val cfg = config ?: return ForwardResult(FloatArray(64), null, false)
        val dim = cfg.dim
        val hiddenDim = cfg.hiddenDim
        val headSize = cfg.headSize
        val kvDim = cfg.kvDim
        val kvMul = cfg.kvMul

        val emb = tokenEmbeddingTable
        val wqArr = wq
        val wkArr = wk
        val wvArr = wv
        val woArr = wo
        val w1Arr = w1
        val w2Arr = w2
        val w3Arr = w3
        val rmsAtt = rmsAttWeight
        val rmsFfn = rmsFfnWeight
        val rmsFinal = rmsFinalWeight
        val wclsArr = wcls

        val kCache = keyCache ?: FloatArray(1)
        val vCache = valCache ?: FloatArray(1)

        val xBuf: FloatArray = if (isInput) {
            val safeToken = token.coerceIn(0, cfg.vocabSize - 1)
            val xOut = FloatArray(dim)
            if (emb != null) {
                System.arraycopy(emb, safeToken * dim, xOut, 0, dim)
            }
            xOut
        } else {
            incomingHidden?.copyOf() ?: FloatArray(dim)
        }

        val localXb = xb ?: FloatArray(dim)
        val localXb2 = xb2 ?: FloatArray(dim)
        val localHb = hb ?: FloatArray(hiddenDim)
        val localHb2 = hb2 ?: FloatArray(hiddenDim)
        val localQ = q ?: FloatArray(dim)
        val localK = k ?: FloatArray(kvDim)
        val localV = v ?: FloatArray(kvDim)
        val localAtt = att ?: FloatArray(cfg.nHeads * cfg.seqLen)

        val boundedPos = min(pos, cfg.seqLen - 1)
        val actualStart = max(0, startLayer)
        val actualEnd = min(endLayer, cfg.nLayers - 1)

        // Iterate through this node's assigned layers
        for (l in actualStart..actualEnd) {
            val blk = l - loadedStartLayer

            // 1. Attention RMSNorm
            if (rmsAtt != null) {
                rmsnorm(localXb, xBuf, rmsAtt, blk * dim, dim)
            }

            // 2. QKV Matmuls
            if (wqArr != null) {
                matmul(localQ, localXb, wqArr, blk * dim * dim, dim, dim)
            }
            if (wkArr != null) {
                matmul(localK, localXb, wkArr, blk * kvDim * dim, dim, kvDim)
            }
            if (wvArr != null) {
                matmul(localV, localXb, wvArr, blk * kvDim * dim, dim, kvDim)
            }

            // 3. RoPE relative positional encoding
            for (i in 0 until dim step 2) {
                val hDim = i % headSize
                val freq = 1.0f / (10000.0f.pow(hDim.toFloat() / headSize))
                val angle = boundedPos * freq
                val fcr = cos(angle)
                val fci = sin(angle)

                val q0 = localQ[i]
                val q1 = localQ[i + 1]
                localQ[i] = q0 * fcr - q1 * fci
                localQ[i + 1] = q0 * fci + q1 * fcr

                if (i < kvDim) {
                    val k0 = localK[i]
                    val k1 = localK[i + 1]
                    localK[i] = k0 * fcr - k1 * fci
                    localK[i + 1] = k0 * fci + k1 * fcr
                }
            }

            // Store K, V in KV cache
            val layerKvOffset = blk * cfg.seqLen * kvDim + boundedPos * kvDim
            if (layerKvOffset + kvDim <= kCache.size) {
                System.arraycopy(localK, 0, kCache, layerKvOffset, kvDim)
                System.arraycopy(localV, 0, vCache, layerKvOffset, kvDim)
            }

            // 4. Multi-Head Attention
            localXb2.fill(0f)
            val invSqrtHead = 1.0f / sqrt(headSize.toFloat())

            for (h in 0 until cfg.nHeads) {
                val qHeadOff = h * headSize
                val attHeadOff = h * cfg.seqLen

                // Attention scores for all positions up to current pos
                var maxScore = -Float.MAX_VALUE
                for (t in 0..boundedPos) {
                    val kvTStepOff = blk * cfg.seqLen * kvDim + t * kvDim + (h / kvMul) * headSize
                    var dot = 0.0f
                    for (d in 0 until headSize) {
                        dot += localQ[qHeadOff + d] * kCache[kvTStepOff + d]
                    }
                    val score = dot * invSqrtHead
                    localAtt[attHeadOff + t] = score
                    if (score > maxScore) maxScore = score
                }

                // Softmax
                var sumExp = 0.0f
                for (t in 0..boundedPos) {
                    val e = exp(localAtt[attHeadOff + t] - maxScore)
                    localAtt[attHeadOff + t] = e
                    sumExp += e
                }
                val invSumExp = 1.0f / sumExp.coerceAtLeast(1e-9f)
                for (t in 0..boundedPos) {
                    localAtt[attHeadOff + t] *= invSumExp
                }

                // Weighted sum over V values
                for (d in 0 until headSize) {
                    var weightedV = 0.0f
                    for (t in 0..boundedPos) {
                        val kvTStepOff = blk * cfg.seqLen * kvDim + t * kvDim + (h / kvMul) * headSize
                        weightedV += localAtt[attHeadOff + t] * vCache[kvTStepOff + d]
                    }
                    localXb2[qHeadOff + d] = weightedV
                }
            }

            // 5. Output projection + Residual Add: x = x + Wo @ xb2
            if (woArr != null) {
                matmul(localXb, localXb2, woArr, blk * dim * dim, dim, dim)
                for (i in 0 until dim) {
                    xBuf[i] += localXb[i]
                }
            }

            // 6. FFN RMSNorm
            if (rmsFfn != null) {
                rmsnorm(localXb, xBuf, rmsFfn, blk * dim, dim)
            }

            // 7. SwiGLU: SiLU(w1 @ xb) * (w3 @ xb)
            if (w1Arr != null && w3Arr != null && w2Arr != null) {
                matmul(localHb, localXb, w1Arr, blk * hiddenDim * dim, dim, hiddenDim)
                matmul(localHb2, localXb, w3Arr, blk * hiddenDim * dim, dim, hiddenDim)

                for (i in 0 until hiddenDim) {
                    val gate = localHb[i] * (1.0f / (1.0f + exp(-localHb[i].toDouble()).toFloat()))
                    localHb[i] = gate * localHb2[i]
                }

                matmul(localXb, localHb, w2Arr, blk * dim * hiddenDim, hiddenDim, dim)
                for (i in 0 until dim) {
                    xBuf[i] += localXb[i]
                }
            }
        }

        // If this is the output head, perform final RMSNorm and sample next token
        var sampledTokenId: Int? = null
        if (isOutput) {
            if (rmsFinal != null) {
                rmsnorm(xBuf, xBuf, rmsFinal, 0, dim)
            }

            val localLogits = logits ?: FloatArray(cfg.vocabSize)
            if (wclsArr != null) {
                matmul(localLogits, xBuf, wclsArr, 0, dim, cfg.vocabSize)
            }

            // Argmax / Temperature sampling with repetition & loop penalties
            sampledTokenId = sampleToken(
                logits = localLogits,
                temperature = 0.75f,
                recentTokens = recentTokens,
                repeatPenalty = 1.25f,
                frequencyPenalty = 0.35f,
                presencePenalty = 0.25f
            )
        }

        return ForwardResult(
            hiddenState = xBuf,
            sampledTokenId = sampledTokenId,
            isFinished = (sampledTokenId == 2) // Token 2 = EOS in LLaMA
        )
    }

    private fun sampleToken(
        logits: FloatArray,
        temperature: Float = 0.75f,
        recentTokens: List<Int> = emptyList(),
        repeatPenalty: Float = 1.25f,
        frequencyPenalty: Float = 0.35f,
        presencePenalty: Float = 0.25f
    ): Int {
        // Clone logits so original buffer isn't permanently corrupted
        val penalizedLogits = logits.clone()

        // 1. Multiplicative Repetition Penalty & Additive Frequency/Presence Penalties
        if (recentTokens.isNotEmpty()) {
            val window = recentTokens.takeLast(64)
            val counts = HashMap<Int, Int>()
            for (tok in window) {
                counts[tok] = (counts[tok] ?: 0) + 1
            }

            for ((tok, count) in counts) {
                if (tok in 0 until penalizedLogits.size) {
                    if (penalizedLogits[tok] > 0) {
                        penalizedLogits[tok] /= repeatPenalty
                    } else {
                        penalizedLogits[tok] *= repeatPenalty
                    }
                    penalizedLogits[tok] -= (count * frequencyPenalty + presencePenalty)
                }
            }

            // 2. Prevent consecutive token repeating (e.g. "the the the")
            val lastTok = recentTokens.lastOrNull()
            if (lastTok != null && lastTok in 0 until penalizedLogits.size) {
                penalizedLogits[lastTok] -= 6.0f
            }

            // 3. Prevent 2-gram and 3-gram repeating loops (e.g. "the most important element the the...")
            val n = recentTokens.size
            if (n >= 4) {
                val last1 = recentTokens[n - 1]
                val last2 = recentTokens[n - 2]
                val last3 = if (n >= 5) recentTokens[n - 3] else -1

                for (i in 0 until n - 3) {
                    if (recentTokens[i] == last2 && recentTokens[i + 1] == last1) {
                        val forbiddenTok = recentTokens[i + 2]
                        if (forbiddenTok in 0 until penalizedLogits.size) {
                            penalizedLogits[forbiddenTok] -= 8.0f
                        }
                    }
                    if (last3 != -1 && i < n - 4 && recentTokens[i] == last3 && recentTokens[i + 1] == last2 && recentTokens[i + 2] == last1) {
                        val forbiddenTok = recentTokens[i + 3]
                        if (forbiddenTok in 0 until penalizedLogits.size) {
                            penalizedLogits[forbiddenTok] = -1e9f
                        }
                    }
                }
            }
        }

        if (temperature <= 0.05f) {
            // Greedy argmax
            var maxIdx = 0
            var maxVal = penalizedLogits[0]
            for (i in 1 until penalizedLogits.size) {
                if (penalizedLogits[i] > maxVal) {
                    maxVal = penalizedLogits[i]
                    maxIdx = i
                }
            }
            return maxIdx
        }

        // Top-K sampling with temperature
        val topK = min(40, penalizedLogits.size)
        // Find top-k values
        val indexedLogits = Array(penalizedLogits.size) { i -> i to penalizedLogits[i] }
        indexedLogits.sortByDescending { it.second }

        val topTokens = IntArray(topK)
        val topProbs = FloatArray(topK)

        var maxVal = indexedLogits[0].second
        var sumExp = 0.0f
        for (i in 0 until topK) {
            val (idx, l) = indexedLogits[i]
            topTokens[i] = idx
            val p = exp((l - maxVal) / temperature)
            topProbs[i] = p
            sumExp += p
        }

        val r = Random.nextFloat() * sumExp
        var accum = 0.0f
        for (i in 0 until topK) {
            accum += topProbs[i]
            if (accum >= r) {
                return topTokens[i]
            }
        }
        return topTokens[0]
    }

    private fun rmsnorm(out: FloatArray, x: FloatArray, w: FloatArray, wOff: Int, size: Int) {
        var ss = 0.0f
        for (i in 0 until size) {
            ss += x[i] * x[i]
        }
        ss = (ss / size) + 1e-5f
        val invRms = 1.0f / sqrt(ss.toDouble()).toFloat()
        for (i in 0 until size) {
            out[i] = w[wOff + i] * (x[i] * invRms)
        }
    }

    private fun matmul(xout: FloatArray, x: FloatArray, w: FloatArray, wOff: Int, n: Int, d: Int) {
        for (i in 0 until d) {
            var sum = 0.0f
            val rowOff = wOff + i * n
            for (j in 0 until n) {
                sum += w[rowOff + j] * x[j]
            }
            xout[i] = sum
        }
    }

    data class ForwardResult(
        val hiddenState: FloatArray,
        val sampledTokenId: Int?,
        val isFinished: Boolean
    )
}
