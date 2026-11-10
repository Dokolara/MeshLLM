package org.meshllm.node.engine

import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * High-performance parser and validator for GGUF (GGML Universal File Format v2 & v3)
 * models downloaded to mobile storage.
 *
 * Extracts architecture details (layers, hidden dim, head count), tensor directory,
 * and tokenizer metadata directly from local disk files without loading the entire
 * multi-gigabyte weight payload into JVM heap.
 */
class GGUFParser {

    companion object {
        private const val TAG = "GGUFParser"
        const val GGUF_MAGIC = 0x46554747 // "GGUF" in little-endian

        // GGUF Value Types
        const val TYPE_UINT8 = 0
        const val TYPE_INT8 = 1
        const val TYPE_UINT16 = 2
        const val TYPE_INT16 = 3
        const val TYPE_UINT32 = 4
        const val TYPE_INT32 = 5
        const val TYPE_FLOAT32 = 6
        const val TYPE_BOOL = 7
        const val TYPE_STRING = 8
        const val TYPE_ARRAY = 9
        const val TYPE_UINT64 = 10
        const val TYPE_INT64 = 11
        const val TYPE_FLOAT64 = 12
    }

    data class GGUFMetadata(
        val version: Int,
        val tensorCount: Long,
        val kvCount: Long,
        val architecture: String,
        val modelName: String,
        val blockCount: Int,
        val hiddenDim: Int,
        val feedForwardDim: Int,
        val headCount: Int,
        val headCountKv: Int,
        val contextLength: Int,
        val vocabSize: Int,
        val sampleVocabTokens: List<String>,
        val fileSizeBytes: Long,
        val tensorNames: List<String>,
        val dataOffset: Long
    ) {
        val fileSizeMb: Double
            get() = Math.round((fileSizeBytes.toDouble() / (1024.0 * 1024.0)) * 10.0) / 10.0

        val summaryString: String
            get() = "GGUF v$version • Arch: $architecture • $blockCount Layers • Dim: $hiddenDim • Heads: $headCount • $tensorCount Tensors (${fileSizeMb} MB)"
    }

    /**
     * Inspects and parses a GGUF file from disk.
     * Returns null if file is not a valid GGUF file.
     */
    fun parse(file: File): GGUFMetadata? {
        if (!file.exists() || file.length() < 32) return null

        try {
            RandomAccessFile(file, "r").use { raf ->
                val channel = raf.channel
                // Map the first 4MB which comfortably holds header and key-value metadata
                val mapSize = Math.min(channel.size(), 4L * 1024 * 1024)
                val buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, mapSize)
                buffer.order(ByteOrder.LITTLE_ENDIAN)

                // 1. Validate Magic
                val magic = buffer.int
                if (magic != GGUF_MAGIC) {
                    Log.d(TAG, "File ${file.name} magic 0x${Integer.toHexString(magic)} != GGUF (0x${Integer.toHexString(GGUF_MAGIC)})")
                    return null
                }

                // 2. Read Version
                val version = buffer.int
                if (version !in 1..4) {
                    Log.w(TAG, "Unsupported GGUF version $version in ${file.name}")
                    return null
                }

                // 3. Tensor and KV counts
                val tensorCount = buffer.long
                val kvCount = buffer.long

                var architecture = "unknown"
                var modelName = file.nameWithoutExtension
                var blockCount = 0
                var hiddenDim = 0
                var feedForwardDim = 0
                var headCount = 0
                var headCountKv = 0
                var contextLength = 2048
                var vocabSize = 0
                val sampleVocab = mutableListOf<String>()

                // 4. Parse Metadata KV pairs
                for (i in 0 until kvCount) {
                    if (buffer.remaining() < 12) break
                    val key = readGGUFString(buffer) ?: break
                    val vType = buffer.int

                    when (vType) {
                        TYPE_UINT8, TYPE_INT8, TYPE_BOOL -> {
                            buffer.get()
                        }
                        TYPE_UINT16, TYPE_INT16 -> {
                            buffer.short
                        }
                        TYPE_UINT32, TYPE_INT32 -> {
                            val v = buffer.int
                            if (key.endsWith(".block_count") || key.endsWith(".layer_count")) blockCount = v
                            else if (key.endsWith(".embedding_length") || key == "general.hidden_size") hiddenDim = v
                            else if (key.endsWith(".feed_forward_length")) feedForwardDim = v
                            else if (key.endsWith(".head_count") && !key.endsWith(".head_count_kv")) headCount = v
                            else if (key.endsWith(".head_count_kv")) headCountKv = v
                            else if (key.endsWith(".context_length")) contextLength = v
                        }
                        TYPE_UINT64, TYPE_INT64 -> {
                            val v = buffer.long
                            if (key.endsWith(".block_count") || key.endsWith(".layer_count")) blockCount = v.toInt()
                            else if (key.endsWith(".embedding_length")) hiddenDim = v.toInt()
                            else if (key.endsWith(".context_length")) contextLength = v.toInt()
                        }
                        TYPE_FLOAT32 -> {
                            buffer.float
                        }
                        TYPE_FLOAT64 -> {
                            buffer.double
                        }
                        TYPE_STRING -> {
                            val strVal = readGGUFString(buffer) ?: ""
                            if (key == "general.architecture") architecture = strVal
                            else if (key == "general.name") modelName = strVal
                        }
                        TYPE_ARRAY -> {
                            val elemType = buffer.int
                            val elemCount = buffer.long
                            if (key == "tokenizer.ggml.tokens" && elemType == TYPE_STRING) {
                                vocabSize = elemCount.toInt()
                                val previewLimit = Math.min(elemCount, 50L)
                                for (t in 0 until previewLimit) {
                                    val tokenStr = readGGUFString(buffer)
                                    if (tokenStr != null) sampleVocab.add(tokenStr)
                                }
                                // Skip remaining string tokens in array to conserve parse time
                                for (t in previewLimit until elemCount) {
                                    val len = buffer.long
                                    if (len in 0..buffer.remaining()) {
                                        buffer.position(buffer.position() + len.toInt())
                                    } else break
                                }
                            } else {
                                skipGGUFArray(buffer, elemType, elemCount)
                            }
                        }
                        else -> {
                            // Unknown type, exit KV loop
                            break
                        }
                    }
                }

                // 5. Read Tensor directory preview (first 10 tensors)
                val tensorNames = mutableListOf<String>()
                val previewTensors = Math.min(tensorCount, 20L)
                for (t in 0 until previewTensors) {
                    if (buffer.remaining() < 16) break
                    val tName = readGGUFString(buffer) ?: break
                    tensorNames.add(tName)
                    val nDims = buffer.int
                    if (nDims in 1..8) {
                        for (d in 0 until nDims) buffer.long
                        buffer.int // type
                        buffer.long // offset
                    } else break
                }

                // Derive fallback architecture specs if not explicitly specified in metadata
                if (blockCount <= 0) blockCount = if (architecture.contains("14b", true)) 48 else if (architecture.contains("7b", true) || architecture.contains("8b", true)) 32 else 24
                if (hiddenDim <= 0) hiddenDim = if (architecture.contains("14b", true)) 5120 else if (architecture.contains("7b", true) || architecture.contains("8b", true)) 4096 else 896
                if (headCount <= 0) headCount = if (architecture.contains("14b", true)) 40 else 32

                Log.i(TAG, "Parsed GGUF header for ${file.name}: arch=$architecture, layers=$blockCount, dim=$hiddenDim, tensors=$tensorCount")

                return GGUFMetadata(
                    version = version,
                    tensorCount = tensorCount,
                    kvCount = kvCount,
                    architecture = architecture,
                    modelName = modelName,
                    blockCount = blockCount,
                    hiddenDim = hiddenDim,
                    feedForwardDim = feedForwardDim,
                    headCount = headCount,
                    headCountKv = headCountKv,
                    contextLength = contextLength,
                    vocabSize = vocabSize,
                    sampleVocabTokens = sampleVocab,
                    fileSizeBytes = file.length(),
                    tensorNames = tensorNames,
                    dataOffset = buffer.position().toLong()
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse GGUF ${file.name}", e)
            return null
        }
    }

    private fun readGGUFString(buffer: ByteBuffer): String? {
        if (buffer.remaining() < 8) return null
        val len = buffer.long
        if (len < 0 || len > buffer.remaining() || len > 1024 * 1024) return null
        val bytes = ByteArray(len.toInt())
        buffer.get(bytes)
        return String(bytes, Charsets.UTF_8)
    }

    private fun skipGGUFArray(buffer: ByteBuffer, elemType: Int, count: Long) {
        if (count <= 0) return
        when (elemType) {
            TYPE_UINT8, TYPE_INT8, TYPE_BOOL -> {
                val skip = Math.min(count, buffer.remaining().toLong()).toInt()
                buffer.position(buffer.position() + skip)
            }
            TYPE_UINT16, TYPE_INT16 -> {
                val skip = Math.min(count * 2, buffer.remaining().toLong()).toInt()
                buffer.position(buffer.position() + skip)
            }
            TYPE_UINT32, TYPE_INT32, TYPE_FLOAT32 -> {
                val skip = Math.min(count * 4, buffer.remaining().toLong()).toInt()
                buffer.position(buffer.position() + skip)
            }
            TYPE_UINT64, TYPE_INT64, TYPE_FLOAT64 -> {
                val skip = Math.min(count * 8, buffer.remaining().toLong()).toInt()
                buffer.position(buffer.position() + skip)
            }
            TYPE_STRING -> {
                for (c in 0 until count) {
                    if (buffer.remaining() < 8) break
                    val sLen = buffer.long
                    if (sLen in 0..buffer.remaining()) {
                        buffer.position(buffer.position() + sLen.toInt())
                    } else break
                }
            }
        }
    }
}
