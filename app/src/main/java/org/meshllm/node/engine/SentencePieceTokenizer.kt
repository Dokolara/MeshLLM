package org.meshllm.node.engine

import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * SentencePiece BPE tokenizer that loads official llama2.c binary tokenizer files
 * (e.g. tokenizer.bin for 32k LLaMA models, tok512.bin for TinyStories 260K).
 */
class SentencePieceTokenizer {

    var maxTokenLength: Int = 32
        private set

    private val vocab = mutableListOf<String>()
    private val scores = mutableListOf<Float>()
    private val tokenToId = mutableMapOf<String, Int>()

    val vocabSize: Int
        get() = vocab.size

    val isLoaded: Boolean
        get() = vocab.isNotEmpty()

    fun loadFromFile(file: File): Boolean {
        if (!file.exists() || file.length() < 16) return false
        return try {
            FileInputStream(file).use { loadFromStream(it) }
        } catch (_: Exception) {
            false
        }
    }

    fun loadFromStream(inputStream: InputStream): Boolean {
        return try {
            val bytes = inputStream.readBytes()
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            if (buffer.remaining() < 4) return false
            maxTokenLength = buffer.int

            vocab.clear()
            scores.clear()
            tokenToId.clear()

            while (buffer.remaining() >= 8) {
                val score = buffer.float
                val len = buffer.int
                if (buffer.remaining() < len) break
                val strBytes = ByteArray(len)
                buffer.get(strBytes)
                val token = String(strBytes, Charsets.UTF_8)
                val id = vocab.size
                vocab.add(token)
                scores.add(score)
                tokenToId[token] = id
            }
            vocab.isNotEmpty()
        } catch (e: Exception) {
            false
        }
    }

    fun decode(tokenId: Int): String {
        if (tokenId !in 0 until vocab.size) return ""
        val piece = vocab[tokenId]
        // Clean special tokens
        return when (piece) {
            "<s>", "</s>", "\n<s>\n", "\n</s>\n", "<unk>" -> ""
            else -> piece
        }
    }

    fun encode(text: String): List<Int> {
        if (vocab.isEmpty()) return listOf(1)
        val tokens = mutableListOf<Int>()
        tokens.add(1) // BOS token

        var normalized = text.trim()
        if (!normalized.startsWith(" ")) {
            normalized = " $normalized"
        }

        var i = 0
        while (i < normalized.length) {
            var bestId = -1
            var bestLen = 0
            val maxL = kotlin.math.min(maxTokenLength, normalized.length - i)
            for (l in maxL downTo 1) {
                val sub = normalized.substring(i, i + l)
                val id = tokenToId[sub]
                if (id != null) {
                    bestId = id
                    bestLen = l
                    break
                }
            }
            if (bestId != -1) {
                tokens.add(bestId)
                i += bestLen
            } else {
                // If single char not in vocab, try case-insensitive or space prefix
                val ch = normalized[i].toString()
                val altId = tokenToId[ch] ?: tokenToId[" $ch"] ?: 0
                tokens.add(altId)
                i++
            }
        }
        return tokens
    }
}
