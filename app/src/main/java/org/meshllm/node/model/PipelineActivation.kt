package org.meshllm.node.model

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64

data class PipelineActivation(
    val sessionToken: String,
    val requestId: String,
    val promptId: String,
    val tokenIndex: Int,
    val blockId: String,
    val senderNodeId: String,
    val receiverNodeId: String,
    val hiddenState: List<Float> = emptyList(),
    val hiddenStateBase64: String? = null,
    val generatedTokenId: Int? = null,
    val generatedTextPiece: String? = null,
    val isFinished: Boolean = false,
    val executionTimeMs: Long = 0,
    val timestamp: Long = System.currentTimeMillis(),
    val seqPos: Int = 0,
    val promptTokensCount: Int = 0
) {
    /**
     * Efficient FloatArray extraction.
     * Decodes compact Base64 binary payload if present; falls back to List<Float>.
     */
    fun extractHiddenState(): FloatArray {
        if (!hiddenStateBase64.isNullOrEmpty()) {
            return decodeBase64ToFloatArray(hiddenStateBase64)
        }
        if (hiddenState.isNotEmpty()) {
            return hiddenState.toFloatArray()
        }
        return FloatArray(0)
    }

    companion object {
        fun encodeFloatArrayToBase64(floats: FloatArray): String {
            if (floats.isEmpty()) return ""
            val byteBuffer = ByteBuffer.allocate(floats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            val fb = byteBuffer.asFloatBuffer()
            fb.put(floats)
            return Base64.getEncoder().encodeToString(byteBuffer.array())
        }

        fun decodeBase64ToFloatArray(base64Str: String): FloatArray {
            if (base64Str.isEmpty()) return FloatArray(0)
            return try {
                val bytes = Base64.getDecoder().decode(base64Str)
                val byteBuffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                val fb = byteBuffer.asFloatBuffer()
                val floats = FloatArray(fb.remaining())
                fb.get(floats)
                floats
            } catch (_: Exception) {
                FloatArray(0)
            }
        }
    }
}
