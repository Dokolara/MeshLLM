package org.meshllm.node.engine

/**
 * Result of a native pipeline forward or prefill step.
 */
data class NativePipelineResult(
    val sampledToken: Int = -1,
    val piece: String = "",
    val isFinished: Boolean = false,
    val outputHidden: FloatArray? = null
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as NativePipelineResult

        if (sampledToken != other.sampledToken) return false
        if (piece != other.piece) return false
        if (isFinished != other.isFinished) return false
        if (outputHidden != null) {
            if (other.outputHidden == null) return false
            if (!outputHidden.contentEquals(other.outputHidden)) return false
        } else if (other.outputHidden != null) return false

        return true
    }

    override fun hashCode(): Int {
        var result = sampledToken
        result = 31 * result + piece.hashCode()
        result = 31 * result + isFinished.hashCode()
        result = 31 * result + (outputHidden?.contentHashCode() ?: 0)
        return result
    }
}
