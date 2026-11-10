package org.meshllm.node.model

/**
 * Standard error codes for distributed LLM inference.
 * Replaces any simulated/fallback responses with clear actionable error diagnostics.
 */
enum class LlmErrorCode(val code: String, val title: String) {
    ERR_MODEL_NOT_DOWNLOADED("E101", "Model Weights Not Downloaded"),
    ERR_WEIGHTS_CORRUPTED("E102", "Corrupted or Invalid Weights Binary"),
    ERR_OUT_OF_MEMORY("E103", "Out Of Memory (Node RAM Exceeded)"),
    ERR_TOKENIZER_MISSING("E104", "Tokenizer Vocabulary Missing"),
    ERR_NODE_DISCONNECTED("E201", "Mesh Node Disconnected"),
    ERR_PIPELINE_TIMEOUT("E202", "Distributed Pipeline Timeout"),
    ERR_SESSION_INVALID("E301", "Invalid Mesh Session Token"),
    ERR_THERMAL_CRITICAL("E401", "Thermal Protection Triggered");

    fun format(detail: String): String = "[$code $title] $detail"
}

class LlmInferenceException(
    val errorCode: LlmErrorCode,
    val detail: String,
    cause: Throwable? = null
) : Exception(errorCode.format(detail), cause)
