package org.meshllm.node.telemetry

import android.util.Log
import com.google.gson.Gson
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.meshllm.node.model.ModelCatalog
import org.meshllm.node.model.ModelSpec
import java.net.URLEncoder
import java.util.concurrent.TimeUnit
import java.util.regex.Pattern

/**
 * Dynamic Hugging Face Hub client for real-time model discovery and multi-quantization selection.
 * Directly queries the official Hugging Face API so no models or download URLs are hardcoded.
 */
class HuggingFaceClient {

    companion object {
        private const val TAG = "HuggingFaceClient"
        private const val HF_API_BASE = "https://huggingface.co/api"
    }

    private val gson = Gson()
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    /**
     * Search Hugging Face dynamically for GGUF models matching [query].
     * Returns a list of [ModelSpec]s with all available quantizations (Q2_K, Q3_K_M, Q4_K_M, etc.).
     */
    suspend fun searchModels(query: String, limit: Int = 15): List<ModelSpec> = withContext(Dispatchers.IO) {
        val cleanQuery = query.trim()
        if (cleanQuery.isEmpty()) return@withContext emptyList()

        try {
            val encodedQuery = URLEncoder.encode(cleanQuery, "UTF-8")
            val url = "$HF_API_BASE/models?search=$encodedQuery&filter=gguf&sort=downloads&direction=-1&limit=$limit"

            val request = Request.Builder()
                .url(url)
                .addHeader("User-Agent", "MeshLLM-Android-Node/1.0")
                .build()

            httpClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.w(TAG, "HF search failed: HTTP ${response.code}")
                    return@withContext emptyList()
                }

                val body = response.body?.string() ?: return@withContext emptyList()
                val jsonArray = gson.fromJson(body, JsonArray::class.java)

                val results = mutableListOf<ModelSpec>()
                for (element in jsonArray) {
                    if (!element.isJsonObject) continue
                    val obj = element.asJsonObject
                    val specs = parseModelObject(obj)
                    results.addAll(specs)
                }

                Log.i(TAG, "Found ${results.size} model quantizations on Hugging Face for '$cleanQuery'")
                return@withContext results
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error searching Hugging Face: ${e.message}", e)
            return@withContext emptyList()
        }
    }

    /**
     * Fetch all available GGUF quantization files with exact file sizes for a specific Hugging Face repository.
     */
    suspend fun fetchRepoQuantizations(repoId: String): List<ModelSpec> = withContext(Dispatchers.IO) {
        try {
            val url = "$HF_API_BASE/models/$repoId/tree/main"
            val request = Request.Builder()
                .url(url)
                .addHeader("User-Agent", "MeshLLM-Android-Node/1.0")
                .build()

            httpClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    return@withContext emptyList()
                }

                val body = response.body?.string() ?: return@withContext emptyList()
                val treeArray = gson.fromJson(body, JsonArray::class.java)

                val results = mutableListOf<ModelSpec>()
                for (item in treeArray) {
                    if (!item.isJsonObject) continue
                    val fileObj = item.asJsonObject
                    val path = fileObj.get("path")?.asString ?: continue
                    if (!path.endsWith(".gguf", ignoreCase = true)) continue

                    val size = fileObj.get("size")?.asLong ?: 0L
                    val spec = createModelSpec(repoId, path, size)
                    results.add(spec)
                }

                return@withContext results.sortedBy { it.fileSizeBytes }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error fetching repo tree for $repoId: ${e.message}", e)
            return@withContext emptyList()
        }
    }

    private fun parseModelObject(obj: JsonObject): List<ModelSpec> {
        val repoId = obj.get("id")?.asString ?: return emptyList()
        val siblings = obj.getAsJsonArray("siblings") ?: return emptyList()

        val results = mutableListOf<ModelSpec>()
        for (sib in siblings) {
            if (!sib.isJsonObject) continue
            val rfilename = sib.asJsonObject.get("rfilename")?.asString ?: continue
            if (rfilename.endsWith(".gguf", ignoreCase = true)) {
                val spec = createModelSpec(repoId, rfilename, 0L)
                results.add(spec)
            }
        }

        return results
    }

    private fun createModelSpec(repoId: String, fileName: String, sizeBytes: Long): ModelSpec {
        val author = repoId.substringBefore('/')
        val modelBaseName = repoId.substringAfter('/').removeSuffix("-GGUF").removeSuffix("-gguf")
        val quant = extractQuantization(fileName)

        val id = "${repoId.replace('/', '_')}__$fileName"
        val displayName = "$modelBaseName [$quant]"

        // Infer parameters from name (e.g. 0.5B, 1B, 3B, 7B, 8B, 14B, 70B)
        val paramCount = inferParamCount(repoId + " " + fileName)
        val layers = inferLayers(paramCount)
        val hiddenDim = inferHiddenDim(paramCount)
        val category = inferCategory(paramCount)

        val downloadUrl = "https://huggingface.co/$repoId/resolve/main/$fileName"

        // Estimated size if not provided by tree endpoint
        val finalSizeBytes = if (sizeBytes > 0L) sizeBytes else estimateSizeBytes(paramCount, quant)
        val sizeMb = Math.round((finalSizeBytes.toDouble() / (1024.0 * 1024.0)) * 10.0) / 10.0

        val memReqStr = if (sizeMb >= 1024.0) {
            String.format("~%.2f GB ($quant)", sizeMb / 1024.0)
        } else {
            "~$sizeMb MB ($quant)"
        }

        val isMassive = paramCount >= 7.0
        val minDevices = when {
            paramCount >= 14.0 -> 6
            paramCount >= 7.0 -> 4
            paramCount >= 2.0 -> 2
            else -> 1
        }

        return ModelSpec(
            id = id,
            name = displayName,
            provider = author,
            category = category,
            totalLayers = layers,
            hiddenDim = hiddenDim,
            intermediateDim = (hiddenDim * 3.5).toInt(),
            numHeads = maxOf(4, hiddenDim / 128),
            contextLength = 4096,
            paramCountStr = if (paramCount >= 1.0) "${paramCount}B" else "${(paramCount * 1000).toInt()}M",
            memoryRequirementStr = memReqStr,
            description = "Dynamic Hugging Face model ($repoId) in $quant quantization. Optimized for pipeline parallel execution across $minDevices+ mobile devices.",
            samplePrompt = "Explain the fundamental advantages of dynamic layer partitioning across distributed edge devices:",
            huggingFaceRepo = repoId,
            downloadUrl = downloadUrl,
            fileName = fileName,
            fileSizeBytes = finalSizeBytes,
            fileFormat = "GGUF",
            recommendedMinDevices = minDevices,
            singleDeviceOomWarning = isMassive,
            isStarterTestModel = paramCount < 0.3
        )
    }

    private fun extractQuantization(fileName: String): String {
        val lower = fileName.lowercase()
        val quants = listOf(
            "q2_k_s", "q2_k_l", "q2_k",
            "q3_k_s", "q3_k_m", "q3_k_l", "q3_k",
            "q4_k_s", "q4_k_m", "q4_k", "q4_0", "q4_1",
            "q5_k_s", "q5_k_m", "q5_k", "q5_0", "q5_1",
            "q6_k",
            "q8_0", "q8_k",
            "fp16", "f16", "bf16", "fp32"
        )

        for (q in quants) {
            if (lower.contains(q)) return q.uppercase()
        }
        return "GGUF"
    }

    private fun inferParamCount(name: String): Double {
        val pattern = Pattern.compile("(?i)(\\d+(?:\\.\\d+)?)\\s*(b|m)")
        val matcher = pattern.matcher(name)
        if (matcher.find()) {
            val num = matcher.group(1)?.toDoubleOrNull() ?: 1.0
            val unit = matcher.group(2)?.lowercase() ?: "b"
            return if (unit == "m") num / 1000.0 else num
        }
        return 1.5 // Default reasonable mobile parameter estimate
    }

    private fun inferLayers(paramCount: Double): Int {
        return when {
            paramCount >= 30.0 -> 60
            paramCount >= 14.0 -> 48
            paramCount >= 7.0 -> 32
            paramCount >= 3.0 -> 28
            paramCount >= 1.0 -> 24
            paramCount >= 0.5 -> 24
            paramCount >= 0.1 -> 16
            else -> 8
        }
    }

    private fun inferHiddenDim(paramCount: Double): Int {
        return when {
            paramCount >= 14.0 -> 5120
            paramCount >= 7.0 -> 4096
            paramCount >= 3.0 -> 3072
            paramCount >= 1.0 -> 2048
            paramCount >= 0.5 -> 896
            else -> 576
        }
    }

    private fun inferCategory(paramCount: Double): String {
        return when {
            paramCount >= 6.0 -> ModelCatalog.CATEGORY_LARGE_MESH
            paramCount >= 1.0 -> ModelCatalog.CATEGORY_STANDARD
            paramCount >= 0.2 -> ModelCatalog.CATEGORY_COMPACT
            else -> ModelCatalog.CATEGORY_ULTRA_SMALL
        }
    }

    private fun estimateSizeBytes(paramCount: Double, quant: String): Long {
        val bytesPerParam = when {
            quant.contains("Q2") -> 0.35
            quant.contains("Q3") -> 0.48
            quant.contains("Q4") -> 0.60
            quant.contains("Q5") -> 0.72
            quant.contains("Q6") -> 0.85
            quant.contains("Q8") -> 1.10
            quant.contains("16") -> 2.00
            else -> 0.65
        }
        val totalBytes = (paramCount * 1_000_000_000.0 * bytesPerParam).toLong()
        return maxOf(50_000_000L, totalBytes)
    }
}
