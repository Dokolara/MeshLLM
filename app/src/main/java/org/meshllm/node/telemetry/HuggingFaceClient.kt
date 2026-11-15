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
 * Dynamic Hugging Face Hub client for real-time model discovery and direct GGUF fetching.
 * Directly queries the official Hugging Face Hub API so models, architectures,
 * and download URLs are dynamically resolved rather than hardcoded.
 */
class HuggingFaceClient {

    companion object {
        private const val TAG = "HuggingFaceClient"
        private const val HF_API_BASE = "https://huggingface.co/api"

        val SUITABLE_QUANTS = setOf(
            "Q2_K", "Q2_K_S", "Q2_K_L", "Q2_0",
            "Q3_K_M", "Q3_K_S", "Q3_K_L", "Q3_K_XL", "Q3_0",
            "Q4_K_M", "Q4_K_S", "Q4_K_L", "Q4_0", "Q4_1",
            "Q5_K_M", "Q5_K_S", "Q5_K_L", "Q5_0", "Q5_1",
            "Q6_K", "Q8_0",
            "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS",
            "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS",
            "IQ4_XS", "IQ4_NL"
        )
    }

    private val gson = Gson()
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(20, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    /**
     * Dynamically fetch popular, top-downloaded and trending GGUF text generation models from Hugging Face.
     */
    suspend fun fetchPopularModels(limit: Int = 40): List<ModelSpec> = withContext(Dispatchers.IO) {
        val collected = mutableListOf<ModelSpec>()
        val seenIds = mutableSetOf<String>()

        fun addSpecs(specs: List<ModelSpec>) {
            for (s in specs) {
                if (seenIds.add(s.id)) {
                    collected.add(s)
                }
            }
        }

        // 1. Top downloaded GGUF text generation models
        try {
            val topUrl = "$HF_API_BASE/models?pipeline_tag=text-generation&filter=gguf&sort=downloads&direction=-1&limit=$limit&full=true"
            val topSpecs = executeModelListQuery(topUrl)
            addSpecs(topSpecs)
        } catch (e: Exception) {
            Log.w(TAG, "Failed top models query: ${e.message}")
        }

        // 2. Curated popular mobile architectures (Qwen, Llama, SmolLM, Mistral, Gemma)
        val queries = listOf(
            "qwen2.5%20gguf",
            "llama-3.2%20gguf",
            "mistral%20gguf",
            "smollm2%20gguf"
        )
        for (q in queries) {
            try {
                val qUrl = "$HF_API_BASE/models?search=$q&pipeline_tag=text-generation&sort=downloads&direction=-1&limit=6&full=true"
                val qSpecs = executeModelListQuery(qUrl)
                addSpecs(qSpecs)
            } catch (e: Exception) {
                Log.w(TAG, "Failed query for '$q': ${e.message}")
            }
        }

        Log.i(TAG, "Successfully fetched ${collected.size} dynamic models from Hugging Face Hub")
        return@withContext collected
    }

    /**
     * Search Hugging Face dynamically for GGUF models matching [query].
     * Returns a list of [ModelSpec]s with all available quantizations.
     */
    suspend fun searchModels(query: String, limit: Int = 30): List<ModelSpec> = withContext(Dispatchers.IO) {
        val cleanQuery = query.trim()
        if (cleanQuery.isEmpty()) return@withContext emptyList()

        val results = mutableListOf<ModelSpec>()
        val seenIds = mutableSetOf<String>()

        fun addSpecs(specs: List<ModelSpec>) {
            for (s in specs) {
                if (seenIds.add(s.id)) {
                    results.add(s)
                }
            }
        }

        try {
            // Direct repository check if query matches author/repo (e.g. bartowski/Llama-3.2-1B-Instruct-GGUF)
            if (cleanQuery.contains('/')) {
                val repoSpecs = fetchRepoQuantizations(cleanQuery)
                if (repoSpecs.isNotEmpty()) {
                    addSpecs(repoSpecs)
                }
            }

            // Primary search: search query with filter=gguf
            val encodedQuery = URLEncoder.encode(cleanQuery, "UTF-8")
            val primaryUrl = "$HF_API_BASE/models?search=$encodedQuery&filter=gguf&sort=downloads&direction=-1&limit=$limit&full=true"
            val primarySpecs = executeModelListQuery(primaryUrl)
            addSpecs(primarySpecs)

            // Fallback: if few results, search with "$cleanQuery gguf" without tag filter
            if (results.size < 5) {
                val fallbackQuery = URLEncoder.encode("$cleanQuery gguf", "UTF-8")
                val fallbackUrl = "$HF_API_BASE/models?search=$fallbackQuery&sort=downloads&direction=-1&limit=$limit&full=true"
                val fallbackSpecs = executeModelListQuery(fallbackUrl)
                addSpecs(fallbackSpecs)
            }

            Log.i(TAG, "Found ${results.size} model quantizations on Hugging Face for '$cleanQuery'")
            return@withContext results
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
                .addHeader("User-Agent", "Mozilla/5.0 (Android; Mobile) MeshLLM/1.0")
                .build()

            httpClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return@withContext emptyList()

                val body = response.body?.string() ?: return@withContext emptyList()
                val treeArray = gson.fromJson(body, JsonArray::class.java)

                val results = mutableListOf<ModelSpec>()
                for (item in treeArray) {
                    if (!item.isJsonObject) continue
                    val fileObj = item.asJsonObject
                    val path = fileObj.get("path")?.asString ?: continue
                    if (!path.endsWith(".gguf", ignoreCase = true)) continue

                    val size = fileObj.get("size")?.asLong ?: 0L
                    val spec = createModelSpec(repoId, path, size, downloads = 0L, likes = 0L)
                    results.add(spec)
                }

                return@withContext results.sortedBy { it.fileSizeBytes }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error fetching repo tree for $repoId: ${e.message}", e)
            return@withContext emptyList()
        }
    }

    private fun executeModelListQuery(url: String): List<ModelSpec> {
        val request = Request.Builder()
            .url(url)
            .addHeader("User-Agent", "Mozilla/5.0 (Android; Mobile) MeshLLM/1.0")
            .build()

        httpClient.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                Log.w(TAG, "HF query failed [${response.code}]: $url")
                return emptyList()
            }

            val body = response.body?.string() ?: return emptyList()
            val jsonArray = gson.fromJson(body, JsonArray::class.java)

            val results = mutableListOf<ModelSpec>()
            for (element in jsonArray) {
                if (!element.isJsonObject) continue
                val obj = element.asJsonObject
                val specs = parseModelObject(obj)
                results.addAll(specs)
            }
            return results
        }
    }

    private fun parseModelObject(obj: JsonObject): List<ModelSpec> {
        val repoId = obj.get("id")?.asString ?: return emptyList()
        val downloads = obj.get("downloads")?.asLong ?: 0L
        val likes = obj.get("likes")?.asLong ?: 0L
        val siblings = obj.getAsJsonArray("siblings")

        val results = mutableListOf<ModelSpec>()

        if (siblings != null && siblings.size() > 0) {
            for (sib in siblings) {
                if (!sib.isJsonObject) continue
                val rfilename = sib.asJsonObject.get("rfilename")?.asString ?: continue
                if (rfilename.endsWith(".gguf", ignoreCase = true)) {
                    val spec = createModelSpec(repoId, rfilename, 0L, downloads, likes)
                    results.add(spec)
                }
            }
        } else {
            val modelName = repoId.substringAfter('/').removeSuffix("-GGUF").removeSuffix("-gguf")
            val defaultFileName = "${modelName.lowercase()}-q4_k_m.gguf"
            val spec = createModelSpec(repoId, defaultFileName, 0L, downloads, likes)
            results.add(spec)
        }

        return results
    }

    private fun createModelSpec(
        repoId: String,
        fileName: String,
        sizeBytes: Long,
        downloads: Long,
        likes: Long
    ): ModelSpec {
        val author = repoId.substringBefore('/')
        val modelBaseName = repoId.substringAfter('/').removeSuffix("-GGUF").removeSuffix("-gguf")
        val cleanFileName = fileName.substringAfterLast('/')
        val quant = extractQuantization(cleanFileName)

        val id = "${repoId.replace('/', '_')}__$cleanFileName"
        val displayName = "$modelBaseName [$quant]"

        // Infer parameter count
        val paramCount = inferParamCount(repoId + " " + cleanFileName)
        val layers = inferLayers(paramCount)
        val hiddenDim = inferHiddenDim(paramCount)
        val category = inferCategory(paramCount)

        val downloadUrl = "https://huggingface.co/$repoId/resolve/main/$fileName"

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

        // Assess suitability for distributed mobile inference
        val lowerFile = cleanFileName.lowercase()
        val isMmproj = lowerFile.contains("mmproj")
        val isSplit = lowerFile.contains("-of-") || lowerFile.contains(".part")
        val isGguf = lowerFile.endsWith(".gguf")
        val isRawFloat = quant == "FP16" || quant == "BF16" || quant == "FP32"
        val isReasonableParam = paramCount in 0.01..20.0
        val isStandardQuant = quant.startsWith("Q") || quant.startsWith("IQ") ||
                SUITABLE_QUANTS.any { quant.contains(it, ignoreCase = true) }

        val isSuitable = isGguf && !isMmproj && !isSplit && !isRawFloat && isReasonableParam && isStandardQuant

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
            description = "Dynamic Hugging Face model ($repoId) in $quant. Distributed cluster recommended: $minDevices+ devices.",
            samplePrompt = "Explain the fundamental advantages of dynamic layer partitioning across distributed edge devices:",
            huggingFaceRepo = repoId,
            downloadUrl = downloadUrl,
            fileName = cleanFileName,
            fileSizeBytes = finalSizeBytes,
            fileFormat = "GGUF",
            recommendedMinDevices = minDevices,
            singleDeviceOomWarning = isMassive,
            isStarterTestModel = false,
            quantization = quant,
            isSuitableForMobile = isSuitable,
            isOnlineHfModel = true,
            downloadsCount = downloads,
            likesCount = likes
        )
    }

    private fun extractQuantization(fileName: String): String {
        val lower = fileName.lowercase()

        // Explicit check for bf16 first so it isn't masked by f16
        if (lower.contains("bf16")) return "BF16"
        if (lower.contains("fp16") || lower.contains("f16")) return "FP16"
        if (lower.contains("fp32")) return "FP32"

        val quants = listOf(
            "q2_k_s", "q2_k_l", "q2_k", "q2_0",
            "q3_k_s", "q3_k_m", "q3_k_l", "q3_k_xl", "q3_k", "q3_0",
            "q4_k_s", "q4_k_m", "q4_k_l", "q4_k", "q4_0_4_4", "q4_0_4_8", "q4_0_8_8", "q4_0", "q4_1",
            "q5_k_s", "q5_k_m", "q5_k_l", "q5_k", "q5_0", "q5_1",
            "q6_k_l", "q6_k",
            "q8_0", "q8_k",
            "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m",
            "iq3_xxs", "iq3_xs", "iq3_s", "iq3_m", "iq4_nl", "iq4_xs"
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
        return 1.5
    }

    private fun inferLayers(paramCount: Double): Int {
        return when {
            paramCount >= 30.0 -> 64
            paramCount >= 14.0 -> 48
            paramCount >= 8.0 -> 32
            paramCount >= 6.0 -> 28
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
            quant.contains("Q2") || quant.contains("IQ2") -> 0.35
            quant.contains("Q3") || quant.contains("IQ3") -> 0.48
            quant.contains("Q4") || quant.contains("IQ4") -> 0.60
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
