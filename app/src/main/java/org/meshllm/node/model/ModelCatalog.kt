package org.meshllm.node.model

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList

object ModelCatalog {

    private const val TAG = "ModelCatalog"
    private const val CACHE_FILE = "hf_models_cache.json"

    const val CATEGORY_ALL = "All"
    const val CATEGORY_ULTRA_SMALL = "Starter & Fast (<100M)"
    const val CATEGORY_COMPACT = "Compact Edge (<1B)"
    const val CATEGORY_STANDARD = "Standard Edge (1B - 3B)"
    const val CATEGORY_LARGE_MESH = "Massive Mesh (7B - 15B)"

    val CATEGORIES = listOf(
        CATEGORY_ALL,
        CATEGORY_ULTRA_SMALL,
        CATEGORY_COMPACT,
        CATEGORY_STANDARD,
        CATEGORY_LARGE_MESH
    )

    // Baseline offline models for initial installation & local testing
    private val FALLBACK_STARTER_MODELS: List<ModelSpec> = listOf(
        ModelSpec(
            id = "tinystories-15m",
            name = "TinyStories-15M",
            provider = "Microsoft / Eldan & Li",
            category = CATEGORY_ULTRA_SMALL,
            totalLayers = 6,
            hiddenDim = 288,
            intermediateDim = 768,
            numHeads = 6,
            contextLength = 512,
            paramCountStr = "15M",
            memoryRequirementStr = "~60 MB",
            description = "Full 6-layer LLaMA architecture trained on synthetic child stories. Generates coherent story continuations across mobile mesh nodes.",
            samplePrompt = "Once upon a time, a small friendly robot wanted to explore the garden",
            huggingFaceRepo = "karpathy/tinyllamas",
            downloadUrl = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
            fileName = "stories15M.bin",
            fileSizeBytes = 60816028L,
            fileFormat = "BIN_LLAMA2",
            recommendedMinDevices = 1,
            isStarterTestModel = true,
            quantization = "BIN",
            isSuitableForMobile = true
        ),
        ModelSpec(
            id = "tinystories-33m",
            name = "TinyStories-33M",
            provider = "Microsoft / Eldan & Li",
            category = CATEGORY_ULTRA_SMALL,
            totalLayers = 8,
            hiddenDim = 384,
            intermediateDim = 1024,
            numHeads = 8,
            contextLength = 512,
            paramCountStr = "33M",
            memoryRequirementStr = "~38 MB",
            description = "8-layer compact transformer trained on synthetic child narratives. Low-latency multi-device pipeline parallel execution.",
            samplePrompt = "Lily found an old key hidden beneath the wooden bench",
            huggingFaceRepo = "roneneldan/TinyStories-33M",
            downloadUrl = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin",
            fileName = "stories33M.bin",
            fileSizeBytes = 38000000L,
            fileFormat = "BIN_LLAMA2",
            recommendedMinDevices = 1,
            isStarterTestModel = true,
            quantization = "BIN",
            isSuitableForMobile = true
        ),
        ModelSpec(
            id = "tinystories-260k",
            name = "TinyStories-260K (Fast 1MB)",
            provider = "Microsoft / Eldan & Li",
            category = CATEGORY_ULTRA_SMALL,
            totalLayers = 4,
            hiddenDim = 64,
            intermediateDim = 172,
            numHeads = 4,
            contextLength = 512,
            paramCountStr = "260K",
            memoryRequirementStr = "~1.0 MB",
            description = "Ultra-fast 1 MB test model from Hugging Face for instant pipeline verification without downloading gigabytes.",
            samplePrompt = "Once upon a time, a small friendly robot wanted to explore the garden",
            huggingFaceRepo = "karpathy/tinyllamas",
            downloadUrl = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin",
            fileName = "stories260K.bin",
            fileSizeBytes = 1056540L,
            fileFormat = "BIN_LLAMA2",
            recommendedMinDevices = 1,
            isStarterTestModel = true,
            quantization = "BIN",
            isSuitableForMobile = true
        ),
        ModelSpec(
            id = "smollm2-135m",
            name = "SmolLM2-135M",
            provider = "Hugging Face",
            category = CATEGORY_COMPACT,
            totalLayers = 30,
            hiddenDim = 576,
            intermediateDim = 1536,
            numHeads = 9,
            contextLength = 2048,
            paramCountStr = "135M",
            memoryRequirementStr = "~96 MB (Q4_K_M)",
            description = "Hugging Face flagship compact edge model with 30 transformer layers. Excellent depth for partitioning across 2 to 6 phones.",
            samplePrompt = "Decentralized mobile edge networks function by distributing transformer layers across",
            huggingFaceRepo = "ggml-org/SmolLM2-135M-GGUF",
            downloadUrl = "https://huggingface.co/ggml-org/SmolLM2-135M-GGUF/resolve/main/SmolLM2-135M-Q4_K_M.gguf",
            fileName = "SmolLM2-135M-Q4_K_M.gguf",
            fileSizeBytes = 96000000L,
            fileFormat = "GGUF",
            recommendedMinDevices = 2,
            isStarterTestModel = true,
            quantization = "Q4_K_M",
            isSuitableForMobile = true
        ),
        ModelSpec(
            id = "qwen2.5-0.5b",
            name = "Qwen2.5-0.5B-Instruct",
            provider = "Alibaba Cloud",
            category = CATEGORY_COMPACT,
            totalLayers = 24,
            hiddenDim = 896,
            intermediateDim = 4864,
            numHeads = 14,
            contextLength = 4096,
            paramCountStr = "490M",
            memoryRequirementStr = "~248 MB (Q2_K)",
            description = "Alibaba's capable 24-layer multilingual sub-billion model. Downloads directly from Hugging Face for 2-4 device mesh pipelines.",
            samplePrompt = "List three benefits of collaborative distributed inference on battery-operated devices:",
            huggingFaceRepo = "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            downloadUrl = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q2_k.gguf",
            fileName = "qwen2.5-0.5b-instruct-q2_k.gguf",
            fileSizeBytes = 248000000L,
            fileFormat = "GGUF",
            recommendedMinDevices = 2,
            quantization = "Q2_K",
            isSuitableForMobile = true
        ),
        ModelSpec(
            id = "llama-3.2-1b",
            name = "Llama-3.2-1B-Instruct",
            provider = "Meta AI",
            category = CATEGORY_STANDARD,
            totalLayers = 16,
            hiddenDim = 2048,
            intermediateDim = 8192,
            numHeads = 32,
            contextLength = 8192,
            paramCountStr = "1.23B",
            memoryRequirementStr = "~550 MB (Q2_K)",
            description = "Meta's lightweight edge model with 16 wide layers and GQA. Runs across 2 to 4 mobile devices.",
            samplePrompt = "Explain how dynamic layer rebalancing prevents device overheating in a mesh cluster:",
            huggingFaceRepo = "bartowski/Llama-3.2-1B-Instruct-GGUF",
            downloadUrl = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q2_K.gguf",
            fileName = "Llama-3.2-1B-Instruct-Q2_K.gguf",
            fileSizeBytes = 550000000L,
            fileFormat = "GGUF",
            recommendedMinDevices = 2,
            quantization = "Q2_K",
            isSuitableForMobile = true
        ),
        ModelSpec(
            id = "tinyllama-1.1b",
            name = "TinyLlama-1.1B-Chat",
            provider = "TinyLlama Project",
            category = CATEGORY_STANDARD,
            totalLayers = 22,
            hiddenDim = 2048,
            intermediateDim = 5632,
            numHeads = 32,
            contextLength = 2048,
            paramCountStr = "1.1B",
            memoryRequirementStr = "~480 MB (Q2_K)",
            description = "Trained on 3 trillion tokens. 22 layers split evenly across 2 to 4 phones for rapid inference.",
            samplePrompt = "In a distributed phone cluster, how does leader election protect against master failure?",
            huggingFaceRepo = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            downloadUrl = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
            fileName = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
            fileSizeBytes = 480000000L,
            fileFormat = "GGUF",
            recommendedMinDevices = 2,
            quantization = "Q2_K",
            isSuitableForMobile = true
        ),
        ModelSpec(
            id = "mistral-7b",
            name = "Mistral-7B-Instruct-v0.3",
            provider = "Mistral AI",
            category = CATEGORY_LARGE_MESH,
            totalLayers = 32,
            hiddenDim = 4096,
            intermediateDim = 14336,
            numHeads = 32,
            contextLength = 8192,
            paramCountStr = "7.25B",
            memoryRequirementStr = "~3.0 GB (Q2_K)",
            description = "High-throughput 32-layer instruction model with sliding window attention. Split across 4 to 6 phones for fast token throughput.",
            samplePrompt = "Describe the role of UDP broadcast beacons in serverless peer-to-peer mobile mesh discovery:",
            huggingFaceRepo = "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            downloadUrl = "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q2_K.gguf",
            fileName = "Mistral-7B-Instruct-v0.3-Q2_K.gguf",
            fileSizeBytes = 3165000000L,
            fileFormat = "GGUF",
            recommendedMinDevices = 4,
            singleDeviceOomWarning = true,
            quantization = "Q2_K",
            isSuitableForMobile = true
        ),
        ModelSpec(
            id = "qwen2.5-7b",
            name = "Qwen2.5-7B-Instruct",
            provider = "Alibaba Cloud",
            category = CATEGORY_LARGE_MESH,
            totalLayers = 28,
            hiddenDim = 3584,
            intermediateDim = 18944,
            numHeads = 28,
            contextLength = 8192,
            paramCountStr = "7.61B",
            memoryRequirementStr = "~2.9 GB (Q2_K)",
            description = "High-efficiency 28-layer LLM with enhanced coding and mathematical capabilities. Partitioned across 4 to 6 phones.",
            samplePrompt = "Provide an outline of how mobile sensors dynamically modulate compute weight assignments:",
            huggingFaceRepo = "Qwen/Qwen2.5-7B-Instruct-GGUF",
            downloadUrl = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q2_k.gguf",
            fileName = "qwen2.5-7b-instruct-q2_k.gguf",
            fileSizeBytes = 3080000000L,
            fileFormat = "GGUF",
            recommendedMinDevices = 4,
            singleDeviceOomWarning = true,
            quantization = "Q2_K",
            isSuitableForMobile = true
        ),
        ModelSpec(
            id = "llama-3.1-8b",
            name = "Llama-3.1-8B-Instruct",
            provider = "Meta AI",
            category = CATEGORY_LARGE_MESH,
            totalLayers = 32,
            hiddenDim = 4096,
            intermediateDim = 14336,
            numHeads = 32,
            contextLength = 8192,
            paramCountStr = "8.03B",
            memoryRequirementStr = "~3.2 GB (Q2_K)",
            description = "Meta's flagship 8B open model. 32 wide transformer layers with Grouped-Query Attention (GQA). Split across 4 to 6 devices.",
            samplePrompt = "Explain how distributed fault tolerance ensures that if an orchestrator node disconnects, a successor node seamlessly resumes generation:",
            huggingFaceRepo = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            downloadUrl = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q2_K.gguf",
            fileName = "Meta-Llama-3.1-8B-Instruct-Q2_K.gguf",
            fileSizeBytes = 3415000000L,
            fileFormat = "GGUF",
            recommendedMinDevices = 4,
            singleDeviceOomWarning = true,
            quantization = "Q2_K",
            isSuitableForMobile = true
        ),
        ModelSpec(
            id = "deepseek-r1-14b",
            name = "DeepSeek-R1-Distill-Qwen-14B",
            provider = "DeepSeek AI",
            category = CATEGORY_LARGE_MESH,
            totalLayers = 48,
            hiddenDim = 5120,
            intermediateDim = 27648,
            numHeads = 40,
            contextLength = 8192,
            paramCountStr = "14.7B",
            memoryRequirementStr = "~5.8 GB (Q2_K)",
            description = "State-of-the-art distilled reasoning model with 48 layers and chain-of-thought logic. Requires 6+ mobile nodes.",
            samplePrompt = "A car travels at 60 mph for 2 hours, then 40 mph for 1 hour. Reason step-by-step through the average speed:",
            huggingFaceRepo = "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            downloadUrl = "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Q2_K.gguf",
            fileName = "DeepSeek-R1-Distill-Qwen-14B-Q2_K.gguf",
            fileSizeBytes = 6250000000L,
            fileFormat = "GGUF",
            recommendedMinDevices = 6,
            singleDeviceOomWarning = true,
            quantization = "Q2_K",
            isSuitableForMobile = true
        ),
        ModelSpec(
            id = "qwen2.5-14b",
            name = "Qwen2.5-14B-Instruct",
            provider = "Alibaba Cloud",
            category = CATEGORY_LARGE_MESH,
            totalLayers = 48,
            hiddenDim = 5120,
            intermediateDim = 27648,
            numHeads = 40,
            contextLength = 4096,
            paramCountStr = "14.7B",
            memoryRequirementStr = "~5.8 GB (Q2_K)",
            description = "High-intelligence 48-layer multilingual foundation model. Exceeds single phone RAM and partitions across 6 to 8 phones.",
            samplePrompt = "Compare the architectural benefits of tensor parallelism versus sequential pipeline parallelism on mobile clusters:",
            huggingFaceRepo = "Qwen/Qwen2.5-14B-Instruct-GGUF",
            downloadUrl = "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q2_k.gguf",
            fileName = "qwen2.5-14b-instruct-q2_k.gguf",
            fileSizeBytes = 6250000000L,
            fileFormat = "GGUF",
            recommendedMinDevices = 6,
            singleDeviceOomWarning = true,
            quantization = "Q2_K",
            isSuitableForMobile = true
        )
    )

    // Dynamic models fetched directly from Hugging Face Hub
    private val dynamicModels = CopyOnWriteArrayList<ModelSpec>()

    fun loadCache(context: Context) {
        try {
            val file = File(context.filesDir, CACHE_FILE)
            if (file.exists() && file.length() > 10L) {
                val json = file.readText()
                val listType = object : TypeToken<List<ModelSpec>>() {}.type
                val cached: List<ModelSpec> = Gson().fromJson(json, listType)
                if (cached.isNotEmpty()) {
                    dynamicModels.clear()
                    dynamicModels.addAll(cached)
                    Log.i(TAG, "Loaded ${cached.size} Hugging Face models from local disk cache")
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load cached models: ${e.message}")
        }
    }

    fun saveCache(context: Context) {
        try {
            val file = File(context.filesDir, CACHE_FILE)
            val json = Gson().toJson(dynamicModels.toList())
            file.writeText(json)
            Log.i(TAG, "Saved ${dynamicModels.size} Hugging Face models to local disk cache")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to save models cache: ${e.message}")
        }
    }

    fun addDynamicModels(models: List<ModelSpec>, context: Context? = null) {
        var changed = false
        for (m in models) {
            val idx = dynamicModels.indexOfFirst { it.id.equals(m.id, ignoreCase = true) }
            if (idx >= 0) {
                dynamicModels[idx] = m
            } else {
                dynamicModels.add(m)
                changed = true
            }
        }
        if (changed && context != null) {
            saveCache(context)
        }
    }

    fun setDynamicModels(models: List<ModelSpec>, context: Context? = null) {
        dynamicModels.clear()
        dynamicModels.addAll(models)
        if (context != null) {
            saveCache(context)
        }
    }

    fun getAllModels(): List<ModelSpec> {
        val all = mutableListOf<ModelSpec>()
        all.addAll(FALLBACK_STARTER_MODELS)
        for (m in dynamicModels) {
            if (all.none { it.id.equals(m.id, ignoreCase = true) }) {
                all.add(m)
            }
        }
        return all
    }

    fun getDefaultModel(): ModelSpec =
        FALLBACK_STARTER_MODELS.firstOrNull { it.id == "tinystories-15m" } ?: getAllModels().first()

    fun getModelById(id: String): ModelSpec? =
        dynamicModels.find { it.id.equals(id, ignoreCase = true) }
            ?: FALLBACK_STARTER_MODELS.find { it.id.equals(id, ignoreCase = true) }

    /**
     * Filter models by category, search query, suitability for mobile, and quantization.
     * [suitableOnly] defaults to true to show tested, mobile-safe models, but can be disabled
     * or altered to view any model on Hugging Face.
     */
    fun filter(
        category: String?,
        query: String?,
        suitableOnly: Boolean = true,
        preferredQuant: String? = null
    ): List<ModelSpec> {
        val pool = getAllModels()
        return pool.filter { model ->
            // Suitability check
            if (suitableOnly && !model.isSuitableForMobile) {
                return@filter false
            }

            // Category check
            val matchCategory = category.isNullOrEmpty() ||
                    category == CATEGORY_ALL ||
                    model.category.equals(category, ignoreCase = true)

            // Quantization check
            val matchQuant = preferredQuant.isNullOrEmpty() ||
                    model.quantization.contains(preferredQuant, ignoreCase = true)

            // Search query check
            val matchQuery = query.isNullOrBlank() ||
                    model.name.contains(query, ignoreCase = true) ||
                    model.provider.contains(query, ignoreCase = true) ||
                    model.description.contains(query, ignoreCase = true) ||
                    model.paramCountStr.contains(query, ignoreCase = true) ||
                    (model.huggingFaceRepo?.contains(query, ignoreCase = true) == true) ||
                    model.quantization.contains(query, ignoreCase = true)

            matchCategory && matchQuant && matchQuery
        }
    }
}
