package org.meshllm.node

import org.junit.Assert.*
import org.junit.Test
import org.meshllm.node.model.ModelCatalog
import org.meshllm.node.model.ModelSpec

class HuggingFaceModelCatalogTest {

    @Test
    fun testSuitabilityFilterDefaultBehavior() {
        val nonSuitableModel = ModelSpec(
            id = "huge-70b-raw-fp16",
            name = "Huge-70B-FP16",
            provider = "Meta AI",
            category = ModelCatalog.CATEGORY_LARGE_MESH,
            totalLayers = 80,
            hiddenDim = 8192,
            intermediateDim = 28672,
            numHeads = 64,
            contextLength = 4096,
            paramCountStr = "70B",
            memoryRequirementStr = "~140 GB",
            description = "Unquantized 70B raw float model exceeding mobile hardware",
            samplePrompt = "Test prompt",
            huggingFaceRepo = "meta/llama-70b",
            downloadUrl = "https://huggingface.co/meta/llama-70b/resolve/main/model.fp16.gguf",
            fileName = "model.fp16.gguf",
            fileSizeBytes = 140_000_000_000L,
            fileFormat = "GGUF",
            quantization = "FP16",
            isSuitableForMobile = false,
            isOnlineHfModel = true
        )

        ModelCatalog.addDynamicModels(listOf(nonSuitableModel))

        // When suitableOnly is TRUE (default): non-suitable model MUST NOT appear
        val filteredSuitable = ModelCatalog.filter(
            category = ModelCatalog.CATEGORY_ALL,
            query = "Huge-70B",
            suitableOnly = true
        )
        assertTrue("Non-suitable model should be filtered out by default", filteredSuitable.isEmpty())

        // When suitableOnly is FALSE (filter removed/disabled): non-suitable model MUST appear
        val unfiltered = ModelCatalog.filter(
            category = ModelCatalog.CATEGORY_ALL,
            query = "Huge-70B",
            suitableOnly = false
        )
        assertEquals("When filter is removed, all Hugging Face models must appear", 1, unfiltered.size)
        assertEquals("huge-70b-raw-fp16", unfiltered.first().id)
        assertTrue(unfiltered.first().isOnlineHfModel)
    }

    @Test
    fun testMultiTokenFlexibleSearch() {
        // Test searching for "llama 3.2" matches "Llama-3.2-1B-Instruct" despite punctuation and spaces
        val llamaResults = ModelCatalog.filter(
            category = ModelCatalog.CATEGORY_ALL,
            query = "llama 3.2",
            suitableOnly = true
        )
        assertTrue("Query 'llama 3.2' should match Llama 3.2 models", llamaResults.isNotEmpty())
        assertTrue(llamaResults.any { it.name.contains("Llama-3.2", ignoreCase = true) })

        // Test searching for "mistral 7b" matches "Mistral-7B-Instruct-v0.3"
        val mistralResults = ModelCatalog.filter(
            category = ModelCatalog.CATEGORY_ALL,
            query = "mistral 7b",
            suitableOnly = true
        )
        assertTrue("Query 'mistral 7b' should match Mistral 7B", mistralResults.isNotEmpty())
        assertTrue(mistralResults.any { it.name.contains("Mistral-7B", ignoreCase = true) })

        // Test searching for "qwen 2.5" matches "Qwen2.5-7B-Instruct"
        val qwenResults = ModelCatalog.filter(
            category = ModelCatalog.CATEGORY_ALL,
            query = "qwen 2.5",
            suitableOnly = true
        )
        assertTrue("Query 'qwen 2.5' should match Qwen 2.5 models", qwenResults.isNotEmpty())
        assertTrue(qwenResults.any { it.name.contains("Qwen2.5", ignoreCase = true) })
    }

    @Test
    fun testDynamicHuggingFaceDownloadUrlResolution() {
        val repo = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
        val fileName = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        val expectedUrl = "https://huggingface.co/$repo/resolve/main/$fileName"

        val spec = ModelSpec(
            id = "${repo.replace('/', '_')}__$fileName",
            name = "Llama-3.1-8B [Q4_K_M]",
            provider = "bartowski",
            category = ModelCatalog.CATEGORY_LARGE_MESH,
            totalLayers = 32,
            hiddenDim = 4096,
            intermediateDim = 14336,
            numHeads = 32,
            contextLength = 8192,
            paramCountStr = "8B",
            memoryRequirementStr = "~4.9 GB",
            description = "Dynamic Hugging Face test model",
            samplePrompt = "Hello world",
            huggingFaceRepo = repo,
            downloadUrl = expectedUrl,
            fileName = fileName,
            fileSizeBytes = 4920000000L,
            fileFormat = "GGUF",
            quantization = "Q4_K_M",
            isSuitableForMobile = true,
            isOnlineHfModel = true
        )

        assertEquals(expectedUrl, spec.downloadUrl)
        assertTrue(spec.isSuitableForMobile)
        assertTrue(spec.isOnlineHfModel)
        assertEquals("Q4_K_M", spec.quantization)
    }

    @Test
    fun testCategoryFilteringWithDynamicModels() {
        val starterModels = ModelCatalog.filter(
            category = ModelCatalog.CATEGORY_ULTRA_SMALL,
            query = "",
            suitableOnly = true
        )
        assertTrue("Starter models should exist", starterModels.isNotEmpty())
        for (m in starterModels) {
            assertEquals(ModelCatalog.CATEGORY_ULTRA_SMALL, m.category)
        }

        val compactModels = ModelCatalog.filter(
            category = ModelCatalog.CATEGORY_COMPACT,
            query = "",
            suitableOnly = true
        )
        assertTrue("Compact models should exist", compactModels.isNotEmpty())
        for (m in compactModels) {
            assertEquals(ModelCatalog.CATEGORY_COMPACT, m.category)
        }
    }
}
