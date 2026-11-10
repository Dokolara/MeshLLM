package org.meshllm.node

import org.junit.Assert.*
import org.junit.Test
import org.meshllm.node.model.ModelCatalog
import org.meshllm.node.model.ModelSpec
import org.meshllm.node.telemetry.HuggingFaceClient

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
            isSuitableForMobile = false
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
            isSuitableForMobile = true
        )

        assertEquals(expectedUrl, spec.downloadUrl)
        assertTrue(spec.isSuitableForMobile)
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
