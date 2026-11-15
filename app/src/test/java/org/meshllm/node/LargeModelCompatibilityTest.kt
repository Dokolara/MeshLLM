package org.meshllm.node

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import org.junit.Assert.*
import org.junit.Test
import org.meshllm.node.engine.DistributedOrchestrator
import org.meshllm.node.engine.GGUFParser
import org.meshllm.node.engine.SentencePieceTokenizer
import org.meshllm.node.model.*
import org.meshllm.node.p2p.P2PClient
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Validates system resilience and compatibility for large models (>5GB raw size,
 * 14B/32B/70B parameter architectures, 128k+ vocabulary tokenizers, and high-dimension
 * inter-device activation passing).
 */
class LargeModelCompatibilityTest {

    private val testScope = CoroutineScope(Dispatchers.Unconfined)

    @Test
    fun testHighDimensionActivationSerializationAndPrecision() {
        // 1. Test 8B / Mistral 7B dimension (4096 floats)
        val hidden7b = FloatArray(4096) { i -> (i * 0.001f) - 2.0f }
        val b64_7b = PipelineActivation.encodeFloatArrayToBase64(hidden7b)
        val decoded7b = PipelineActivation.decodeBase64ToFloatArray(b64_7b)
        assertEquals(4096, decoded7b.size)
        assertArrayEquals(hidden7b, decoded7b, 0.000001f)

        // 2. Test 14B / DeepSeek R1 dimension (5120 floats)
        val hidden14b = FloatArray(5120) { i -> kotlin.math.sin(i.toDouble()).toFloat() }
        val b64_14b = PipelineActivation.encodeFloatArrayToBase64(hidden14b)
        val decoded14b = PipelineActivation.decodeBase64ToFloatArray(b64_14b)
        assertEquals(5120, decoded14b.size)
        assertArrayEquals(hidden14b, decoded14b, 0.000001f)

        // 3. Test 70B dimension (8192 floats)
        val hidden70b = FloatArray(8192) { i -> kotlin.math.cos(i.toDouble()).toFloat() }
        val b64_70b = PipelineActivation.encodeFloatArrayToBase64(hidden70b)
        val decoded70b = PipelineActivation.decodeBase64ToFloatArray(b64_70b)
        assertEquals(8192, decoded70b.size)
        assertArrayEquals(hidden70b, decoded70b, 0.000001f)

        // 4. Test Multi-token prompt prefill embedding batch: 32 tokens * 5120 dim = 163,840 floats (655 KB payload)
        val batchEmbd = FloatArray(32 * 5120) { i -> (i % 1000) * 0.005f }
        val b64_batch = PipelineActivation.encodeFloatArrayToBase64(batchEmbd)
        val decodedBatch = PipelineActivation.decodeBase64ToFloatArray(b64_batch)
        assertEquals(163840, decodedBatch.size)
        assertArrayEquals(batchEmbd, decodedBatch, 0.000001f)
    }

    @Test
    fun testSentencePieceTokenizerLargeVocabCapacity() {
        val tokenizer = SentencePieceTokenizer()

        // Generate synthetic 128k vocabulary (like Llama-3 128,256 tokens)
        val vocabSize = 128_256
        val syntheticTokens = ArrayList<String>(vocabSize)
        syntheticTokens.add("<unk>")
        syntheticTokens.add("<s>")
        syntheticTokens.add("</s>")
        for (i in 3 until vocabSize) {
            syntheticTokens.add("tok_$i")
        }

        val loaded = tokenizer.loadFromTokens(syntheticTokens)
        assertTrue("Tokenizer should successfully load 128k vocabulary", loaded)
        assertEquals(vocabSize, tokenizer.vocabSize)

        // Test encoding and decoding with high token IDs
        val encoded = tokenizer.encode("tok_100 tok_50000 tok_120000")
        assertTrue("Encoded tokens must not be empty", encoded.isNotEmpty())
        assertEquals("tok_100", tokenizer.decode(100))
        assertEquals("tok_50000", tokenizer.decode(50000))
        assertEquals("tok_120000", tokenizer.decode(120000))
    }

    @Test
    fun testGGUFParserSyntheticLargeModelHeader() {
        val tempFile = File.createTempFile("test_qwen_14b_", ".gguf")
        tempFile.deleteOnExit()

        // Build a synthetic valid GGUF v3 header with architecture metadata
        val buffer = ByteBuffer.allocate(64 * 1024).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(GGUFParser.GGUF_MAGIC)
        buffer.putInt(3) // Version 3
        buffer.putLong(360L) // Tensor count (14B model has ~360 tensors)
        buffer.putLong(6L)   // 6 KV pairs

        fun writeGgufStr(s: String) {
            val bytes = s.toByteArray(Charsets.UTF_8)
            buffer.putLong(bytes.size.toLong())
            buffer.put(bytes)
        }

        // KV 1: general.architecture = "qwen2"
        writeGgufStr("general.architecture")
        buffer.putInt(GGUFParser.TYPE_STRING)
        writeGgufStr("qwen2")

        // KV 2: general.name = "Qwen2.5-14B-Instruct"
        writeGgufStr("general.name")
        buffer.putInt(GGUFParser.TYPE_STRING)
        writeGgufStr("Qwen2.5-14B-Instruct")

        // KV 3: qwen2.block_count = 48 (14B standard)
        writeGgufStr("qwen2.block_count")
        buffer.putInt(GGUFParser.TYPE_INT32)
        buffer.putInt(48)

        // KV 4: qwen2.embedding_length = 5120
        writeGgufStr("qwen2.embedding_length")
        buffer.putInt(GGUFParser.TYPE_INT32)
        buffer.putInt(5120)

        // KV 5: qwen2.head_count = 40
        writeGgufStr("qwen2.head_count")
        buffer.putInt(GGUFParser.TYPE_INT32)
        buffer.putInt(40)

        // KV 6: tokenizer.ggml.tokens (sample 10 tokens)
        writeGgufStr("tokenizer.ggml.tokens")
        buffer.putInt(GGUFParser.TYPE_ARRAY)
        buffer.putInt(GGUFParser.TYPE_STRING) // elemType
        buffer.putLong(5L) // 5 tokens
        for (i in 0 until 5) {
            writeGgufStr("token_$i")
        }

        FileOutputStream(tempFile).use { fos ->
            fos.write(buffer.array(), 0, buffer.position())
        }

        val parser = GGUFParser()
        val meta = parser.parse(tempFile)

        assertNotNull("GGUF metadata should parse successfully", meta)
        assertEquals("qwen2", meta?.architecture)
        assertEquals("Qwen2.5-14B-Instruct", meta?.modelName)
        assertEquals(48, meta?.blockCount)
        assertEquals(5120, meta?.hiddenDim)
        assertEquals(40, meta?.headCount)
        assertEquals(5, meta?.vocabSize)
        assertEquals(360L, meta?.tensorCount)

        // Test extractVocabulary directly
        val extractedVocab = parser.extractVocabulary(tempFile)
        assertEquals(5, extractedVocab.size)
        assertEquals("token_0", extractedVocab[0])
        assertEquals("token_4", extractedVocab[4])
    }

    @Test
    fun testDistributedOrchestrator48LayerSequentialAllotment() {
        val model14b = ModelSpec(
            id = "deepseek-r1-14b-test",
            name = "DeepSeek-R1-14B",
            provider = "DeepSeek",
            category = ModelCatalog.CATEGORY_LARGE_MESH,
            totalLayers = 48,
            hiddenDim = 5120,
            intermediateDim = 27648,
            numHeads = 40,
            contextLength = 8192,
            paramCountStr = "14.7B",
            memoryRequirementStr = "~6.25 GB",
            description = "14B test model",
            samplePrompt = "Test reasoning",
            fileSizeBytes = 6_250_000_000L,
            recommendedMinDevices = 6,
            singleDeviceOomWarning = true
        )

        val orchestrator = DistributedOrchestrator(
            sessionToken = "MESH-TEST-14B-OK",
            localNodeId = "master_node",
            localAddress = "192.168.1.100",
            localPort = 8080,
            localBenchmark = BenchmarkProfile(100.0, 5000.0, 8, 8000L, 100.0),
            p2pClient = P2PClient(),
            scope = testScope,
            activeModel = model14b
        )

        // Register 5 additional slave nodes (total 6 nodes in cluster)
        for (i in 1..5) {
            val slave = NodeInfo(
                nodeId = "slave_node_$i",
                deviceName = "Phone Node $i",
                ipAddress = "192.168.1.10$i",
                port = 8080,
                role = NodeRole.SLAVE,
                joinOrder = i,
                benchmark = BenchmarkProfile(100.0, 5000.0, 8, 8000L, 100.0),
                metrics = SensorMetrics(availableRamMb = 4000)
            )
            orchestrator.registerSlaveNode(P2PClient.RegistrationRequest(slave, "MESH-TEST-14B-OK"))
        }

        val state = orchestrator.clusterState.value
        assertEquals(6, state.nodes.size)
        assertEquals(48, state.totalLayers)

        // Verify that 48 layers are partitioned sequentially across all 6 nodes with 0 gaps
        var expectedNextLayer = 0
        for ((idx, node) in state.nodes.withIndex()) {
            assertEquals(1, node.assignedBlocks.size)
            val block = node.assignedBlocks.first()

            assertEquals("Start layer of node $idx must be continuous", expectedNextLayer, block.startLayer)
            assertTrue("End layer must be >= start layer", block.endLayer >= block.startLayer)
            expectedNextLayer = block.endLayer + 1

            if (idx == 0) {
                assertEquals(BlockType.EMBEDDING_INPUT, block.blockType)
            } else if (idx == 5) {
                assertEquals(BlockType.OUTPUT_LM_HEAD, block.blockType)
            } else {
                assertEquals(BlockType.TRANSFORMER_BLOCK, block.blockType)
            }
        }
        assertEquals("Total allocated layers must reach 48", 48, expectedNextLayer)
    }

    @Test
    fun testDynamicLoadRebalancingWhenSlaveUnderRamStress() {
        val model8b = ModelSpec(
            id = "llama-8b-test",
            name = "Llama-3.1-8B",
            provider = "Meta AI",
            category = ModelCatalog.CATEGORY_LARGE_MESH,
            totalLayers = 32,
            hiddenDim = 4096,
            intermediateDim = 14336,
            numHeads = 32,
            contextLength = 4096,
            paramCountStr = "8B",
            memoryRequirementStr = "~5.2 GB",
            description = "8B test model",
            samplePrompt = "Test",
            fileSizeBytes = 5_200_000_000L,
            recommendedMinDevices = 4,
            singleDeviceOomWarning = true
        )

        val orchestrator = DistributedOrchestrator(
            sessionToken = "MESH-TEST-8B-RAM",
            localNodeId = "master_0",
            localAddress = "192.168.1.100",
            localPort = 8080,
            localBenchmark = BenchmarkProfile(100.0, 5000.0, 8, 8000L, 100.0),
            p2pClient = P2PClient(),
            scope = testScope,
            activeModel = model8b
        )

        // Add 3 slaves
        for (i in 1..3) {
            val slave = NodeInfo(
                nodeId = "slave_$i",
                deviceName = "Phone $i",
                ipAddress = "192.168.1.10$i",
                port = 8080,
                role = NodeRole.SLAVE,
                joinOrder = i,
                benchmark = BenchmarkProfile(100.0, 5000.0, 8, 8000L, 100.0),
                metrics = SensorMetrics(availableRamMb = 4000)
            )
            orchestrator.registerSlaveNode(P2PClient.RegistrationRequest(slave, "MESH-TEST-8B-RAM"))
        }

        val initialNodes = orchestrator.clusterState.value.nodes
        val slave1InitialLayers = initialNodes.find { it.nodeId == "slave_1" }!!.assignedBlocks.first().layerCount

        // Slave 1 experiences heavy RAM pressure and overheating: available RAM drops to 250MB, temp 46C
        val stressedMetrics = SensorMetrics(
            availableRamMb = 250,
            batteryTemperatureC = 46.0f,
            thermalStatus = 3
        )
        orchestrator.handleHeartbeat(P2PClient.HeartbeatRequest("slave_1", "MESH-TEST-8B-RAM", stressedMetrics))
        orchestrator.rebalanceLayers()

        val rebalancedNodes = orchestrator.clusterState.value.nodes
        val slave1StressedLayers = rebalancedNodes.find { it.nodeId == "slave_1" }!!.assignedBlocks.first().layerCount

        assertTrue(
            "Stressed phone with low RAM and high temp should be allotted fewer layers ($slave1StressedLayers < $slave1InitialLayers)",
            slave1StressedLayers < slave1InitialLayers
        )

        // Verify total layers across all nodes still sum to exactly 32
        var sum = 0
        for (n in rebalancedNodes) {
            sum += n.assignedBlocks.first().layerCount
        }
        assertEquals(32, sum)
    }
}
