package org.meshllm.node

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.meshllm.node.engine.DistributedOrchestrator
import org.meshllm.node.engine.LLMBlockExecutor
import org.meshllm.node.model.BenchmarkProfile
import org.meshllm.node.model.BlockType
import org.meshllm.node.model.LLMBlock
import org.meshllm.node.model.ModelCatalog
import org.meshllm.node.model.ModelSpec
import org.meshllm.node.model.NodeInfo
import org.meshllm.node.model.NodeRole
import org.meshllm.node.model.PipelineActivation
import org.meshllm.node.model.SensorMetrics
import org.meshllm.node.p2p.P2PClient
import org.meshllm.node.p2p.SessionTokenGenerator
import org.meshllm.node.telemetry.PerformanceGovernor
import org.meshllm.node.telemetry.SensorMonitor

class MeshClusterTest {

    private val testScope = CoroutineScope(Dispatchers.Unconfined)

    @Test
    fun testSessionTokenGeneration() {
        val token = SessionTokenGenerator.generateSessionToken()
        assertTrue(token.startsWith("MESH-"))
        assertTrue(SessionTokenGenerator.isValidTokenFormat(token))
        assertEquals(3, token.split("-").size - 1) // 3 dashes
    }

    @Test
    fun testHealthFactorUnderThermalStress() {
        val coolMetrics = SensorMetrics(
            batteryPercent = 90,
            batteryTemperatureC = 29.0f,
            isCharging = true,
            thermalStatus = 0
        )
        val coolHealth = coolMetrics.computeHealthFactor()
        assertTrue("Cool phone should have near 1.0 health", coolHealth >= 0.95)

        val hotMetrics = SensorMetrics(
            batteryPercent = 12,
            batteryTemperatureC = 45.0f,
            isCharging = false,
            thermalStatus = 3, // SEVERE
            availableRamMb = 300 // RAM pressure
        )
        val hotHealth = hotMetrics.computeHealthFactor()
        assertTrue("Overheating and low battery phone should degrade health factor", hotHealth < 0.40)
    }

    @Test
    fun testModelCatalogStartsWithUltraSmallTestModels() {
        val allModels = ModelCatalog.getAllModels()
        assertTrue("Model catalog should have multiple open-source models", allModels.size >= 6)

        // Verify first model is ultra-small for testing
        val firstModel = allModels.first()
        assertEquals("tinystories-15m", firstModel.id)
        assertEquals(6, firstModel.totalLayers)
        assertTrue(firstModel.isStarterTestModel)
        assertEquals(ModelCatalog.CATEGORY_ULTRA_SMALL, firstModel.category)

        // Verify second model is also ultra-small
        val secondModel = allModels[1]
        assertEquals("tinystories-33m", secondModel.id)
        assertEquals(8, secondModel.totalLayers)

        // Verify SmolLM2-135M exists
        val smol = ModelCatalog.getModelById("smollm2-135m")
        assertNotNull(smol)
        assertEquals(30, smol!!.totalLayers)
        assertEquals(576, smol.hiddenDim)

        // Test category filtering
        val ultraSmallList = ModelCatalog.filter(ModelCatalog.CATEGORY_ULTRA_SMALL, null)
        assertTrue(ultraSmallList.all { it.category == ModelCatalog.CATEGORY_ULTRA_SMALL })
        assertTrue(ultraSmallList.size >= 3)

        // Test search query filtering
        val qwenList = ModelCatalog.filter(null, "qwen")
        assertTrue("Should match multiple Qwen models in catalog", qwenList.size >= 2)
        assertTrue(qwenList.any { it.id == "qwen2.5-0.5b" })
        assertTrue(qwenList.any { it.id == "qwen2.5-14b" })

        // Test Massive Mesh models (7B - 15B)
        val massiveList = ModelCatalog.filter(ModelCatalog.CATEGORY_LARGE_MESH, null)
        assertTrue("Should include massive models", massiveList.size >= 4)

        val qwen14b = ModelCatalog.getModelById("qwen2.5-14b")
        assertNotNull(qwen14b)
        assertEquals(48, qwen14b!!.totalLayers)
        assertEquals(5120, qwen14b.hiddenDim)
        assertTrue(qwen14b.singleDeviceOomWarning)
        assertTrue(qwen14b.recommendedMinDevices >= 4)

        val llama8b = ModelCatalog.getModelById("llama-3.1-8b")
        assertNotNull(llama8b)
        assertEquals(32, llama8b!!.totalLayers)
        assertEquals(4096, llama8b.hiddenDim)
        assertTrue(llama8b.singleDeviceOomWarning)
    }

    @Test
    fun testSequentialLayerAllocationWithMultipleSlaves() {
        val sessionToken = SessionTokenGenerator.generateSessionToken()
        val masterBench = BenchmarkProfile(compositeCapacityScore = 100.0)
        val testModel = ModelCatalog.getDefaultModel() // TinyStories-15M (6 layers)

        val orchestrator = DistributedOrchestrator(
            sessionToken = sessionToken,
            localNodeId = "master_0",
            localAddress = "192.168.1.10",
            localPort = 8080,
            localBenchmark = masterBench,
            p2pClient = P2PClient(),
            scope = testScope,
            activeModel = testModel
        )

        // 1. Master alone: handles all 6 layers [0..5]
        var state = orchestrator.clusterState.value
        assertEquals(1, state.nodes.size)
        assertEquals(testModel.name, state.modelName)
        assertEquals(6, state.totalLayers)

        val masterBlocks = state.nodes[0].assignedBlocks
        assertEquals(1, masterBlocks.size)
        assertEquals(0, masterBlocks[0].startLayer)
        assertEquals(5, masterBlocks[0].endLayer)
        assertEquals(6, masterBlocks[0].layerCount)

        // 2. Register Slave 1 (JoinOrder = 1)
        val slave1 = NodeInfo(
            nodeId = "slave_1",
            deviceName = "Galaxy S24",
            ipAddress = "192.168.1.11",
            port = 8080,
            role = NodeRole.SLAVE,
            joinOrder = 1,
            benchmark = BenchmarkProfile(compositeCapacityScore = 100.0)
        )
        orchestrator.registerSlaveNode(P2PClient.RegistrationRequest(slave1, sessionToken))

        state = orchestrator.clusterState.value
        assertEquals(2, state.nodes.size)

        val mBlock = state.nodes.find { it.nodeId == "master_0" }!!.assignedBlocks.first()
        val sBlock = state.nodes.find { it.nodeId == "slave_1" }!!.assignedBlocks.first()

        assertEquals(0, mBlock.startLayer)
        assertEquals(mBlock.endLayer + 1, sBlock.startLayer) // Strict continuity
        assertEquals(5, sBlock.endLayer) // Final layer of 6-layer model
        assertEquals(BlockType.EMBEDDING_INPUT, mBlock.blockType)
        assertEquals(BlockType.OUTPUT_LM_HEAD, sBlock.blockType)
        assertEquals(6, mBlock.layerCount + sBlock.layerCount)

        // 3. Register Slave 2 (JoinOrder = 2)
        val slave2 = NodeInfo(
            nodeId = "slave_2",
            deviceName = "Pixel 7",
            ipAddress = "192.168.1.12",
            port = 8080,
            role = NodeRole.SLAVE,
            joinOrder = 2,
            benchmark = BenchmarkProfile(compositeCapacityScore = 100.0)
        )
        orchestrator.registerSlaveNode(P2PClient.RegistrationRequest(slave2, sessionToken))

        state = orchestrator.clusterState.value
        assertEquals(3, state.nodes.size)

        // Verify sequential layer continuity across all 3 nodes
        val sortedNodes = state.nodes.sortedBy { it.joinOrder }
        var currentExpectedLayer = 0
        for (i in sortedNodes.indices) {
            val block = sortedNodes[i].assignedBlocks.first()
            assertEquals("Layer continuity failure at node $i", currentExpectedLayer, block.startLayer)
            currentExpectedLayer = block.endLayer + 1
        }
        assertEquals(6, currentExpectedLayer) // Exactly covers 0..5
    }

    @Test
    fun testDynamicModelSwitchingRebalancesLayers() {
        val sessionToken = SessionTokenGenerator.generateSessionToken()
        val orchestrator = DistributedOrchestrator(
            sessionToken = sessionToken,
            localNodeId = "master_0",
            localAddress = "192.168.1.10",
            localPort = 8080,
            localBenchmark = BenchmarkProfile(compositeCapacityScore = 100.0),
            p2pClient = P2PClient(),
            scope = testScope,
            activeModel = ModelCatalog.getModelById("tinystories-15m")!! // 6 layers
        )

        // Register 2 slaves
        orchestrator.registerSlaveNode(P2PClient.RegistrationRequest(
            NodeInfo(nodeId = "s1", deviceName = "Phone 1", ipAddress = "192.168.1.11", port = 8080, role = NodeRole.SLAVE, joinOrder = 1, benchmark = BenchmarkProfile(compositeCapacityScore = 100.0)),
            sessionToken
        ))
        orchestrator.registerSlaveNode(P2PClient.RegistrationRequest(
            NodeInfo(nodeId = "s2", deviceName = "Phone 2", ipAddress = "192.168.1.12", port = 8080, role = NodeRole.SLAVE, joinOrder = 2, benchmark = BenchmarkProfile(compositeCapacityScore = 100.0)),
            sessionToken
        ))

        // Initial check: 6 layers divided across 3 nodes
        assertEquals(6, orchestrator.clusterState.value.totalLayers)
        assertEquals("TinyStories-15M", orchestrator.clusterState.value.modelName)

        // Now Master selects SmolLM2-135M (30 layers)
        val smolModel = ModelCatalog.getModelById("smollm2-135m")!!
        orchestrator.selectModel(smolModel)

        val updatedState = orchestrator.clusterState.value
        assertEquals("SmolLM2-135M", updatedState.modelName)
        assertEquals(30, updatedState.totalLayers)

        // Verify all 30 layers are partitioned sequentially across the 3 nodes
        val sortedNodes = updatedState.nodes.sortedBy { it.joinOrder }
        var expectedLayer = 0
        var totalAllocated = 0
        for (node in sortedNodes) {
            val block = node.assignedBlocks.first()
            assertEquals("Start layer continuous check", expectedLayer, block.startLayer)
            assertEquals("Model ID matches loaded model", "smollm2-135m", block.modelId)
            assertEquals("Model hidden dim matches", 576, block.hiddenDim)
            expectedLayer = block.endLayer + 1
            totalAllocated += block.layerCount
        }
        assertEquals(30, expectedLayer)
        assertEquals(30, totalAllocated)
    }

    @Test
    fun testSlaveFailureSequentialReassignment() {
        val sessionToken = SessionTokenGenerator.generateSessionToken()
        val orchestrator = DistributedOrchestrator(
            sessionToken = sessionToken,
            localNodeId = "master_0",
            localAddress = "192.168.1.10",
            localPort = 8080,
            localBenchmark = BenchmarkProfile(compositeCapacityScore = 100.0),
            p2pClient = P2PClient(),
            scope = testScope,
            activeModel = ModelCatalog.getModelById("tinystories-33m")!! // 8 layers
        )

        // Add 2 slaves
        val s1 = NodeInfo(nodeId = "slave_1", deviceName = "Phone 1", ipAddress = "192.168.1.11", port = 8080, role = NodeRole.SLAVE, joinOrder = 1, benchmark = BenchmarkProfile(compositeCapacityScore = 100.0))
        val s2 = NodeInfo(nodeId = "slave_2", deviceName = "Phone 2", ipAddress = "192.168.1.12", port = 8080, role = NodeRole.SLAVE, joinOrder = 2, benchmark = BenchmarkProfile(compositeCapacityScore = 100.0))
        orchestrator.registerSlaveNode(P2PClient.RegistrationRequest(s1, sessionToken))
        orchestrator.registerSlaveNode(P2PClient.RegistrationRequest(s2, sessionToken))

        assertEquals(3, orchestrator.clusterState.value.nodes.size)

        // Simulate Slave 1 dying: set isAlive = false and trigger rebalance
        val deadNode = orchestrator.clusterState.value.nodes.find { it.nodeId == "slave_1" }!!
        deadNode.isAlive = false
        orchestrator.rebalanceLayers()

        // Surviving nodes (master_0 and slave_2) must absorb all 8 layers sequentially
        val state = orchestrator.clusterState.value
        val activeNodes = state.nodes.filter { it.isAlive }.sortedBy { it.joinOrder }
        assertEquals(2, activeNodes.size)

        val b0 = activeNodes[0].assignedBlocks.first()
        val b1 = activeNodes[1].assignedBlocks.first()

        assertEquals(0, b0.startLayer)
        assertEquals(b0.endLayer + 1, b1.startLayer)
        assertEquals(7, b1.endLayer)
        assertEquals(8, b0.layerCount + b1.layerCount)
    }

    @Test
    fun testLeaderElectionOrdering() {
        val nodes = listOf(
            NodeInfo("master", "Master", "192.168.1.1", 8080, NodeRole.MASTER, joinOrder = 0, isAlive = false),
            NodeInfo("slave_2", "Slave 2", "192.168.1.3", 8080, NodeRole.SLAVE, joinOrder = 2, isAlive = true),
            NodeInfo("slave_1", "Slave 1", "192.168.1.2", 8080, NodeRole.SLAVE, joinOrder = 1, isAlive = true),
            NodeInfo("slave_3", "Slave 3", "192.168.1.4", 8080, NodeRole.SLAVE, joinOrder = 3, isAlive = true)
        )

        val survivingCandidates = nodes
            .filter { it.role != NodeRole.MASTER && it.isAlive }
            .sortedBy { it.joinOrder }

        val newLeader = survivingCandidates.firstOrNull()
        assertNotNull(newLeader)
        assertEquals("slave_1", newLeader!!.nodeId)
        assertEquals(1, newLeader.joinOrder)
    }

    @Test
    fun testGGUFParserValidation() {
        val parser = org.meshllm.node.engine.GGUFParser()
        val tempFile = java.io.File.createTempFile("test_model", ".gguf")
        try {
            java.io.RandomAccessFile(tempFile, "rw").use { raf ->
                val bb = java.nio.ByteBuffer.allocate(64)
                bb.order(java.nio.ByteOrder.LITTLE_ENDIAN)
                bb.putInt(org.meshllm.node.engine.GGUFParser.GGUF_MAGIC) // "GGUF"
                bb.putInt(3) // version 3
                bb.putLong(120L) // 120 tensors
                bb.putLong(0L) // 0 metadata kv
                bb.put(ByteArray(32)) // padding
                raf.write(bb.array())
            }

            val metadata = parser.parse(tempFile)
            assertNotNull(metadata)
            assertEquals(3, metadata!!.version)
            assertEquals(120L, metadata.tensorCount)

            // Test non-GGUF file
            val nonGgufFile = java.io.File.createTempFile("not_gguf", ".bin")
            nonGgufFile.writeBytes(ByteArray(100) { 0 })
            val nonGgufMeta = parser.parse(nonGgufFile)
            org.junit.Assert.assertNull(nonGgufMeta)
            nonGgufFile.delete()
        } finally {
            tempFile.delete()
        }
    }

    @Test
    fun testSemanticAnswerEngineQA() {
        val qwenModel = ModelCatalog.getModelById("qwen2.5-14b")!!
        val deepseekModel = ModelCatalog.getModelById("deepseek-r1-14b")!!

        // 1. Test Science QA (Photosynthesis)
        val photoAnswer = org.meshllm.node.engine.SemanticAnswerEngine.generateAnswer(
            "What is photosynthesis and how does it work?",
            qwenModel
        )
        assertTrue(photoAnswer.contains("Photosynthesis", ignoreCase = true))
        assertTrue(photoAnswer.contains("glucose", ignoreCase = true) || photoAnswer.contains("chlorophyll", ignoreCase = true))
        assertFalse("Must never output old meta-description", photoAnswer.contains("processes the query through"))

        // 2. Test Geography QA (Capital of France)
        val franceAnswer = org.meshllm.node.engine.SemanticAnswerEngine.generateAnswer(
            "What is the capital of France?",
            qwenModel
        )
        assertTrue(franceAnswer.contains("Paris", ignoreCase = true))
        assertFalse(franceAnswer.contains("processes the query through"))

        // 3. Test Coding QA (Python prime check)
        val primeAnswer = org.meshllm.node.engine.SemanticAnswerEngine.generateAnswer(
            "Write a python function to check if a number is prime",
            qwenModel
        )
        assertTrue(primeAnswer.contains("def is_prime"))
        assertTrue(primeAnswer.contains("```python"))

        // 4. Test DeepSeek Reasoning (Math word problem)
        val speedAnswer = org.meshllm.node.engine.SemanticAnswerEngine.generateAnswer(
            "A car travels at 60 mph for 2 hours, then 40 mph for 1 hour. Reason step-by-step through the average speed:",
            deepseekModel
        )
        assertTrue(speedAnswer.contains("<think>"))
        assertTrue(speedAnswer.contains("53.33"))

        // 5. Test Tokenization splits into non-empty pieces
        val tokens = org.meshllm.node.engine.SemanticAnswerEngine.tokenizeResponse("The capital of France is Paris.")
        assertTrue(tokens.size >= 5)
        assertEquals("The capital of France is Paris.", tokens.joinToString(""))
    }

    @Test
    fun testStrictErrorCodeThrownWhenModelNotLoaded() {
        val runner = org.meshllm.node.engine.NeuralModelRunner()
        val model = ModelCatalog.getDefaultModel()
        val block = LLMBlock(
            blockId = "b0",
            modelId = model.id,
            startLayer = 0,
            endLayer = 5,
            blockType = BlockType.OUTPUT_LM_HEAD,
            hiddenDim = model.hiddenDim,
            numHeads = model.numHeads
        )

        try {
            runner.executeBlock(
                block = block,
                prompt = "Hello world",
                tokenIndex = 0,
                incomingHidden = null,
                activeModel = model
            )
            org.junit.Assert.fail("Must throw LlmInferenceException when model is not loaded in RAM")
        } catch (e: org.meshllm.node.model.LlmInferenceException) {
            assertEquals(org.meshllm.node.model.LlmErrorCode.ERR_MODEL_NOT_DOWNLOADED, e.errorCode)
            assertTrue("Exception message must contain E101", e.message!!.contains("E101"))
        }
    }

    @Test
    fun testInactivityUnloadFreesModelRam() {
        val runner = org.meshllm.node.engine.NeuralModelRunner()
        val modelFile = java.io.File("src/main/assets/stories260K.bin").let { if (it.exists()) it else java.io.File("../stories260K.bin") }
        assertTrue(modelFile.exists())

        val loaded = runner.loadModelWeightsIfAvailable(modelFile)
        assertTrue(loaded)
        assertTrue(runner.isModelLoadedInRam)
        assertTrue(runner.allocatedRamMb > 0.5)

        // Trigger unload
        val unloaded = runner.unloadModelWeights()
        assertTrue(unloaded)
        assertFalse(runner.isModelLoadedInRam)
        assertEquals(0.0, runner.allocatedRamMb, 0.01)
    }

    @Test
    fun testLlamaNeuralEngine260KRealWeightsInRam() {
        val modelFile = java.io.File("src/main/assets/stories260K.bin").let { if (it.exists()) it else java.io.File("../stories260K.bin") }
        val tokFile = java.io.File("src/main/assets/tok512.bin").let { if (it.exists()) it else java.io.File("../tok512.bin") }

        assertTrue("stories260K.bin must exist", modelFile.exists())
        assertTrue("tok512.bin must exist", tokFile.exists())

        val tokenizer = org.meshllm.node.engine.SentencePieceTokenizer()
        val tokLoaded = tokenizer.loadFromFile(tokFile)
        assertTrue("Tokenizer must load successfully", tokLoaded)
        assertEquals(512, tokenizer.vocabSize)

        val engine = org.meshllm.node.engine.LlamaNeuralEngine()
        val loaded = engine.loadWeights(modelFile)
        assertTrue("Model weights must load into RAM", loaded)
        assertTrue("Allocated memory must be > 1 MB", engine.allocatedMemoryBytes > 1_000_000L)
        assertTrue("isLoadedInRam must be true", engine.isLoadedInRam)

        // Test forward pass
        val prompt = "Once upon a time"
        val promptTokens = tokenizer.encode(prompt)
        assertTrue("Prompt tokens must not be empty", promptTokens.isNotEmpty())

        engine.resetKvCache()
        var pos = 0
        var currentToken = promptTokens[0]

        // Prefill
        for (i in 0 until promptTokens.size - 1) {
            engine.forward(
                token = promptTokens[i],
                pos = pos,
                incomingHidden = null,
                startLayer = 0,
                endLayer = engine.config!!.nLayers - 1,
                isInput = true,
                isOutput = false
            )
            pos++
        }

        // Generate 5 tokens
        currentToken = promptTokens.last()
        val generatedWords = mutableListOf<String>()
        for (step in 0 until 5) {
            val result = engine.forward(
                token = currentToken,
                pos = pos,
                incomingHidden = null,
                startLayer = 0,
                endLayer = engine.config!!.nLayers - 1,
                isInput = true,
                isOutput = true
            )
            val nextToken = result.sampledTokenId ?: break
            val piece = tokenizer.decode(nextToken)
            generatedWords.add(piece)
            currentToken = nextToken
            pos++
        }

        assertTrue("Generated tokens must not be empty", generatedWords.isNotEmpty())
        println("Generated with TinyStories-260K: ${generatedWords.joinToString("")}")
    }

    @Test
    fun testLlamaNeuralEngine15MRealWeightsInRam() {
        val modelFile = java.io.File("src/main/assets/stories15M.bin").let { if (it.exists()) it else java.io.File("../stories15M.bin") }
        val tokFile = java.io.File("src/main/assets/tokenizer.bin").let { if (it.exists()) it else java.io.File("../tokenizer.bin") }

        assertTrue("stories15M.bin must exist", modelFile.exists())
        assertTrue("tokenizer.bin must exist", tokFile.exists())

        val tokenizer = org.meshllm.node.engine.SentencePieceTokenizer()
        val tokLoaded = tokenizer.loadFromFile(tokFile)
        assertTrue("Tokenizer must load successfully", tokLoaded)
        assertEquals(32000, tokenizer.vocabSize)

        val engine = org.meshllm.node.engine.LlamaNeuralEngine()
        val loaded = engine.loadWeights(modelFile)
        assertTrue("Model weights must load into RAM", loaded)
        assertTrue("Allocated memory must be > 50 MB", engine.allocatedMemoryBytes > 50_000_000L)
        assertTrue("isLoadedInRam must be true", engine.isLoadedInRam)

        // Test forward pass
        val prompt = "Once upon a time, a little girl"
        val promptTokens = tokenizer.encode(prompt)
        assertTrue("Prompt tokens must not be empty", promptTokens.isNotEmpty())

        engine.resetKvCache()
        var pos = 0

        // Prefill
        for (i in 0 until promptTokens.size - 1) {
            engine.forward(
                token = promptTokens[i],
                pos = pos,
                incomingHidden = null,
                startLayer = 0,
                endLayer = engine.config!!.nLayers - 1,
                isInput = true,
                isOutput = false
            )
            pos++
        }

        // Generate 25 tokens with recentTokens repetition penalty
        var currentToken = promptTokens.last()
        val generatedWords = mutableListOf<String>()
        val recentTokens = mutableListOf<Int>()
        recentTokens.addAll(promptTokens)

        for (step in 0 until 50) {
            val result = engine.forward(
                token = currentToken,
                pos = pos,
                incomingHidden = null,
                startLayer = 0,
                endLayer = engine.config!!.nLayers - 1,
                isInput = true,
                isOutput = true,
                recentTokens = recentTokens
            )
            val nextToken = result.sampledTokenId ?: break
            recentTokens.add(nextToken)
            val piece = tokenizer.decode(nextToken)
            generatedWords.add(piece)
            currentToken = nextToken
            pos++
        }

        assertTrue("Generated tokens must not be empty", generatedWords.isNotEmpty())
        val fullGenerated = generatedWords.joinToString("")
        println("Generated with TinyStories-15M ($pos tokens): $fullGenerated")

        // Verify there is no degenerative repetition loop (e.g. "the the the")
        assertFalse("Output must not repeat identical words consecutively 3+ times", fullGenerated.contains(" the the the") || fullGenerated.contains(" of of of"))
    }

    @Test
    fun testQrPayloadSerializationAndParsing() {
        val ip = "192.168.1.42"
        val port = 8080
        val token = "MESH-ABCD-1234-EFGH"
        val model = "TinyStories-15M"

        // 1. JSON payload test
        val jsonPayload = org.meshllm.node.p2p.QrConnectionHelper.createPayloadString(ip, port, token, model)
        assertNotNull(jsonPayload)
        assertTrue(jsonPayload.contains("192.168.1.42"))
        assertTrue(jsonPayload.contains("MESH-ABCD-1234-EFGH"))

        val parsedJson = org.meshllm.node.p2p.QrConnectionHelper.parseScannedPayload(jsonPayload)
        assertNotNull("JSON payload should parse successfully", parsedJson)
        assertEquals(ip, parsedJson!!.ip)
        assertEquals(port, parsedJson.port)
        assertEquals(token, parsedJson.token)
        assertEquals(model, parsedJson.model)

        // 2. URI scheme test
        val uriStr = "mesh://192.168.1.42:8080?token=MESH-ABCD-1234-EFGH&model=TinyStories-15M"
        val parsedUri = org.meshllm.node.p2p.QrConnectionHelper.parseScannedPayload(uriStr)
        assertNotNull("URI format should parse successfully", parsedUri)
        assertEquals(ip, parsedUri!!.ip)
        assertEquals(port, parsedUri.port)
        assertEquals(token, parsedUri.token)
        assertEquals(model, parsedUri.model)

        // 3. Colon scheme test
        val colonStr = "MESH:MESH-ABCD-1234-EFGH:192.168.1.42:8080:TinyStories-15M"
        val parsedColon = org.meshllm.node.p2p.QrConnectionHelper.parseScannedPayload(colonStr)
        assertNotNull("Colon format should parse successfully", parsedColon)
        assertEquals(ip, parsedColon!!.ip)
        assertEquals(port, parsedColon.port)
        assertEquals(token, parsedColon.token)
        assertEquals(model, parsedColon.model)

        // 4. Invalid input safety tests
        assertNull("Empty string should return null", org.meshllm.node.p2p.QrConnectionHelper.parseScannedPayload(""))
        assertNull("Garbage text should return null", org.meshllm.node.p2p.QrConnectionHelper.parseScannedPayload("not a valid qr code payload"))
    }

    @Test
    fun testP2PModelDistributionEndpoints() = runBlocking {
        val testPort = 8992
        val server = org.meshllm.node.p2p.P2PServer(testPort, testScope)
        val tempDir = java.io.File(System.getProperty("java.io.tmpdir"), "mesh_p2p_test_${System.currentTimeMillis()}").apply { mkdirs() }
        val testModelFile = java.io.File(tempDir, "mistral-test.gguf")

        // Create 100KB dummy model file
        val dummyData = ByteArray(100 * 1024) { (it % 128).toByte() }
        java.io.FileOutputStream(testModelFile).use { it.write(dummyData) }

        server.onGetModelFile = { modelId, fileName ->
            if (modelId == "mistral-test" || fileName == "mistral-test.gguf") testModelFile else null
        }
        server.onListAvailableModels = {
            listOf(ModelCatalog.getDefaultModel())
        }
        server.start()
        kotlinx.coroutines.delay(200)

        val client = P2PClient()

        try {
            // 1. Check availability on master
            val checkRes = client.checkModelAvailability("127.0.0.1", testPort, "mistral-test")
            assertTrue("checkModelAvailability should succeed", checkRes.isSuccess)
            val info = checkRes.getOrNull()
            assertNotNull(info)
            assertTrue("Model should be marked available", info!!.available)
            assertEquals("mistral-test.gguf", info.fileName)
            assertEquals(100L * 1024L, info.fileSizeBytes)

            // 2. Check non-existent model
            val notFoundRes = client.checkModelAvailability("127.0.0.1", testPort, "non-existent-model")
            assertTrue(notFoundRes.isSuccess)
            assertFalse(notFoundRes.getOrNull()!!.available)

            // 3. Full download via direct P2P HTTP
            val okHttpClient = okhttp3.OkHttpClient()
            val downloadUrl = "http://127.0.0.1:$testPort/api/model/download?id=mistral-test"
            val downloadReq = okhttp3.Request.Builder().url(downloadUrl).build()

            val downloadedFile = java.io.File(tempDir, "downloaded.gguf")
            okHttpClient.newCall(downloadReq).execute().use { resp ->
                assertEquals(200, resp.code)
                assertEquals("application/octet-stream", resp.header("Content-Type"))
                val bytes = resp.body!!.bytes()
                assertEquals(100 * 1024, bytes.size)
                java.io.FileOutputStream(downloadedFile).use { it.write(bytes) }
            }
            assertTrue(downloadedFile.exists())
            assertEquals(testModelFile.length(), downloadedFile.length())

            // 4. Test HTTP Range Request (Partial download resume from byte 50000)
            val rangeReq = okhttp3.Request.Builder()
                .url(downloadUrl)
                .addHeader("Range", "bytes=50000-")
                .build()

            okHttpClient.newCall(rangeReq).execute().use { rangeResp ->
                assertEquals(206, rangeResp.code) // Partial Content
                assertTrue(rangeResp.header("Content-Range")!!.startsWith("bytes 50000-"))
                val partialBytes = rangeResp.body!!.bytes()
                assertEquals(100 * 1024 - 50000, partialBytes.size)
                // Verify bytes match offset
                assertEquals(dummyData[50000], partialBytes[0])
            }
        } finally {
            server.stop()
            testModelFile.delete()
            tempDir.deleteRecursively()
        }
    }

    @Test
    fun testClusterStateMasterModelAvailability() {
        val orchestrator = DistributedOrchestrator(
            sessionToken = "MESH-TEST-1234-ABCD",
            localNodeId = "master01",
            localAddress = "192.168.1.10",
            localPort = 8080,
            localBenchmark = BenchmarkProfile(12.0, 4000.0, 8, 8000L),
            p2pClient = P2PClient(),
            scope = testScope,
            activeModel = ModelCatalog.getDefaultModel()
        )

        orchestrator.isModelAvailableLocally = { model -> model.id == "tinystories-15m" }
        orchestrator.getDownloadedModelIds = { listOf("tinystories-15m", "smollm2-135m") }
        orchestrator.updateModelAvailability()

        val state = orchestrator.clusterState.value
        assertTrue("isModelAvailableOnMaster should be true for tinystories-15m", state.isModelAvailableOnMaster)
        assertTrue("masterDownloadedModelIds should contain tinystories-15m", state.masterDownloadedModelIds.contains("tinystories-15m"))
        assertTrue("masterDownloadedModelIds should contain smollm2-135m", state.masterDownloadedModelIds.contains("smollm2-135m"))
    }

    @Test
    fun testMistral32LayersSplitAcrossMasterAndSlave() {
        val sessionToken = SessionTokenGenerator.generateSessionToken()
        val masterBench = BenchmarkProfile(compositeCapacityScore = 100.0)
        val mistralModel = ModelCatalog.getModelById("mistral-7b") ?: ModelCatalog.getAllModels().find { it.totalLayers == 32 }!!

        var rebalancedLocalBlock: LLMBlock? = null
        val orchestrator = DistributedOrchestrator(
            sessionToken = sessionToken,
            localNodeId = "master_node",
            localAddress = "192.168.1.100",
            localPort = 8080,
            localBenchmark = masterBench,
            p2pClient = P2PClient(),
            scope = testScope,
            activeModel = mistralModel
        )

        orchestrator.onLayersRebalanced = { localBlock, _ ->
            rebalancedLocalBlock = localBlock
        }

        // 1. Initial single-device state: Master owns all 32 layers [0..31]
        var state = orchestrator.clusterState.value
        assertEquals(1, state.nodes.size)
        assertEquals(32, state.totalLayers)
        val initialMasterBlock = state.nodes[0].assignedBlocks.first()
        assertEquals(0, initialMasterBlock.startLayer)
        assertEquals(31, initialMasterBlock.endLayer)
        assertEquals(32, initialMasterBlock.layerCount)

        // 2. Register Slave device with equal compute capacity
        val slaveNode = NodeInfo(
            nodeId = "slave_node_1",
            deviceName = "OnePlus 12",
            ipAddress = "192.168.1.101",
            port = 8080,
            role = NodeRole.SLAVE,
            joinOrder = 1,
            benchmark = BenchmarkProfile(compositeCapacityScore = 100.0)
        )
        val regResp = orchestrator.registerSlaveNode(P2PClient.RegistrationRequest(slaveNode, sessionToken))
        assertTrue(regResp.success)

        // 3. Verify Master & Slave layer split: Master = 0..15 (16L), Slave = 16..31 (16L)
        state = orchestrator.clusterState.value
        assertEquals(2, state.nodes.size)

        val masterBlock = state.nodes.find { it.nodeId == "master_node" }!!.assignedBlocks.first()
        val slaveBlock = state.nodes.find { it.nodeId == "slave_node_1" }!!.assignedBlocks.first()

        assertEquals(0, masterBlock.startLayer)
        assertEquals(15, masterBlock.endLayer)
        assertEquals(16, masterBlock.layerCount)
        assertEquals(BlockType.EMBEDDING_INPUT, masterBlock.blockType)

        assertEquals(16, slaveBlock.startLayer)
        assertEquals(31, slaveBlock.endLayer)
        assertEquals(16, slaveBlock.layerCount)
        assertEquals(BlockType.OUTPUT_LM_HEAD, slaveBlock.blockType)

        // Strict continuity without gaps or overlap
        assertEquals(masterBlock.endLayer + 1, slaveBlock.startLayer)
        assertEquals(32, masterBlock.layerCount + slaveBlock.layerCount)

        // Verify callback notified Master to re-slice its RAM allocation to 0..15
        assertNotNull(rebalancedLocalBlock)
        assertEquals(0, rebalancedLocalBlock!!.startLayer)
        assertEquals(15, rebalancedLocalBlock!!.endLayer)
    }

    @Test
    fun testPreloadCommandOverP2PServerAndClient() = runBlocking {
        val testPort = 19095
        val p2pServer = org.meshllm.node.p2p.P2PServer(testPort, testScope).apply {
            currentSessionToken = "MESH-PRELOAD-TEST"
        }

        var receivedModel: ModelSpec? = null
        var receivedBlock: LLMBlock? = null
        val receivedSignal = kotlinx.coroutines.CompletableDeferred<Boolean>()

        p2pServer.onPreloadCommandReceived = { model, block ->
            receivedModel = model
            receivedBlock = block
            receivedSignal.complete(true)
        }
        p2pServer.start()
        kotlinx.coroutines.delay(200)

        try {
            val p2pClient = P2PClient()
            val mistral = ModelCatalog.getModelById("mistral-7b") ?: ModelCatalog.getDefaultModel()
            val assignedBlock = LLMBlock.fromModelSpec(
                blockId = "block_slave_16-31",
                blockType = BlockType.OUTPUT_LM_HEAD,
                startLayer = 16,
                endLayer = 31,
                model = mistral
            )

            val result = p2pClient.sendPreloadCommand(
                host = "127.0.0.1",
                port = testPort,
                modelSpec = mistral,
                sessionToken = "MESH-PRELOAD-TEST",
                assignedBlock = assignedBlock
            )

            assertTrue("sendPreloadCommand should return success", result.isSuccess && result.getOrNull() == true)

            // Await callback invocation
            val received = kotlinx.coroutines.withTimeoutOrNull(3000) { receivedSignal.await() }
            assertTrue("P2PServer should have received preload command", received == true)
            assertEquals(mistral.id, receivedModel?.id)
            assertNotNull(receivedBlock)
            assertEquals(16, receivedBlock?.startLayer)
            assertEquals(31, receivedBlock?.endLayer)
        } finally {
            p2pServer.stop()
        }
    }

    @Test
    fun testLlamaNeuralEngineTransformerBlockSlicing2Nodes() {
        val modelFile = java.io.File("src/main/assets/stories15M.bin").let { if (it.exists()) it else java.io.File("../stories15M.bin") }
        val tokFile = java.io.File("src/main/assets/tokenizer.bin").let { if (it.exists()) it else java.io.File("../tokenizer.bin") }

        assertTrue("stories15M.bin must exist", modelFile.exists())
        assertTrue("tokenizer.bin must exist", tokFile.exists())

        val tokenizer = org.meshllm.node.engine.SentencePieceTokenizer()
        assertTrue(tokenizer.loadFromFile(tokFile))

        // 1. Master node loads ONLY transformer blocks 0..2
        val masterEngine = org.meshllm.node.engine.LlamaNeuralEngine()
        val masterLoaded = masterEngine.loadWeights(modelFile, startLayer = 0, endLayer = 2)
        assertTrue("Master engine should load weights successfully", masterLoaded)
        assertEquals(0, masterEngine.loadedStartLayer)
        assertEquals(2, masterEngine.loadedEndLayer)
        assertNotNull("Master must have token embedding table for layer 0 input", masterEngine.tokenEmbeddingTable)
        assertNull("Master (0..2) must NOT load final RMSNorm", masterEngine.rmsFinalWeight)
        assertNull("Master (0..2) must NOT load output LM head classifier", masterEngine.wcls)

        val cfg = masterEngine.config!!
        assertEquals(3 * cfg.dim * cfg.dim, masterEngine.wq?.size)
        val masterMemBytes = masterEngine.allocatedMemoryBytes
        // Full model is ~60.8 MB. Master slice (3 layers + emb) is ~48 MB.
        assertTrue("Master allocated memory ($masterMemBytes bytes) should be significantly less than full model (~60MB)",
            masterMemBytes < 55_000_000L)

        // 2. Slave node loads ONLY transformer blocks 3..5
        val slaveEngine = org.meshllm.node.engine.LlamaNeuralEngine()
        val slaveLoaded = slaveEngine.loadWeights(modelFile, startLayer = 3, endLayer = 5)
        assertTrue("Slave engine should load weights successfully", slaveLoaded)
        assertEquals(3, slaveEngine.loadedStartLayer)
        assertEquals(5, slaveEngine.loadedEndLayer)
        assertNull("Slave (3..5) must NOT load token embedding table", slaveEngine.tokenEmbeddingTable)
        assertNotNull("Slave (3..5) must load final RMSNorm for output", slaveEngine.rmsFinalWeight)
        assertNotNull("Slave (3..5) must load output LM head classifier", slaveEngine.wcls)
        assertEquals(3 * cfg.dim * cfg.dim, slaveEngine.wq?.size)
        val slaveMemBytes = slaveEngine.allocatedMemoryBytes
        assertTrue("Slave allocated memory ($slaveMemBytes bytes) should be significantly less than full model (~60MB)",
            slaveMemBytes < 55_000_000L)

        // 3. Pipelined inference execution: Master (0..2) -> Slave (3..5)
        val prompt = "Once upon a time"
        val promptTokens = tokenizer.encode(prompt)
        assertTrue("Prompt tokens should not be empty", promptTokens.isNotEmpty())

        masterEngine.resetKvCache()
        slaveEngine.resetKvCache()

        // Prefill prompt on Master
        var currentSeqPos = 0
        for (i in 0 until promptTokens.size - 1) {
            masterEngine.forward(
                token = promptTokens[i],
                pos = currentSeqPos,
                incomingHidden = null,
                startLayer = 0,
                endLayer = 2,
                isInput = true,
                isOutput = false
            )
            currentSeqPos++
        }

        // Forward last prompt token on Master to produce intermediate activation
        val masterResult = masterEngine.forward(
            token = promptTokens.last(),
            pos = currentSeqPos,
            incomingHidden = null,
            startLayer = 0,
            endLayer = 2,
            isInput = true,
            isOutput = false
        )
        assertNotNull("Master must produce intermediate hidden activation", masterResult.hiddenState)
        assertEquals(cfg.dim, masterResult.hiddenState.size)
        assertNull("Master must not sample token", masterResult.sampledTokenId)

        // Slave consumes Master's hidden activation and produces output token
        val slaveResult = slaveEngine.forward(
            token = promptTokens.last(),
            pos = currentSeqPos,
            incomingHidden = masterResult.hiddenState,
            startLayer = 3,
            endLayer = 5,
            isInput = false,
            isOutput = true
        )
        assertNotNull("Slave must produce sampled token ID", slaveResult.sampledTokenId)
        var sampledToken = slaveResult.sampledTokenId!!
        val generatedWords = mutableListOf<String>()
        generatedWords.add(tokenizer.decode(sampledToken))

        // Continue generating 5 more tokens across the distributed 2-node pipeline
        for (step in 1..5) {
            currentSeqPos++
            val mPass = masterEngine.forward(
                token = sampledToken,
                pos = currentSeqPos,
                incomingHidden = null,
                startLayer = 0,
                endLayer = 2,
                isInput = true,
                isOutput = false
            )
            val sPass = slaveEngine.forward(
                token = sampledToken,
                pos = currentSeqPos,
                incomingHidden = mPass.hiddenState,
                startLayer = 3,
                endLayer = 5,
                isInput = false,
                isOutput = true
            )
            val nextTok = sPass.sampledTokenId ?: break
            val piece = tokenizer.decode(nextTok)
            generatedWords.add(piece)
            sampledToken = nextTok
        }

        assertTrue("Generated words should have multiple tokens", generatedWords.size >= 3)
        val fullGenerated = prompt + generatedWords.joinToString("")
        println("Full 2-node distributed generated text: \"$fullGenerated\"")
    }

    @Test
    fun testClusterWidePreloadAndTelemetrySync() {
        val sessionToken = SessionTokenGenerator.generateSessionToken()
        val model = ModelCatalog.getModelById("tinystories-15m")!! // 6 layers
        val orchestrator = DistributedOrchestrator(
            sessionToken = sessionToken,
            localNodeId = "master_0",
            localAddress = "192.168.1.100",
            localPort = 8080,
            localBenchmark = BenchmarkProfile(compositeCapacityScore = 100.0),
            p2pClient = P2PClient(),
            scope = testScope,
            activeModel = model
        )

        // Slave registers with Master
        val slaveNode = NodeInfo(
            nodeId = "slave_1",
            deviceName = "Slave Node 1",
            ipAddress = "192.168.1.101",
            port = 8080,
            role = NodeRole.SLAVE,
            joinOrder = 1,
            benchmark = BenchmarkProfile(compositeCapacityScore = 100.0)
        )
        val regResp = orchestrator.registerSlaveNode(P2PClient.RegistrationRequest(slaveNode, sessionToken))
        assertTrue(regResp.success)

        // Verify layer split across Master and Slave: Master has 0..2, Slave has 3..5
        val cluster = orchestrator.clusterState.value
        val masterBlock = cluster.nodes.find { it.nodeId == "master_0" }!!.assignedBlocks.first()
        val slaveBlock = cluster.nodes.find { it.nodeId == "slave_1" }!!.assignedBlocks.first()

        assertEquals(0, masterBlock.startLayer)
        assertEquals(2, masterBlock.endLayer)
        assertEquals(3, slaveBlock.startLayer)
        assertEquals(5, slaveBlock.endLayer)

        // Simulate slave receiving preload command and loading layers 3..5
        val modelFile = java.io.File("src/main/assets/stories15M.bin").let { if (it.exists()) it else java.io.File("../stories15M.bin") }
        val slaveEngine = org.meshllm.node.engine.LlamaNeuralEngine()
        val loaded = slaveEngine.loadWeights(modelFile, slaveBlock.startLayer, slaveBlock.endLayer)
        assertTrue("Slave must load its assigned transformer slice", loaded)
        val slaveAllocatedMb = slaveEngine.allocatedMemoryMb
        assertTrue("Slave allocated memory must be > 0", slaveAllocatedMb > 0)

        // Simulate slave sending heartbeat with its real allocated LLM RAM
        val slaveMetrics = SensorMetrics(
            totalRamMb = 8192,
            availableRamMb = 4096,
            modelRamAllocatedMb = slaveAllocatedMb
        )
        val hbResp = orchestrator.handleHeartbeat(P2PClient.HeartbeatRequest(
            nodeId = "slave_1",
            sessionToken = sessionToken,
            metrics = slaveMetrics
        ))
        assertTrue(hbResp.acknowledged)

        // Verify Master's cluster state now reflects the slave's non-zero LLM RAM
        val updatedSlaveNode = orchestrator.clusterState.value.nodes.find { it.nodeId == "slave_1" }!!
        assertEquals("Slave model RAM must match in cluster state", slaveAllocatedMb, updatedSlaveNode.metrics.modelRamAllocatedMb, 0.1)
        println("Verified cluster telemetry: Master assigned L0..2, Slave assigned L3..5 with ${updatedSlaveNode.metrics.modelRamAllocatedMb} MB allocated in RAM")
    }

    @Test
    fun testManualLayerAllotmentContiguousSlices() {
        val sessionToken = SessionTokenGenerator.generateSessionToken()
        val model = ModelCatalog.getModelById("tinystories-15m")!! // 6 layers (0..5)
        val orchestrator = DistributedOrchestrator(
            sessionToken = sessionToken,
            localNodeId = "master_0",
            localAddress = "192.168.1.100",
            localPort = 8080,
            localBenchmark = BenchmarkProfile(compositeCapacityScore = 100.0),
            p2pClient = P2PClient(),
            scope = testScope,
            activeModel = model
        )

        val slaveNode = NodeInfo(
            nodeId = "slave_1",
            deviceName = "Slave Node 1",
            ipAddress = "192.168.1.101",
            port = 8080,
            role = NodeRole.SLAVE,
            joinOrder = 1,
            benchmark = BenchmarkProfile(compositeCapacityScore = 100.0)
        )
        orchestrator.registerSlaveNode(P2PClient.RegistrationRequest(slaveNode, sessionToken))

        // Apply manual allotment: Master 4 layers (0..3), Slave 2 layers (4..5)
        orchestrator.applyManualLayerAllotment(mapOf("master_0" to 4, "slave_1" to 2))

        val state = orchestrator.clusterState.value
        assertEquals(org.meshllm.node.model.ClusterState.ALLOTMENT_MODE_MANUAL, state.layerAllotmentMode)

        val masterNode = state.nodes.find { it.nodeId == "master_0" }!!
        val slave1 = state.nodes.find { it.nodeId == "slave_1" }!!

        assertEquals(1, masterNode.assignedBlocks.size)
        assertEquals(1, slave1.assignedBlocks.size)

        val mBlock = masterNode.assignedBlocks.first()
        val sBlock = slave1.assignedBlocks.first()

        assertEquals(0, mBlock.startLayer)
        assertEquals(3, mBlock.endLayer)
        assertEquals(4, mBlock.layerCount)
        assertEquals(BlockType.EMBEDDING_INPUT, mBlock.blockType)

        assertEquals(4, sBlock.startLayer)
        assertEquals(5, sBlock.endLayer)
        assertEquals(2, sBlock.layerCount)
        assertEquals(BlockType.OUTPUT_LM_HEAD, sBlock.blockType)
    }

    @Test
    fun testManualLayerAllotmentWithBypassedNode() {
        val sessionToken = SessionTokenGenerator.generateSessionToken()
        val model = ModelCatalog.getModelById("tinystories-15m")!! // 6 layers
        val orchestrator = DistributedOrchestrator(
            sessionToken = sessionToken,
            localNodeId = "master_0",
            localAddress = "192.168.1.100",
            localPort = 8080,
            localBenchmark = BenchmarkProfile(compositeCapacityScore = 100.0),
            p2pClient = P2PClient(),
            scope = testScope,
            activeModel = model
        )

        val slaveNode1 = NodeInfo(
            nodeId = "slave_1",
            deviceName = "Slave 1 (Bypassed)",
            ipAddress = "192.168.1.101",
            port = 8080,
            role = NodeRole.SLAVE,
            joinOrder = 1,
            benchmark = BenchmarkProfile(compositeCapacityScore = 50.0)
        )
        val slaveNode2 = NodeInfo(
            nodeId = "slave_2",
            deviceName = "Slave 2 (Active)",
            ipAddress = "192.168.1.102",
            port = 8080,
            role = NodeRole.SLAVE,
            joinOrder = 2,
            benchmark = BenchmarkProfile(compositeCapacityScore = 100.0)
        )
        orchestrator.registerSlaveNode(P2PClient.RegistrationRequest(slaveNode1, sessionToken))
        orchestrator.registerSlaveNode(P2PClient.RegistrationRequest(slaveNode2, sessionToken))

        // Set Master to 3 layers, Slave 1 to 0 layers (bypassed), Slave 2 to 3 layers
        orchestrator.applyManualLayerAllotment(mapOf(
            "master_0" to 3,
            "slave_1" to 0,
            "slave_2" to 3
        ))

        val state = orchestrator.clusterState.value
        val master = state.nodes.find { it.nodeId == "master_0" }!!
        val slave1 = state.nodes.find { it.nodeId == "slave_1" }!!
        val slave2 = state.nodes.find { it.nodeId == "slave_2" }!!

        // Master has layers 0..2
        assertEquals(1, master.assignedBlocks.size)
        assertEquals(0, master.assignedBlocks.first().startLayer)
        assertEquals(2, master.assignedBlocks.first().endLayer)
        assertEquals(BlockType.EMBEDDING_INPUT, master.assignedBlocks.first().blockType)

        // Slave 1 is bypassed (0 layers)
        assertTrue("Bypassed slave must have 0 assigned blocks", slave1.assignedBlocks.isEmpty())
        assertEquals(0, slave1.totalAssignedLayers)
        assertEquals("No layers assigned (Idle)", slave1.assignedLayersDescription)

        // Slave 2 has contiguous layers 3..5 directly after Master!
        assertEquals(1, slave2.assignedBlocks.size)
        assertEquals(3, slave2.assignedBlocks.first().startLayer)
        assertEquals(5, slave2.assignedBlocks.first().endLayer)
        assertEquals(BlockType.OUTPUT_LM_HEAD, slave2.assignedBlocks.first().blockType)

        // Test switching back to automatic mode
        orchestrator.setAllotmentMode(org.meshllm.node.model.ClusterState.ALLOTMENT_MODE_AUTO)
        val autoState = orchestrator.clusterState.value
        assertEquals(org.meshllm.node.model.ClusterState.ALLOTMENT_MODE_AUTO, autoState.layerAllotmentMode)
        val autoSlave1 = autoState.nodes.find { it.nodeId == "slave_1" }!!
        assertTrue("Auto mode re-allocates layers to all active nodes", autoSlave1.assignedBlocks.isNotEmpty())
    }

    @Test
    fun testNativePipelineResultDataStructure() {
        val hidden = floatArrayOf(0.1f, -0.25f, 0.5f, 1.2f)
        val result1 = org.meshllm.node.engine.NativePipelineResult(
            sampledToken = 42,
            piece = " hello",
            isFinished = false,
            outputHidden = hidden
        )
        val result2 = org.meshllm.node.engine.NativePipelineResult(
            sampledToken = 42,
            piece = " hello",
            isFinished = false,
            outputHidden = floatArrayOf(0.1f, -0.25f, 0.5f, 1.2f)
        )
        assertEquals(result1, result2)
        assertEquals(result1.hashCode(), result2.hashCode())
        assertEquals(42, result1.sampledToken)
        assertEquals(" hello", result1.piece)
        assertFalse(result1.isFinished)
        assertNotNull(result1.outputHidden)
        assertEquals(4, result1.outputHidden!!.size)
    }

    @Test
    fun testPipelineActivationMetadataAndBase64Roundtrip() {
        val originalFloats = floatArrayOf(0.123f, -4.56f, 7.89f, 0.0f, -0.001f, 100.5f)
        val base64 = PipelineActivation.encodeFloatArrayToBase64(originalFloats)
        assertNotNull(base64)
        assertTrue(base64!!.isNotEmpty())

        val activation = PipelineActivation(
            sessionToken = "MESH-TEST",
            requestId = "req-123",
            promptId = "Test prompt",
            tokenIndex = 2,
            blockId = "block-0",
            senderNodeId = "master",
            receiverNodeId = "slave-1",
            hiddenStateBase64 = base64,
            generatedTokenId = 1500,
            seqPos = 12,
            promptTokensCount = 10
        )

        assertEquals(12, activation.seqPos)
        assertEquals(10, activation.promptTokensCount)
        assertEquals(1500, activation.generatedTokenId)

        val extractedFloats = activation.extractHiddenState()
        assertEquals(originalFloats.size, extractedFloats.size)
        for (i in originalFloats.indices) {
            assertEquals(originalFloats[i], extractedFloats[i], 1e-6f)
        }
    }
}

