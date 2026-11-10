package org.meshllm.node.engine

import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.meshllm.node.model.BenchmarkProfile
import org.meshllm.node.model.BlockType
import org.meshllm.node.model.ClusterState
import org.meshllm.node.model.LLMBlock
import org.meshllm.node.model.ModelCatalog
import org.meshllm.node.model.ModelSpec
import org.meshllm.node.model.NodeInfo
import org.meshllm.node.model.NodeRole
import org.meshllm.node.model.PipelineActivation
import org.meshllm.node.model.SensorMetrics
import org.meshllm.node.p2p.P2PClient
import java.util.UUID

class DistributedOrchestrator(
    val sessionToken: String,
    val localNodeId: String,
    val localAddress: String,
    val localPort: Int,
    val localBenchmark: BenchmarkProfile,
    private val p2pClient: P2PClient,
    private val scope: CoroutineScope,
    var activeModel: ModelSpec = ModelCatalog.getDefaultModel()
) {
    companion object {
        private const val TAG = "DistributedOrchestrator"
        const val HEARTBEAT_TIMEOUT_MS = 6500L
    }

    private val _clusterState = MutableStateFlow(
        ClusterState(
            sessionToken = sessionToken,
            modelSpec = activeModel,
            modelName = activeModel.name,
            totalLayers = activeModel.totalLayers,
            masterNodeId = localNodeId,
            masterAddress = "$localAddress:$localPort"
        )
    )
    val clusterState: StateFlow<ClusterState> = _clusterState.asStateFlow()

    // Thread-safe node registry
    private val registeredNodes = mutableMapOf<String, NodeInfo>()
    private var nextJoinOrder = 1
    private var healthCheckJob: Job? = null

    // Callbacks
    var onClusterUpdated: ((ClusterState) -> Unit)? = null
    var onTokenGenerated: ((tokenPiece: String, totalTokens: Int, isFinished: Boolean) -> Unit)? = null
    var onPipelineTrace: ((trace: String, tokPerSec: Double, ttftMs: Long) -> Unit)? = null
    var onLayersRebalanced: ((localBlock: LLMBlock, clusterState: ClusterState) -> Unit)? = null
    var isModelAvailableLocally: ((ModelSpec) -> Boolean)? = null
    var getDownloadedModelIds: (() -> List<String>)? = null

    fun updateModelAvailability() {
        val updated = _clusterState.value.copy(
            isModelAvailableOnMaster = isModelAvailableLocally?.invoke(activeModel) ?: false,
            masterDownloadedModelIds = getDownloadedModelIds?.invoke() ?: emptyList(),
            generationSequence = _clusterState.value.generationSequence + 1
        )
        _clusterState.value = updated
        onClusterUpdated?.invoke(updated)
    }

    private var inferenceJob: Job? = null
    private val localExecutor: LLMBlockExecutor by lazy {
        val appContext = try { org.meshllm.node.MeshLLMApp.instance } catch (_: Throwable) { null }
        val monitor = if (appContext != null) {
            try { org.meshllm.node.telemetry.SensorMonitor(appContext) } catch (_: Throwable) { null }
        } else null

        LLMBlockExecutor(
            sensorMonitor = monitor,
            performanceGovernor = org.meshllm.node.telemetry.PerformanceGovernor(),
            context = appContext
        )
    }
    val executor: LLMBlockExecutor get() = localExecutor

    init {
        // Requirement 7: Master node also runs a local slave node on the device
        val masterSelfWorker = NodeInfo(
            nodeId = localNodeId,
            deviceName = "Master Device (Local Worker)",
            ipAddress = localAddress,
            port = localPort,
            role = NodeRole.MASTER,
            joinOrder = 0,
            benchmark = localBenchmark,
            metrics = SensorMetrics(),
            isAlive = true
        )
        registeredNodes[localNodeId] = masterSelfWorker
        rebalanceLayers()
        startClusterHealthMonitor()
    }

    /**
     * Requirement 8 & User Request: Master selects open-source model to load into the mesh.
     * Slices the new model's totalLayers across all active devices.
     */
    @Synchronized
    fun selectModel(model: ModelSpec) {
        Log.i(TAG, "Master selected model: ${model.name} (${model.totalLayers} layers, ${model.paramCountStr})")
        activeModel = model
        _clusterState.value = _clusterState.value.copy(
            modelSpec = model,
            modelName = model.name,
            totalLayers = model.totalLayers,
            generationSequence = _clusterState.value.generationSequence + 1
        )

        // Dynamically recalculate layer slices for the new model
        rebalanceLayers()

        // Inform all connected slave nodes over P2P network
        scope.launch(Dispatchers.IO) {
            broadcastModelChangeToSlaves(model)
        }
    }

    private suspend fun broadcastModelChangeToSlaves(model: ModelSpec) {
        val remoteSlaves = registeredNodes.values.filter { it.isAlive && it.nodeId != localNodeId }
        for (slave in remoteSlaves) {
            try {
                p2pClient.notifyModelChange(slave.ipAddress, slave.port, model, sessionToken)
            } catch (e: Exception) {
                Log.w(TAG, "Failed notifying slave ${slave.nodeId} about model switch", e)
            }
        }
    }

    /**
     * Requirement 4 & 5: Slave node registration with compute capacity determination.
     */
    @Synchronized
    fun registerSlaveNode(request: P2PClient.RegistrationRequest): P2PClient.RegistrationResponse {
        val incomingNode = request.node
        Log.i(TAG, "Registering node ${incomingNode.nodeId} (${incomingNode.deviceName}) from ${incomingNode.ipAddress}")

        val assignedOrder = nextJoinOrder++
        val updatedNode = incomingNode.copy(
            role = NodeRole.SLAVE,
            joinOrder = assignedOrder,
            lastHeartbeat = System.currentTimeMillis(),
            isAlive = true
        )

        registeredNodes[incomingNode.nodeId] = updatedNode

        // Rebalance layers across all active nodes sequentially for the current model
        rebalanceLayers()

        return P2PClient.RegistrationResponse(
            success = true,
            message = "Node successfully registered into inference cluster",
            sessionToken = sessionToken,
            assignedJoinOrder = assignedOrder,
            clusterState = _clusterState.value
        )
    }

    /**
     * Requirement 3: Process heartbeat and adjust layers based on sensor readings.
     */
    @Synchronized
    fun handleHeartbeat(request: P2PClient.HeartbeatRequest): P2PClient.HeartbeatResponse {
        val node = registeredNodes[request.nodeId]
        if (node != null) {
            node.lastHeartbeat = System.currentTimeMillis()
            node.metrics = request.metrics

            // Check if node is dying or severely overheating
            val oldHealth = node.metrics.computeHealthFactor()
            val newHealth = request.metrics.computeHealthFactor()

            // If capacity changed significantly (>25%), trigger layer rebalance
            if (Math.abs(newHealth - oldHealth) > 0.25) {
                rebalanceLayers()
            }

            return P2PClient.HeartbeatResponse(
                acknowledged = true,
                clusterState = _clusterState.value
            )
        }

        return P2PClient.HeartbeatResponse(acknowledged = false, clusterState = null)
    }

    /**
     * Update master's own local sensor readings.
     */
    @Synchronized
    fun updateLocalMetrics(metrics: SensorMetrics) {
        val masterNode = registeredNodes[localNodeId]
        if (masterNode != null) {
            masterNode.metrics = metrics
            masterNode.lastHeartbeat = System.currentTimeMillis()
        }
    }

    var allotmentMode: String = ClusterState.ALLOTMENT_MODE_AUTO
        private set
    private val manualLayerCounts = mutableMapOf<String, Int>()

    fun setAllotmentMode(mode: String) {
        allotmentMode = mode
        if (mode == ClusterState.ALLOTMENT_MODE_AUTO) {
            rebalanceLayers()
        }
    }

    /**
     * Requirement 3 & 4: Dynamic sequential layer allocation for the active model.
     * Divides [0..totalLayers-1] proportionally among active nodes in sequential order.
     */
    @Synchronized
    fun rebalanceLayers() {
        allotmentMode = ClusterState.ALLOTMENT_MODE_AUTO
        val activeNodes = registeredNodes.values
            .filter { it.isAlive }
            .sortedBy { it.joinOrder } // Strict sequential order: Master (0), Slave 1 (1), Slave 2 (2)...

        if (activeNodes.isEmpty()) return

        val totalLayers = activeModel.totalLayers
        val n = activeNodes.size

        // Calculate dynamic effective capacity score for each node:
        // Capacity = BenchmarkScore * RealTimeSensorHealthFactor
        val capacities = activeNodes.map { node ->
            val eff = node.benchmark.getEffectiveCapacity(node.metrics)
            eff.coerceAtLeast(1.0)
        }
        val sumCapacity = capacities.sum()

        // Allocate layers sequentially without gaps
        var currentLayer = 0
        for (i in 0 until n) {
            val node = activeNodes[i]
            val isFirst = (i == 0)
            val isLast = (i == n - 1)

            val share = capacities[i] / sumCapacity
            val targetCount = if (isLast) {
                (totalLayers - currentLayer).coerceAtLeast(1) // Last node absorbs all remaining layers
            } else {
                val maxAllowed = (totalLayers - currentLayer - (n - 1 - i)).coerceAtLeast(1)
                val count = Math.round(share * totalLayers).toInt().coerceIn(1, maxAllowed)
                count
            }

            val startL = currentLayer
            val endL = (startL + targetCount - 1).coerceAtMost(totalLayers - 1)
            currentLayer = endL + 1

            // Determine LLM block type
            val blockType = when {
                isFirst && isLast -> BlockType.OUTPUT_LM_HEAD // Single node handles entire model
                isFirst -> BlockType.EMBEDDING_INPUT
                isLast -> BlockType.OUTPUT_LM_HEAD
                else -> BlockType.TRANSFORMER_BLOCK
            }

            val block = LLMBlock.fromModelSpec(
                blockId = "block_${node.nodeId}_$startL-$endL",
                blockType = blockType,
                startLayer = startL,
                endLayer = endL,
                model = activeModel
            )

            val updatedNode = node.copy(
                assignedBlocks = listOf(block),
                role = if (node.nodeId == localNodeId) NodeRole.MASTER else NodeRole.SLAVE
            )
            registeredNodes[node.nodeId] = updatedNode
        }

        val updatedState = _clusterState.value.copy(
            modelSpec = activeModel,
            modelName = activeModel.name,
            totalLayers = activeModel.totalLayers,
            nodes = registeredNodes.values.toList().sortedBy { it.joinOrder },
            generationSequence = _clusterState.value.generationSequence + 1,
            isModelAvailableOnMaster = isModelAvailableLocally?.invoke(activeModel) ?: false,
            masterDownloadedModelIds = getDownloadedModelIds?.invoke() ?: emptyList(),
            layerAllotmentMode = ClusterState.ALLOTMENT_MODE_AUTO
        )
        _clusterState.value = updatedState
        onClusterUpdated?.invoke(updatedState)

        val localBlock = registeredNodes[localNodeId]?.assignedBlocks?.firstOrNull()
        if (localBlock != null) {
            onLayersRebalanced?.invoke(localBlock, updatedState)
        }
    }

    /**
     * Manual continuous layer allotment.
     * Master node selects exact layer counts per node with sliders.
     * Nodes with 0 layers are bypassed entirely with 0 data/computation.
     */
    @Synchronized
    fun applyManualLayerAllotment(layerCounts: Map<String, Int>) {
        allotmentMode = ClusterState.ALLOTMENT_MODE_MANUAL
        manualLayerCounts.clear()
        manualLayerCounts.putAll(layerCounts)

        val activeNodes = registeredNodes.values
            .filter { it.isAlive }
            .sortedBy { it.joinOrder } // Strict sequential order: Master (0), Slave 1 (1)...

        if (activeNodes.isEmpty()) return

        val totalLayers = activeModel.totalLayers
        var currentLayer = 0

        // Determine which nodes have layers > 0
        val counts = activeNodes.map { node ->
            node to (manualLayerCounts[node.nodeId] ?: 0).coerceAtLeast(0)
        }

        val nonZeroNodes = counts.filter { it.second > 0 }
        val firstActiveNodeId = nonZeroNodes.firstOrNull()?.first?.nodeId
        val lastActiveNodeId = nonZeroNodes.lastOrNull()?.first?.nodeId

        for ((node, count) in counts) {
            if (count <= 0 || currentLayer >= totalLayers) {
                // 0 layers assigned: completely bypassed from pipeline & execution
                val updatedNode = node.copy(
                    assignedBlocks = emptyList(),
                    role = if (node.nodeId == localNodeId) NodeRole.MASTER else NodeRole.SLAVE
                )
                registeredNodes[node.nodeId] = updatedNode
                continue
            }

            val isFirstActive = (node.nodeId == firstActiveNodeId)
            val isLastActive = (node.nodeId == lastActiveNodeId)

            val startL = currentLayer
            val endL = if (isLastActive) {
                totalLayers - 1
            } else {
                (startL + count - 1).coerceAtMost(totalLayers - 1)
            }
            currentLayer = endL + 1

            val blockType = when {
                isFirstActive && isLastActive -> BlockType.OUTPUT_LM_HEAD
                isFirstActive -> BlockType.EMBEDDING_INPUT
                isLastActive -> BlockType.OUTPUT_LM_HEAD
                else -> BlockType.TRANSFORMER_BLOCK
            }

            val block = LLMBlock.fromModelSpec(
                blockId = "block_${node.nodeId}_$startL-$endL",
                blockType = blockType,
                startLayer = startL,
                endLayer = endL,
                model = activeModel
            )

            val updatedNode = node.copy(
                assignedBlocks = listOf(block),
                role = if (node.nodeId == localNodeId) NodeRole.MASTER else NodeRole.SLAVE
            )
            registeredNodes[node.nodeId] = updatedNode
        }

        val updatedState = _clusterState.value.copy(
            modelSpec = activeModel,
            modelName = activeModel.name,
            totalLayers = activeModel.totalLayers,
            nodes = registeredNodes.values.toList().sortedBy { it.joinOrder },
            generationSequence = _clusterState.value.generationSequence + 1,
            isModelAvailableOnMaster = isModelAvailableLocally?.invoke(activeModel) ?: false,
            masterDownloadedModelIds = getDownloadedModelIds?.invoke() ?: emptyList(),
            layerAllotmentMode = ClusterState.ALLOTMENT_MODE_MANUAL
        )
        _clusterState.value = updatedState
        onClusterUpdated?.invoke(updatedState)

        val localBlock = registeredNodes[localNodeId]?.assignedBlocks?.firstOrNull()
        if (localBlock != null) {
            onLayersRebalanced?.invoke(localBlock, updatedState)
        }
        broadcastPreloadToSlaves(activeModel)
    }

    /**
     * Broadcasts a preload instruction to all connected slave nodes for their assigned slices.
     */
    fun broadcastPreloadToSlaves(model: ModelSpec = activeModel) {
        val remoteSlaves = registeredNodes.values.filter { it.isAlive && it.nodeId != localNodeId }
        scope.launch(Dispatchers.IO) {
            for (slave in remoteSlaves) {
                try {
                    val assignedBlock = slave.assignedBlocks.firstOrNull()
                    Log.i(TAG, "Broadcasting preload command to slave ${slave.nodeId} (${slave.ipAddress}:${slave.port}) for block: ${assignedBlock?.startLayer}..${assignedBlock?.endLayer}")
                    p2pClient.sendPreloadCommand(slave.ipAddress, slave.port, model, sessionToken, assignedBlock)
                } catch (e: Exception) {
                    Log.w(TAG, "Failed sending preload command to slave ${slave.nodeId}", e)
                }
            }
        }
    }

    private val lastHealthScores = mutableMapOf<String, Double>()

    /**
     * Periodic node health watchdog.
     * 1. Detects dead slaves and triggers sequential layer reassignment.
     * 2. Checks CPU/RAM/Battery sensor pressure and dynamically rebalances layers.
     * 3. Checks 10-minute inactivity timer and auto-unloads LLM weights from RAM.
     */
    private fun startClusterHealthMonitor() {
        healthCheckJob?.cancel()
        healthCheckJob = scope.launch(Dispatchers.Default) {
            while (true) {
                delay(3000)
                checkNodeTimeouts()
                checkSensorPressureAndRebalance()
                localExecutor.runner.checkInactivityUnload()
            }
        }
    }

    @Synchronized
    private fun checkSensorPressureAndRebalance() {
        if (_clusterState.value.isRunningInference) return // Avoid rebalancing mid-generation
        if (allotmentMode == ClusterState.ALLOTMENT_MODE_MANUAL) return // Respect user's manual allotment

        val activeNodes = registeredNodes.values.filter { it.isAlive }
        if (activeNodes.size < 2) return

        var needsRebalance = false
        for (node in activeNodes) {
            val currentHealth = node.metrics.computeHealthFactor()
            val prevHealth = lastHealthScores[node.nodeId] ?: currentHealth
            if (kotlin.math.abs(currentHealth - prevHealth) > 0.12) {
                Log.i(TAG, "Dynamic load shift detected on ${node.deviceName} (Health changed from $prevHealth to $currentHealth). Rebalancing layers...")
                needsRebalance = true
                break
            }
        }

        if (needsRebalance) {
            for (node in activeNodes) {
                lastHealthScores[node.nodeId] = node.metrics.computeHealthFactor()
            }
            rebalanceLayers()
        }
    }

    @Synchronized
    private fun checkNodeTimeouts() {
        val now = System.currentTimeMillis()
        var nodeDied = false

        for ((id, node) in registeredNodes) {
            if (id == localNodeId) continue // Don't time out local master
            if (node.isAlive && (now - node.lastHeartbeat > HEARTBEAT_TIMEOUT_MS)) {
                Log.w(TAG, "Slave node ${node.nodeId} (${node.deviceName}) missed heartbeats! Marking DEAD.")
                node.isAlive = false
                node.assignedBlocks = emptyList()
                nodeDied = true
            }
        }

        if (nodeDied) {
            Log.i(TAG, "Cluster membership changed due to slave timeout. Reassigning layers sequentially...")
            rebalanceLayers()
        }
    }

    fun stopInference() {
        inferenceJob?.cancel()
        inferenceJob = null
        _clusterState.value = _clusterState.value.copy(isRunningInference = false)
    }

    /**
     * Coordinate distributed inference pipeline for a prompt with the active model.
     * Reports exact error codes upon failure without any synthetic fallback responses.
     */
    fun startDistributedInference(prompt: String, maxTokens: Int = 120) {
        val activeNodes = registeredNodes.values.filter { it.isAlive }.sortedBy { it.joinOrder }
        if (activeNodes.isEmpty()) return

        stopInference()

        val requestId = UUID.randomUUID().toString()
        _clusterState.value = _clusterState.value.copy(
            isRunningInference = true,
            activePrompt = prompt,
            generatedOutput = "",
            tokensGenerated = 0
        )

        inferenceJob = scope.launch(Dispatchers.IO) {
            val startTimeMs = System.currentTimeMillis()
            var firstTokenMs = 0L

            try {
                var currentOutput = ""
                var lastTokenId: Int? = null
                var promptTokensCount = 0

                for (tokenIdx in 0 until maxTokens) {
                    val currentPos = if (tokenIdx == 0) 0 else (promptTokensCount + tokenIdx - 1)
                    val initialActivation = PipelineActivation(
                        sessionToken = sessionToken,
                        requestId = requestId,
                        promptId = prompt,
                        tokenIndex = tokenIdx,
                        blockId = activeModel.id,
                        senderNodeId = localNodeId,
                        receiverNodeId = activeNodes.first().nodeId,
                        generatedTokenId = lastTokenId,
                        seqPos = currentPos,
                        promptTokensCount = promptTokensCount
                    )

                    // Execute through the sequential pipeline of nodes
                    val hopLogs = StringBuilder()
                    val result = executePipelinePass(activeNodes, initialActivation, hopLogs)

                    if (tokenIdx == 0) {
                        firstTokenMs = System.currentTimeMillis() - startTimeMs
                        if (result != null && result.promptTokensCount > 0) {
                            promptTokensCount = result.promptTokensCount
                        }
                    }

                    val totalElapsedSec = (System.currentTimeMillis() - startTimeMs).toDouble() / 1000.0
                    val currentTokPerSec = if (totalElapsedSec > 0) {
                        Math.round(((tokenIdx + 1).toDouble() / totalElapsedSec) * 10.0) / 10.0
                    } else 0.0

                    if (result != null) {
                        var textPiece = result.generatedTextPiece ?: ""
                        val sampledTok = result.generatedTokenId

                        // Dual decoding safeguard: If piece is empty but token ID was sampled, decode via Master's native vocabulary
                        if (textPiece.isEmpty() && sampledTok != null && sampledTok >= 0) {
                            try {
                                val decoded = localExecutor.runner.nativeBridge.decodeToken(sampledTok)
                                if (decoded.isNotEmpty()) {
                                    textPiece = decoded
                                }
                            } catch (_: Exception) {}
                        }

                        currentOutput += textPiece
                        lastTokenId = result.generatedTokenId

                        // High-frequency token callback decoupled from heavyweight clusterState emissions
                        onTokenGenerated?.invoke(textPiece, tokenIdx + 1, result.isFinished)
                        hopLogs.append("• Token ${tokenIdx + 1}: ID=$sampledTok ('${textPiece.replace("\n", "\\n")}')\n")

                        // Throttled pipeline hop trace and cluster state (update every 5 tokens or upon completion)
                        val shouldEmitTelemetry = (tokenIdx == 0 || (tokenIdx + 1) % 5 == 0 || result.isFinished)
                        if (shouldEmitTelemetry) {
                            _clusterState.value = _clusterState.value.copy(
                                generatedOutput = currentOutput,
                                tokensGenerated = tokenIdx + 1
                            )
                            onPipelineTrace?.invoke(hopLogs.toString(), currentTokPerSec, firstTokenMs)
                        }

                        if (result.isFinished) break
                    } else {
                        val missingHeadMsg = if (activeNodes.none { it.assignedBlocks.any { b -> b.isOutputBlock } }) {
                            "Pipeline error: No node in cluster hosts the LM Output Head (Layer ${activeModel.totalLayers - 1}) for ${activeModel.name}. Please ensure layers are allotted across active nodes in the Clusters tab."
                        } else {
                            "Pipeline pass at token $tokenIdx completed without generating a token piece from the output head."
                        }
                        throw org.meshllm.node.model.LlmInferenceException(
                            org.meshllm.node.model.LlmErrorCode.ERR_WEIGHTS_CORRUPTED,
                            missingHeadMsg
                        )
                    }
                }
            } catch (e: org.meshllm.node.model.LlmInferenceException) {
                Log.e(TAG, "Inference error: ${e.message}", e)
                val errOutput = "\n[ERROR ${e.errorCode.code}: ${e.errorCode.title}]\n${e.detail}"
                _clusterState.value = _clusterState.value.copy(
                    generatedOutput = errOutput,
                    isRunningInference = false
                )
                onTokenGenerated?.invoke(errOutput, 0, true)
            } catch (e: Exception) {
                Log.e(TAG, "Pipeline execution error: ${e.message}", e)
                val errOutput = "\n[ERROR E500 System Exception]\n${e.message ?: "Pipeline failure"}"
                _clusterState.value = _clusterState.value.copy(
                    generatedOutput = errOutput,
                    isRunningInference = false
                )
                onTokenGenerated?.invoke(errOutput, 0, true)
            } finally {
                _clusterState.value = _clusterState.value.copy(isRunningInference = false)
            }
        }
    }

    private suspend fun executePipelinePass(
        nodes: List<NodeInfo>,
        initialActivation: PipelineActivation,
        hopLogs: StringBuilder
    ): PipelineActivation? {
        var currentActivation = initialActivation

        for (i in nodes.indices) {
            val node = nodes[i]
            val block = node.assignedBlocks.firstOrNull() ?: continue
            if (block.layerCount <= 0) continue
            val hopStart = System.currentTimeMillis()

            if (node.nodeId == localNodeId) {
                // Execute locally on Master's worker instance using persistent localExecutor
                currentActivation = localExecutor.executeForward(block, currentActivation)
                val duration = System.currentTimeMillis() - hopStart
                hopLogs.append("• [Node ${node.joinOrder} (Local)] L${block.startLayer}..${block.endLayer} ($duration ms)\n")
            } else {
                // Send over LAN to remote slave node
                val targetActivation = currentActivation.copy(
                    receiverNodeId = node.nodeId,
                    senderNodeId = localNodeId
                )
                val forwardResult = p2pClient.forwardActivation(node.ipAddress, node.port, targetActivation)
                val duration = System.currentTimeMillis() - hopStart
                if (forwardResult.isSuccess && forwardResult.getOrNull() != null) {
                    currentActivation = forwardResult.getOrNull()!!
                    hopLogs.append("• [Node ${node.joinOrder} (${node.ipAddress})] L${block.startLayer}..${block.endLayer} ($duration ms)\n")
                } else {
                    val errorMsg = forwardResult.exceptionOrNull()?.message ?: "Slave node unreachable"
                    Log.w(TAG, "Failed forwarding to ${node.nodeId}: $errorMsg")
                    hopLogs.append("• [Node ${node.joinOrder} (${node.ipAddress})] FAILED ($errorMsg)\n")
                    throw org.meshllm.node.model.LlmInferenceException(
                        org.meshllm.node.model.LlmErrorCode.ERR_NODE_DISCONNECTED,
                        "Slave device '${node.deviceName}' (Node ${node.joinOrder}) failed during layer ${block.startLayer}..${block.endLayer}: $errorMsg"
                    )
                }
            }
        }

        return currentActivation
    }

    fun stop() {
        stopInference()
        healthCheckJob?.cancel()
        healthCheckJob = null
    }
}
