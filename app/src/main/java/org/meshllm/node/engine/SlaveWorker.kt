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
import org.meshllm.node.model.ClusterState
import org.meshllm.node.model.LLMBlock
import org.meshllm.node.model.ModelCatalog
import org.meshllm.node.model.ModelSpec
import org.meshllm.node.model.NodeInfo
import org.meshllm.node.model.NodeRole
import org.meshllm.node.model.PipelineActivation
import org.meshllm.node.p2p.P2PClient
import org.meshllm.node.telemetry.PerformanceGovernor
import org.meshllm.node.telemetry.SensorMonitor

class SlaveWorker(
    var masterHost: String,
    var masterPort: Int,
    val sessionToken: String,
    val localNode: NodeInfo,
    private val p2pClient: P2PClient,
    private val sensorMonitor: SensorMonitor,
    private val performanceGovernor: PerformanceGovernor,
    private val scope: CoroutineScope,
    var currentModelSpec: ModelSpec = ModelCatalog.getDefaultModel()
) {
    companion object {
        private const val TAG = "SlaveWorker"
        const val MASTER_TIMEOUT_MS = 6500L
    }

    private var heartbeatJob: Job? = null
    private var lastMasterSeenTime = System.currentTimeMillis()
    private var latestClusterState: ClusterState? = null

    private val _assignedBlocks = MutableStateFlow<List<LLMBlock>>(localNode.assignedBlocks)
    val assignedBlocks: StateFlow<List<LLMBlock>> = _assignedBlocks.asStateFlow()

    fun setAssignedBlocks(blocks: List<LLMBlock>) {
        _assignedBlocks.value = blocks
    }

    val executor: LLMBlockExecutor by lazy {
        val appContext = try { org.meshllm.node.MeshLLMApp.instance } catch (_: Throwable) { null }
        LLMBlockExecutor(sensorMonitor, performanceGovernor, appContext)
    }

    var onPromotedToMaster: ((newClusterState: ClusterState?) -> Unit)? = null
    var onClusterStateUpdated: ((ClusterState) -> Unit)? = null
    var onModelChanged: ((ModelSpec) -> Unit)? = null
    var onAssignedBlockChanged: ((LLMBlock) -> Unit)? = null

    fun start() {
        lastMasterSeenTime = System.currentTimeMillis()
        startHeartbeatLoop()
    }

    private fun startHeartbeatLoop() {
        heartbeatJob?.cancel()
        heartbeatJob = scope.launch(Dispatchers.IO) {
            while (true) {
                delay(2000)
                sendHeartbeat()
                checkMasterLiveness()
            }
        }
    }

    private suspend fun sendHeartbeat() {
        val currentMetrics = sensorMonitor.updateMetrics()
        val result = p2pClient.sendHeartbeat(
            host = masterHost,
            port = masterPort,
            nodeId = localNode.nodeId,
            sessionToken = sessionToken,
            metrics = currentMetrics
        )

        if (result.isSuccess) {
            lastMasterSeenTime = System.currentTimeMillis()
            val resp = result.getOrNull()
            if (resp?.acknowledged == true && resp.clusterState != null) {
                val cluster = resp.clusterState
                latestClusterState = cluster

                if (cluster.modelSpec.id != currentModelSpec.id) {
                    currentModelSpec = cluster.modelSpec
                    onModelChanged?.invoke(cluster.modelSpec)
                }

                onClusterStateUpdated?.invoke(cluster)

                // Update my assigned blocks from cluster state
                val me = cluster.nodes.find { it.nodeId == localNode.nodeId }
                if (me != null) {
                    val oldBlock = _assignedBlocks.value.firstOrNull()
                    val newBlock = me.assignedBlocks.firstOrNull()
                    _assignedBlocks.value = me.assignedBlocks
                    if (newBlock != null && (oldBlock == null || oldBlock.modelId != newBlock.modelId || oldBlock.startLayer != newBlock.startLayer || oldBlock.endLayer != newBlock.endLayer)) {
                        Log.i(TAG, "Assigned block changed on slave: L${newBlock.startLayer}..${newBlock.endLayer}")
                        onAssignedBlockChanged?.invoke(newBlock)
                    } else if (newBlock == null && oldBlock != null) {
                        Log.i(TAG, "Slave reassigned to 0 layers (bypassed). Clearing model from RAM.")
                        executor.runner.unloadModelWeights()
                        sensorMonitor.updateModelRam(0.0)
                        triggerHeartbeat()
                    }
                }
            }
        } else {
            Log.w(TAG, "Heartbeat to master ($masterHost:$masterPort) failed")
        }
    }

    fun updateModelSpec(model: ModelSpec) {
        currentModelSpec = model
        onModelChanged?.invoke(model)
    }

    /**
     * Requirement 6: Master failure detection & Leader Election.
     * If the master node dies or gets disconnected, the first device connected
     * to the network after the master node (lowest joinOrder >= 1) becomes the new master.
     */
    private suspend fun checkMasterLiveness() {
        val timeSinceMasterSeen = System.currentTimeMillis() - lastMasterSeenTime
        if (timeSinceMasterSeen > MASTER_TIMEOUT_MS) {
            Log.w(TAG, "Master node timed out! Initiating leader election...")

            val survivingNodes = latestClusterState?.nodes
                ?.filter { it.role != NodeRole.MASTER && it.isAlive }
                ?.sortedBy { it.joinOrder } ?: emptyList()

            val nextInLine = survivingNodes.firstOrNull()

            if (nextInLine != null && nextInLine.nodeId == localNode.nodeId) {
                // I am the first connected device after the master!
                Log.i(TAG, "I am the next device in line (joinOrder=${localNode.joinOrder}). Promoting myself to MASTER!")
                stop()
                onPromotedToMaster?.invoke(latestClusterState)
            } else if (nextInLine != null) {
                // Another slave is first in line; switch my master pointer to that node
                Log.i(TAG, "Device ${nextInLine.nodeId} (joinOrder=${nextInLine.joinOrder}) is taking over as new master.")
                masterHost = nextInLine.ipAddress
                masterPort = nextInLine.port
                lastMasterSeenTime = System.currentTimeMillis()
            }
        }
    }

    /**
     * Process incoming pipeline activation sent by predecessor node and return result.
     */
    fun handlePipelineActivation(activation: PipelineActivation): PipelineActivation? {
        val blocks = _assignedBlocks.value
        if (blocks.isEmpty()) {
            Log.w(TAG, "Received activation but no blocks assigned to this slave.")
            return null
        }

        // Execute forward pass for my block strictly on the background inference thread pool
        val myBlock = blocks.first()
        return kotlinx.coroutines.runBlocking(LLMBlockExecutor.inferenceDispatcher) {
            val res = executor.executeForward(myBlock, activation)
            triggerHeartbeat()
            res
        }
    }

    fun triggerHeartbeat() {
        scope.launch(Dispatchers.IO) {
            try {
                sendHeartbeat()
            } catch (_: Exception) {}
        }
    }

    fun stop() {
        heartbeatJob?.cancel()
        heartbeatJob = null
    }
}
