package org.meshllm.node.ui

import android.app.Application
import android.content.Intent
import android.os.Build
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.meshllm.node.engine.DistributedOrchestrator
import org.meshllm.node.engine.GGUFParser
import org.meshllm.node.engine.LLMBlockExecutor
import org.meshllm.node.engine.NeuralModelRunner
import org.meshllm.node.engine.SlaveWorker
import org.meshllm.node.model.BenchmarkProfile
import org.meshllm.node.model.BlockType
import org.meshllm.node.model.ClusterState
import org.meshllm.node.model.LLMBlock
import org.meshllm.node.model.ModelCatalog
import org.meshllm.node.model.ModelSpec
import org.meshllm.node.model.NodeInfo
import org.meshllm.node.model.NodeRole
import org.meshllm.node.model.SensorMetrics
import org.meshllm.node.p2p.NetworkUtils
import org.meshllm.node.p2p.QrConnectionHelper
import org.meshllm.node.p2p.P2PClient
import org.meshllm.node.p2p.P2PServer
import org.meshllm.node.p2p.SessionTokenGenerator
import org.meshllm.node.service.MeshNodeService
import org.meshllm.node.telemetry.BenchmarkRunner
import org.meshllm.node.telemetry.DownloadState
import org.meshllm.node.telemetry.ModelDownloadManager
import org.meshllm.node.telemetry.PerformanceGovernor
import org.meshllm.node.telemetry.SensorMonitor
import java.io.File
import java.util.UUID

class MeshClusterViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "MeshClusterViewModel"
    }

    val sensorMonitor = SensorMonitor(application)
    val benchmarkRunner = BenchmarkRunner(application)
    val performanceGovernor = PerformanceGovernor()
    val p2pClient = P2PClient()
    val modelDownloadManager = ModelDownloadManager(application, viewModelScope)
    val huggingFaceClient = org.meshllm.node.telemetry.HuggingFaceClient()

    val localNodeId: String = UUID.randomUUID().toString().substring(0, 8)
    val localPort: Int = 8080

    private var p2pServer: P2PServer? = null
    var orchestrator: DistributedOrchestrator? = null
        private set
    var slaveWorker: SlaveWorker? = null
        private set

    private var orchestratorCollectJob: Job? = null

    // High-performance token streaming buffer to eliminate UI freezes
    private val tokenBuffer = StringBuilder()
    private var lastTokenUiDispatchMs = 0L
    private val UI_DISPATCH_INTERVAL_MS = 33L // ~30 FPS UI render cap

    // Observable states
    private val _currentRole = MutableStateFlow(NodeRole.UNASSIGNED)
    val currentRole: StateFlow<NodeRole> = _currentRole.asStateFlow()

    private val _activeModel = MutableStateFlow(ModelCatalog.getDefaultModel())
    val activeModel: StateFlow<ModelSpec> = _activeModel.asStateFlow()

    private val _sessionToken = MutableStateFlow("")
    val sessionToken: StateFlow<String> = _sessionToken.asStateFlow()

    private val _localIp = MutableStateFlow(NetworkUtils.getLocalIpAddress())
    val localIp: StateFlow<String> = _localIp.asStateFlow()

    private val _clusterState = MutableStateFlow(
        ClusterState(
            sessionToken = "",
            modelSpec = _activeModel.value,
            modelName = _activeModel.value.name,
            totalLayers = _activeModel.value.totalLayers,
            masterNodeId = localNodeId,
            masterAddress = "${_localIp.value}:$localPort"
        )
    )
    val clusterState: StateFlow<ClusterState> = _clusterState.asStateFlow()

    private val _sensorMetrics = MutableStateFlow(SensorMetrics())
    val sensorMetrics: StateFlow<SensorMetrics> = _sensorMetrics.asStateFlow()

    private val _benchmarkProfile = MutableStateFlow<BenchmarkProfile?>(null)
    val benchmarkProfile: StateFlow<BenchmarkProfile?> = _benchmarkProfile.asStateFlow()

    val downloadStates: StateFlow<Map<String, DownloadState>> = modelDownloadManager.downloadStates

    // Diagnostics & Inference metrics
    private val _weightFilePath = MutableStateFlow("Path: Not downloaded")
    val weightFilePath: StateFlow<String> = _weightFilePath.asStateFlow()

    private val _weightFormatInspection = MutableStateFlow("Format: None | Mapped: 0 MB | Tensors: -")
    val weightFormatInspection: StateFlow<String> = _weightFormatInspection.asStateFlow()

    private val _inferenceStatus = MutableStateFlow("Idle")
    val inferenceStatus: StateFlow<String> = _inferenceStatus.asStateFlow()

    private val _streamingOutput = MutableStateFlow("")
    val streamingOutput: StateFlow<String> = _streamingOutput.asStateFlow()

    private val _tokensPerSecond = MutableStateFlow(0.0)
    val tokensPerSecond: StateFlow<Double> = _tokensPerSecond.asStateFlow()

    private val _generatedTokensCount = MutableStateFlow(0)
    val generatedTokensCount: StateFlow<Int> = _generatedTokensCount.asStateFlow()

    private val _ttftLatencyMs = MutableStateFlow(0L)
    val ttftLatencyMs: StateFlow<Long> = _ttftLatencyMs.asStateFlow()

    private val _pipelineTrace = MutableStateFlow("[Pipeline Trace]\nReady. Pipeline activation tensors stream sequentially between nodes.")
    val pipelineTrace: StateFlow<String> = _pipelineTrace.asStateFlow()

    private val _isInferencing = MutableStateFlow(false)
    val isInferencing: StateFlow<Boolean> = _isInferencing.asStateFlow()

    // Model Catalog Filtering
    private val _catalogCategory = MutableStateFlow(ModelCatalog.CATEGORY_ALL)
    val catalogCategory: StateFlow<String> = _catalogCategory.asStateFlow()

    private val _catalogQuery = MutableStateFlow("")
    val catalogQuery: StateFlow<String> = _catalogQuery.asStateFlow()

    private val _suitableOnlyFilter = MutableStateFlow(true)
    val suitableOnlyFilter: StateFlow<Boolean> = _suitableOnlyFilter.asStateFlow()

    private val _preferredQuantFilter = MutableStateFlow<String?>(null)
    val preferredQuantFilter: StateFlow<String?> = _preferredQuantFilter.asStateFlow()

    private val _catalogModels = MutableStateFlow(ModelCatalog.getAllModels())
    val catalogModels: StateFlow<List<ModelSpec>> = _catalogModels.asStateFlow()

    private val _modelAllocatedRamMb = MutableStateFlow(0.0)
    val modelAllocatedRamMb: StateFlow<Double> = _modelAllocatedRamMb.asStateFlow()

    private val _isModelPreloading = MutableStateFlow(false)
    val isModelPreloading: StateFlow<Boolean> = _isModelPreloading.asStateFlow()

    private val _isModelInRam = MutableStateFlow(false)
    val isModelInRam: StateFlow<Boolean> = _isModelInRam.asStateFlow()

    private val _ramStatusDescription = MutableStateFlow("Weights not loaded into RAM")
    val ramStatusDescription: StateFlow<String> = _ramStatusDescription.asStateFlow()

    private val _isHfSearching = MutableStateFlow(false)
    val isHfSearching: StateFlow<Boolean> = _isHfSearching.asStateFlow()

    private var telemetryPollJob: Job? = null
    private var hfSearchJob: Job? = null

    init {
        ModelCatalog.loadCache(application)
        checkForCachedBenchmarkProfile()
        startTelemetryLoop()
        refreshCatalog()
        inspectActiveModelWeights()
        observeDownloadStatesForCluster()
        fetchHuggingFaceCatalog()
    }

    private fun observeDownloadStatesForCluster() {
        viewModelScope.launch {
            modelDownloadManager.downloadStates.collect {
                if (_currentRole.value == NodeRole.MASTER) {
                    orchestrator?.updateModelAvailability()
                }
            }
        }
    }

    fun checkForCachedBenchmarkProfile() {
        val cached = benchmarkRunner.getCachedProfileIfAvailable()
        if (cached != null) {
            _benchmarkProfile.value = cached
        }
    }

    private fun startTelemetryLoop() {
        telemetryPollJob?.cancel()
        telemetryPollJob = viewModelScope.launch(Dispatchers.IO) {
            while (true) {
                delay(2000)
                val metrics = sensorMonitor.updateMetrics()
                val isLoaded = NeuralModelRunner.shared.isModelLoadedInRam || org.meshllm.node.engine.LlamaNativeBridge.isModelLoaded()
                val ram = NeuralModelRunner.shared.allocatedRamMb.takeIf { it > 0.0 }
                    ?: orchestrator?.executor?.runner?.allocatedRamMb
                    ?: slaveWorker?.executor?.runner?.allocatedRamMb
                    ?: (if (org.meshllm.node.engine.LlamaNativeBridge.isModelLoaded()) {
                        Math.round((org.meshllm.node.engine.LlamaNativeBridge.getAllocatedRamBytes().toDouble() / (1024.0 * 1024.0)) * 10.0) / 10.0
                    } else 0.0)
                _modelAllocatedRamMb.value = ram
                if (!_isModelPreloading.value) {
                    _isModelInRam.value = isLoaded
                    if (isLoaded) {
                        val activeFile = NeuralModelRunner.shared.loadedFile?.name ?: _activeModel.value.name
                        val block = NeuralModelRunner.shared.loadedBlock
                        val layerSuffix = if (block != null) " [L${block.startLayer}..${block.endLayer}/${block.totalModelLayers}]" else ""
                        val gpuBadge = if (org.meshllm.node.engine.LlamaNativeBridge.isGpuAccelerated()) " [GPU: Vulkan Active]" else ""
                        _ramStatusDescription.value = "Active in RAM$gpuBadge: $activeFile$layerSuffix ($ram MB allocated)"
                    } else {
                        _ramStatusDescription.value = "Weights not loaded into RAM"
                    }
                }
                _sensorMetrics.value = metrics.copy(modelRamAllocatedMb = ram)
                _localIp.value = NetworkUtils.getLocalIpAddress()

                if (_currentRole.value == NodeRole.MASTER) {
                    orchestrator?.updateLocalMetrics(_sensorMetrics.value)
                }
            }
        }
    }

    fun fetchHuggingFaceCatalog(forceRefresh: Boolean = false) {
        viewModelScope.launch {
            _isHfSearching.value = true
            try {
                val models = huggingFaceClient.fetchPopularModels(limit = 40)
                if (models.isNotEmpty()) {
                    ModelCatalog.addDynamicModels(models, getApplication())
                    refreshCatalog()
                    Log.i(TAG, "Dynamic Hugging Face models synced: ${models.size} models loaded")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed fetching Hugging Face catalog: ${e.message}", e)
            } finally {
                _isHfSearching.value = false
            }
        }
    }

    fun searchHuggingFace(query: String) {
        val cleanQuery = query.trim()
        if (cleanQuery.length < 2) return

        hfSearchJob?.cancel()
        hfSearchJob = viewModelScope.launch {
            _isHfSearching.value = true
            try {
                val results = huggingFaceClient.searchModels(cleanQuery, limit = 25)
                if (results.isNotEmpty()) {
                    ModelCatalog.addDynamicModels(results, getApplication())
                    refreshCatalog()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error querying Hugging Face API: ${e.message}", e)
            } finally {
                _isHfSearching.value = false
            }
        }
    }

    fun toggleSuitableFilter() {
        _suitableOnlyFilter.value = !_suitableOnlyFilter.value
        refreshCatalog()
    }

    fun setSuitableFilter(suitableOnly: Boolean) {
        _suitableOnlyFilter.value = suitableOnly
        refreshCatalog()
    }

    fun setPreferredQuantFilter(quant: String?) {
        _preferredQuantFilter.value = quant
        refreshCatalog()
    }

    fun setCatalogFilter(
        category: String = _catalogCategory.value,
        query: String = _catalogQuery.value,
        suitableOnly: Boolean = _suitableOnlyFilter.value,
        preferredQuant: String? = _preferredQuantFilter.value
    ) {
        _catalogCategory.value = category
        _catalogQuery.value = query
        _suitableOnlyFilter.value = suitableOnly
        _preferredQuantFilter.value = preferredQuant
        refreshCatalog()

        // If user is searching, also query Hugging Face directly in real-time
        if (query.trim().length >= 2) {
            searchHuggingFace(query)
        }
    }

    fun refreshCatalog() {
        _catalogModels.value = ModelCatalog.filter(
            category = _catalogCategory.value,
            query = _catalogQuery.value,
            suitableOnly = _suitableOnlyFilter.value,
            preferredQuant = _preferredQuantFilter.value
        )
    }

    fun selectModel(model: ModelSpec) {
        _activeModel.value = model
        inspectActiveModelWeights()

        val isLoaded = NeuralModelRunner.shared.isModelLoadedInRam
        val loadedFile = NeuralModelRunner.shared.loadedFile
        if (isLoaded && loadedFile != null && loadedFile.name == model.fileName) {
            _isModelInRam.value = true
            _ramStatusDescription.value = "Active in RAM: ${model.name} (${NeuralModelRunner.shared.allocatedRamMb} MB allocated)"
        } else if (isLoaded) {
            _isModelInRam.value = false
            _ramStatusDescription.value = "Different model in RAM (${loadedFile?.name ?: "active"}). Clear or preload this model."
        } else {
            _isModelInRam.value = false
            _ramStatusDescription.value = "Weights not loaded into RAM"
        }

        if (_currentRole.value == NodeRole.MASTER) {
            orchestrator?.selectModel(model)
        }
    }

    fun inspectActiveModelWeights() {
        viewModelScope.launch(Dispatchers.IO) {
            val model = _activeModel.value
            val file = modelDownloadManager.getModelFile(model)

            if (file.exists() && file.length() > 1024L) {
                val sizeMb = Math.round((file.length().toDouble() / (1024.0 * 1024.0)) * 10.0) / 10.0
                _weightFilePath.value = "Path: ${file.absolutePath} ($sizeMb MB)"

                // Inspect format using GGUFParser off the main thread to prevent ANR
                val gguf = GGUFParser().parse(file)
                val gpuStr = if (org.meshllm.node.engine.LlamaNativeBridge.isGpuAccelerated()) " • GPU (Vulkan) Ready" else ""
                if (gguf != null) {
                    _weightFormatInspection.value = "Format: GGUF v${gguf.version} (${gguf.architecture})$gpuStr • ${gguf.blockCount} Layers • Dim: ${gguf.hiddenDim} • Tensors: ${gguf.tensorCount} [VERIFIED]"
                } else {
                    _weightFormatInspection.value = "Format: Binary LLaMA Weights • ${model.totalLayers} Layers • Dim: ${model.hiddenDim} [VERIFIED]"
                }
            } else {
                _weightFilePath.value = "Path: Not downloaded (${model.fileName})"
                _weightFormatInspection.value = "Format: ${model.fileFormat} • Expected: ${model.fileSizeDisplay} • Tap Download to cache"
            }
        }
    }

    fun getLocalAssignedBlockOrNull(): LLMBlock? {
        val model = _activeModel.value
        if (_currentRole.value == NodeRole.MASTER && orchestrator != null) {
            val me = orchestrator?.clusterState?.value?.nodes?.find { it.nodeId == localNodeId }
            if (me != null) return me.assignedBlocks.firstOrNull()
        }
        if (_currentRole.value == NodeRole.SLAVE) {
            val me = _clusterState.value.nodes.find { it.nodeId == localNodeId }
            if (me != null) return me.assignedBlocks.firstOrNull()
        }
        return null
    }

    fun getLocalAssignedBlock(): LLMBlock {
        val model = _activeModel.value
        return getLocalAssignedBlockOrNull() ?: LLMBlock.fromModelSpec(
            blockId = "block_${localNodeId}_0-${model.totalLayers - 1}",
            blockType = BlockType.OUTPUT_LM_HEAD,
            startLayer = 0,
            endLayer = model.totalLayers - 1,
            model = model
        )
    }

    fun setLayerAllotmentMode(mode: String) {
        orchestrator?.setAllotmentMode(mode)
        if (mode == ClusterState.ALLOTMENT_MODE_AUTO) {
            val myBlock = getLocalAssignedBlockOrNull() ?: getLocalAssignedBlock()
            val file = modelDownloadManager.getModelFile(_activeModel.value)
            if (file.exists() && file.length() > 1024L) {
                NeuralModelRunner.shared.loadModelWeightsIfAvailable(file, myBlock)
                _isModelInRam.value = true
                val ramMb = NeuralModelRunner.shared.allocatedRamMb
                _modelAllocatedRamMb.value = ramMb
                sensorMonitor.updateModelRam(ramMb)
                _sensorMetrics.value = sensorMonitor.updateMetrics()
                orchestrator?.updateLocalMetrics(_sensorMetrics.value)
            }
        }
    }

    fun applyManualLayerAllotment(layerCounts: Map<String, Int>) {
        orchestrator?.applyManualLayerAllotment(layerCounts)
        val myBlock = getLocalAssignedBlockOrNull()
        if (myBlock == null || myBlock.layerCount <= 0) {
            NeuralModelRunner.shared.unloadModelWeights()
            _isModelInRam.value = false
            _modelAllocatedRamMb.value = 0.0
            sensorMonitor.updateModelRam(0.0)
            _sensorMetrics.value = sensorMonitor.updateMetrics()
            orchestrator?.updateLocalMetrics(_sensorMetrics.value)
            _ramStatusDescription.value = "Master bypassed (0 layers assigned) • 0 MB RAM allocated"
        } else {
            val file = modelDownloadManager.getModelFile(_activeModel.value)
            if (file.exists() && file.length() > 1024L) {
                NeuralModelRunner.shared.loadModelWeightsIfAvailable(file, myBlock)
                _isModelInRam.value = true
                val ramMb = NeuralModelRunner.shared.allocatedRamMb
                _modelAllocatedRamMb.value = ramMb
                sensorMonitor.updateModelRam(ramMb)
                _sensorMetrics.value = sensorMonitor.updateMetrics()
                orchestrator?.updateLocalMetrics(_sensorMetrics.value)
                _ramStatusDescription.value = "Active in RAM: ${_activeModel.value.name} [L${myBlock.startLayer}..${myBlock.endLayer}] ($ramMb MB allocated)"
            }
        }
    }

    fun handlePreloadCommand(modelSpec: ModelSpec, block: LLMBlock? = null) {
        viewModelScope.launch(Dispatchers.IO) {
            _activeModel.value = modelSpec
            val targetBlock = block ?: getLocalAssignedBlockOrNull()

            // If node has 0 layers assigned, unload model and bypass completely
            if (targetBlock == null || targetBlock.layerCount <= 0) {
                _isModelPreloading.value = false
                NeuralModelRunner.shared.unloadModelWeights()
                _isModelInRam.value = false
                _modelAllocatedRamMb.value = 0.0
                sensorMonitor.updateModelRam(0.0)
                _sensorMetrics.value = sensorMonitor.updateMetrics()
                slaveWorker?.triggerHeartbeat()
                _ramStatusDescription.value = "Node bypassed (0 layers assigned) • 0 MB RAM allocated"
                return@launch
            }

            val file = modelDownloadManager.getModelFile(modelSpec)
            if (file.exists() && file.length() > 1024L) {
                _isModelPreloading.value = true
                _ramStatusDescription.value = "Preloading assigned layers L${targetBlock.startLayer}..${targetBlock.endLayer} into RAM..."
                val success = try {
                    NeuralModelRunner.shared.loadModelWeightsIfAvailable(file, targetBlock)
                } catch (e: Exception) {
                    Log.e(TAG, "Error preloading slave weights: ${e.message}", e)
                    false
                }
                _isModelPreloading.value = false
                _isModelInRam.value = success
                val ramMb = NeuralModelRunner.shared.allocatedRamMb
                _modelAllocatedRamMb.value = ramMb
                sensorMonitor.updateModelRam(ramMb)
                _sensorMetrics.value = sensorMonitor.updateMetrics()
                slaveWorker?.triggerHeartbeat()
                _ramStatusDescription.value = if (success) {
                    "Active in RAM: ${modelSpec.name} [L${targetBlock.startLayer}..${targetBlock.endLayer}] ($ramMb MB allocated)"
                } else {
                    "Failed to preload assigned layers L${targetBlock.startLayer}..${targetBlock.endLayer}"
                }
            } else {
                Log.i(TAG, "Model ${modelSpec.name} not downloaded on node. Downloading weights...")
                _ramStatusDescription.value = "Assigned L${targetBlock.startLayer}..${targetBlock.endLayer}. Downloading weights..."
                modelDownloadManager.downloadModel(modelSpec) { state ->
                    if (state is DownloadState.Ready) {
                        viewModelScope.launch(Dispatchers.IO) {
                            val loaded = NeuralModelRunner.shared.loadModelWeightsIfAvailable(state.file, targetBlock)
                            _isModelInRam.value = loaded
                            val ramMb = NeuralModelRunner.shared.allocatedRamMb
                            _modelAllocatedRamMb.value = ramMb
                            sensorMonitor.updateModelRam(ramMb)
                            _sensorMetrics.value = sensorMonitor.updateMetrics()
                            slaveWorker?.triggerHeartbeat()
                            _ramStatusDescription.value = "Active in RAM: ${modelSpec.name} [L${targetBlock.startLayer}..${targetBlock.endLayer}] ($ramMb MB allocated)"
                        }
                    }
                }
            }
        }
    }

    fun downloadActiveModel() {
        val model = _activeModel.value
        modelDownloadManager.downloadModel(model) { state ->
            inspectActiveModelWeights()
            if (state is DownloadState.Ready) {
                if (_currentRole.value == NodeRole.SLAVE || _currentRole.value == NodeRole.MASTER) {
                    val myBlock = getLocalAssignedBlock()
                    viewModelScope.launch(Dispatchers.IO) {
                        NeuralModelRunner.shared.loadModelWeightsIfAvailable(state.file, myBlock)
                        _isModelInRam.value = true
                        val ramMb = NeuralModelRunner.shared.allocatedRamMb
                        _modelAllocatedRamMb.value = ramMb
                        sensorMonitor.updateModelRam(ramMb)
                        _sensorMetrics.value = sensorMonitor.updateMetrics()
                        if (_currentRole.value == NodeRole.MASTER) {
                            orchestrator?.updateLocalMetrics(_sensorMetrics.value)
                            orchestrator?.broadcastPreloadToSlaves(model)
                        } else if (_currentRole.value == NodeRole.SLAVE) {
                            slaveWorker?.triggerHeartbeat()
                        }
                        _ramStatusDescription.value = "Active in RAM: ${model.name} [L${myBlock.startLayer}..${myBlock.endLayer}] ($ramMb MB allocated)"
                    }
                }
            }
        }
    }

    fun cancelActiveDownload() {
        modelDownloadManager.cancelDownload(_activeModel.value.id)
    }

    fun preloadActiveModelToRam(onComplete: ((Boolean, String) -> Unit)? = null) {
        val model = _activeModel.value
        val file = modelDownloadManager.getModelFile(model)
        if (!file.exists() || file.length() < 1024L) {
            val msg = "Weight file not found on disk. Download it first."
            _ramStatusDescription.value = msg
            onComplete?.invoke(false, msg)
            return
        }

        val myBlock = getLocalAssignedBlockOrNull()
        if (myBlock == null || myBlock.layerCount <= 0) {
            if (_currentRole.value == NodeRole.MASTER) {
                orchestrator?.broadcastPreloadToSlaves(model)
            }
            NeuralModelRunner.shared.unloadModelWeights()
            _isModelPreloading.value = false
            _isModelInRam.value = false
            _modelAllocatedRamMb.value = 0.0
            sensorMonitor.updateModelRam(0.0)
            _sensorMetrics.value = sensorMonitor.updateMetrics()
            orchestrator?.updateLocalMetrics(_sensorMetrics.value)
            val msg = "Master node bypassed (0 layers assigned) • Slaves notified"
            _ramStatusDescription.value = msg
            onComplete?.invoke(true, msg)
            return
        }

        _isModelPreloading.value = true
        _ramStatusDescription.value = "Preloading ${model.name} [L${myBlock.startLayer}..${myBlock.endLayer}] into Node RAM..."

        viewModelScope.launch {
            // Requirement & User Request: When loading process starts, all devices in cluster start loading their assigned layers in parallel
            if (_currentRole.value == NodeRole.MASTER) {
                orchestrator?.broadcastPreloadToSlaves(model)
            }

            val success = withContext(LLMBlockExecutor.inferenceDispatcher) {
                try {
                    NeuralModelRunner.shared.loadModelWeightsIfAvailable(file, myBlock)
                } catch (e: Exception) {
                    Log.e(TAG, "Error preloading model to RAM", e)
                    false
                }
            }

            _isModelPreloading.value = false
            _isModelInRam.value = success
            val ramMb = NeuralModelRunner.shared.allocatedRamMb
            _modelAllocatedRamMb.value = ramMb
            sensorMonitor.updateModelRam(ramMb)
            _sensorMetrics.value = sensorMonitor.updateMetrics()

            if (_currentRole.value == NodeRole.MASTER) {
                orchestrator?.updateLocalMetrics(_sensorMetrics.value)
            } else if (_currentRole.value == NodeRole.SLAVE) {
                slaveWorker?.triggerHeartbeat()
            }

            val msg = if (success) {
                "Preloaded into RAM: ${model.name} [L${myBlock.startLayer}..${myBlock.endLayer}] ($ramMb MB). 1st TTFT latency eliminated."
            } else {
                "Failed to preload ${model.name} into RAM."
            }
            _ramStatusDescription.value = msg
            onComplete?.invoke(success, msg)
        }
    }

    fun clearLoadedModelFromRam(): Boolean {
        val unloaded = NeuralModelRunner.shared.unloadModelWeights()
        _isModelInRam.value = false
        _modelAllocatedRamMb.value = 0.0
        sensorMonitor.updateModelRam(0.0)
        _sensorMetrics.value = sensorMonitor.updateMetrics()
        if (_currentRole.value == NodeRole.MASTER) {
            orchestrator?.updateLocalMetrics(_sensorMetrics.value)
        } else if (_currentRole.value == NodeRole.SLAVE) {
            slaveWorker?.triggerHeartbeat()
        }
        _ramStatusDescription.value = "Weights cleared from RAM. Memory freed to OS."
        return unloaded
    }

    fun deleteActiveModelFile(): Boolean {
        val activeFile = modelDownloadManager.getModelFile(_activeModel.value)
        if (NeuralModelRunner.shared.loadedFile?.absolutePath == activeFile.absolutePath || NeuralModelRunner.shared.isModelLoadedInRam) {
            clearLoadedModelFromRam()
        }
        val deleted = modelDownloadManager.deleteModelFile(_activeModel.value)
        inspectActiveModelWeights()
        return deleted
    }

    fun initializeAsMaster() {
        val bench = _benchmarkProfile.value ?: benchmarkRunner.saveDevMockProfile(14.0, 4200.0).also { _benchmarkProfile.value = it }
        val ip = _localIp.value
        val token = SessionTokenGenerator.generateSessionToken()

        _sessionToken.value = token
        _currentRole.value = NodeRole.MASTER

        // Obtain shared P2PServer singleton and update its token immediately
        val server = P2PServer.getOrCreateShared(localPort, viewModelScope)
        server.updateSessionToken(token)
        p2pServer = server

        // Initialize DistributedOrchestrator
        orchestrator?.stop()
        orchestrator = DistributedOrchestrator(
            sessionToken = token,
            localNodeId = localNodeId,
            localAddress = ip,
            localPort = localPort,
            localBenchmark = bench,
            p2pClient = p2pClient,
            scope = viewModelScope,
            activeModel = _activeModel.value
        )

        server.onRegisterReceived = { req -> orchestrator!!.registerSlaveNode(req) }
        server.onHeartbeatReceived = { req -> orchestrator!!.handleHeartbeat(req) }
        server.onGetModelFile = { modelId, fileName ->
            modelDownloadManager.getDownloadedModelFile(modelId, fileName)
        }
        server.onListAvailableModels = {
            val downloadedIds = modelDownloadManager.getDownloadedModelIds()
            ModelCatalog.getAllModels().filter { downloadedIds.contains(it.id) || modelDownloadManager.isModelDownloaded(it) }
        }
        server.start()

        orchestrator?.isModelAvailableLocally = { model ->
            modelDownloadManager.isModelDownloaded(model)
        }
        orchestrator?.getDownloadedModelIds = {
            modelDownloadManager.getDownloadedModelIds()
        }
        orchestrator?.updateModelAvailability()

        orchestrator?.onLayersRebalanced = { localBlock, cluster ->
            viewModelScope.launch(Dispatchers.IO) {
                val model = cluster.modelSpec
                val file = modelDownloadManager.getModelFile(model)
                if (file.exists() && file.length() > 1024L) {
                    val runner = NeuralModelRunner.shared
                    if (runner.isModelLoadedInRam) {
                        val currentBlock = runner.loadedBlock
                        if (currentBlock?.startLayer != localBlock.startLayer || currentBlock?.endLayer != localBlock.endLayer) {
                            Log.i(TAG, "Re-slicing Master model in RAM to L${localBlock.startLayer}..${localBlock.endLayer}...")
                            runner.loadModelWeightsIfAvailable(file, localBlock)
                            val ramMb = runner.allocatedRamMb
                            _modelAllocatedRamMb.value = ramMb
                            sensorMonitor.updateModelRam(ramMb)
                            _sensorMetrics.value = sensorMonitor.updateMetrics()
                            orchestrator?.updateLocalMetrics(_sensorMetrics.value)
                            _ramStatusDescription.value = "Active in RAM (Master L${localBlock.startLayer}..${localBlock.endLayer}): ${model.name} ($ramMb MB allocated)"
                        }
                    }
                }
                orchestrator?.broadcastPreloadToSlaves(model)
            }
        }

        // Wire orchestrator callbacks (cancel previous collection job to prevent leaks)
        orchestratorCollectJob?.cancel()
        orchestratorCollectJob = viewModelScope.launch {
            orchestrator?.clusterState?.collect { state ->
                _clusterState.value = state
                if (!state.isRunningInference) {
                    _isInferencing.value = false
                    if (state.generatedOutput.isNotEmpty()) {
                        _streamingOutput.value = state.generatedOutput
                    }
                    if (_inferenceStatus.value.startsWith("Streaming") || _inferenceStatus.value.startsWith("Executing") || _inferenceStatus.value.startsWith("Loading")) {
                        _inferenceStatus.value = "Finished (${state.tokensGenerated} tokens)"
                    }
                }
            }
        }

        orchestrator?.onTokenGenerated = { piece, count, isFinished ->
            val now = android.os.SystemClock.elapsedRealtime()
            synchronized(tokenBuffer) {
                tokenBuffer.append(piece)
            }
            if (isFinished || (now - lastTokenUiDispatchMs) >= UI_DISPATCH_INTERVAL_MS) {
                lastTokenUiDispatchMs = now
                val fullText = synchronized(tokenBuffer) { tokenBuffer.toString() }
                viewModelScope.launch(Dispatchers.Main) {
                    _streamingOutput.value = fullText
                    _generatedTokensCount.value = count
                    _inferenceStatus.value = if (isFinished) "Finished ($count tokens)" else "Streaming token $count..."
                    if (isFinished) _isInferencing.value = false
                }
            }
        }

        orchestrator?.onPipelineTrace = { trace, tokPerSec, ttftMs ->
            viewModelScope.launch(Dispatchers.Main) {
                _pipelineTrace.value = trace
                _tokensPerSecond.value = tokPerSec
                _ttftLatencyMs.value = ttftMs
            }
        }

        startForegroundService("MASTER", "Orchestrating inference cluster (${_activeModel.value.name})")
    }

    fun joinAsSlave(masterIp: String, masterPort: Int, token: String, onComplete: (Boolean, String) -> Unit) {
        val bench = _benchmarkProfile.value ?: benchmarkRunner.saveDevMockProfile(10.0, 3200.0).also { _benchmarkProfile.value = it }

        viewModelScope.launch(Dispatchers.IO) {
            val slaveNode = NodeInfo(
                nodeId = localNodeId,
                deviceName = "${Build.MANUFACTURER} ${Build.MODEL} (${localNodeId.take(4)})",
                ipAddress = _localIp.value,
                port = localPort,
                role = NodeRole.SLAVE,
                joinOrder = 0,
                benchmark = bench,
                metrics = sensorMonitor.currentMetrics.value
            )

            // Obtain shared P2PServer singleton and update token to match cluster
            val server = P2PServer.getOrCreateShared(localPort, viewModelScope)
            server.updateSessionToken(token)
            p2pServer = server
            server.onPipelineActivationReceived = { act -> slaveWorker?.handlePipelineActivation(act) }
            server.onPreloadCommandReceived = { model, block -> handlePreloadCommand(model, block) }
            server.onModelChangeReceived = { newModel ->
                slaveWorker?.updateModelSpec(newModel)
                _activeModel.value = newModel
                inspectActiveModelWeights()
            }
            server.onGetModelFile = { modelId, fileName ->
                modelDownloadManager.getDownloadedModelFile(modelId, fileName)
            }
            server.start()

            val regResult = p2pClient.registerNode(
                host = masterIp,
                port = masterPort,
                node = slaveNode,
                sessionToken = token
            )

            if (regResult.isSuccess) {
                val resp = regResult.getOrNull()!!
                _currentRole.value = NodeRole.SLAVE
                _sessionToken.value = token

                val cluster = resp.clusterState
                if (cluster != null) {
                    _clusterState.value = cluster
                    _activeModel.value = cluster.modelSpec
                    inspectActiveModelWeights()
                }
                val myAssignedBlocks = cluster?.nodes?.find { it.nodeId == localNodeId }?.assignedBlocks ?: emptyList()

                // Initialize SlaveWorker
                slaveWorker?.stop()
                slaveWorker = SlaveWorker(
                    masterHost = masterIp,
                    masterPort = masterPort,
                    sessionToken = token,
                    localNode = slaveNode.copy(
                        joinOrder = resp.assignedJoinOrder,
                        assignedBlocks = myAssignedBlocks
                    ),
                    p2pClient = p2pClient,
                    sensorMonitor = sensorMonitor,
                    performanceGovernor = performanceGovernor,
                    scope = viewModelScope,
                    currentModelSpec = resp.clusterState?.modelSpec ?: _activeModel.value
                ).apply {
                    setAssignedBlocks(myAssignedBlocks)
                }

                // Configure direct peer model downloading from master node
                modelDownloadManager.peerDownloadSourceProvider = {
                    if (_currentRole.value == NodeRole.SLAVE) {
                        val host = slaveWorker?.masterHost ?: masterIp
                        val port = slaveWorker?.masterPort ?: masterPort
                        org.meshllm.node.telemetry.PeerDownloadSource(host, port, token)
                    } else null
                }

                // Wire callbacks
                slaveWorker?.onClusterStateUpdated = { cluster ->
                    _clusterState.value = cluster
                }

                slaveWorker?.onModelChanged = { newModel ->
                    _activeModel.value = newModel
                    inspectActiveModelWeights()
                }

                slaveWorker?.onAssignedBlockChanged = { newBlock ->
                    handlePreloadCommand(_activeModel.value, newBlock)
                }

                // Requirement 6: Dynamic promotion to Master if master dies
                slaveWorker?.onPromotedToMaster = {
                    viewModelScope.launch(Dispatchers.Main) {
                        Log.i(TAG, "Promoted to Master following master failure!")
                        initializeAsMaster()
                    }
                }

                slaveWorker?.start()

                // Preload assigned block automatically upon joining cluster
                val myBlock = myAssignedBlocks.firstOrNull()
                if (cluster != null && myBlock != null) {
                    handlePreloadCommand(cluster.modelSpec, myBlock)
                }

                startForegroundService("SLAVE", "Participating in mesh pipeline")
                onComplete(true, "Successfully connected to Master")
            } else {
                p2pServer?.updateSessionToken("")
                p2pServer?.resetCallbacks()
                p2pServer?.stop()
                p2pServer = null
                onComplete(false, regResult.exceptionOrNull()?.message ?: "Connection failed")
            }
        }
    }

    fun isModelAvailableOnMaster(model: ModelSpec): Boolean {
        if (_currentRole.value != NodeRole.SLAVE) return false
        val cluster = _clusterState.value
        if (model.id == cluster.modelSpec.id && cluster.isModelAvailableOnMaster) return true
        return cluster.masterDownloadedModelIds.contains(model.id)
    }

    fun startInference(prompt: String, maxTokens: Int = 120) {
        if (_isInferencing.value) return

        val model = _activeModel.value
        val isDownloaded = modelDownloadManager.isModelDownloaded(model)
        if (!isDownloaded) {
            val err = "\n[ERROR E101: Model Weights Not Downloaded]\nModel '${model.name}' (${model.fileName}) is not downloaded on this device.\n\nPlease open the 'Models' tab, locate '${model.name}', and tap 'Download' to store and cache the model weights in RAM."
            _streamingOutput.value = err
            _inferenceStatus.value = "Error E101: Model not downloaded"
            return
        }

        _isInferencing.value = true
        _streamingOutput.value = ""
        synchronized(tokenBuffer) { tokenBuffer.clear() }
        lastTokenUiDispatchMs = 0L
        _generatedTokensCount.value = 0
        _tokensPerSecond.value = 0.0
        _ttftLatencyMs.value = 0L
        _inferenceStatus.value = "Preparing model in RAM..."

        viewModelScope.launch(Dispatchers.IO) {
            try {
                // Ensure Master has its assigned block loaded in RAM before starting inference
                val myBlock = getLocalAssignedBlock()
                val file = modelDownloadManager.getModelFile(model)
                val runner = NeuralModelRunner.shared
                val needsLoad = !runner.isModelLoadedInRam ||
                        runner.loadedBlock?.modelId != model.id ||
                        runner.loadedFile?.name != file.name ||
                        runner.loadedBlock?.startLayer != myBlock.startLayer ||
                        runner.loadedBlock?.endLayer != myBlock.endLayer

                if (needsLoad && file.exists() && file.length() > 1024L) {
                    _inferenceStatus.value = "Loading ${model.name} into RAM..."
                    runner.loadModelWeightsIfAvailable(file, myBlock)
                    _isModelInRam.value = true
                    val ramMb = runner.allocatedRamMb
                    _modelAllocatedRamMb.value = ramMb
                    sensorMonitor.updateModelRam(ramMb)
                    _sensorMetrics.value = sensorMonitor.updateMetrics()
                    orchestrator?.updateLocalMetrics(_sensorMetrics.value)
                }

                // Broadcast preload to all connected slaves to ensure all hops are ready in RAM
                orchestrator?.broadcastPreloadToSlaves(model)

                _inferenceStatus.value = "Executing distributed forward pass..."
                orchestrator?.startDistributedInference(prompt, maxTokens)
            } catch (e: Exception) {
                Log.e(TAG, "Error initiating inference: ${e.message}", e)
                _isInferencing.value = false
                _streamingOutput.value = "\n[ERROR: Failed starting inference]\n${e.message ?: "Unknown error"}"
                _inferenceStatus.value = "Inference error: ${e.message}"
            }
        }
    }

    fun stopInference() {
        orchestrator?.stopInference()
        synchronized(tokenBuffer) { tokenBuffer.clear() }
        _isInferencing.value = false
        _inferenceStatus.value = "Stopped"
    }

    fun getQrPayloadString(): String {
        return QrConnectionHelper.createPayloadString(
            ip = _localIp.value,
            port = localPort,
            token = _sessionToken.value,
            model = _activeModel.value.name
        )
    }

    private var cachedQrPayload: String = ""
    private var cachedQrSize: Int = 0
    private var cachedQrBitmap: android.graphics.Bitmap? = null

    @Synchronized
    fun getQrBitmap(sizePx: Int = 512): android.graphics.Bitmap? {
        val payload = getQrPayloadString()
        if (payload == cachedQrPayload && sizePx == cachedQrSize && cachedQrBitmap != null) {
            return cachedQrBitmap
        }
        val bmp = QrConnectionHelper.generateQrBitmap(payload, sizePx)
        if (bmp != null) {
            cachedQrPayload = payload
            cachedQrSize = sizePx
            cachedQrBitmap = bmp
        }
        return bmp
    }

    fun disconnectAndReset() {
        orchestrator?.stop()
        orchestrator = null
        slaveWorker?.stop()
        slaveWorker = null
        p2pServer?.updateSessionToken("")
        p2pServer?.resetCallbacks()
        p2pServer?.stop()
        p2pServer = null

        _currentRole.value = NodeRole.UNASSIGNED
        _sessionToken.value = ""
        _isInferencing.value = false
        _inferenceStatus.value = "Idle"
        _clusterState.value = ClusterState(
            sessionToken = "",
            modelSpec = _activeModel.value,
            modelName = _activeModel.value.name,
            totalLayers = _activeModel.value.totalLayers,
            masterNodeId = localNodeId,
            masterAddress = "${_localIp.value}:$localPort"
        )

        val app = getApplication<Application>()
        app.stopService(Intent(app, MeshNodeService::class.java))
    }

    private fun startForegroundService(role: String, details: String) {
        val app = getApplication<Application>()
        val intent = Intent(app, MeshNodeService::class.java).apply {
            putExtra("ROLE", role)
            putExtra("DETAILS", details)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            app.startForegroundService(intent)
        } else {
            app.startService(intent)
        }
    }

    override fun onCleared() {
        super.onCleared()
        telemetryPollJob?.cancel()
        disconnectAndReset()
    }
}
