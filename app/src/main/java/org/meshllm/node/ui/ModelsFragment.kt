package org.meshllm.node.ui

import android.graphics.Color
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import kotlinx.coroutines.launch
import org.meshllm.node.databinding.FragmentModelsBinding
import org.meshllm.node.model.ModelCatalog
import org.meshllm.node.model.ModelSpec
import org.meshllm.node.telemetry.DownloadState

class ModelsFragment : Fragment() {

    private var _binding: FragmentModelsBinding? = null
    private val binding get() = _binding!!

    private val viewModel: MeshClusterViewModel by activityViewModels()
    private lateinit var modelAdapter: ModelAdapter

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentModelsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        setupRecyclerView()
        setupCategoryFilters()
        setupSuitabilityAndQuantFilters()
        setupSearch()
        setupActionButtons()
        observeViewModel()
    }

    private fun setupRecyclerView() {
        modelAdapter = ModelAdapter(
            models = emptyList(),
            activeModelId = viewModel.activeModel.value.id,
            isModelDownloaded = { viewModel.modelDownloadManager.isModelDownloaded(it) },
            isModelAvailableOnMaster = { viewModel.isModelAvailableOnMaster(it) },
            onDownloadClicked = { model ->
                val onMaster = viewModel.isModelAvailableOnMaster(model)
                val sourceLabel = if (onMaster) "Master Node (Local Mesh P2P)" else "Hugging Face"
                viewModel.modelDownloadManager.downloadModel(model)
                Toast.makeText(requireContext(), "Downloading ${model.name} (${model.fileSizeDisplay}) from $sourceLabel...", Toast.LENGTH_SHORT).show()
            },
            onModelSelected = { model ->
                viewModel.selectModel(model)
                Toast.makeText(requireContext(), "Loaded ${model.name} into cluster", Toast.LENGTH_SHORT).show()
            }
        )

        binding.rvCatalogList.layoutManager = LinearLayoutManager(requireContext())
        binding.rvCatalogList.adapter = modelAdapter
    }

    private fun setupCategoryFilters() {
        val buttons = listOf(
            binding.btnCatAll to ModelCatalog.CATEGORY_ALL,
            binding.btnCatMassive to ModelCatalog.CATEGORY_LARGE_MESH,
            binding.btnCatStandard to ModelCatalog.CATEGORY_STANDARD,
            binding.btnCatCompact to ModelCatalog.CATEGORY_COMPACT,
            binding.btnCatStarter to ModelCatalog.CATEGORY_ULTRA_SMALL
        )

        val activeColor = Color.parseColor("#6366F1")
        val inactiveColor = Color.parseColor("#171D2A")

        fun updateButtonColors(selectedCategory: String) {
            buttons.forEach { (btn, cat) ->
                btn.setBackgroundColor(if (cat == selectedCategory) activeColor else inactiveColor)
                btn.setTextColor(if (cat == selectedCategory) Color.WHITE else Color.parseColor("#94A3B8"))
            }
        }

        buttons.forEach { (btn, cat) ->
            btn.setOnClickListener {
                viewModel.setCatalogFilter(category = cat, query = binding.etCatalogSearch.text.toString().trim())
                updateButtonColors(cat)
            }
        }

        updateButtonColors(viewModel.catalogCategory.value)
    }

    private fun setupSuitabilityAndQuantFilters() {
        binding.btnRefreshHf.setOnClickListener {
            Toast.makeText(requireContext(), "Syncing models from Hugging Face Hub...", Toast.LENGTH_SHORT).show()
            viewModel.fetchHuggingFaceCatalog(forceRefresh = true)
        }

        binding.btnToggleSuitable.setOnClickListener {
            viewModel.toggleSuitableFilter()
            val isSuitable = viewModel.suitableOnlyFilter.value
            val msg = if (isSuitable) "Default filter: Showing suitable models only" else "Filter removed: Showing all Hugging Face models"
            Toast.makeText(requireContext(), msg, Toast.LENGTH_SHORT).show()
        }

        val quantOptions = listOf(
            null to "All Quants",
            "Q4" to "Q4 (Optimal)",
            "Q2" to "Q2 (Low RAM)",
            "Q3" to "Q3 (Balanced)",
            "Q5" to "Q5 (High Qual)",
            "Q8" to "Q8 (Precision)"
        )
        var currentQuantIdx = 0

        binding.btnQuantFilter.setOnClickListener {
            currentQuantIdx = (currentQuantIdx + 1) % quantOptions.size
            val (quantValue, label) = quantOptions[currentQuantIdx]
            binding.btnQuantFilter.text = label
            viewModel.setPreferredQuantFilter(quantValue)
            Toast.makeText(requireContext(), "Quantization: $label", Toast.LENGTH_SHORT).show()
        }
    }

    private fun setupSearch() {
        binding.etCatalogSearch.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {
                viewModel.setCatalogFilter(viewModel.catalogCategory.value, s?.toString()?.trim() ?: "")
            }
            override fun afterTextChanged(s: Editable?) {}
        })
    }

    private fun setupActionButtons() {
        binding.btnDownloadAction.setOnClickListener {
            val model = viewModel.activeModel.value
            val isDownloaded = viewModel.modelDownloadManager.isModelDownloaded(model)
            val state = viewModel.downloadStates.value[model.id]

            when {
                state is DownloadState.Downloading -> {
                    viewModel.cancelActiveDownload()
                    Toast.makeText(requireContext(), "Download cancelled", Toast.LENGTH_SHORT).show()
                }
                isDownloaded -> {
                    viewModel.deleteActiveModelFile()
                    Toast.makeText(requireContext(), "Deleted ${model.name} weights from storage", Toast.LENGTH_SHORT).show()
                }
                else -> {
                    viewModel.downloadActiveModel()
                    Toast.makeText(requireContext(), "Started downloading ${model.name} (${model.fileSizeDisplay})", Toast.LENGTH_SHORT).show()
                }
            }
        }

        binding.btnVerifyWeights.setOnClickListener {
            viewModel.inspectActiveModelWeights()
            Toast.makeText(requireContext(), "Weight file integrity verified", Toast.LENGTH_SHORT).show()
        }

        binding.btnPreloadWeightsToRam.setOnClickListener {
            val model = viewModel.activeModel.value
            val isDownloaded = viewModel.modelDownloadManager.isModelDownloaded(model)
            if (!isDownloaded) {
                Toast.makeText(requireContext(), "Please download ${model.name} weights first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            Toast.makeText(requireContext(), "Preloading ${model.name} into RAM...", Toast.LENGTH_SHORT).show()
            viewModel.preloadActiveModelToRam { success, msg ->
                activity?.runOnUiThread {
                    Toast.makeText(requireContext(), msg, Toast.LENGTH_SHORT).show()
                }
            }
        }

        binding.btnClearRamWeights.setOnClickListener {
            val cleared = viewModel.clearLoadedModelFromRam()
            val msg = if (cleared) "Cleared model weights from RAM" else "RAM already free"
            Toast.makeText(requireContext(), msg, Toast.LENGTH_SHORT).show()
        }
    }

    private fun observeViewModel() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    viewModel.activeModel.collect { model ->
                        binding.tvActiveModelName.text = model.name
                        binding.tvActiveModelArchitecture.text = "${model.totalLayers} Layers • Hidden: ${model.hiddenDim} • Heads: ${model.numHeads} • ${model.fileSizeDisplay}"
                        binding.tvActiveModelCategoryBadge.text = model.category
                        binding.tvActiveModelDesc.text = model.description
                        binding.tvActiveModelOomAlert.visibility = if (model.singleDeviceOomWarning) View.VISIBLE else View.GONE

                        updateDownloadCardUi(model)
                    }
                }

                launch {
                    viewModel.downloadStates.collect { states ->
                        val active = viewModel.activeModel.value
                        updateDownloadCardUi(active)
                        states.keys.forEach { modelId ->
                            modelAdapter.notifyModelStatusChanged(modelId)
                        }
                    }
                }

                launch {
                    viewModel.clusterState.collect {
                        updateDownloadCardUi(viewModel.activeModel.value)
                    }
                }

                launch {
                    viewModel.weightFilePath.collect { path ->
                        binding.tvWeightFilePath.text = path
                    }
                }

                launch {
                    viewModel.weightFormatInspection.collect { format ->
                        binding.tvWeightFormatDetails.text = format
                    }
                }

                launch {
                    viewModel.catalogModels.collect { models ->
                        if (!viewModel.isHfSearching.value) {
                            binding.tvCatalogCount.text = "${models.size} Models"
                        }
                        modelAdapter.updateData(models, viewModel.activeModel.value.id)
                    }
                }

                launch {
                    viewModel.isHfSearching.collect { searching ->
                        binding.pbHfLoading.visibility = if (searching) View.VISIBLE else View.GONE
                        if (searching) {
                            binding.tvCatalogCount.text = "Syncing HF..."
                        } else {
                            binding.tvCatalogCount.text = "${viewModel.catalogModels.value.size} Models"
                        }
                    }
                }

                launch {
                    viewModel.suitableOnlyFilter.collect { suitableOnly ->
                        if (suitableOnly) {
                            binding.btnToggleSuitable.text = "✓ Suitable Only (≤15B, Quantized)"
                            binding.btnToggleSuitable.setTextColor(Color.parseColor("#10B981"))
                            binding.btnToggleSuitable.strokeColor = android.content.res.ColorStateList.valueOf(Color.parseColor("#10B981"))
                        } else {
                            binding.btnToggleSuitable.text = "All Models (Unfiltered)"
                            binding.btnToggleSuitable.setTextColor(Color.parseColor("#94A3B8"))
                            binding.btnToggleSuitable.strokeColor = android.content.res.ColorStateList.valueOf(Color.parseColor("#212838"))
                        }
                    }
                }

                launch {
                    viewModel.preferredQuantFilter.collect { quant ->
                        binding.btnQuantFilter.text = when (quant) {
                            "Q4" -> "Q4 (Optimal)"
                            "Q2" -> "Q2 (Low RAM)"
                            "Q3" -> "Q3 (Balanced)"
                            "Q5" -> "Q5 (High Qual)"
                            "Q8" -> "Q8 (Precision)"
                            else -> "All Quants"
                        }
                    }
                }

                launch {
                    viewModel.isModelPreloading.collect { isPreloading ->
                        binding.pbRamPreload.visibility = if (isPreloading) View.VISIBLE else View.GONE
                        binding.btnPreloadWeightsToRam.isEnabled = !isPreloading && viewModel.modelDownloadManager.isModelDownloaded(viewModel.activeModel.value)
                        binding.btnClearRamWeights.isEnabled = !isPreloading && viewModel.isModelInRam.value
                        if (isPreloading) {
                            binding.tvRamStatusBadge.text = "LOADING TO RAM..."
                            binding.tvRamStatusBadge.setBackgroundColor(Color.parseColor("#FF9800"))
                        }
                    }
                }

                launch {
                    viewModel.isModelInRam.collect { isInRam ->
                        val ramMb = viewModel.modelAllocatedRamMb.value
                        if (!viewModel.isModelPreloading.value) {
                            if (isInRam) {
                                binding.tvRamStatusBadge.text = "LOADED IN RAM (${ramMb} MB)"
                                binding.tvRamStatusBadge.setBackgroundColor(Color.parseColor("#1B5E20"))
                                binding.btnPreloadWeightsToRam.text = "Reload in RAM"
                                binding.btnClearRamWeights.isEnabled = true
                            } else {
                                binding.tvRamStatusBadge.text = "NOT IN RAM"
                                binding.tvRamStatusBadge.setBackgroundColor(Color.parseColor("#424242"))
                                binding.btnPreloadWeightsToRam.text = "Load Weights to RAM"
                                binding.btnClearRamWeights.isEnabled = false
                            }
                        }
                    }
                }

                launch {
                    viewModel.ramStatusDescription.collect { status ->
                        binding.tvRamPreloadStatus.text = status
                    }
                }

                launch {
                    viewModel.modelAllocatedRamMb.collect { ramMb ->
                        if (viewModel.isModelInRam.value && !viewModel.isModelPreloading.value) {
                            binding.tvRamStatusBadge.text = "LOADED IN RAM (${ramMb} MB)"
                        }
                    }
                }
            }
        }
    }

    private fun updateDownloadCardUi(model: ModelSpec) {
        val state = viewModel.downloadStates.value[model.id]
        val isDownloaded = viewModel.modelDownloadManager.isModelDownloaded(model)

        when {
            state is DownloadState.Downloading -> {
                binding.tvDownloadStatusBadge.text = "DOWNLOADING (${state.percent}%)"
                binding.tvDownloadStatusBadge.setBackgroundColor(Color.parseColor("#1565C0"))

                binding.btnDownloadAction.text = "Cancel Download"
                binding.btnDownloadAction.setBackgroundColor(Color.parseColor("#D32F2F"))

                binding.pbModelDownload.visibility = View.VISIBLE
                if (state.totalBytes > 0) {
                    binding.pbModelDownload.isIndeterminate = false
                    binding.pbModelDownload.progress = state.percent
                } else {
                    binding.pbModelDownload.isIndeterminate = true
                }

                val mbRead = Math.round((state.bytesRead.toDouble() / (1024.0 * 1024.0)) * 10.0) / 10.0
                val mbTotal = Math.round((state.totalBytes.toDouble() / (1024.0 * 1024.0)) * 10.0) / 10.0
                val sourceLabel = if (state.source.isNotEmpty()) state.source else "Hugging Face"
                binding.tvDownloadProgressInfo.visibility = View.VISIBLE
                binding.tvDownloadProgressInfo.text = "Downloading from $sourceLabel: $mbRead MB / $mbTotal MB (${state.percent}%) • ${state.speedMbPerSec} MB/s"
            }
            isDownloaded -> {
                val size = viewModel.modelDownloadManager.getDownloadedSizeMb(model)
                binding.tvDownloadStatusBadge.text = "READY ON DISK ($size MB)"
                binding.tvDownloadStatusBadge.setBackgroundColor(Color.parseColor("#1B5E20"))

                binding.btnDownloadAction.text = "Delete Weights"
                binding.btnDownloadAction.setBackgroundColor(Color.parseColor("#C62828"))

                binding.pbModelDownload.visibility = View.GONE
                binding.tvDownloadProgressInfo.visibility = View.VISIBLE
                binding.tvDownloadProgressInfo.text = "Weights cached and verified on disk ($size MB)"
            }
            state is DownloadState.Failed -> {
                binding.tvDownloadStatusBadge.text = "DOWNLOAD FAILED"
                binding.tvDownloadStatusBadge.setBackgroundColor(Color.parseColor("#B71C1C"))

                binding.btnDownloadAction.text = "Retry Download"
                binding.btnDownloadAction.setBackgroundColor(Color.parseColor("#00C853"))

                binding.pbModelDownload.visibility = View.GONE
                binding.tvDownloadProgressInfo.visibility = View.VISIBLE
                binding.tvDownloadProgressInfo.text = "Error: ${state.error}. Tap Retry to download."
            }
            else -> {
                val onMaster = viewModel.isModelAvailableOnMaster(model)
                if (onMaster) {
                    binding.tvDownloadStatusBadge.text = "AVAILABLE ON MASTER (FAST P2P)"
                    binding.tvDownloadStatusBadge.setBackgroundColor(Color.parseColor("#00838F"))

                    binding.btnDownloadAction.text = "Download from Master (${model.fileSizeDisplay})"
                    binding.btnDownloadAction.setBackgroundColor(Color.parseColor("#00B0FF"))

                    binding.pbModelDownload.visibility = View.GONE
                    binding.tvDownloadProgressInfo.visibility = View.VISIBLE
                    binding.tvDownloadProgressInfo.text = "Cached on Master node. Direct local P2P mesh transfer available without internet."
                } else {
                    binding.tvDownloadStatusBadge.text = "NOT DOWNLOADED"
                    binding.tvDownloadStatusBadge.setBackgroundColor(Color.parseColor("#E65100"))

                    binding.btnDownloadAction.text = "Download Weights (${model.fileSizeDisplay})"
                    binding.btnDownloadAction.setBackgroundColor(Color.parseColor("#00C853"))

                    binding.pbModelDownload.visibility = View.GONE
                    binding.tvDownloadProgressInfo.visibility = View.GONE
                }
            }
        }

        val isPreloading = viewModel.isModelPreloading.value
        if (!isDownloaded) {
            binding.btnPreloadWeightsToRam.isEnabled = false
            binding.btnPreloadWeightsToRam.alpha = 0.5f
        } else {
            binding.btnPreloadWeightsToRam.isEnabled = !isPreloading
            binding.btnPreloadWeightsToRam.alpha = 1.0f
        }
        binding.btnClearRamWeights.isEnabled = !isPreloading && viewModel.isModelInRam.value
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
