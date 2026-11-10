package org.meshllm.node.ui

import android.graphics.Color
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import org.meshllm.node.databinding.FragmentInferenceBinding
import org.meshllm.node.model.NodeRole

class InferenceFragment : Fragment() {

    private var _binding: FragmentInferenceBinding? = null
    private val binding get() = _binding!!

    private val viewModel: MeshClusterViewModel by activityViewModels()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentInferenceBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        setupListeners()
        observeViewModel()
    }

    private fun setupListeners() {
        binding.btnUseSamplePrompt.setOnClickListener {
            binding.etPrompt.setText(viewModel.activeModel.value.samplePrompt)
            Toast.makeText(requireContext(), "Sample prompt loaded", Toast.LENGTH_SHORT).show()
        }

        binding.btnClearPrompt.setOnClickListener {
            binding.etPrompt.setText("")
        }

        binding.btnRunInference.setOnClickListener {
            val prompt = binding.etPrompt.text.toString().trim()
            if (prompt.isEmpty()) {
                Toast.makeText(requireContext(), "Please enter a prompt", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            if (viewModel.currentRole.value != NodeRole.MASTER) {
                // If not master, initialize master automatically or prompt user
                if (viewModel.currentRole.value == NodeRole.UNASSIGNED) {
                    viewModel.initializeAsMaster()
                    Toast.makeText(requireContext(), "Initialized as Master node for inference", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(requireContext(), "Inference is orchestrated by Master node", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }
            }

            // Immediately clear any old dialogue/response from the UI
            binding.tvInferenceOutput.text = "⚡ Evaluating prompt across mesh... Streaming tokens will appear here."
            binding.tvTokenCount.text = "0 tokens"
            binding.tvThroughput.text = "-- tok/s"
            binding.tvTtftLatency.text = "-- ms"
            binding.tvPipelineTrace.text = "Initiating distributed pipeline for ${viewModel.activeModel.value.name}..."

            viewModel.startInference(prompt)
        }

        binding.btnStopInference.setOnClickListener {
            viewModel.stopInference()
            Toast.makeText(requireContext(), "Inference stopped", Toast.LENGTH_SHORT).show()
        }
    }

    private fun observeViewModel() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.activeModel.collect { model ->
                updateClusterBanner()
                binding.etPrompt.setText(model.samplePrompt)
                binding.tvInferenceOutput.text = "Ready. Tap 'Run Distributed Inference' to stream tokens for ${model.name}."
                binding.tvTokenCount.text = "0 tokens"
                binding.tvThroughput.text = "-- tok/s"
                binding.tvTtftLatency.text = "-- ms"
                binding.tvPipelineTrace.text = ""
            }
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.clusterState.collect {
                updateClusterBanner()
            }
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.isInferencing.collect { isRunning ->
                val isSlave = viewModel.currentRole.value == NodeRole.SLAVE
                binding.btnRunInference.isEnabled = !isRunning && !isSlave
                binding.btnStopInference.visibility = if (isRunning) View.VISIBLE else View.GONE
                if (isSlave) {
                    binding.btnRunInference.text = "Mesh Worker Active (Listening for Master)"
                    binding.btnRunInference.setBackgroundColor(Color.parseColor("#37474F"))
                } else if (isRunning) {
                    binding.btnRunInference.text = "⚡ Generating Tokens..."
                    binding.btnRunInference.setBackgroundColor(Color.parseColor("#455A64"))
                } else {
                    binding.btnRunInference.text = "🚀 Run Distributed Inference"
                    binding.btnRunInference.setBackgroundColor(Color.parseColor("#00C853"))
                }
            }
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.currentRole.collect { role ->
                val isSlave = role == NodeRole.SLAVE
                if (isSlave) {
                    binding.btnRunInference.isEnabled = false
                    binding.btnRunInference.text = "Mesh Worker Active (Listening for Master)"
                    binding.btnRunInference.setBackgroundColor(Color.parseColor("#37474F"))
                    binding.tvInferenceStatus.text = "Worker node connected. Awaiting pipeline activations from Master."
                }
            }
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.inferenceStatus.collect { status ->
                binding.tvInferenceStatus.text = status
            }
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.streamingOutput.collect { output ->
                if (output.isNotEmpty()) {
                    binding.tvInferenceOutput.text = output
                } else if (viewModel.isInferencing.value) {
                    val count = viewModel.generatedTokensCount.value
                    if (count > 0) {
                        binding.tvInferenceOutput.text = "⚡ Generating tokens: $count tokens produced across mesh..."
                    } else {
                        binding.tvInferenceOutput.text = "⚡ Evaluating prompt across mesh... Streaming tokens will appear here."
                    }
                }
            }
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.generatedTokensCount.collect { count ->
                binding.tvTokenCount.text = "$count tokens"
            }
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.tokensPerSecond.collect { tps ->
                binding.tvThroughput.text = if (tps > 0) "$tps tok/s" else "-- tok/s"
            }
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.ttftLatencyMs.collect { ttft ->
                binding.tvTtftLatency.text = if (ttft > 0) "$ttft ms" else "-- ms"
            }
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.pipelineTrace.collect { trace ->
                binding.tvPipelineTrace.text = trace
            }
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.weightFormatInspection.collect { format ->
                val isDownloaded = viewModel.modelDownloadManager.isModelDownloaded(viewModel.activeModel.value)
                val modelRam = viewModel.modelAllocatedRamMb.value
                if (isDownloaded) {
                    binding.tvWeightSourceStatus.text = "✓ REAL WEIGHTS IN RAM: $format (Allocated: $modelRam MB)"
                    binding.tvWeightSourceStatus.setBackgroundColor(Color.parseColor("#1B5E20"))
                    binding.tvWeightSourceStatus.setTextColor(Color.parseColor("#A5D6A7"))
                } else {
                    binding.tvWeightSourceStatus.text = "⚠ MODEL NOT DOWNLOADED: Open 'Models' tab to download ${viewModel.activeModel.value.name} (${viewModel.activeModel.value.fileSizeDisplay}) before inference."
                    binding.tvWeightSourceStatus.setBackgroundColor(Color.parseColor("#B71C1C"))
                    binding.tvWeightSourceStatus.setTextColor(Color.parseColor("#FFCDD2"))
                }
            }
        }
    }

    private fun updateClusterBanner() {
        val model = viewModel.activeModel.value
        val state = viewModel.clusterState.value
        val nodeCount = state.nodes.size.coerceAtLeast(1)
        val roleStr = when (viewModel.currentRole.value) {
            NodeRole.MASTER -> "Master"
            NodeRole.SLAVE -> "Slave"
            else -> "Local"
        }
        binding.tvInferenceClusterInfo.text = "Model: ${model.name} (${model.paramCountStr}) • Pipeline: ${model.totalLayers} Layers across $nodeCount Nodes ($roleStr)"
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
