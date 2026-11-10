package org.meshllm.node.ui

import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.graphics.Color
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import com.journeyapps.barcodescanner.ScanContract
import com.journeyapps.barcodescanner.ScanOptions
import kotlinx.coroutines.launch
import org.meshllm.node.R
import org.meshllm.node.databinding.FragmentClusterBinding
import org.meshllm.node.model.NodeRole
import org.meshllm.node.p2p.QrConnectionHelper

class ClusterFragment : Fragment() {

    private var _binding: FragmentClusterBinding? = null
    private val binding get() = _binding!!

    private val viewModel: MeshClusterViewModel by activityViewModels()
    private val nodeAdapter = NodeAdapter()

    // ActivityResultLauncher for ZXing Camera QR Code Scanner
    private val qrScanLauncher = registerForActivityResult(ScanContract()) { result ->
        val contents = result.contents
        if (!contents.isNullOrEmpty()) {
            handleScannedQr(contents)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentClusterBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        binding.rvClusterNodes.layoutManager = LinearLayoutManager(requireContext())
        binding.rvClusterNodes.adapter = nodeAdapter

        setupListeners()
        setupLayerAllotmentUi()
        observeViewModel()
    }

    private fun launchQrScanner() {
        val options = ScanOptions().apply {
            setDesiredBarcodeFormats(ScanOptions.QR_CODE)
            setPrompt("Point camera at Master node's QR Code")
            setCameraId(0)
            setBeepEnabled(true)
            setBarcodeImageEnabled(false)
            setOrientationLocked(false)
        }
        qrScanLauncher.launch(options)
    }

    private fun handleScannedQr(contents: String) {
        val payload = QrConnectionHelper.parseScannedPayload(contents)
        if (payload != null) {
            Toast.makeText(requireContext(), "Found Master at ${payload.ip}:${payload.port}. Connecting...", Toast.LENGTH_SHORT).show()
            viewModel.joinAsSlave(payload.ip, payload.port, payload.token) { success, msg ->
                requireActivity().runOnUiThread {
                    if (success) {
                        Toast.makeText(requireContext(), "Joined Cluster as Slave via QR!", Toast.LENGTH_SHORT).show()
                    } else {
                        Toast.makeText(requireContext(), "QR Join failed: $msg", Toast.LENGTH_LONG).show()
                    }
                }
            }
        } else {
            Toast.makeText(requireContext(), "Invalid QR Code for MeshLLM Cluster", Toast.LENGTH_LONG).show()
        }
    }

    private fun showEnlargedQrDialog() {
        val qrBitmap = viewModel.getQrBitmap(512) ?: viewModel.getQrBitmap(800) ?: return
        val dialogView = LayoutInflater.from(requireContext()).inflate(R.layout.dialog_enlarged_qr, null, false)
        val iv = dialogView.findViewById<ImageView>(R.id.ivEnlargedQr)
        val tvDetails = dialogView.findViewById<TextView>(R.id.tvEnlargedDetails)

        iv.setImageBitmap(qrBitmap)
        tvDetails.text = "Master: ${viewModel.localIp.value}:${viewModel.localPort}\nToken: ${viewModel.sessionToken.value}\nModel: ${viewModel.activeModel.value.name}"

        AlertDialog.Builder(requireContext())
            .setView(dialogView)
            .setPositiveButton("Close", null)
            .show()
    }

    private fun setupListeners() {
        binding.btnStartMaster.setOnClickListener {
            viewModel.initializeAsMaster()
            Toast.makeText(requireContext(), "Master initialized with ${viewModel.activeModel.value.name}", Toast.LENGTH_SHORT).show()
        }

        binding.btnScanQrJoin.setOnClickListener {
            launchQrScanner()
        }

        binding.btnJoinSlave.setOnClickListener {
            showJoinDialog()
        }

        binding.btnDisconnect.setOnClickListener {
            viewModel.disconnectAndReset()
            Toast.makeText(requireContext(), "Disconnected from network", Toast.LENGTH_SHORT).show()
        }

        binding.ivMasterQrCode.setOnClickListener {
            showEnlargedQrDialog()
        }

        binding.btnCopyToken.setOnClickListener {
            val token = viewModel.sessionToken.value
            val cm = requireContext().getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("Session Token", token))
            Toast.makeText(requireContext(), "Session token copied to clipboard", Toast.LENGTH_SHORT).show()
        }
    }

    private fun refreshMasterQrDisplay() {
        if (viewModel.currentRole.value == NodeRole.MASTER) {
            viewLifecycleOwner.lifecycleScope.launch(kotlinx.coroutines.Dispatchers.Default) {
                val qrBitmap = viewModel.getQrBitmap(512)
                kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
                    if (_binding != null && viewModel.currentRole.value == NodeRole.MASTER) {
                        if (qrBitmap != null) {
                            binding.ivMasterQrCode.setImageBitmap(qrBitmap)
                            binding.layoutMasterQr.visibility = View.VISIBLE
                            binding.tvQrPayloadInfo.text = "Master: ${viewModel.localIp.value}:${viewModel.localPort} • Model: ${viewModel.activeModel.value.name}"
                        } else {
                            binding.layoutMasterQr.visibility = View.GONE
                        }
                    }
                }
            }
        } else {
            binding.layoutMasterQr.visibility = View.GONE
        }
    }

    private val nodeSliderViews = mutableMapOf<String, com.google.android.material.slider.Slider>()
    private var lastSliderNodeIds: List<String> = emptyList()

    private fun setupLayerAllotmentUi() {
        binding.rgAllotmentMode.setOnCheckedChangeListener { _, checkedId ->
            if (checkedId == R.id.rbAllotmentAuto) {
                binding.layoutManualAllotment.visibility = View.GONE
                viewModel.setLayerAllotmentMode(org.meshllm.node.model.ClusterState.ALLOTMENT_MODE_AUTO)
                Toast.makeText(requireContext(), "Automatic benchmark-based layer allotment active", Toast.LENGTH_SHORT).show()
            } else if (checkedId == R.id.rbAllotmentManual) {
                binding.layoutManualAllotment.visibility = View.VISIBLE
                refreshNodeSliders(viewModel.clusterState.value)
            }
        }

        binding.btnDistributeEvenly.setOnClickListener {
            val state = viewModel.clusterState.value
            val activeNodes = state.nodes.filter { it.isAlive }.sortedBy { it.joinOrder }
            if (activeNodes.isEmpty()) return@setOnClickListener
            val total = state.totalLayers
            val base = total / activeNodes.size
            val remainder = total % activeNodes.size

            activeNodes.forEachIndexed { idx, node ->
                val count = base + (if (idx < remainder) 1 else 0)
                nodeSliderViews[node.nodeId]?.value = count.toFloat().coerceIn(0f, total.toFloat())
            }
            recalculateManualPreview(state.totalLayers)
        }

        binding.btnApplyAllotment.setOnClickListener {
            val layerCounts = mutableMapOf<String, Int>()
            nodeSliderViews.forEach { (nodeId, slider) ->
                layerCounts[nodeId] = slider.value.toInt()
            }
            val totalAssigned = layerCounts.values.sum()
            val totalLayers = viewModel.clusterState.value.totalLayers
            viewModel.applyManualLayerAllotment(layerCounts)
            Toast.makeText(requireContext(), "Manual layer allotment applied ($totalAssigned/$totalLayers L)!", Toast.LENGTH_SHORT).show()
        }
    }

    private fun refreshNodeSliders(state: org.meshllm.node.model.ClusterState) {
        val activeNodes = state.nodes.filter { it.isAlive }.sortedBy { it.joinOrder }
        val currentIds = activeNodes.map { it.nodeId }

        if (currentIds != lastSliderNodeIds) {
            lastSliderNodeIds = currentIds
            binding.layoutSlidersContainer.removeAllViews()
            nodeSliderViews.clear()

            val inflater = LayoutInflater.from(requireContext())
            for (node in activeNodes) {
                val itemView = inflater.inflate(R.layout.item_node_slider, binding.layoutSlidersContainer, false)
                val tvTitle = itemView.findViewById<TextView>(R.id.tvSliderNodeTitle)
                val slider = itemView.findViewById<com.google.android.material.slider.Slider>(R.id.sliderNodeLayers)

                val roleStr = if (node.role == NodeRole.MASTER) "Master" else "Slave"
                tvTitle.text = "Node ${node.joinOrder} ($roleStr) - ${node.deviceName}"

                slider.valueFrom = 0.0f
                slider.valueTo = state.totalLayers.toFloat()
                slider.stepSize = 1.0f

                val currentCount = node.totalAssignedLayers.toFloat().coerceIn(0f, state.totalLayers.toFloat())
                slider.value = currentCount

                slider.addOnChangeListener { _, _, _ ->
                    recalculateManualPreview(state.totalLayers)
                }

                nodeSliderViews[node.nodeId] = slider
                binding.layoutSlidersContainer.addView(itemView)
            }
        }
        recalculateManualPreview(state.totalLayers)
    }

    private fun recalculateManualPreview(totalLayers: Int) {
        val activeNodes = viewModel.clusterState.value.nodes.filter { it.isAlive }.sortedBy { it.joinOrder }
        var currentLayer = 0
        var totalAssigned = 0

        for ((index, node) in activeNodes.withIndex()) {
            val slider = nodeSliderViews[node.nodeId] ?: continue
            val count = slider.value.toInt()
            val itemView = binding.layoutSlidersContainer.getChildAt(index)
            val tvRange = itemView?.findViewById<TextView>(R.id.tvSliderLayerRange)

            if (count == 0) {
                tvRange?.text = "0 layers (Bypassed - 0 MB)"
                tvRange?.setTextColor(Color.parseColor("#757575"))
            } else {
                val start = currentLayer
                val end = (start + count - 1).coerceAtMost(totalLayers - 1)
                tvRange?.text = "$count layers (L$start..L$end)"
                tvRange?.setTextColor(Color.parseColor("#00E676"))
                currentLayer = end + 1
                totalAssigned += count
            }
        }

        binding.tvAllotmentSummary.text = "Total Assigned: $totalAssigned / $totalLayers Layers"
        if (totalAssigned == totalLayers) {
            binding.tvAllotmentSummary.setTextColor(Color.parseColor("#00E676"))
        } else {
            binding.tvAllotmentSummary.setTextColor(Color.parseColor("#FFC107"))
        }
    }

    private fun observeViewModel() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    viewModel.currentRole.collect { role ->
                        when (role) {
                            NodeRole.MASTER -> {
                                binding.tvCurrentRoleStatus.text = "Role: MASTER (Orchestrator + Worker)"
                                binding.tvCurrentRoleStatus.setTextColor(Color.parseColor("#B388FF"))
                                binding.cardSessionToken.visibility = View.VISIBLE
                                binding.cardLayerAllotment.visibility = View.VISIBLE
                                binding.layoutRoleButtons.visibility = View.GONE
                                binding.btnDisconnect.visibility = View.VISIBLE
                                binding.tvSeniorityInfo.text = "Local Seniority: Master Orchestrator (Node 0)"
                                refreshMasterQrDisplay()
                            }
                            NodeRole.SLAVE -> {
                                binding.tvCurrentRoleStatus.text = "Role: SLAVE (Mesh Worker)"
                                binding.tvCurrentRoleStatus.setTextColor(Color.parseColor("#80D8FF"))
                                binding.cardSessionToken.visibility = View.VISIBLE
                                binding.cardLayerAllotment.visibility = View.GONE
                                binding.layoutMasterQr.visibility = View.GONE
                                binding.layoutRoleButtons.visibility = View.GONE
                                binding.btnDisconnect.visibility = View.VISIBLE
                            }
                            else -> {
                                binding.tvCurrentRoleStatus.text = "Role: Unassigned (Idle)"
                                binding.tvCurrentRoleStatus.setTextColor(Color.parseColor("#AAAAAA"))
                                binding.cardSessionToken.visibility = View.GONE
                                binding.cardLayerAllotment.visibility = View.GONE
                                binding.layoutMasterQr.visibility = View.GONE
                                binding.layoutRoleButtons.visibility = View.VISIBLE
                                binding.btnDisconnect.visibility = View.GONE
                                binding.tvSeniorityInfo.text = "Local Seniority: - | Heartbeat: Idle"
                            }
                        }
                    }
                }

                launch {
                    viewModel.sessionToken.collect { token ->
                        binding.tvSessionToken.text = if (token.isNotEmpty()) token else "MESH-XXXX-XXXX"
                        refreshMasterQrDisplay()
                    }
                }

                launch {
                    viewModel.activeModel.collect {
                        refreshMasterQrDisplay()
                        if (binding.layoutManualAllotment.visibility == View.VISIBLE) {
                            refreshNodeSliders(viewModel.clusterState.value)
                        }
                    }
                }

                launch {
                    viewModel.localIp.collect { ip ->
                        binding.tvLocalIpPort.text = "IP: $ip:${viewModel.localPort} | Node ID: ${viewModel.localNodeId}"
                        refreshMasterQrDisplay()
                    }
                }

                launch {
                    viewModel.clusterState.collect { state ->
                        binding.tvClusterSummary.text = "${state.nodes.size} Nodes • Total ${state.totalLayers} Layers (${state.modelName})"
                        nodeAdapter.updateNodes(state.nodes)

                        // Sync allotment mode radio selection
                        if (state.layerAllotmentMode == org.meshllm.node.model.ClusterState.ALLOTMENT_MODE_MANUAL) {
                            if (!binding.rbAllotmentManual.isChecked) binding.rbAllotmentManual.isChecked = true
                            binding.layoutManualAllotment.visibility = View.VISIBLE
                            refreshNodeSliders(state)
                        } else {
                            if (!binding.rbAllotmentAuto.isChecked) binding.rbAllotmentAuto.isChecked = true
                            binding.layoutManualAllotment.visibility = View.GONE
                        }

                        // Update Sequential Pipeline visualizer
                        if (state.nodes.isEmpty()) {
                            binding.tvPipelineDiagram.text = "[Pipeline Idle]\nStart as Master or Join an existing network to form an inference pipeline."
                        } else {
                            val sb = StringBuilder()
                            sb.append("[Model: ${state.modelName} • ${state.totalLayers} Layers]\n")
                            state.nodes.sortedBy { it.joinOrder }.forEachIndexed { index, node ->
                                val block = node.assignedBlocks.firstOrNull()
                                val layerRange = if (block != null) "Layers ${block.startLayer}..${block.endLayer} (${block.layerCount}L)" else "0 Layers Assigned (Bypassed)"
                                val arrow = if (index < state.nodes.size - 1) " ➔ " else " ➔ [Output Head]"
                                sb.append("• [Node ${node.joinOrder} (${node.role.name})]: $layerRange$arrow\n")
                            }
                            binding.tvPipelineDiagram.text = sb.toString().trimEnd()
                        }

                        // If slave, show seniority & failover rank
                        if (viewModel.currentRole.value == NodeRole.SLAVE) {
                            val me = state.nodes.find { it.nodeId == viewModel.localNodeId }
                            if (me != null) {
                                val isNext = me.joinOrder == 1
                                binding.tvSeniorityInfo.text = "Local Join Order: #${me.joinOrder} ${if (isNext) "★ (NEXT IN LINE TO BECOME MASTER)" else "(Successor Backup)"}"
                            }
                        }
                    }
                }
            }
        }
    }

    private fun showJoinDialog() {
        val dialogView = LayoutInflater.from(requireContext()).inflate(R.layout.dialog_join_network, null)
        val etIp = dialogView.findViewById<EditText>(R.id.etMasterIp)
        val etPort = dialogView.findViewById<EditText>(R.id.etMasterPort)
        val etToken = dialogView.findViewById<EditText>(R.id.etSessionToken)
        val btnScanQr = dialogView.findViewById<Button>(R.id.btnScanQrDialog)

        var dialog: AlertDialog? = null

        btnScanQr.setOnClickListener {
            dialog?.dismiss()
            launchQrScanner()
        }

        dialog = AlertDialog.Builder(requireContext())
            .setView(dialogView)
            .setPositiveButton("Connect") { _, _ ->
                val ip = etIp.text.toString().trim()
                val port = etPort.text.toString().toIntOrNull() ?: 8080
                val token = etToken.text.toString().trim()

                if (ip.isEmpty() || token.isEmpty()) {
                    Toast.makeText(requireContext(), "Please specify Master IP and Session Token", Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }

                viewModel.joinAsSlave(ip, port, token) { success, msg ->
                    requireActivity().runOnUiThread {
                        if (success) {
                            Toast.makeText(requireContext(), "Connected as Slave node", Toast.LENGTH_SHORT).show()
                        } else {
                            Toast.makeText(requireContext(), "Connection failed: $msg", Toast.LENGTH_LONG).show()
                        }
                    }
                }
            }
            .setNegativeButton("Cancel") { d, _ ->
                d.dismiss()
            }
            .create()

        dialog.show()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
