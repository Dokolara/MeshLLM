package org.meshllm.node.ui

import android.graphics.Color
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import org.meshllm.node.databinding.ItemModelBinding
import org.meshllm.node.model.ModelSpec

import androidx.recyclerview.widget.DiffUtil

class ModelAdapter(
    private var models: List<ModelSpec> = emptyList(),
    private var activeModelId: String = "",
    private val isModelDownloaded: (ModelSpec) -> Boolean = { false },
    private val isModelAvailableOnMaster: (ModelSpec) -> Boolean = { false },
    private val onDownloadClicked: (ModelSpec) -> Unit = {},
    val onModelSelected: (ModelSpec) -> Unit
) : RecyclerView.Adapter<ModelAdapter.ModelViewHolder>() {

    fun updateData(newModels: List<ModelSpec>, currentActiveModelId: String) {
        this.models = ArrayList(newModels)
        this.activeModelId = currentActiveModelId
        notifyDataSetChanged()
    }

    fun notifyModelStatusChanged(modelId: String) {
        val idx = models.indexOfFirst { it.id.equals(modelId, ignoreCase = true) }
        if (idx != -1) {
            notifyItemChanged(idx)
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ModelViewHolder {
        val binding = ItemModelBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return ModelViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ModelViewHolder, position: Int) {
        holder.bind(models[position], models[position].id.equals(activeModelId, ignoreCase = true))
    }

    override fun getItemCount(): Int = models.size

    inner class ModelViewHolder(private val binding: ItemModelBinding) : RecyclerView.ViewHolder(binding.root) {

        fun bind(model: ModelSpec, isActive: Boolean) {
            binding.tvModelName.text = model.name
            val repoText = if (!model.huggingFaceRepo.isNullOrEmpty()) " • ${model.huggingFaceRepo}" else ""
            binding.tvModelProvider.text = "${model.provider}$repoText"
            binding.tvModelCategory.text = model.category

            if (model.isOnlineHfModel) {
                binding.tvStarterBadge.visibility = View.VISIBLE
                binding.tvStarterBadge.text = "HF HUB"
                binding.tvStarterBadge.setTextColor(Color.parseColor("#818CF8"))
            } else if (model.isStarterTestModel) {
                binding.tvStarterBadge.visibility = View.VISIBLE
                binding.tvStarterBadge.text = "STARTER"
                binding.tvStarterBadge.setTextColor(Color.parseColor("#10B981"))
            } else {
                binding.tvStarterBadge.visibility = View.GONE
            }
            binding.tvModelParams.text = "${model.paramCountStr} Params"
            binding.tvModelLayers.text = "${model.totalLayers} Layers"
            binding.tvQuantBadge.text = model.quantization
            binding.tvModelRam.text = model.fileSizeDisplay
            binding.tvModelDim.text = "${model.hiddenDim} Dim"
            binding.tvModelDesc.text = model.description
            binding.tvOomWarningBanner.visibility = if (model.singleDeviceOomWarning) View.VISIBLE else View.GONE

            val downloaded = isModelDownloaded(model)
            val onMaster = isModelAvailableOnMaster(model)

            if (downloaded) {
                binding.tvItemDownloadStatus.text = "Downloaded (${model.fileSizeDisplay})"
                binding.tvItemDownloadStatus.setTextColor(Color.parseColor("#00E676"))
                binding.btnItemDownload.visibility = View.GONE
            } else if (onMaster) {
                binding.tvItemDownloadStatus.text = "Available on Master (Fast Mesh P2P)"
                binding.tvItemDownloadStatus.setTextColor(Color.parseColor("#00E5FF"))
                binding.btnItemDownload.visibility = View.VISIBLE
                binding.btnItemDownload.text = "P2P Download"
                binding.btnItemDownload.setTextColor(Color.parseColor("#00E5FF"))
                binding.btnItemDownload.setOnClickListener {
                    onDownloadClicked(model)
                }
            } else {
                binding.tvItemDownloadStatus.text = "Ready to fetch (${model.fileSizeDisplay})"
                binding.tvItemDownloadStatus.setTextColor(Color.parseColor("#FF9800"))
                binding.btnItemDownload.visibility = View.VISIBLE
                binding.btnItemDownload.text = "Download"
                binding.btnItemDownload.setTextColor(Color.parseColor("#00E676"))
                binding.btnItemDownload.setOnClickListener {
                    onDownloadClicked(model)
                }
            }

            if (isActive) {
                binding.tvActiveStatus.visibility = View.VISIBLE
                binding.btnSelectModel.isEnabled = false
                binding.btnSelectModel.text = "Loaded"
                binding.cardModelItem.strokeColor = Color.parseColor("#00E676")
                binding.cardModelItem.strokeWidth = 2
            } else {
                binding.tvActiveStatus.visibility = View.GONE
                binding.btnSelectModel.isEnabled = true
                binding.btnSelectModel.text = "Load into Mesh"
                binding.cardModelItem.strokeColor = Color.parseColor("#3A3A3A")
                binding.cardModelItem.strokeWidth = 1

                binding.btnSelectModel.setOnClickListener {
                    onModelSelected(model)
                }
                binding.cardModelItem.setOnClickListener {
                    onModelSelected(model)
                }
            }
        }
    }
}
