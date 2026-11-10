package org.meshllm.node.ui

import android.graphics.Color
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import org.meshllm.node.R
import androidx.recyclerview.widget.DiffUtil
import org.meshllm.node.model.NodeInfo
import org.meshllm.node.model.NodeRole

class NodeAdapter : RecyclerView.Adapter<NodeAdapter.NodeViewHolder>() {

    private val nodes = mutableListOf<NodeInfo>()

    fun updateNodes(newNodes: List<NodeInfo>) {
        val diffCallback = object : DiffUtil.Callback() {
            override fun getOldListSize(): Int = nodes.size
            override fun getNewListSize(): Int = newNodes.size
            override fun areItemsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean =
                nodes[oldItemPosition].nodeId == newNodes[newItemPosition].nodeId
            override fun areContentsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean {
                val old = nodes[oldItemPosition]
                val new = newNodes[newItemPosition]
                return old.assignedLayersDescription == new.assignedLayersDescription &&
                        old.role == new.role &&
                        old.isAlive == new.isAlive &&
                        old.joinOrder == new.joinOrder &&
                        old.metrics.batteryPercent == new.metrics.batteryPercent &&
                        old.metrics.batteryTemperatureC == new.metrics.batteryTemperatureC &&
                        old.metrics.thermalStatus == new.metrics.thermalStatus &&
                        old.metrics.modelRamAllocatedMb == new.metrics.modelRamAllocatedMb
            }
        }
        val diffResult = DiffUtil.calculateDiff(diffCallback)
        nodes.clear()
        nodes.addAll(newNodes.map { it.copy(assignedBlocks = it.assignedBlocks.toList()) })
        diffResult.dispatchUpdatesTo(this)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): NodeViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_node, parent, false)
        return NodeViewHolder(view)
    }

    override fun onBindViewHolder(holder: NodeViewHolder, position: Int) {
        holder.bind(nodes[position])
    }

    override fun getItemCount(): Int = nodes.size

    class NodeViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvDeviceName: TextView = itemView.findViewById(R.id.tvDeviceName)
        private val tvRoleBadge: TextView = itemView.findViewById(R.id.tvRoleBadge)
        private val tvJoinOrderAndIp: TextView = itemView.findViewById(R.id.tvJoinOrderAndIp)
        private val tvAssignedLayers: TextView = itemView.findViewById(R.id.tvAssignedLayers)
        private val tvNodeBattery: TextView = itemView.findViewById(R.id.tvNodeBattery)
        private val tvNodeThermal: TextView = itemView.findViewById(R.id.tvNodeThermal)
        private val tvNodeCapacity: TextView = itemView.findViewById(R.id.tvNodeCapacity)

        private val nodeRamLlm: View = itemView.findViewById(R.id.nodeRamLlm)
        private val nodeRamSystem: View = itemView.findViewById(R.id.nodeRamSystem)
        private val nodeRamEmpty: View = itemView.findViewById(R.id.nodeRamEmpty)

        private val tvNodeLlmRam: TextView = itemView.findViewById(R.id.tvNodeLlmRam)
        private val tvNodeSysRam: TextView = itemView.findViewById(R.id.tvNodeSysRam)

        fun bind(node: NodeInfo) {
            tvDeviceName.text = node.deviceName

            if (node.role == NodeRole.MASTER) {
                tvRoleBadge.text = "MASTER"
                tvRoleBadge.setBackgroundColor(Color.parseColor("#311B92"))
                tvRoleBadge.setTextColor(Color.parseColor("#B388FF"))
            } else {
                tvRoleBadge.text = "SLAVE"
                tvRoleBadge.setBackgroundColor(Color.parseColor("#01579B"))
                tvRoleBadge.setTextColor(Color.parseColor("#80D8FF"))
            }

            val orderDesc = if (node.joinOrder == 0) {
                "Order #0 (Primary Master)"
            } else if (node.joinOrder == 1) {
                "Order #1 (Next in line for Failover)"
            } else {
                "Order #${node.joinOrder}"
            }
            tvJoinOrderAndIp.text = "$orderDesc • ${node.ipAddress}:${node.port}"

            tvAssignedLayers.text = node.assignedLayersDescription

            val chargeStatus = if (node.metrics.isCharging) "(Charging)" else ""
            tvNodeBattery.text = "Bat: ${node.metrics.batteryPercent}% (${node.metrics.batteryTemperatureC}°C) $chargeStatus"
            tvNodeThermal.text = "Thermal: ${node.metrics.thermalStatusString}"

            val effectiveCap = Math.round(node.benchmark.getEffectiveCapacity(node.metrics)).toInt()
            tvNodeCapacity.text = "Cap: $effectiveCap pts"

            // Resource distribution breakdown (Red=LLM, Grey=System & Other Apps, Empty=Free)
            val metrics = node.metrics
            val llmPct = metrics.llmRamPercent
            val sysPct = metrics.systemRamPercent
            val emptyPct = (100f - llmPct - sysPct).coerceAtLeast(0f)

            setSegmentWeight(nodeRamLlm, llmPct)
            setSegmentWeight(nodeRamSystem, sysPct)
            setSegmentWeight(nodeRamEmpty, emptyPct)

            val llmMb = Math.round(metrics.modelRamAllocatedMb).toInt()
            val sysMb = Math.round(metrics.systemAndOtherAppsRamMb).toInt()

            tvNodeLlmRam.text = "LLM: ${llmMb}MB"
            tvNodeSysRam.text = "Sys: ${sysMb}MB"

            if (!node.isAlive) {
                tvRoleBadge.text = "DISCONNECTED"
                tvRoleBadge.setBackgroundColor(Color.parseColor("#B71C1C"))
                tvRoleBadge.setTextColor(Color.parseColor("#FF8A80"))
            }
        }

        private fun setSegmentWeight(view: View, weightPercent: Float) {
            val lp = view.layoutParams as? android.widget.LinearLayout.LayoutParams ?: return
            val target = weightPercent.coerceAtLeast(0.01f)
            if (kotlin.math.abs(lp.weight - target) >= 0.5f) {
                lp.weight = target
                view.layoutParams = lp
            }
        }
    }
}
