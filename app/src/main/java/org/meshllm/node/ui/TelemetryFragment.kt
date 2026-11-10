package org.meshllm.node.ui

import android.app.AlertDialog
import android.graphics.Color
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.meshllm.node.R
import org.meshllm.node.databinding.FragmentTelemetryBinding
import org.meshllm.node.telemetry.BenchmarkRunner

class TelemetryFragment : Fragment() {

    private var _binding: FragmentTelemetryBinding? = null
    private val binding get() = _binding!!

    private val viewModel: MeshClusterViewModel by activityViewModels()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentTelemetryBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        setupButtons()
        observeViewModel()
    }

    private fun setupButtons() {
        binding.btnRunBenchmark.setOnClickListener {
            runBenchmarkSuite()
        }

        binding.btnDevMode.setOnClickListener {
            showDevModeDialog()
        }
    }

    private fun observeViewModel() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.sensorMetrics.collect { metrics ->
                binding.tvBatteryPct.text = "${metrics.batteryPercent}%"
                binding.tvChargingState.text = if (metrics.isCharging) "Charging (${metrics.chargingSource})" else "Discharging"

                binding.tvBatteryTemp.text = "${metrics.batteryTemperatureC}°C"
                when {
                    metrics.batteryTemperatureC > 42.0 -> binding.tvBatteryTemp.setTextColor(Color.parseColor("#FF5252"))
                    metrics.batteryTemperatureC > 38.0 -> binding.tvBatteryTemp.setTextColor(Color.parseColor("#FFA726"))
                    else -> binding.tvBatteryTemp.setTextColor(Color.parseColor("#66BB6A"))
                }

                binding.tvBatteryVoltage.text = "Voltage: ${metrics.batteryVoltageMv} mV"

                binding.tvThermalStatus.text = "Thermal: ${metrics.thermalStatusString}"
                val healthMult = Math.round(metrics.computeHealthFactor() * 100.0) / 100.0
                binding.tvHealthScore.text = "Health: ${healthMult}x"

                val throttleMs = when {
                    metrics.batteryTemperatureC > 44.0 -> "200ms (Heavy Throttle)"
                    metrics.batteryTemperatureC > 40.0 -> "50ms (Light Throttle)"
                    else -> "0ms (Normal)"
                }
                binding.tvGovernorDetails.text = "Throttle Delay: $throttleMs • Priority: THREAD_PRIORITY_BACKGROUND • Dynamic Health Factor: ${healthMult}x"

                // RAM Breakdown (Red=LLM, Grey=System & Other Apps, Empty=Free)
                val usedRamMb = Math.round((metrics.totalRamMb - metrics.availableRamMb) * 10.0) / 10.0
                binding.tvRamTotal.text = "Total: ${metrics.totalRamMb} MB • Used: ${usedRamMb} MB"

                val llmRamPercent = metrics.llmRamPercent
                val sysRamPercent = metrics.systemRamPercent
                val emptyRamPercent = (100f - llmRamPercent - sysRamPercent).coerceAtLeast(0f)

                setSegmentWeight(binding.viewRamLlm, llmRamPercent)
                setSegmentWeight(binding.viewRamSystem, sysRamPercent)
                setSegmentWeight(binding.viewRamEmpty, emptyRamPercent)

                val llmMb = Math.round(metrics.modelRamAllocatedMb * 10.0) / 10.0
                val sysMb = Math.round(metrics.systemAndOtherAppsRamMb * 10.0) / 10.0

                binding.tvRamLlm.text = "● LLM: ${llmMb} MB (${String.format("%.1f", llmRamPercent)}%)"
                binding.tvRamSystem.text = "● System & Other Apps: ${sysMb} MB (${String.format("%.1f", sysRamPercent)}%)"

                // CPU Breakdown (Red=LLM, Grey=System & Other Apps, Empty=Idle)
                val cores = Runtime.getRuntime().availableProcessors()
                val totalCpu = String.format("%.1f", metrics.totalCpuPercent)
                binding.tvCpuCores.text = "$cores Cores • Load: $totalCpu%"

                val llmCpuPercent = metrics.llmCpuPercent
                val sysCpuPercent = metrics.systemCpuPercent
                val emptyCpuPercent = (100f - llmCpuPercent - sysCpuPercent).coerceAtLeast(0f)

                setSegmentWeight(binding.viewCpuLlm, llmCpuPercent)
                setSegmentWeight(binding.viewCpuSystem, sysCpuPercent)
                setSegmentWeight(binding.viewCpuEmpty, emptyCpuPercent)

                binding.tvCpuLlm.text = "● LLM: ${String.format("%.1f", llmCpuPercent)}%"
                binding.tvCpuSystem.text = "● System & Other Apps: ${String.format("%.1f", sysCpuPercent)}%"
            }
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.benchmarkProfile.collect { profile ->
                if (profile != null) {
                    binding.tvBenchmarkStatus.text = "Profile Loaded from Cache (Tests Skipped)"
                    binding.tvBenchmarkStatus.setTextColor(Color.parseColor("#00E676"))
                    binding.tvBenchmarkSummary.text = "Compute: ${profile.computeScoreGflops} GFLOPS • Mem: ${profile.memoryBandwidthMbPerSec} MB/s • Cores: ${profile.cpuCoreCount} • Capacity: ${profile.compositeCapacityScore} pts"
                    binding.btnRunBenchmark.text = "Re-Run Benchmark"
                } else {
                    binding.tvBenchmarkStatus.text = "Not Benchmarked (First Launch)"
                    binding.tvBenchmarkStatus.setTextColor(Color.parseColor("#FFAB00"))
                    binding.tvBenchmarkSummary.text = "Run benchmark to profile device compute and capacity"
                    binding.btnRunBenchmark.text = "Run Benchmark"
                }
            }
        }
    }

    private fun runBenchmarkSuite() {
        binding.pbBenchmark.visibility = View.VISIBLE
        binding.btnRunBenchmark.isEnabled = false

        viewLifecycleOwner.lifecycleScope.launch {
            val profile = viewModel.benchmarkRunner.executeBenchmark { phase, pct ->
                lifecycleScope.launch(Dispatchers.Main) {
                    binding.tvBenchmarkStatus.text = phase
                    binding.pbBenchmark.progress = pct
                }
            }

            viewModel.checkForCachedBenchmarkProfile()
            binding.pbBenchmark.visibility = View.GONE
            binding.btnRunBenchmark.isEnabled = true
            Toast.makeText(requireContext(), "Benchmark complete and saved to cache", Toast.LENGTH_SHORT).show()
        }
    }

    private fun showDevModeDialog() {
        val dialogView = LayoutInflater.from(requireContext()).inflate(R.layout.dialog_dev_mode, null)
        val etGflops = dialogView.findViewById<EditText>(R.id.etDevGflops)
        val etMemBw = dialogView.findViewById<EditText>(R.id.etDevMemBw)
        val tvPath = dialogView.findViewById<TextView>(R.id.tvCachePathInfo)
        val btnSave = dialogView.findViewById<Button>(R.id.btnSaveDevProfile)
        val btnClear = dialogView.findViewById<Button>(R.id.btnClearCacheFile)

        tvPath.text = "Cache File: ${requireContext().filesDir.absolutePath}/${BenchmarkRunner.CACHE_FILENAME}"

        val dialog = AlertDialog.Builder(requireContext())
            .setView(dialogView)
            .create()

        btnSave.setOnClickListener {
            val gflops = etGflops.text.toString().toDoubleOrNull() ?: 15.0
            val memBw = etMemBw.text.toString().toDoubleOrNull() ?: 4500.0
            viewModel.benchmarkRunner.saveDevMockProfile(gflops, memBw)
            viewModel.checkForCachedBenchmarkProfile()
            dialog.dismiss()
            Toast.makeText(requireContext(), "Dev mode: Profile saved to cache file", Toast.LENGTH_SHORT).show()
        }

        btnClear.setOnClickListener {
            viewModel.benchmarkRunner.clearCache()
            viewModel.checkForCachedBenchmarkProfile()
            dialog.dismiss()
            Toast.makeText(requireContext(), "Benchmark cache file deleted", Toast.LENGTH_SHORT).show()
        }

        dialog.show()
    }

    private fun setSegmentWeight(view: View, weightPercent: Float) {
        val lp = view.layoutParams as? android.widget.LinearLayout.LayoutParams ?: return
        val target = weightPercent.coerceAtLeast(0.01f)
        if (kotlin.math.abs(lp.weight - target) >= 0.5f) {
            lp.weight = target
            view.layoutParams = lp
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
