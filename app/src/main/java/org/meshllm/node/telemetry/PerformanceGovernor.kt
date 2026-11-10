package org.meshllm.node.telemetry

import android.os.PowerManager
import android.os.Process
import kotlinx.coroutines.delay
import org.meshllm.node.model.SensorMetrics

class PerformanceGovernor {

    /**
     * Set thread priority to background so UI and other apps are never starved.
     */
    fun applyBackgroundPriority() {
        try {
            Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
        } catch (_: Exception) {}
    }

    /**
     * Requirement 10: dynamically limit compute concurrency based on device health.
     * Prevents saturated CPU load when phone is in active daily use.
     */
    fun determineConcurrency(metrics: SensorMetrics): Int {
        val totalCores = Runtime.getRuntime().availableProcessors()
        return when {
            // Severe thermal throttling or critically low battery -> minimum background work
            metrics.thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE ||
                    (!metrics.isCharging && metrics.batteryPercent < 15) -> 1

            // Moderate thermal or low battery -> use at most 2 cores or 25% of cores
            metrics.thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE ||
                    (!metrics.isCharging && metrics.batteryPercent < 30) -> (totalCores / 4).coerceIn(1, 2)

            // Plugged in or cool -> up to half of cores, leaving the other half for system/UI
            metrics.isCharging -> (totalCores / 2).coerceIn(2, 4)

            // Normal battery usage -> balanced
            else -> (totalCores / 3).coerceIn(1, 3)
        }
    }

    /**
     * Yield CPU cycles and introduce throttling delays between tensor operations
     * if thermal throttling or high load is detected.
     */
    suspend fun throttleIfOverheating(metrics: SensorMetrics) {
        when {
            metrics.thermalStatus >= PowerManager.THERMAL_STATUS_CRITICAL ||
                    metrics.batteryTemperatureC > 45f -> {
                // Emergency cooldown delay
                delay(80)
            }
            metrics.thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE ||
                    metrics.batteryTemperatureC > 42f -> {
                delay(40)
            }
            metrics.thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE ||
                    metrics.batteryTemperatureC > 39f -> {
                delay(15)
            }
            !metrics.isCharging && metrics.batteryPercent < 20 -> {
                delay(20)
            }
            else -> {
                // Normal operation: brief yield to allow OS UI thread scheduling
                Thread.yield()
            }
        }
    }

    /**
     * Evaluates whether this node is under too much pressure and must offload layers.
     */
    fun shouldRequestLayerOffload(metrics: SensorMetrics): Boolean {
        return metrics.thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE ||
                metrics.batteryTemperatureC >= 43.5f ||
                (!metrics.isCharging && metrics.batteryPercent <= 12) ||
                metrics.availableRamMb < 350
    }
}
