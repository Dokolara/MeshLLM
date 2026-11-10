package org.meshllm.node.model

data class SensorMetrics(
    val batteryPercent: Int = 100,
    val batteryTemperatureC: Float = 28.0f,
    val batteryVoltageMv: Int = 4000,
    val isCharging: Boolean = false,
    val chargingSource: String = "Discharging",
    val thermalStatus: Int = 0,
    val thermalStatusString: String = "NORMAL",
    val availableRamMb: Long = 2048,
    val totalRamMb: Long = 8192,
    val ramUsagePercent: Int = 50,
    val cpuLoadEstimate: Float = 0.15f,
    val networkRttMs: Long = 10,
    val modelRamAllocatedMb: Double = 0.0,
    val appRamAllocatedMb: Double = 0.0,
    val timestamp: Long = System.currentTimeMillis()
) {
    /**
     * Other apps and system resource usage in MB (Grey).
     * Total used RAM on the device minus LLM model RAM.
     */
    val systemAndOtherAppsRamMb: Double
        get() = (totalRamMb.toDouble() - availableRamMb.toDouble() - modelRamAllocatedMb).coerceAtLeast(0.0)

    val freeRamMb: Double
        get() = availableRamMb.toDouble().coerceAtLeast(0.0)

    // RAM Percentages: Only LLM (Red) and System & Other Apps (Grey)
    val llmRamPercent: Float
        get() = ((modelRamAllocatedMb / totalRamMb.coerceAtLeast(1).toDouble()) * 100.0).toFloat().coerceIn(0f, 100f)

    val systemRamPercent: Float
        get() = ((systemAndOtherAppsRamMb / totalRamMb.coerceAtLeast(1).toDouble()) * 100.0).toFloat().coerceIn(0f, 100f)

    val appRamPercent: Float
        get() = 0f

    val freeRamPercent: Float
        get() = ((freeRamMb / totalRamMb.coerceAtLeast(1).toDouble()) * 100.0).toFloat().coerceIn(0f, 100f)

    // CPU Breakdown (Percentages 0..100): Only LLM (Red) and System & Other Apps (Grey)
    val totalCpuPercent: Float
        get() = (cpuLoadEstimate * 100f).coerceIn(0f, 100f)

    val llmCpuPercent: Float
        get() = if (modelRamAllocatedMb > 0.0) {
            (totalCpuPercent * 0.65f).coerceIn(0f, totalCpuPercent)
        } else 0f

    val systemCpuPercent: Float
        get() = (totalCpuPercent - llmCpuPercent).coerceAtLeast(0f)

    val appCpuPercent: Float
        get() = 0f

    val idleCpuPercent: Float
        get() = (100f - totalCpuPercent).coerceAtLeast(0f)
    /**
     * Compute dynamic health & capability multiplier (0.0 to 1.0).
     * Degrades severely if device overheats or battery drops while discharging.
     */
    fun computeHealthFactor(): Double {
        // CPU utilization penalty
        val cpuFactor = when {
            cpuLoadEstimate > 0.85f -> 0.25
            cpuLoadEstimate > 0.70f -> 0.50
            cpuLoadEstimate > 0.50f -> 0.75
            else -> 1.0
        }

        // Thermal penalty
        val thermalFactor = when {
            thermalStatus >= 4 -> 0.15 // CRITICAL
            thermalStatus >= 3 -> 0.35 // SEVERE
            thermalStatus >= 2 -> 0.60 // MODERATE
            thermalStatus >= 1 -> 0.85 // LIGHT
            batteryTemperatureC > 44f -> 0.20
            batteryTemperatureC > 40f -> 0.50
            batteryTemperatureC > 36f -> 0.80
            else -> 1.0
        }

        // Battery penalty
        val batteryFactor = when {
            isCharging -> 1.0
            batteryPercent < 15 -> 0.15
            batteryPercent < 25 -> 0.40
            batteryPercent < 40 -> 0.70
            else -> 1.0
        }

        // RAM memory penalty
        val memoryFactor = when {
            ramUsagePercent > 85 || availableRamMb < 400 -> 0.20
            ramUsagePercent > 70 || availableRamMb < 800 -> 0.50
            ramUsagePercent > 55 || availableRamMb < 1500 -> 0.80
            else -> 1.0
        }

        // Combined health factor: determines proportional layer capacity in the cluster
        return (thermalFactor * 0.25 + batteryFactor * 0.25 + memoryFactor * 0.30 + cpuFactor * 0.20).coerceIn(0.1, 1.0)
    }
}
