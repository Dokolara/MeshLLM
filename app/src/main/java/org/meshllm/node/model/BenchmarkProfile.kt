package org.meshllm.node.model

data class BenchmarkProfile(
    val computeScoreGflops: Double = 0.0,
    val memoryBandwidthMbPerSec: Double = 0.0,
    val cpuCoreCount: Int = Runtime.getRuntime().availableProcessors(),
    val totalRamMb: Long = 0,
    val compositeCapacityScore: Double = 0.0,
    val benchmarkTimestamp: Long = System.currentTimeMillis(),
    val isFromDevCache: Boolean = false,
    val cacheFilePath: String? = null
) {
    /**
     * Effective compute capacity dynamically scaled by real-time sensor health.
     */
    fun getEffectiveCapacity(metrics: SensorMetrics): Double {
        val health = metrics.computeHealthFactor()
        return (compositeCapacityScore * health).coerceAtLeast(1.0)
    }
}
