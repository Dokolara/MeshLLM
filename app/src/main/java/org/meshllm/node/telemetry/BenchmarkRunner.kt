package org.meshllm.node.telemetry

import android.content.Context
import android.os.Environment
import android.util.Log
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.meshllm.node.model.BenchmarkProfile
import java.io.File

class BenchmarkRunner(private val context: Context) {

    private val gson = Gson()

    // File locations checked for cached benchmark profile
    private val internalCacheFile: File
        get() = File(context.filesDir, CACHE_FILENAME)

    private val externalCacheFile: File
        get() = File(Environment.getExternalStorageDirectory(), CACHE_FILENAME)

    companion object {
        private const val TAG = "BenchmarkRunner"
        const val CACHE_FILENAME = "meshllm_bench_profile.json"
    }

    /**
     * Requirement 5: Check if benchmark cache file exists.
     * If found, returns the cached profile immediately without running tests.
     */
    fun getCachedProfileIfAvailable(): BenchmarkProfile? {
        // 1. Check internal storage first
        if (internalCacheFile.exists() && internalCacheFile.length() > 0) {
            try {
                val json = internalCacheFile.readText()
                val profile = gson.fromJson(json, BenchmarkProfile::class.java)
                Log.i(TAG, "Found cached benchmark profile at: ${internalCacheFile.absolutePath}")
                return profile.copy(isFromDevCache = true, cacheFilePath = internalCacheFile.absolutePath)
            } catch (e: Exception) {
                Log.e(TAG, "Error parsing internal cache file", e)
            }
        }

        // 2. Check external storage / sdcard fallback
        try {
            if (externalCacheFile.exists() && externalCacheFile.length() > 0) {
                val json = externalCacheFile.readText()
                val profile = gson.fromJson(json, BenchmarkProfile::class.java)
                Log.i(TAG, "Found cached benchmark profile at: ${externalCacheFile.absolutePath}")
                return profile.copy(isFromDevCache = true, cacheFilePath = externalCacheFile.absolutePath)
            }
        } catch (_: Exception) {
            // May lack external storage permission on Android 11+
        }

        return null
    }

    /**
     * Run full compute (GEMM GFLOPS) and memory bandwidth benchmarks.
     * Automatically writes results to the cache file.
     */
    suspend fun executeBenchmark(
        onProgress: (phase: String, progressPct: Int) -> Unit
    ): BenchmarkProfile = withContext(Dispatchers.Default) {
        onProgress("Profiling CPU architecture and cores...", 10)
        val cores = Runtime.getRuntime().availableProcessors()
        val totalRamMb = Runtime.getRuntime().maxMemory() / (1024 * 1024)

        // 1. Matrix Multiplication (GEMM) benchmark
        onProgress("Running GEMM compute benchmark (GFLOPS)...", 30)
        val matrixSize = 256
        val matrixA = FloatArray(matrixSize * matrixSize) { 0.5f }
        val matrixB = FloatArray(matrixSize * matrixSize) { 0.8f }
        val matrixC = FloatArray(matrixSize * matrixSize)

        val gemmStart = System.nanoTime()
        val iterations = 15
        for (iter in 0 until iterations) {
            for (i in 0 until matrixSize) {
                for (k in 0 until matrixSize) {
                    val aVal = matrixA[i * matrixSize + k]
                    for (j in 0 until matrixSize) {
                        matrixC[i * matrixSize + j] += aVal * matrixB[k * matrixSize + j]
                    }
                }
            }
        }
        val gemmElapsedNs = System.nanoTime() - gemmStart
        val totalOps = 2.0 * matrixSize * matrixSize * matrixSize * iterations
        val gflops = (totalOps / (gemmElapsedNs / 1_000_000_000.0)) / 1_000_000_000.0

        onProgress("Measuring memory bandwidth...", 70)
        // 2. Memory bandwidth test (Sequential allocation & traversal)
        val memBufferSize = 16 * 1024 * 1024 // 16MB buffer
        val memBuffer = ByteArray(memBufferSize)
        val memStart = System.nanoTime()
        val memIters = 20
        var checksum: Byte = 0
        for (iter in 0 until memIters) {
            for (i in 0 until memBufferSize step 64) {
                memBuffer[i] = (checksum + iter).toByte()
                checksum = (checksum + memBuffer[i]).toByte()
            }
        }
        val memElapsedNs = System.nanoTime() - memStart
        val totalBytesMoved = memBufferSize.toDouble() * memIters
        val memoryMbPerSec = (totalBytesMoved / (1024.0 * 1024.0)) / (memElapsedNs / 1_000_000_000.0)

        onProgress("Computing composite capacity score...", 90)
        // Composite score weights: compute (70%), memory bandwidth (20%), core parallelism (10%)
        val composite = (gflops * 15.0) + (memoryMbPerSec / 100.0) + (cores * 3.0)

        val profile = BenchmarkProfile(
            computeScoreGflops = Math.round(gflops * 100.0) / 100.0,
            memoryBandwidthMbPerSec = Math.round(memoryMbPerSec * 10.0) / 10.0,
            cpuCoreCount = cores,
            totalRamMb = totalRamMb,
            compositeCapacityScore = Math.round(composite * 10.0) / 10.0,
            benchmarkTimestamp = System.currentTimeMillis(),
            isFromDevCache = false,
            cacheFilePath = internalCacheFile.absolutePath
        )

        // Save to cache file
        saveProfileToCache(profile)

        onProgress("Benchmark completed!", 100)
        profile
    }

    /**
     * Persist profile to disk so subsequent launches skip benchmarks.
     */
    fun saveProfileToCache(profile: BenchmarkProfile) {
        try {
            val json = gson.toJson(profile)
            internalCacheFile.writeText(json)
            Log.i(TAG, "Saved benchmark profile to cache: ${internalCacheFile.absolutePath}")

            // Attempt external backup if possible
            try {
                externalCacheFile.writeText(json)
            } catch (_: Exception) {}
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write benchmark cache file", e)
        }
    }

    /**
     * Dev mode quick override: saves a mock high-performance or balanced profile.
     */
    fun saveDevMockProfile(gflops: Double = 12.5, memBw: Double = 4200.0): BenchmarkProfile {
        val cores = Runtime.getRuntime().availableProcessors()
        val composite = (gflops * 15.0) + (memBw / 100.0) + (cores * 3.0)
        val profile = BenchmarkProfile(
            computeScoreGflops = gflops,
            memoryBandwidthMbPerSec = memBw,
            cpuCoreCount = cores,
            totalRamMb = Runtime.getRuntime().maxMemory() / (1024 * 1024),
            compositeCapacityScore = composite,
            benchmarkTimestamp = System.currentTimeMillis(),
            isFromDevCache = true,
            cacheFilePath = internalCacheFile.absolutePath
        )
        saveProfileToCache(profile)
        return profile
    }

    fun clearCache(): Boolean {
        var deleted = internalCacheFile.delete()
        try {
            if (externalCacheFile.exists()) {
                deleted = externalCacheFile.delete() || deleted
            }
        } catch (_: Exception) {}
        return deleted
    }
}
