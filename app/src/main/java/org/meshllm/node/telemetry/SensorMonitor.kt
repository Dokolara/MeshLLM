package org.meshllm.node.telemetry

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.meshllm.node.model.SensorMetrics
import java.io.RandomAccessFile

class SensorMonitor(private val context: Context) {

    private val _currentMetrics = MutableStateFlow(SensorMetrics())
    val currentMetrics: StateFlow<SensorMetrics> = _currentMetrics.asStateFlow()

    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager

    private var thermalStatus = 0
    private var lastCpuTotal: Long = 0
    private var lastCpuIdle: Long = 0

    init {
        setupThermalListener()
    }

    private fun setupThermalListener() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            try {
                powerManager.addThermalStatusListener { status ->
                    thermalStatus = status
                    updateMetrics()
                }
            } catch (_: Exception) {
                // Ignore if not supported on this device/ROM
            }
        }
    }

    private var lastBatteryCheckMs: Long = 0
    private var cachedBatteryPct: Int = 100
    private var cachedBatteryTempC: Float = 28.0f
    private var cachedBatteryVoltageMv: Int = 4000
    private var cachedIsCharging: Boolean = false
    private var cachedChargingSource: String = "Discharging"

    private var lastCpuSampleTime: Long = 0
    private var cachedCpuLoad: Float = 0.20f

    /**
     * Sample all hardware sensors and update state flow.
     */
    @Synchronized
    fun updateMetrics(): SensorMetrics {
        val now = System.currentTimeMillis()

        // 1. Battery & Temperature (throttled to avoid IPC churn on low-end CPUs)
        if (now - lastBatteryCheckMs >= 1500L) {
            try {
                val batteryIntent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
                val level = batteryIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: 100
                val scale = batteryIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: 100
                cachedBatteryPct = if (level >= 0 && scale > 0) (level * 100) / scale else 100

                val tempRaw = batteryIntent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 280) ?: 280
                cachedBatteryTempC = tempRaw / 10.0f

                val status = batteryIntent?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
                cachedIsCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                        status == BatteryManager.BATTERY_STATUS_FULL
                cachedBatteryVoltageMv = batteryIntent?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 4000) ?: 4000
                val plugged = batteryIntent?.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1) ?: 0
                cachedChargingSource = when (plugged) {
                    BatteryManager.BATTERY_PLUGGED_AC -> "AC"
                    BatteryManager.BATTERY_PLUGGED_USB -> "USB"
                    BatteryManager.BATTERY_PLUGGED_WIRELESS -> "Wireless"
                    else -> if (cachedIsCharging) "Charging" else "Discharging"
                }
                lastBatteryCheckMs = now
            } catch (_: Exception) {}
        }

        // 2. RAM Information
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        val availRamMb = memInfo.availMem / (1024 * 1024)
        val totalRamMb = memInfo.totalMem / (1024 * 1024)
        val ramUsagePct = (((totalRamMb - availRamMb).toDouble() / totalRamMb.coerceAtLeast(1)) * 100).toInt()

        // Measure App RAM usage with zero-overhead runtime heap measurement (avoids blocking Debug.getPss smaps scan)
        val rt = Runtime.getRuntime()
        val appRamMb = Math.round(((rt.totalMemory() - rt.freeMemory()) / (1024.0 * 1024.0)) * 10.0) / 10.0

        // 3. Thermal Status String
        val thermalString = when (thermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> "NORMAL"
            PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT_THROTTLE"
            PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE_THROTTLE"
            PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE_THROTTLE"
            PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL_THROTTLE"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN"
            else -> if (cachedBatteryTempC > 40f) "WARM" else "NORMAL"
        }

        // 4. CPU usage estimate from /proc/stat (throttled to avoid disk I/O thrashing)
        val cpuLoad = if (now - lastCpuSampleTime >= 1500L) {
            sampleCpuLoad().also {
                cachedCpuLoad = it
                lastCpuSampleTime = now
            }
        } else {
            cachedCpuLoad
        }

        val modelRam = try {
            org.meshllm.node.engine.NeuralModelRunner.shared.allocatedRamMb
        } catch (_: Throwable) {
            _currentMetrics.value.modelRamAllocatedMb
        }

        val metrics = SensorMetrics(
            batteryPercent = cachedBatteryPct,
            batteryTemperatureC = cachedBatteryTempC,
            batteryVoltageMv = cachedBatteryVoltageMv,
            isCharging = cachedIsCharging,
            chargingSource = cachedChargingSource,
            thermalStatus = thermalStatus,
            thermalStatusString = thermalString,
            availableRamMb = availRamMb,
            totalRamMb = totalRamMb,
            ramUsagePercent = ramUsagePct,
            cpuLoadEstimate = cpuLoad,
            networkRttMs = _currentMetrics.value.networkRttMs,
            modelRamAllocatedMb = modelRam,
            appRamAllocatedMb = appRamMb,
            timestamp = now
        )

        _currentMetrics.value = metrics
        return metrics
    }

    @Synchronized
    fun updateModelRam(ramMb: Double) {
        _currentMetrics.value = _currentMetrics.value.copy(modelRamAllocatedMb = ramMb)
    }

    fun updateNetworkRtt(rttMs: Long) {
        _currentMetrics.value = _currentMetrics.value.copy(networkRttMs = rttMs)
    }

    private fun sampleCpuLoad(): Float {
        return try {
            val file = java.io.File("/proc/stat")
            if (!file.exists()) return 0.20f
            java.io.BufferedReader(java.io.FileReader(file), 512).use { reader ->
                val load = reader.readLine() ?: return 0.20f
                val toks = load.split(" +".toRegex())
                if (toks.size >= 5) {
                    val idle = toks[4].toLong()
                    val total = toks.drop(1).take(7).mapNotNull { it.toLongOrNull() }.sum()
                    val diffIdle = idle - lastCpuIdle
                    val diffTotal = total - lastCpuTotal
                    lastCpuIdle = idle
                    lastCpuTotal = total
                    if (diffTotal > 0) {
                        ((diffTotal - diffIdle).toFloat() / diffTotal).coerceIn(0f, 1f)
                    } else {
                        0.20f
                    }
                } else {
                    0.20f
                }
            }
        } catch (_: Exception) {
            0.20f
        }
    }
}
