package org.meshllm.node.telemetry

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.meshllm.node.model.ModelSpec
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit

data class PeerDownloadSource(
    val host: String,
    val port: Int,
    val sessionToken: String? = null
)

sealed class DownloadState {
    object Idle : DownloadState()
    data class Downloading(
        val modelId: String,
        val percent: Int,
        val bytesRead: Long,
        val totalBytes: Long,
        val speedMbPerSec: Double,
        val source: String = "Internet"
    ) : DownloadState()
    data class Ready(val modelId: String, val file: File, val sizeMb: Double) : DownloadState()
    data class Failed(val modelId: String, val error: String) : DownloadState()
}

class ModelDownloadManager(
    private val context: Context,
    private val scope: CoroutineScope
) {
    companion object {
        private const val TAG = "ModelDownloadManager"
    }

    var peerDownloadSourceProvider: (() -> PeerDownloadSource?)? = null

    private val httpClient = OkHttpClient.Builder()
        .followRedirects(true)
        .followSslRedirects(true)
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .build()

    private val _downloadStates = MutableStateFlow<Map<String, DownloadState>>(emptyMap())
    val downloadStates: StateFlow<Map<String, DownloadState>> = _downloadStates.asStateFlow()

    private val activeDownloadJobs = ConcurrentHashMap<String, Job>()

    private val modelsDir: File
        get() = File(context.filesDir, "models").apply { if (!exists()) mkdirs() }

    init {
        extractBundledAssetsIfPresent()
    }

    private fun extractBundledAssetsIfPresent() {
        scope.launch(Dispatchers.IO) {
            val assetManager = context.assets
            val assetList = try { assetManager.list("")?.toList() ?: emptyList() } catch (_: Throwable) { emptyList() }

            // Extract tokenizers
            for (tok in listOf("tok512.bin", "tokenizer.bin")) {
                if (tok in assetList) {
                    val destTok = File(modelsDir, tok)
                    if (!destTok.exists() || destTok.length() < 100L) {
                        try {
                            assetManager.open(tok).use { input ->
                                FileOutputStream(destTok).use { output ->
                                    input.copyTo(output)
                                }
                            }
                            Log.i(TAG, "Extracted asset tokenizer: $tok")
                        } catch (e: Exception) {
                            Log.w(TAG, "Failed extracting $tok: ${e.message}")
                        }
                    }
                }
            }

            // Extract bundled starter models
            val bundledModels = listOf(
                "tinystories-15m" to "stories15M.bin",
                "tinystories-260k" to "stories260K.bin"
            )

            for ((modelId, fileName) in bundledModels) {
                if (fileName in assetList) {
                    val targetDir = File(modelsDir, modelId).apply { if (!exists()) mkdirs() }
                    val targetFile = File(targetDir, fileName)
                    if (!targetFile.exists() || targetFile.length() < 1000L) {
                        try {
                            Log.i(TAG, "Unpacking bundled model asset: $fileName ...")
                            assetManager.open(fileName).use { input ->
                                FileOutputStream(targetFile).use { output ->
                                    input.copyTo(output)
                                }
                            }
                            Log.i(TAG, "Finished unpacking $fileName (${targetFile.length() / 1024} KB)")
                        } catch (e: Exception) {
                            Log.e(TAG, "Failed unpacking $fileName: ${e.message}")
                        }
                    }
                    if (targetFile.exists() && targetFile.length() > 1024L) {
                        val sizeMb = Math.round((targetFile.length().toDouble() / (1024.0 * 1024.0)) * 10.0) / 10.0
                        updateState(modelId, DownloadState.Ready(modelId, targetFile, sizeMb))
                    }
                }
            }
        }
    }

    fun getModelFile(model: ModelSpec): File {
        val dir = File(modelsDir, model.id).apply { if (!exists()) mkdirs() }
        val filename = if (model.fileName.isNotEmpty()) model.fileName else "${model.id}.bin"
        return File(dir, filename)
    }

    fun isModelDownloaded(model: ModelSpec): Boolean {
        val file = getModelFile(model)
        return file.exists() && file.length() > 1024L
    }

    fun getDownloadedSizeMb(model: ModelSpec): Double {
        val file = getModelFile(model)
        if (!file.exists()) return 0.0
        return Math.round((file.length().toDouble() / (1024.0 * 1024.0)) * 10.0) / 10.0
    }

    fun getDownloadedModelFile(modelId: String, preferredFileName: String? = null): File? {
        val dir = File(modelsDir, modelId)
        if (dir.exists() && dir.isDirectory) {
            if (!preferredFileName.isNullOrEmpty()) {
                val f = File(dir, preferredFileName)
                if (f.exists() && f.length() > 1024L && !f.name.endsWith(".part")) return f
            }
            val firstValid = dir.listFiles()?.firstOrNull { it.isFile && it.length() > 1024L && !it.name.endsWith(".part") }
            if (firstValid != null) return firstValid
        }
        val direct = File(modelsDir, modelId)
        if (direct.exists() && direct.isFile && direct.length() > 1024L && !direct.name.endsWith(".part")) {
            return direct
        }
        val spec = org.meshllm.node.model.ModelCatalog.getModelById(modelId)
        if (spec != null) {
            val f = getModelFile(spec)
            if (f.exists() && f.length() > 1024L && !f.name.endsWith(".part")) return f
        }
        return null
    }

    fun getDownloadedModelIds(): List<String> {
        val result = mutableSetOf<String>()
        modelsDir.listFiles()?.forEach { fileOrDir ->
            if (fileOrDir.isDirectory) {
                val hasValid = fileOrDir.listFiles()?.any { it.isFile && it.length() > 1024L && !it.name.endsWith(".part") } ?: false
                if (hasValid) result.add(fileOrDir.name)
            } else if (fileOrDir.isFile && fileOrDir.length() > 1024L && !fileOrDir.name.endsWith(".part")) {
                result.add(fileOrDir.nameWithoutExtension)
            }
        }
        return result.toList()
    }

    fun downloadModel(
        model: ModelSpec,
        onProgress: ((DownloadState) -> Unit)? = null
    ) {
        if (isModelDownloaded(model)) {
            val ready = DownloadState.Ready(model.id, getModelFile(model), getDownloadedSizeMb(model))
            updateState(model.id, ready)
            onProgress?.invoke(ready)
            return
        }

        // Cancel any existing job for this model
        activeDownloadJobs[model.id]?.cancel()

        val job = scope.launch(Dispatchers.IO) {
            val destFile = getModelFile(model)
            val partFile = File(destFile.parentFile, "${destFile.name}.part")

            // 0. Storage Space Pre-flight Verification for Large Models (>5GB)
            val availableSpace = destFile.parentFile?.usableSpace ?: Long.MAX_VALUE
            val requiredBytes = if (model.fileSizeBytes > 0L) model.fileSizeBytes else 500_000_000L
            val safetyMargin = 300_000_000L // 300MB buffer for OS operation
            if (availableSpace < (requiredBytes + safetyMargin)) {
                val reqGb = String.format("%.2f", requiredBytes.toDouble() / (1024.0 * 1024.0 * 1024.0))
                val freeGb = String.format("%.2f", availableSpace.toDouble() / (1024.0 * 1024.0 * 1024.0))
                val errMsg = "Insufficient storage space: Model requires $reqGb GB, but device only has $freeGb GB free"
                Log.e(TAG, errMsg)
                val failedState = DownloadState.Failed(model.id, errMsg)
                updateState(model.id, failedState)
                withContext(Dispatchers.Main) { onProgress?.invoke(failedState) }
                return@launch
            }

            // 1. Check if a peer (Master Node) has this model available locally
            var effectiveUrl = model.downloadUrl
            var downloadSource = "Internet (Hugging Face)"

            val peerSource = peerDownloadSourceProvider?.invoke()
            if (peerSource != null) {
                try {
                    val encodedId = java.net.URLEncoder.encode(model.id, "UTF-8")
                    val nameParam = if (model.fileName.isNotEmpty()) "&filename=" + java.net.URLEncoder.encode(model.fileName, "UTF-8") else ""
                    val checkUrl = "http://${peerSource.host}:${peerSource.port}/api/model/status?id=$encodedId$nameParam"

                    val checkReq = Request.Builder()
                        .url(checkUrl)
                        .get()
                        .build()

                    val quickClient = httpClient.newBuilder()
                        .connectTimeout(3, TimeUnit.SECONDS)
                        .readTimeout(3, TimeUnit.SECONDS)
                        .build()

                    quickClient.newCall(checkReq).execute().use { checkResp ->
                        if (checkResp.isSuccessful) {
                            val checkBody = checkResp.body?.string() ?: ""
                            val json = com.google.gson.JsonParser.parseString(checkBody).asJsonObject
                            if (json.has("available") && json.get("available").asBoolean) {
                                effectiveUrl = "http://${peerSource.host}:${peerSource.port}/api/model/download?id=$encodedId$nameParam"
                                downloadSource = "Master Node (Local Mesh P2P)"
                                Log.i(TAG, "⚡ Found ${model.name} on Master Node (${peerSource.host}:${peerSource.port})! Using direct P2P mesh transfer.")
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Master node status check failed, falling back to Internet: ${e.message}")
                }
            }

            if (effectiveUrl.isEmpty()) {
                val error = DownloadState.Failed(model.id, "No download URL available on Master node or Internet")
                updateState(model.id, error)
                withContext(Dispatchers.Main) { onProgress?.invoke(error) }
                return@launch
            }

            Log.i(TAG, "Starting download of ${model.name} from $downloadSource: $effectiveUrl")
            val startState = DownloadState.Downloading(model.id, 0, 0L, model.fileSizeBytes, 0.0, downloadSource)
            updateState(model.id, startState)
            withContext(Dispatchers.Main) { onProgress?.invoke(startState) }

            try {
                var downloadSuccess = false
                var attempt = 0
                val maxAttempts = 4
                var lastError: Exception? = null

                while (attempt < maxAttempts && !downloadSuccess && scope.isActive) {
                    attempt++
                    try {
                        performDownload(model, effectiveUrl, downloadSource, destFile, partFile, onProgress)
                        downloadSuccess = true
                    } catch (e: CancellationException) {
                        Log.w(TAG, "Download cancelled for ${model.name}")
                        updateState(model.id, DownloadState.Idle)
                        return@launch
                    } catch (e: Exception) {
                        lastError = e
                        Log.w(TAG, "Download attempt $attempt/$maxAttempts for ${model.name} interrupted: ${e.message}. Resuming...")
                        if (attempt < maxAttempts) {
                            kotlinx.coroutines.delay(1000L * attempt)
                        }
                    }
                }

                if (!downloadSuccess) {
                    // If peer download failed and we have an internet URL, try fallback
                    if (downloadSource != "Internet (Hugging Face)" && model.downloadUrl.isNotEmpty()) {
                        Log.w(TAG, "P2P download from master failed: ${lastError?.message}. Retrying via Internet...")
                        try {
                            val fallbackSource = "Internet (Fallback)"
                            val fallbackState = DownloadState.Downloading(model.id, 0, 0L, model.fileSizeBytes, 0.0, fallbackSource)
                            updateState(model.id, fallbackState)
                            scope.launch(Dispatchers.Main) { onProgress?.invoke(fallbackState) }
                            performDownload(model, model.downloadUrl, fallbackSource, destFile, partFile, onProgress)
                            return@launch
                        } catch (fbEx: Exception) {
                            Log.e(TAG, "Fallback download also failed", fbEx)
                        }
                    }

                    Log.e(TAG, "Download failed for ${model.name}", lastError)
                    val errState = DownloadState.Failed(model.id, lastError?.message ?: "Download connection error")
                    updateState(model.id, errState)
                    scope.launch(Dispatchers.Main) { onProgress?.invoke(errState) }
                }
            } finally {
                activeDownloadJobs.remove(model.id)
            }
        }

        activeDownloadJobs[model.id] = job
    }

    private suspend fun performDownload(
        model: ModelSpec,
        url: String,
        source: String,
        destFile: File,
        partFile: File,
        onProgress: ((DownloadState) -> Unit)?
    ) {
        val existingBytes = if (partFile.exists()) partFile.length() else 0L

        val reqBuilder = Request.Builder()
            .url(url)
            .addHeader("User-Agent", "MeshLLM-Android-Node/1.0")

        if (existingBytes > 0L) {
            reqBuilder.addHeader("Range", "bytes=$existingBytes-")
        }

        httpClient.newCall(reqBuilder.build()).execute().use { response ->
            if (!response.isSuccessful && response.code != 206) {
                throw IOException("HTTP ${response.code}: ${response.message}")
            }

            val body = response.body ?: throw IOException("Empty response body from server")
            val isPartial = response.code == 206
            val startingByte = if (isPartial) existingBytes else 0L
            val serverContentLength = body.contentLength()
            val totalBytes = if (serverContentLength > 0) startingByte + serverContentLength else model.fileSizeBytes

            var bytesCopied = startingByte
            val buffer = ByteArray(256 * 1024) // 256KB buffer for optimal multi-gigabyte throughput without GC churn
            var lastUpdateMs = System.currentTimeMillis()
            var bytesSinceLastUpdate: Long = 0
            var bytesSinceFlush: Long = 0
            var currentSpeed = 0.0

            java.io.BufferedInputStream(body.byteStream(), 512 * 1024).use { input ->
                java.io.BufferedOutputStream(FileOutputStream(partFile, isPartial), 512 * 1024).use { output ->
                    var read: Int
                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                        bytesCopied += read
                        bytesSinceLastUpdate += read
                        bytesSinceFlush += read

                        // Periodically flush buffer to disk every 4MB to prevent flash write stalls on >5GB downloads
                        if (bytesSinceFlush >= 4 * 1024 * 1024) {
                            output.flush()
                            bytesSinceFlush = 0L
                        }

                        val now = System.currentTimeMillis()
                        val elapsed = now - lastUpdateMs
                        if (elapsed >= 500) {
                            currentSpeed = (bytesSinceLastUpdate.toDouble() / (1024.0 * 1024.0)) / (elapsed.toDouble() / 1000.0)
                            val pct = if (totalBytes > 0) ((bytesCopied * 100) / totalBytes).toInt().coerceIn(0, 99) else 0

                            val progress = DownloadState.Downloading(
                                modelId = model.id,
                                percent = pct,
                                bytesRead = bytesCopied,
                                totalBytes = totalBytes,
                                speedMbPerSec = Math.round(currentSpeed * 10.0) / 10.0,
                                source = source
                            )
                            updateState(model.id, progress)
                            // Non-blocking UI progress notification to avoid starving the I/O thread
                            scope.launch(Dispatchers.Main) { onProgress?.invoke(progress) }

                            lastUpdateMs = now
                            bytesSinceLastUpdate = 0
                        }
                    }
                    output.flush()
                }
            }

            // Rename part file to dest file with atomic fallback
            if (partFile.exists()) {
                if (destFile.exists()) destFile.delete()
                val renamed = partFile.renameTo(destFile)
                if (!renamed && partFile.exists()) {
                    try {
                        java.nio.file.Files.move(
                            partFile.toPath(),
                            destFile.toPath(),
                            java.nio.file.StandardCopyOption.REPLACE_EXISTING
                        )
                    } catch (mvEx: Exception) {
                        Log.w(TAG, "Files.move fallback: ${mvEx.message}")
                    }
                }
            }

            val finalSizeMb = Math.round((destFile.length().toDouble() / (1024.0 * 1024.0)) * 10.0) / 10.0
            Log.i(TAG, "Download complete for ${model.name} ($source): ${destFile.absolutePath} ($finalSizeMb MB)")

            val readyState = DownloadState.Ready(model.id, destFile, finalSizeMb)
            updateState(model.id, readyState)
            scope.launch(Dispatchers.Main) { onProgress?.invoke(readyState) }
        }
    }

    fun cancelDownload(modelId: String) {
        activeDownloadJobs[modelId]?.cancel()
        activeDownloadJobs.remove(modelId)
        updateState(modelId, DownloadState.Idle)
    }

    fun deleteModelFile(model: ModelSpec): Boolean {
        cancelDownload(model.id)
        val file = getModelFile(model)
        val deleted = if (file.exists()) file.delete() else true
        updateState(model.id, DownloadState.Idle)
        return deleted
    }

    private fun updateState(modelId: String, state: DownloadState) {
        val current = _downloadStates.value.toMutableMap()
        current[modelId] = state
        _downloadStates.value = current
    }
}
