package org.meshllm.node.p2p

import android.util.Log
import com.google.gson.Gson
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.meshllm.node.model.LLMBlock
import org.meshllm.node.model.ModelSpec
import org.meshllm.node.model.PipelineActivation
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.ServerSocket
import java.net.Socket
import android.os.Process
import java.util.concurrent.Executors

class P2PServer(
    val port: Int = 8080,
    private val scope: CoroutineScope
) {
    private val gson = Gson()

    // Dedicated background thread pool with THREAD_PRIORITY_BACKGROUND
    // Cached thread pool dynamically grows so inference never starves heartbeats or registration
    private val backgroundThreadPool = Executors.newCachedThreadPool { runnable ->
        Thread {
            try {
                Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
            } catch (_: Exception) {}
            runnable.run()
        }.apply {
            name = "MeshLLM-P2PWorker"
            isDaemon = true
        }
    }

    private var serverSocket: ServerSocket? = null
    @Volatile
    private var isRunning = false
    private val activeSockets = java.util.concurrent.ConcurrentHashMap.newKeySet<Socket>()

    @Volatile
    var currentSessionToken: String = ""

    // Callbacks provided by Orchestrator or Worker
    var onRegisterReceived: ((req: P2PClient.RegistrationRequest) -> P2PClient.RegistrationResponse)? = null
    var onHeartbeatReceived: ((req: P2PClient.HeartbeatRequest) -> P2PClient.HeartbeatResponse)? = null
    var onPipelineActivationReceived: ((activation: PipelineActivation) -> PipelineActivation?)? = null
    var onModelChangeReceived: ((modelSpec: ModelSpec) -> Unit)? = null
    var onPreloadCommandReceived: ((modelSpec: ModelSpec, assignedBlock: LLMBlock?) -> Unit)? = null
    var onFailoverPromotedReceived: ((req: P2PClient.FailoverPromotionRequest) -> Boolean)? = null
    var onGetModelFile: ((modelId: String, fileName: String?) -> java.io.File?)? = null
    var onListAvailableModels: (() -> List<ModelSpec>)? = null

    companion object {
        private const val TAG = "P2PServer"
        @Volatile
        private var sharedServer: P2PServer? = null

        fun getOrCreateShared(port: Int = 8080, scope: CoroutineScope): P2PServer {
            val s = sharedServer
            if (s != null && s.port == port) return s
            return synchronized(this) {
                val cur = sharedServer
                if (cur != null && cur.port == port) {
                    cur
                } else {
                    cur?.stop()
                    P2PServer(port, scope).also { sharedServer = it }
                }
            }
        }
    }

    fun updateSessionToken(token: String) {
        currentSessionToken = token
        Log.i(TAG, "P2PServer session token updated to '$token' on port $port")
    }

    fun resetCallbacks() {
        onRegisterReceived = null
        onHeartbeatReceived = null
        onPipelineActivationReceived = null
        onModelChangeReceived = null
        onPreloadCommandReceived = null
        onFailoverPromotedReceived = null
        onGetModelFile = null
        onListAvailableModels = null
    }

    fun isServerRunning(): Boolean = isRunning

    fun start() {
        synchronized(this) {
            if (isRunning) {
                Log.i(TAG, "P2PServer already running on port $port (Session token: '$currentSessionToken')")
                return
            }
            isRunning = true
        }

        backgroundThreadPool.execute {
            try {
                val ss = ServerSocket()
                ss.reuseAddress = true
                ss.bind(java.net.InetSocketAddress(port))
                serverSocket = ss
                Log.i(TAG, "P2PServer listening on port $port (Background Thread Priority, reuseAddress=true)")
                while (isRunning) {
                    val socket = ss.accept() ?: break
                    activeSockets.add(socket)
                    backgroundThreadPool.execute {
                        try {
                            handleConnection(socket)
                        } finally {
                            activeSockets.remove(socket)
                            try { socket.close() } catch (_: Exception) {}
                        }
                    }
                }
            } catch (e: Exception) {
                if (isRunning) {
                    Log.e(TAG, "ServerSocket error on port $port", e)
                }
            } finally {
                synchronized(this) {
                    isRunning = false
                    try { serverSocket?.close() } catch (_: Exception) {}
                    serverSocket = null
                }
            }
        }
    }

    fun stop() {
        synchronized(this) {
            isRunning = false
            try {
                serverSocket?.close()
            } catch (_: Exception) {}
            serverSocket = null
        }
        for (sock in activeSockets) {
            try { sock.close() } catch (_: Exception) {}
        }
        activeSockets.clear()
        Log.i(TAG, "P2PServer stopped on port $port")
    }

    private fun handleConnection(socket: Socket) {
        try {
            socket.tcpNoDelay = true
            socket.soTimeout = 120_000 // 2 minutes to accommodate heavy prompt prefill on >5GB models
            socket.receiveBufferSize = 512 * 1024
            socket.sendBufferSize = 512 * 1024
            val input = BufferedReader(InputStreamReader(socket.getInputStream(), Charsets.UTF_8))
            val output = socket.getOutputStream()

            while (isRunning && !socket.isClosed) {
                val requestLine = input.readLine() ?: break
                val parts = requestLine.split(" ")
                if (parts.size < 2) break
                val method = parts[0]
                val rawUri = parts[1]
                val path = rawUri.substringBefore('?')
                val queryString = if (rawUri.contains('?')) rawUri.substringAfter('?') else ""

                val queryParams = mutableMapOf<String, String>()
                if (queryString.isNotEmpty()) {
                    queryString.split("&").forEach { param ->
                        val eq = param.indexOf('=')
                        if (eq != -1) {
                            val k = try { java.net.URLDecoder.decode(param.substring(0, eq), "UTF-8") } catch (_: Exception) { param.substring(0, eq) }
                            val v = try { java.net.URLDecoder.decode(param.substring(eq + 1), "UTF-8") } catch (_: Exception) { param.substring(eq + 1) }
                            queryParams[k] = v
                        } else {
                            val k = try { java.net.URLDecoder.decode(param, "UTF-8") } catch (_: Exception) { param }
                            queryParams[k] = ""
                        }
                    }
                }

                // Read headers
                val headers = mutableMapOf<String, String>()
                var line: String?
                while (true) {
                    line = input.readLine()
                    if (line.isNullOrEmpty()) break
                    val idx = line.indexOf(':')
                    if (idx != -1) {
                        val key = line.substring(0, idx).trim().lowercase()
                        val value = line.substring(idx + 1).trim()
                        headers[key] = value
                    }
                }

                // AETHER-LINK: Read body safely and efficiently without per-chunk string allocation
                val contentLength = headers["content-length"]?.toIntOrNull() ?: 0
                val body = if (contentLength > 0) {
                    val buffer = StringBuilder(contentLength)
                    var readChars = 0
                    val chunk = CharArray(minOf(65536, contentLength))
                    while (readChars < contentLength) {
                        val toRead = minOf(chunk.size, contentLength - readChars)
                        val r = input.read(chunk, 0, toRead)
                        if (r == -1) break
                        buffer.append(chunk, 0, r)
                        readChars += r
                    }
                    buffer.toString()
                } else {
                    ""
                }

                val connHeader = headers["connection"]?.lowercase() ?: ""
                val keepAlive = connHeader != "close" && isRunning

                val sessionTokenHeader = headers[P2PClient.HEADER_SESSION_TOKEN.lowercase()] ?: ""

                // Dispatch routes
                when {
                    path == "/api/status" && method == "GET" -> {
                        sendJsonResponse(output, 200, mapOf("status" to "OK", "port" to port), keepAlive)
                    }

                    // Direct Master-to-Slave P2P Model Distribution Routes
                    path == "/api/model/status" && (method == "GET" || method == "HEAD") -> {
                        val modelId = queryParams["id"] ?: ""
                        val fileName = queryParams["filename"]
                        val file = onGetModelFile?.invoke(modelId, fileName)
                        if (file != null && file.exists() && file.length() > 1024L) {
                            val sizeMb = Math.round((file.length().toDouble() / (1024.0 * 1024.0)) * 10.0) / 10.0
                            sendJsonResponse(output, 200, mapOf(
                                "available" to true,
                                "modelId" to modelId,
                                "fileName" to file.name,
                                "fileSizeBytes" to file.length(),
                                "fileSizeMb" to sizeMb,
                                "downloadPath" to "/api/model/download?id=$modelId"
                            ), keepAlive)
                        } else {
                            sendJsonResponse(output, 200, mapOf(
                                "available" to false,
                                "modelId" to modelId,
                                "message" to "Model weights not downloaded on master node"
                            ), keepAlive)
                        }
                    }

                    (path == "/api/model/available" || path == "/api/model/list") && method == "GET" -> {
                        val models = onListAvailableModels?.invoke() ?: emptyList()
                        sendJsonResponse(output, 200, mapOf("models" to models), keepAlive)
                    }

                    (path == "/api/model/download" || path == "/api/model/file") && (method == "GET" || method == "HEAD") -> {
                        val modelId = queryParams["id"] ?: ""
                        val fileName = queryParams["filename"]
                        val file = onGetModelFile?.invoke(modelId, fileName)
                        if (file == null || !file.exists() || file.length() <= 1024L) {
                            sendJsonResponse(output, 404, mapOf("error" to "Model file not found on master node"), keepAlive = false)
                            break
                        } else {
                            serveModelFile(socket, output, file, headers, method == "HEAD")
                            break
                        }
                    }

                    path == "/api/register" && method == "POST" -> {
                        val req = try { gson.fromJson(body, P2PClient.RegistrationRequest::class.java) } catch (_: Exception) { null }
                        if (req == null) {
                            sendJsonResponse(output, 400, mapOf("error" to "Invalid registration request payload"), keepAlive)
                        } else if (currentSessionToken.isEmpty() || onRegisterReceived == null) {
                            sendJsonResponse(output, 503, P2PClient.RegistrationResponse(
                                success = false,
                                message = "Node is not active as a cluster Master.",
                                sessionToken = currentSessionToken,
                                assignedJoinOrder = -1,
                                clusterState = null
                            ), keepAlive)
                        } else if (req.sessionToken != currentSessionToken) {
                            Log.w(TAG, "Registration rejected: Token mismatch. Received '${req.sessionToken}', expected '$currentSessionToken'")
                            sendJsonResponse(output, 403, P2PClient.RegistrationResponse(
                                success = false,
                                message = "Invalid or expired session token. Expected '$currentSessionToken', received '${req.sessionToken}'. Orphan rejected.",
                                sessionToken = currentSessionToken,
                                assignedJoinOrder = -1,
                                clusterState = null
                            ), keepAlive)
                        } else {
                            val resp = onRegisterReceived?.invoke(req) ?: P2PClient.RegistrationResponse(
                                success = true,
                                message = "Registered",
                                sessionToken = currentSessionToken,
                                assignedJoinOrder = 1,
                                clusterState = null
                            )
                            sendJsonResponse(output, 200, resp, keepAlive)
                        }
                    }

                    path == "/api/heartbeat" && method == "POST" -> {
                        if (currentSessionToken.isNotEmpty() && sessionTokenHeader != currentSessionToken) {
                            sendJsonResponse(output, 403, mapOf("error" to "Session token mismatch"), keepAlive)
                        } else {
                            val req = gson.fromJson(body, P2PClient.HeartbeatRequest::class.java)
                            val resp = onHeartbeatReceived?.invoke(req) ?: P2PClient.HeartbeatResponse(true, null)
                            sendJsonResponse(output, 200, resp, keepAlive)
                        }
                    }

                    path == "/api/cluster/model" && method == "POST" -> {
                        if (currentSessionToken.isNotEmpty() && sessionTokenHeader != currentSessionToken) {
                            sendJsonResponse(output, 403, mapOf("error" to "Session token mismatch"), keepAlive)
                        } else {
                            val notice = gson.fromJson(body, P2PClient.ModelChangeNotice::class.java)
                            scope.launch {
                                onModelChangeReceived?.invoke(notice.modelSpec)
                            }
                            sendJsonResponse(output, 200, mapOf("status" to "Model updated", "model" to notice.modelSpec.name), keepAlive)
                        }
                    }

                    path == "/api/cluster/preload" && method == "POST" -> {
                        if (currentSessionToken.isNotEmpty() && sessionTokenHeader != currentSessionToken) {
                            sendJsonResponse(output, 403, mapOf("error" to "Session token mismatch"), keepAlive)
                        } else {
                            val req = gson.fromJson(body, P2PClient.PreloadCommandRequest::class.java)
                            scope.launch {
                                onPreloadCommandReceived?.invoke(req.modelSpec, req.assignedBlock)
                            }
                            sendJsonResponse(output, 200, mapOf("status" to "Preload scheduled", "model" to req.modelSpec.name), keepAlive)
                        }
                    }

                    path == "/api/pipeline/forward" && method == "POST" -> {
                        if (currentSessionToken.isNotEmpty() && sessionTokenHeader != currentSessionToken) {
                            sendJsonResponse(output, 403, mapOf("error" to "Session token mismatch"), keepAlive)
                        } else {
                            val activation = gson.fromJson(body, PipelineActivation::class.java)
                            val resultAct = onPipelineActivationReceived?.invoke(activation)
                            if (resultAct != null) {
                                sendJsonResponse(output, 200, resultAct, keepAlive)
                            } else {
                                sendJsonResponse(output, 200, mapOf("status" to "Accepted"), keepAlive)
                            }
                        }
                    }

                    path == "/api/failover/promote" && method == "POST" -> {
                        val req = gson.fromJson(body, P2PClient.FailoverPromotionRequest::class.java)
                        val ok = onFailoverPromotedReceived?.invoke(req) ?: true
                        sendJsonResponse(output, 200, mapOf("accepted" to ok), keepAlive)
                    }

                    else -> {
                        sendJsonResponse(output, 404, mapOf("error" to "Not Found"), keepAlive)
                    }
                }

                if (!keepAlive) {
                    break
                }
            }
        } catch (_: java.net.SocketTimeoutException) {
            // Normal keep-alive socket timeout when client is done streaming
        } catch (e: Exception) {
            Log.w(TAG, "Error handling socket request", e)
        } finally {
            try {
                socket.close()
            } catch (_: Exception) {}
        }
    }

    private fun sendJsonResponse(output: OutputStream, statusCode: Int, data: Any, keepAlive: Boolean = true) {
        val json = gson.toJson(data)
        val bytes = json.toByteArray(Charsets.UTF_8)
        val statusText = when (statusCode) {
            200 -> "OK"
            403 -> "Forbidden"
            404 -> "Not Found"
            else -> "Error"
        }
        val connVal = if (keepAlive) "keep-alive" else "close"
        val header = "HTTP/1.1 $statusCode $statusText\r\n" +
                "Content-Type: application/json; charset=utf-8\r\n" +
                "Content-Length: ${bytes.size}\r\n" +
                "Connection: $connVal\r\n\r\n"
        output.write(header.toByteArray(Charsets.UTF_8))
        output.write(bytes)
        output.flush()
    }

    private fun serveModelFile(
        socket: Socket,
        output: OutputStream,
        file: java.io.File,
        headers: Map<String, String>,
        isHeadOnly: Boolean
    ) {
        val totalBytes = file.length()
        val rangeHeader = headers["range"]
        var startOffset = 0L
        var endOffset = totalBytes - 1

        var isRange = false
        if (!rangeHeader.isNullOrEmpty() && rangeHeader.startsWith("bytes=")) {
            try {
                val rangeSpec = rangeHeader.removePrefix("bytes=").trim()
                val dashIdx = rangeSpec.indexOf('-')
                if (dashIdx != -1) {
                    val startStr = rangeSpec.substring(0, dashIdx).trim()
                    val endStr = rangeSpec.substring(dashIdx + 1).trim()
                    if (startStr.isNotEmpty()) {
                        startOffset = startStr.toLong().coerceIn(0L, totalBytes - 1)
                    }
                    if (endStr.isNotEmpty()) {
                        endOffset = endStr.toLong().coerceIn(startOffset, totalBytes - 1)
                    }
                    isRange = true
                }
            } catch (_: Exception) {
                isRange = false
                startOffset = 0L
                endOffset = totalBytes - 1
            }
        }

        val contentLength = endOffset - startOffset + 1
        val statusCode = if (isRange) 206 else 200
        val statusText = if (isRange) "Partial Content" else "OK"
        val rangeLine = if (isRange) "Content-Range: bytes $startOffset-$endOffset/$totalBytes\r\n" else ""

        val header = "HTTP/1.1 $statusCode $statusText\r\n" +
                "Content-Type: application/octet-stream\r\n" +
                "Accept-Ranges: bytes\r\n" +
                rangeLine +
                "Content-Length: $contentLength\r\n" +
                "Content-Disposition: attachment; filename=\"${file.name}\"\r\n" +
                "Connection: close\r\n\r\n"

        output.write(header.toByteArray(Charsets.UTF_8))
        output.flush()

        if (isHeadOnly) return

        try {
            // Generous timeout (5 minutes) and socket buffering for large model transfers (>5GB) across LAN Wi-Fi
            socket.soTimeout = 300_000
            socket.keepAlive = true
            socket.sendBufferSize = 1024 * 1024

            java.io.BufferedInputStream(java.io.FileInputStream(file), 512 * 1024).use { bis ->
                if (startOffset > 0L) {
                    var skipped = 0L
                    while (skipped < startOffset) {
                        val n = bis.skip(startOffset - skipped)
                        if (n <= 0) break
                        skipped += n
                    }
                }
                java.io.BufferedOutputStream(output, 512 * 1024).use { bos ->
                    val buffer = ByteArray(512 * 1024)
                    var bytesRemaining = contentLength
                    var bytesSinceFlush = 0L
                    while (bytesRemaining > 0L && isRunning) {
                        val toRead = Math.min(buffer.size.toLong(), bytesRemaining).toInt()
                        val read = bis.read(buffer, 0, toRead)
                        if (read == -1) break
                        bos.write(buffer, 0, read)
                        bytesRemaining -= read
                        bytesSinceFlush += read
                        if (bytesSinceFlush >= 4 * 1024 * 1024) {
                            bos.flush()
                            bytesSinceFlush = 0L
                        }
                    }
                    bos.flush()
                }
            }
            Log.i(TAG, "Completed direct P2P model transfer for ${file.name} ($contentLength bytes)")
        } catch (e: Exception) {
            Log.w(TAG, "P2P model stream ended or aborted by client: ${e.message}")
        }
    }
}
