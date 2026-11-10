package org.meshllm.node.p2p

import android.util.Log
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.meshllm.node.model.ClusterState
import org.meshllm.node.model.LLMBlock
import org.meshllm.node.model.ModelSpec
import org.meshllm.node.model.NodeInfo
import org.meshllm.node.model.PipelineActivation
import org.meshllm.node.model.SensorMetrics
import java.io.IOException
import java.util.concurrent.TimeUnit

class P2PClient {

    private val gson = Gson()
    private val jsonMedia = "application/json; charset=utf-8".toMediaType()

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(180, TimeUnit.SECONDS)
        .writeTimeout(180, TimeUnit.SECONDS)
        .connectionPool(okhttp3.ConnectionPool(8, 5, TimeUnit.MINUTES))
        .retryOnConnectionFailure(true)
        .build()

    companion object {
        private const val TAG = "P2PClient"
        const val HEADER_SESSION_TOKEN = "X-Session-Token"
        const val HEADER_SENDER_ID = "X-Sender-Node-Id"
    }

    data class RegistrationRequest(
        val node: NodeInfo,
        val sessionToken: String
    )

    data class RegistrationResponse(
        val success: Boolean,
        val message: String,
        val sessionToken: String,
        val assignedJoinOrder: Int,
        val clusterState: ClusterState?
    )

    data class HeartbeatRequest(
        val nodeId: String,
        val sessionToken: String,
        val metrics: SensorMetrics
    )

    data class HeartbeatResponse(
        val acknowledged: Boolean,
        val clusterState: ClusterState?
    )

    data class FailoverPromotionRequest(
        val newMasterNodeId: String,
        val newMasterAddress: String,
        val newMasterPort: Int,
        val sessionToken: String
    )

    data class ModelChangeNotice(
        val modelSpec: ModelSpec,
        val sessionToken: String
    )

    data class PreloadCommandRequest(
        val modelSpec: ModelSpec,
        val sessionToken: String,
        val assignedBlock: LLMBlock? = null
    )

    data class ModelAvailabilityResponse(
        val available: Boolean,
        val modelId: String,
        val fileName: String? = null,
        val fileSizeBytes: Long = 0L,
        val fileSizeMb: Double = 0.0,
        val downloadPath: String? = null
    )

    suspend fun registerNode(
        host: String,
        port: Int,
        node: NodeInfo,
        sessionToken: String
    ): Result<RegistrationResponse> = withContext(Dispatchers.IO) {
        try {
            val url = "http://$host:$port/api/register"
            val reqBody = gson.toJson(RegistrationRequest(node, sessionToken)).toRequestBody(jsonMedia)
            val request = Request.Builder()
                .url(url)
                .post(reqBody)
                .addHeader(HEADER_SESSION_TOKEN, sessionToken)
                .addHeader(HEADER_SENDER_ID, node.nodeId)
                .build()

            client.newCall(request).execute().use { response ->
                val body = response.body?.string() ?: ""
                if (!response.isSuccessful) {
                    return@withContext Result.failure(IOException("Registration failed: HTTP ${response.code} - $body"))
                }
                val resp = gson.fromJson(body, RegistrationResponse::class.java)
                Result.success(resp)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to register with master at $host:$port", e)
            Result.failure(e)
        }
    }

    suspend fun sendHeartbeat(
        host: String,
        port: Int,
        nodeId: String,
        sessionToken: String,
        metrics: SensorMetrics
    ): Result<HeartbeatResponse> = withContext(Dispatchers.IO) {
        try {
            val url = "http://$host:$port/api/heartbeat"
            val reqBody = gson.toJson(HeartbeatRequest(nodeId, sessionToken, metrics)).toRequestBody(jsonMedia)
            val request = Request.Builder()
                .url(url)
                .post(reqBody)
                .addHeader(HEADER_SESSION_TOKEN, sessionToken)
                .addHeader(HEADER_SENDER_ID, nodeId)
                .build()

            client.newCall(request).execute().use { response ->
                val body = response.body?.string() ?: ""
                if (!response.isSuccessful) {
                    return@withContext Result.failure(IOException("Heartbeat failed: HTTP ${response.code}"))
                }
                val resp = gson.fromJson(body, HeartbeatResponse::class.java)
                Result.success(resp)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun forwardActivation(
        host: String,
        port: Int,
        activation: PipelineActivation
    ): Result<PipelineActivation> = withContext(Dispatchers.IO) {
        try {
            val url = "http://$host:$port/api/pipeline/forward"
            val reqBody = gson.toJson(activation).toRequestBody(jsonMedia)
            val request = Request.Builder()
                .url(url)
                .post(reqBody)
                .addHeader(HEADER_SESSION_TOKEN, activation.sessionToken)
                .addHeader(HEADER_SENDER_ID, activation.senderNodeId)
                .build()

            client.newCall(request).execute().use { response ->
                if (response.isSuccessful) {
                    val respBody = response.body?.string() ?: ""
                    try {
                        val resultAct = gson.fromJson(respBody, PipelineActivation::class.java)
                        if (resultAct != null && resultAct.tokenIndex == activation.tokenIndex) {
                            return@withContext Result.success(resultAct)
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed parsing forward response JSON: $respBody", e)
                        return@withContext Result.failure(IOException("Failed parsing forward activation response: ${e.message}"))
                    }
                    Result.failure(IOException("Invalid or mismatched activation payload from slave node"))
                } else {
                    Result.failure(IOException("Forward failed: HTTP ${response.code}"))
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed forward activation to $host:$port", e)
            Result.failure(e)
        }
    }

    suspend fun notifyModelChange(
        host: String,
        port: Int,
        modelSpec: ModelSpec,
        sessionToken: String
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        try {
            val url = "http://$host:$port/api/cluster/model"
            val reqBody = gson.toJson(ModelChangeNotice(modelSpec, sessionToken)).toRequestBody(jsonMedia)
            val request = Request.Builder()
                .url(url)
                .post(reqBody)
                .addHeader(HEADER_SESSION_TOKEN, sessionToken)
                .build()

            client.newCall(request).execute().use { response ->
                Result.success(response.isSuccessful)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed notifying model change to $host:$port", e)
            Result.failure(e)
        }
    }

    suspend fun notifyFailover(
        host: String,
        port: Int,
        requestData: FailoverPromotionRequest
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        try {
            val url = "http://$host:$port/api/failover/promote"
            val reqBody = gson.toJson(requestData).toRequestBody(jsonMedia)
            val request = Request.Builder()
                .url(url)
                .post(reqBody)
                .addHeader(HEADER_SESSION_TOKEN, requestData.sessionToken)
                .build()

            client.newCall(request).execute().use { response ->
                Result.success(response.isSuccessful)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun checkModelAvailability(
        host: String,
        port: Int,
        modelId: String,
        fileName: String? = null
    ): Result<ModelAvailabilityResponse> = withContext(Dispatchers.IO) {
        try {
            val encodedId = java.net.URLEncoder.encode(modelId, "UTF-8")
            val nameParam = if (!fileName.isNullOrEmpty()) "&filename=" + java.net.URLEncoder.encode(fileName, "UTF-8") else ""
            val url = "http://$host:$port/api/model/status?id=$encodedId$nameParam"

            val request = Request.Builder()
                .url(url)
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                val body = response.body?.string() ?: ""
                if (!response.isSuccessful) {
                    return@withContext Result.failure(IOException("Model check failed: HTTP ${response.code}"))
                }
                val resp = gson.fromJson(body, ModelAvailabilityResponse::class.java)
                Result.success(resp)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to check model availability on master $host:$port: ${e.message}")
            Result.failure(e)
        }
    }

    suspend fun sendPreloadCommand(
        host: String,
        port: Int,
        modelSpec: ModelSpec,
        sessionToken: String,
        assignedBlock: LLMBlock? = null
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        try {
            val url = "http://$host:$port/api/cluster/preload"
            val reqBody = gson.toJson(PreloadCommandRequest(modelSpec, sessionToken, assignedBlock)).toRequestBody(jsonMedia)
            val request = Request.Builder()
                .url(url)
                .post(reqBody)
                .addHeader(HEADER_SESSION_TOKEN, sessionToken)
                .build()

            client.newCall(request).execute().use { response ->
                Result.success(response.isSuccessful)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed sending preload command to $host:$port", e)
            Result.failure(e)
        }
    }
}
