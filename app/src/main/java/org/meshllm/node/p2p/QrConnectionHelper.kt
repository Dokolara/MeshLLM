package org.meshllm.node.p2p

import android.graphics.Bitmap
import android.graphics.Color
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import com.google.zxing.BarcodeFormat
import com.google.zxing.EncodeHintType
import com.google.zxing.qrcode.QRCodeWriter
import com.google.zxing.qrcode.decoder.ErrorCorrectionLevel
import java.util.EnumMap

data class QrClusterPayload(
    @SerializedName("type") val type: String = "mesh_cluster_v1",
    @SerializedName("ip") val ip: String,
    @SerializedName("port") val port: Int,
    @SerializedName("token") val token: String,
    @SerializedName("model") val model: String = ""
)

object QrConnectionHelper {
    private val gson = Gson()

    /**
     * Serializes cluster connection params into a JSON string payload.
     */
    fun createPayloadString(ip: String, port: Int, token: String, model: String = ""): String {
        val payload = QrClusterPayload(
            ip = ip,
            port = port,
            token = token,
            model = model
        )
        return gson.toJson(payload)
    }

    /**
     * Generates a high-contrast Android Bitmap representation of the QR code.
     * Uses ZXing's QRCodeWriter directly for fast, pure-bitmap rendering without UI dependencies.
     */
    fun generateQrBitmap(content: String, sizePx: Int = 512): Bitmap? {
        return try {
            val hints = EnumMap<EncodeHintType, Any>(EncodeHintType::class.java).apply {
                put(EncodeHintType.CHARACTER_SET, "UTF-8")
                put(EncodeHintType.MARGIN, 1)
                put(EncodeHintType.ERROR_CORRECTION, ErrorCorrectionLevel.M)
            }
            val bitMatrix = QRCodeWriter().encode(content, BarcodeFormat.QR_CODE, sizePx, sizePx, hints)
            val width = bitMatrix.width
            val height = bitMatrix.height
            val pixels = IntArray(width * height)

            for (y in 0 until height) {
                val offset = y * width
                for (x in 0 until width) {
                    pixels[offset + x] = if (bitMatrix.get(x, y)) Color.BLACK else Color.WHITE
                }
            }

            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
            bitmap
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Parses a scanned QR string. Supports:
     * 1. Structured JSON payload: {"type":"mesh_cluster_v1","ip":"192.168.1.5","port":8080,"token":"MESH-...","model":"..."}
     * 2. URI scheme: mesh://192.168.1.5:8080?token=MESH-...&model=...
     * 3. Colon format: MESH:token:ip:port:model
     */
    fun parseScannedPayload(raw: String): QrClusterPayload? {
        val trimmed = raw.trim()
        if (trimmed.isEmpty()) return null

        // Try JSON parsing first
        try {
            if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
                val parsed = gson.fromJson(trimmed, QrClusterPayload::class.java)
                if (!parsed.ip.isNullOrEmpty() && parsed.port > 0 && !parsed.token.isNullOrEmpty()) {
                    return parsed
                }
            }
        } catch (_: Exception) {}

        // Try URI format: mesh://192.168.1.5:8080?token=MESH-XXXX&model=...
        try {
            if (trimmed.startsWith("mesh://")) {
                val withoutScheme = trimmed.removePrefix("mesh://")
                val hostPortAndQuery = withoutScheme.split("?")
                val hostPort = hostPortAndQuery[0].split(":")
                val host = hostPort[0]
                val port = if (hostPort.size > 1) hostPort[1].toIntOrNull() ?: 8080 else 8080
                var token = ""
                var model = ""
                if (hostPortAndQuery.size > 1) {
                    val queryParams = hostPortAndQuery[1].split("&")
                    for (param in queryParams) {
                        val kv = param.split("=")
                        if (kv.size == 2) {
                            if (kv[0] == "token") token = kv[1]
                            if (kv[0] == "model") model = kv[1]
                        }
                    }
                }
                if (host.isNotEmpty() && token.isNotEmpty()) {
                    return QrClusterPayload(ip = host, port = port, token = token, model = model)
                }
            }
        } catch (_: Exception) {}

        // Try colon format: MESH:token:ip:port:model or MESH_BEACON:token:ip:port:model
        try {
            if (trimmed.startsWith("MESH:") || trimmed.startsWith("MESH_BEACON:")) {
                val payload = if (trimmed.startsWith("MESH_BEACON:")) {
                    trimmed.removePrefix("MESH_BEACON:")
                } else {
                    trimmed.removePrefix("MESH:")
                }
                val parts = payload.split(":")
                if (parts.size >= 3) {
                    val token = parts[0]
                    val ip = parts[1]
                    val port = parts[2].toIntOrNull() ?: 8080
                    val model = if (parts.size > 3) parts[3] else ""
                    return QrClusterPayload(ip = ip, port = port, token = token, model = model)
                }
            }
        } catch (_: Exception) {}

        return null
    }
}
