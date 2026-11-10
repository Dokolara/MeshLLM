package org.meshllm.node.p2p

import java.security.SecureRandom

object SessionTokenGenerator {
    private val random = SecureRandom()
    private const val CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

    fun generateSessionToken(): String {
        fun chunk(len: Int): String = (1..len)
            .map { CHARS[random.nextInt(CHARS.length)] }
            .joinToString("")

        return "MESH-${chunk(4)}-${chunk(4)}-${chunk(4)}"
    }

    fun isValidTokenFormat(token: String): Boolean {
        return token.startsWith("MESH-") && token.length >= 14
    }
}
