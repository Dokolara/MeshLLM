package org.meshllm.node.p2p

import java.net.NetworkInterface

object NetworkUtils {
    /**
     * Resolves the primary local IPv4 address (e.g. 192.168.x.x, 10.x.x.x)
     * from active network interfaces (Wi-Fi, Ethernet), ignoring loopback.
     */
    fun getLocalIpAddress(): String {
        try {
            val interfaces = NetworkInterface.getNetworkInterfaces()
            while (interfaces.hasMoreElements()) {
                val iface = interfaces.nextElement()
                if (iface.isLoopback || !iface.isUp) continue
                val addresses = iface.inetAddresses
                while (addresses.hasMoreElements()) {
                    val addr = addresses.nextElement()
                    if (!addr.isLoopbackAddress && addr is java.net.Inet4Address) {
                        val host = addr.hostAddress
                        if (!host.isNullOrEmpty() && host != "127.0.0.1") {
                            return host
                        }
                    }
                }
            }
        } catch (_: Exception) {}
        return "127.0.0.1"
    }
}
