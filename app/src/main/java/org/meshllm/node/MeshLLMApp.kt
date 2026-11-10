package org.meshllm.node

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build

class MeshLLMApp : Application() {

    companion object {
        lateinit var instance: MeshLLMApp
            private set
        const val CHANNEL_ID = "mesh_node_service_channel"
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        createNotificationChannel()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Mesh Node Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Keeps MeshLLM P2P background pipeline active"
            }
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }
}
