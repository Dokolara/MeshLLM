package org.meshllm.node.ui

import android.graphics.Color
import android.os.Bundle
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.viewpager2.adapter.FragmentStateAdapter
import androidx.viewpager2.widget.ViewPager2
import kotlinx.coroutines.launch
import org.meshllm.node.R
import org.meshllm.node.databinding.ActivityMainBinding
import org.meshllm.node.model.NodeRole

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    val viewModel: MeshClusterViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Elevate UI Main thread priority to ensure Android UI, Touch, and Choreographer never starve or trigger ANRs
        try {
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_DISPLAY)
        } catch (_: Exception) {}

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupNavigation()
        observeHeader()
        checkAndRequestBatteryOptimizationExemption()
    }

    private fun checkAndRequestBatteryOptimizationExemption() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
            val pm = getSystemService(POWER_SERVICE) as android.os.PowerManager
            if (!pm.isIgnoringBatteryOptimizations(packageName)) {
                try {
                    val intent = android.content.Intent(android.provider.Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
                        data = android.net.Uri.parse("package:$packageName")
                    }
                    startActivity(intent)
                } catch (_: Exception) {
                    val fallbackIntent = android.content.Intent(android.provider.Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
                    try { startActivity(fallbackIntent) } catch (_: Exception) {}
                }
            }
        }

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            if (checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                requestPermissions(arrayOf(android.Manifest.permission.POST_NOTIFICATIONS), 101)
            }
        }
    }

    private fun setupNavigation() {
        val pagerAdapter = MainPagerAdapter(this)
        binding.viewPager.adapter = pagerAdapter
        binding.viewPager.offscreenPageLimit = 1 // Balanced for older devices to prevent simultaneous view hierarchy measuring

        // Sync ViewPager2 with BottomNavigationView
        binding.bottomNavigation.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.nav_cluster -> {
                    binding.viewPager.currentItem = 0
                    true
                }
                R.id.nav_models -> {
                    binding.viewPager.currentItem = 1
                    true
                }
                R.id.nav_telemetry -> {
                    binding.viewPager.currentItem = 2
                    true
                }
                R.id.nav_inference -> {
                    binding.viewPager.currentItem = 3
                    true
                }
                else -> false
            }
        }

        binding.viewPager.registerOnPageChangeCallback(object : ViewPager2.OnPageChangeCallback() {
            override fun onPageSelected(position: Int) {
                super.onPageSelected(position)
                val itemId = when (position) {
                    0 -> R.id.nav_cluster
                    1 -> R.id.nav_models
                    2 -> R.id.nav_telemetry
                    3 -> R.id.nav_inference
                    else -> R.id.nav_cluster
                }
                if (binding.bottomNavigation.selectedItemId != itemId) {
                    binding.bottomNavigation.selectedItemId = itemId
                }
            }
        })
    }

    private fun observeHeader() {
        lifecycleScope.launch {
            viewModel.currentRole.collect { role ->
                when (role) {
                    NodeRole.MASTER -> {
                        binding.tvHeaderRoleBadge.text = "● MASTER"
                        binding.tvHeaderRoleBadge.setBackgroundColor(Color.parseColor("#1E1B4B"))
                        binding.tvHeaderRoleBadge.setTextColor(Color.parseColor("#A78BFA"))
                    }
                    NodeRole.SLAVE -> {
                        binding.tvHeaderRoleBadge.text = "● WORKER"
                        binding.tvHeaderRoleBadge.setBackgroundColor(Color.parseColor("#082F49"))
                        binding.tvHeaderRoleBadge.setTextColor(Color.parseColor("#38BDF8"))
                    }
                    else -> {
                        binding.tvHeaderRoleBadge.text = "● IDLE"
                        binding.tvHeaderRoleBadge.setBackgroundColor(Color.parseColor("#1E293B"))
                        binding.tvHeaderRoleBadge.setTextColor(Color.parseColor("#94A3B8"))
                    }
                }
            }
        }

        lifecycleScope.launch {
            viewModel.activeModel.collect { model ->
                binding.tvHeaderSubtitle.text = "Model: ${model.name} (${model.paramCountStr} • ${model.totalLayers}L)"
            }
        }
    }

    private inner class MainPagerAdapter(activity: AppCompatActivity) : FragmentStateAdapter(activity) {
        override fun getItemCount(): Int = 4

        override fun createFragment(position: Int): Fragment {
            return when (position) {
                0 -> ClusterFragment()
                1 -> ModelsFragment()
                2 -> TelemetryFragment()
                3 -> InferenceFragment()
                else -> ClusterFragment()
            }
        }
    }
}
