# MeshLLM: Decentralized P2P Mobile LLM Pipeline Inference

MeshLLM is an Android application designed for distributed peer-to-peer (P2P) Large Language Model (LLM) pipeline inference across multiple mobile devices. It operates entirely serverless—all coordination, failure recovery, sensor-driven dynamic layer scheduling, and pipeline execution happen directly on the mobile devices over local Wi-Fi or mobile hotspots.

---

## Key System Architectures & Capabilities

### 1. Dynamic Master / Slave Role Management
- Any device running the app can act as either a **Master (Orchestrator)** or a **Slave (Worker)** node.
- **Dual-Role Master Device**: The Master node device does not sit idle as a pure coordinator—it simultaneously runs a local Worker instance (`joinOrder = 0`), actively processing assigned transformer layers alongside all other connected nodes.

### 2. Sensor-Driven Dynamic Layer Rebalancing
- Real-time hardware telemetry is continuously monitored on each device:
  - **Thermal State & Throttling** (`PowerManager.addThermalStatusListener`, `getThermalHeadroom()`)
  - **Battery Level & Temperature** (`Intent.ACTION_BATTERY_CHANGED`)
  - **RAM Memory Pressure** (`ActivityManager.MemoryInfo`)
  - **CPU Load Estimate & Network RTT**
- Master calculates an effective real-time capacity index:
  $$\text{EffectiveCapacity} = \text{CompositeBenchmarkScore} \times \text{HealthMultiplier}$$
- If a phone overheats ($>42^\circ\text{C}$ or thermal status $\ge \text{SEVERE}$) or battery drains low without charging, its health factor degrades. The Master automatically downscales that device's layer allocation and dynamically migrates layers sequentially to cooler, higher-capacity devices.

### 3. Fault Tolerance & Sequential Layer Reassignment
- If a Slave node disconnects, drops off Wi-Fi, or crashes (heartbeat timeout $>6.5\text{s}$):
  - Master immediately detects the dead node.
  - The dead node's layers are automatically redistributed across all surviving active nodes.
  - Sequential pipeline continuity is strictly maintained without holes or gaps:
    $$\text{Node } 0: [0..i], \quad \text{Node } 1: [i+1..j], \quad \dots, \quad \text{Node } N: [k+1..L-1]$$

### 4. Zero Single Point of Failure (Automated Master Failover)
- When slave devices join the network, each is assigned a deterministic **Join Order** index:
  - `joinOrder = 0`: Original Master
  - `joinOrder = 1`: First device connected after Master
  - `joinOrder = 2, 3, \dots`: Subsequent slave devices
- Slaves run a watchdog monitoring Master heartbeats.
- If the Master dies or disconnects, the surviving node with the lowest join order (`joinOrder = 1`) automatically promotes itself to **Master**, starts its local Orchestrator, and resumes cluster management. Peer nodes seamlessly redirect their heartbeats to the new Master.

### 5. First-Install Compute & Capacity Profiling with Dev Mode Cache
- On first install, the app runs:
  - **GEMM Compute Benchmark**: Matrix multiplication stress test measuring GFLOPS.
  - **Memory Bandwidth Test**: Sequential buffer allocation and throughput test in MB/s.
  - Hardware specs profiling (CPU cores and total RAM).
- **Cache File Checking**:
  - The app checks for the existence of `meshllm_bench_profile.json` (stored in internal storage at `/data/data/org.meshllm.node/files/` and external storage).
  - If the cache file exists, **all benchmark tests are skipped immediately** on boot/install, loading the cached capacity profile.
- **Dev Mode**:
  - A dedicated Dev Mode menu allows skipping benchmarks, setting custom mock GFLOPS and memory scores, and saving/clearing the profile cache file.

### 6. Session Token Isolation & LLM Block Distinction
- **Session Tokens**: Every newly initialized network generates a cryptographic session token (e.g., `MESH-7A4B-99FC-42A1`).
  - All P2P requests (`/api/register`, `/api/heartbeat`, `/api/pipeline/forward`) must validate the session token.
  - Orphan nodes from previous sessions or unauthorized devices on the local Wi-Fi cannot interfere.
- **LLM Block Distinction**:
  - `BlockType.EMBEDDING_INPUT`: Token embedding table, RoPE positional encoding, and initial transformer layers $[0..k]$.
  - `BlockType.TRANSFORMER_BLOCK`: Intermediate transformer decoder layers with Multi-Head Attention, RMSNorm, SwiGLU / MLP feed-forward, and residual adds $[k+1..m]$.
  - `BlockType.OUTPUT_LM_HEAD`: Final transformer layers $[m+1..L-1]$, final RMSNorm, Language Model Head logits projection, and next-token sampler.

### 7. Open-Source Model Catalog, Real Downloader & Prompt Tokenization
- **Direct Downloads from Hugging Face**:
  - `ModelDownloadManager` streams real model weights directly from Hugging Face to local device storage (`/data/user/0/org.meshllm.node/files/models/{id}/{file}`):
    1. **TinyStories-15M** (6 layers, 288 dim, ~60 MB) — Full LLaMA architecture trained on synthetic child stories (`stories15M.bin`).
    2. **TinyStories-33M** (8 layers, 384 dim, ~38 MB) — Compact synthetic reasoning model across 8 layers.
    3. **TinyStories-260K** (4 layers, 64 dim, **1.0 MB**) — Ultra-compact test model (`stories260K.bin`). Downloads in 1-2 seconds on mobile data for instant verification of download, storage caching, and pipeline inference.
    4. **SmolLM2-135M** (30 layers, 576 dim, ~96 MB) — GGUF quantized model (`SmolLM2-135M-Q4_K_M.gguf`).
    5. **Qwen2.5-0.5B** (24 layers, 896 dim, ~248 MB) — Alibaba's instruction foundation model (`qwen2.5-0.5b-instruct-q2_k.gguf`).
    6. **Llama-3.2-1B** (16 layers, 2048 dim, ~550 MB) — Meta's 16-layer edge model.
    7. **TinyLlama-1.1B** (22 layers, 2048 dim, ~480 MB) — 22-layer edge model trained on 3T tokens.
- **Live Download Progress & Storage Verification**:
  - Real-time download progress bar displaying percentage, MB downloaded / total MB, and download speed in MB/s.
  - Delete / Re-download controls to clear device storage at will.
- **Real Prompt Tokenizer & Neural Continuation Engine**:
  - `LlamaTokenizer` encodes user-entered prompts into token IDs and decodes generated token IDs back to text pieces.
  - `NeuralModelRunner` memory-maps downloaded binary weights (`.bin`) and computes embedding lookup, multi-head attention, RMSNorm, SwiGLU FFN, and LM Head vocabulary projection.
  - Text generated across the pipeline is **dynamically conditioned on the prompt context**—never a static hardcoded string.
- **Dynamic Layer Re-slicing**:
  - When the Master selects or switches the model, the orchestrator updates `ClusterState.totalLayers` and automatically re-slices the pipeline across all participating nodes.
  - P2P notification endpoint `/api/cluster/model` and regular heartbeats broadcast model changes to all active slave nodes.

### 8. Pure Serverless P2P Networking
- Embedded multi-threaded HTTP server (`P2PServer`) and client (`P2PClient` with OkHttp) inside the app.
- UDP broadcast discovery beacon on LAN port `55555` for zero-configuration network pairing (advertising the active model name).
- Zero reliance on external servers, cloud relays, or internet connectivity.

### 9. Device Performance Preservation
- Background thread scheduling with low process nice level (`Process.THREAD_PRIORITY_BACKGROUND`).
- `PerformanceGovernor` restricts CPU concurrency to a fraction of available cores and introduces micro-sleep yields during tensor forward passes when thermal or battery thresholds are exceeded, ensuring normal phone usage and foreground apps remain smooth.

---

## Directory Structure

```
/home/foor/MeshLLM/
├── app/
│   ├── build.gradle.kts
│   ├── src/
│   │   ├── main/
│   │   │   ├── AndroidManifest.xml
│   │   │   ├── java/org/meshllm/node/
│   │   │   │   ├── MeshLLMApp.kt                 # Application class & notification channels
│   │   │   │   ├── model/
│   │   │   │   │   ├── NodeRole.kt               # MASTER, SLAVE, UNASSIGNED
│   │   │   │   │   ├── NodeInfo.kt               # Node metadata, join order, assigned layers
│   │   │   │   │   ├── SensorMetrics.kt          # Battery %, temp, thermal status, RAM
│   │   │   │   │   ├── BenchmarkProfile.kt       # GFLOPS, memory bandwidth, cores
│   │   │   │   │   ├── ModelSpec.kt              # Open-source LLM architecture metadata
│   │   │   │   │   ├── ModelCatalog.kt           # Catalog of models (TinyStories, SmolLM2, Qwen...)
│   │   │   │   │   ├── LLMBlock.kt               # Block types (Embedding, Transformer, LM Head)
│   │   │   │   │   ├── PipelineActivation.kt     # Pipeline payload between nodes
│   │   │   │   │   └── ClusterState.kt           # Cluster snapshot and active layers
│   │   │   │   ├── telemetry/
│   │   │   │   │   ├── SensorMonitor.kt          # Real-time hardware telemetry
│   │   │   │   │   ├── BenchmarkRunner.kt        # GEMM & RAM benchmark with file caching
│   │   │   │   │   └── PerformanceGovernor.kt    # Thermal throttling and thread governor
│   │   │   │   ├── p2p/
│   │   │   │   │   ├── P2PServer.kt              # Embedded HTTP server
│   │   │   │   │   ├── P2PClient.kt              # HTTP client for peer communication
│   │   │   │   │   ├── NetworkDiscovery.kt       # UDP broadcast beacon discovery
│   │   │   │   │   └── SessionTokenGenerator.kt  # Secure session token generation
│   │   │   │   ├── engine/
│   │   │   │   │   ├── DistributedOrchestrator.kt# Master orchestrator & rebalancer
│   │   │   │   │   ├── SlaveWorker.kt            # Slave pipeline worker & failover watchdog
│   │   │   │   │   └── LLMBlockExecutor.kt       # Transformer block forward execution
│   │   │   │   ├── service/
│   │   │   │   │   └── MeshNodeService.kt        # Foreground Android service
│   │   │   │   └── ui/
│   │   │   │       ├── MainActivity.kt           # Main dashboard and control console
│   │   │   │       ├── NodeAdapter.kt            # Cluster node RecyclerView adapter
│   │   │   │       └── ModelAdapter.kt           # Model catalog list RecyclerView adapter
│   │   │   └── res/
│   │   │       ├── layout/
│   │   │       │   ├── activity_main.xml
│   │   │       │   ├── item_node.xml
│   │   │       │   ├── item_model.xml
│   │   │       │   ├── dialog_model_selector.xml
│   │   │       │   ├── dialog_join_network.xml
│   │   │       │   └── dialog_dev_mode.xml
│   │   │       └── values/
│   │   │           ├── strings.xml, colors.xml, themes.xml
│   │   └── test/java/org/meshllm/node/
│   │       └── MeshClusterTest.kt                # Unit test suite verifying cluster failover
│   └── build/outputs/apk/debug/
│       └── app-debug.apk                         # Compiled ready-to-install Android APK
├── build.gradle.kts
├── settings.gradle.kts
└── gradlew
```

---

## Installation & Testing

### 1. Build the APK
```bash
JAVA_HOME=/home/foor/.jdk/jdk-17 ./gradlew assembleDebug
```
The compiled APK is at:
`/home/foor/MeshLLM/app/build/outputs/apk/debug/app-debug.apk`

### 2. Install onto Android Device via ADB
Connect the device via USB or Wi-Fi debugging (`adb connect <device-ip>:5555`) and run:
```bash
adb install -r /home/foor/MeshLLM/app/build/outputs/apk/debug/app-debug.apk
```

### 3. Run Automated Tests
```bash
JAVA_HOME=/home/foor/.jdk/jdk-17 ./gradlew test
```
All unit tests verify:
- Secure session token generation
- Thermal and battery health degradation formula
- Sequential layer assignment with continuous layer ranges across master and slave nodes
- Automated sequential layer re-absorption when a slave node drops
- Failover leader election picking the first connected slave node
