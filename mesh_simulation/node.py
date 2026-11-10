#!/usr/bin/env python3
"""
MeshLLM Silicon Node Simulator (mesh_simulation/node.py)

Simulates distributed pipeline-parallel LLM execution across heterogeneous mobile devices
inside isolated Docker containers. Faithfully reproduces:
  - Wire protocol matching P2PServer.kt / P2PClient.kt (/api/register, /api/heartbeat, /api/pipeline/forward)
  - Continuous layer slicing (NEXUS-AIR)
  - INT8 dynamic symmetric activation quantization (TENSOR-SHRINK)
  - Configurable SoC compute delays, Wi-Fi LAN latency jitter, and thermal throttling (VITAL-SENSE)
  - End-to-end token generation & telemetry recording
"""

import os
import sys
import time
import json
import uuid
import struct
import base64
import random
import argparse
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# ----------------------------------------------------------------------
# INT8 Quantization Subsystem (TENSOR-SHRINK Algorithm)
# ----------------------------------------------------------------------
def quantize_int8(floats: list[float]) -> tuple[str, float]:
    """Quantizes FP32 list to INT8 symmetric payload, returns Base64 string and scale."""
    if not floats:
        return "", 1.0
    max_val = max(abs(x) for x in floats)
    scale = (max_val / 127.0) if max_val > 1e-9 else 1.0
    raw = bytearray(len(floats))
    for i, x in enumerate(floats):
        q = int(round(x / scale))
        q = max(-127, min(127, q))
        raw[i] = q & 0xFF
    return base64.b64encode(raw).decode("ascii"), scale

def dequantize_int8(b64_str: str, scale: float) -> list[float]:
    """Dequantizes Base64 INT8 payload back into FP32 floats."""
    if not b64_str:
        return []
    raw = base64.b64decode(b64_str)
    res = []
    for b in raw:
        signed = b if b < 128 else b - 256
        res.append(signed * scale)
    return res

def compute_mae(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

# ----------------------------------------------------------------------
# Token Vocabulary & Synthetic Story Generator (Mocking LLaMA/Mistral)
# ----------------------------------------------------------------------
VOCABULARY = [
    "Once", "upon", "a", "time", "in", "a", "distributed", "cluster", "five", "smartphones",
    "connected", "over", "a", "high-speed", "mesh", "network.", "The", "master", "node", "partitioned",
    "the", "transformer", "layers", "dynamically,", "ensuring", "each", "device", "computed", "its", "assigned",
    "forward", "pass", "without", "exhausting", "RAM.", "Activation", "tensors", "streamed", "between", "devices",
    "using", "INT8", "quantized", "transport,", "achieving", "low", "latency", "and", "flawless", "throughput.",
    "The", "pipeline", "completed", "each", "token", "step", "with", "precision,", "proving", "that",
    "edge", "intelligence", "thrives", "across", "collaborative", "mesh", "networks."
]

# ----------------------------------------------------------------------
# Node State & Data Classes
# ----------------------------------------------------------------------
class NodeInfo:
    def __init__(self, node_id, device_name, ip_address, port, join_order, soc_name, gflops, ram_mb):
        self.node_id = node_id
        self.device_name = device_name
        self.ip_address = ip_address
        self.port = port
        self.join_order = join_order
        self.soc_name = soc_name
        self.gflops = gflops
        self.ram_mb = ram_mb
        self.assigned_start_layer = 0
        self.assigned_end_layer = -1
        self.is_output_block = False
        self.last_heartbeat = time.time()
        self.battery_temp_c = 32.0

    @property
    def layer_count(self):
        if self.assigned_end_layer >= self.assigned_start_layer:
            return self.assigned_end_layer - self.assigned_start_layer + 1
        return 0

    def to_dict(self):
        return {
            "nodeId": self.node_id,
            "deviceName": self.device_name,
            "ipAddress": self.ip_address,
            "port": self.port,
            "joinOrder": self.join_order,
            "socName": self.soc_name,
            "gflops": self.gflops,
            "ramMb": self.ram_mb,
            "assignedBlocks": [
                {
                    "blockId": f"block_{self.node_id}",
                    "startLayer": self.assigned_start_layer,
                    "endLayer": self.assigned_end_layer,
                    "layerCount": self.layer_count,
                    "isInputBlock": (self.assigned_start_layer == 0),
                    "isOutputBlock": self.is_output_block
                }
            ],
            "totalAssignedLayers": self.layer_count,
            "isAlive": (time.time() - self.last_heartbeat < 15.0)
        }

# ----------------------------------------------------------------------
# Threaded HTTP Server
# ----------------------------------------------------------------------
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class MeshNodeServer:
    def __init__(self, role, node_id, port, master_url, total_layers=30, soc_name="Generic SoC",
                 layer_compute_ms=2.0, net_latency_ms=2.0, temp_c=32.0, enable_quant=True):
        self.role = role
        self.node_id = node_id
        self.port = port
        self.master_url = master_url
        self.total_layers = total_layers
        self.soc_name = soc_name
        self.layer_compute_ms = layer_compute_ms
        self.net_latency_ms = net_latency_ms
        self.temp_c = temp_c
        self.enable_quant = enable_quant

        self.session_token = f"MESH-{uuid.uuid4().hex[:4].upper()}-{uuid.uuid4().hex[:4].upper()}" if role == "master" else ""
        self.registered_nodes: list[NodeInfo] = []
        self.local_start_layer = 0
        self.local_end_layer = -1
        self.is_output_block = False
        self.hidden_dim = 288
        self.server = None
        self.running = True

        if self.role == "master":
            # Add self as Node 0
            self_node = NodeInfo(
                node_id=self.node_id,
                device_name=f"Master ({self.soc_name})",
                ip_address="mesh-master",
                port=self.port,
                join_order=0,
                soc_name=self.soc_name,
                gflops=1200.0,
                ram_mb=3800
            )
            self.registered_nodes.append(self_node)
            self.rebalance_layers()

    def rebalance_layers(self):
        """Nexus-Air Continuous Layer Allocation across all active nodes."""
        active = sorted([n for n in self.registered_nodes], key=lambda x: x.join_order)
        if not active:
            return
        n_nodes = len(active)
        base = self.total_layers // n_nodes
        remainder = self.total_layers % n_nodes

        current_layer = 0
        for i, node in enumerate(active):
            count = base + (1 if i < remainder else 0)
            node.assigned_start_layer = current_layer
            node.assigned_end_layer = current_layer + count - 1
            node.is_output_block = (i == n_nodes - 1)
            current_layer += count

        # Sync local master layer bounds
        master_info = self.registered_nodes[0]
        self.local_start_layer = master_info.assigned_start_layer
        self.local_end_layer = master_info.assigned_end_layer
        self.is_output_block = master_info.is_output_block

    def get_cluster_state(self):
        return {
            "sessionToken": self.session_token,
            "modelName": f"Distributed-MeshLLM ({self.total_layers}L)",
            "totalLayers": self.total_layers,
            "masterNodeId": self.node_id,
            "nodes": [n.to_dict() for n in self.registered_nodes]
        }

    def start_http_server(self):
        handler_cls = make_handler(self)
        self.server = ThreadedHTTPServer(("0.0.0.0", self.port), handler_cls)
        t = threading.Thread(target=self.server.serve_forever, daemon=True)
        t.start()
        print(f"[{self.node_id}] Listening on 0.0.0.0:{self.port} (Role: {self.role.upper()}, SoC: {self.soc_name})")

    def register_with_master(self):
        """Worker registers with the Master node."""
        payload = {
            "nodeId": self.node_id,
            "deviceName": f"Worker-{self.node_id} ({self.soc_name})",
            "port": self.port,
            "ipAddress": self.node_id,  # Docker container DNS host
            "sessionToken": "",
            "socName": self.soc_name,
            "gflops": 950.0,
            "ramMb": 3400
        }
        url = f"{self.master_url}/api/register"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        for attempt in range(20):
            try:
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        if data.get("success"):
                            self.session_token = data.get("sessionToken", "")
                            join_order = data.get("assignedJoinOrder", 1)
                            self.local_start_layer = data.get("assignedStartLayer", 0)
                            self.local_end_layer = data.get("assignedEndLayer", -1)
                            self.is_output_block = data.get("isOutputBlock", False)
                            print(f"[{self.node_id}] Successfully registered with Master! Join Order: #{join_order} | Layers: {self.local_start_layer}..{self.local_end_layer} (Output: {self.is_output_block})")
                            return True
            except Exception as e:
                time.sleep(1.0)
        print(f"[{self.node_id}] Failed to register with Master after 20 attempts.")
        return False

    def start_worker_heartbeat(self):
        def loop():
            while self.running:
                time.sleep(3.0)
                try:
                    payload = {
                        "nodeId": self.node_id,
                        "sessionToken": self.session_token,
                        "metrics": {
                            "batteryPercent": 85,
                            "batteryTemperatureC": self.temp_c,
                            "isCharging": False,
                            "totalRamMb": 3800,
                            "availableRamMb": 2400
                        }
                    }
                    req = urllib.request.Request(
                        f"{self.master_url}/api/heartbeat",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json", "X-Session-Token": self.session_token}
                    )
                    with urllib.request.urlopen(req, timeout=2.0) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read().decode("utf-8"))
                            c_state = data.get("clusterState")
                            if c_state:
                                for n in c_state.get("nodes", []):
                                    if n.get("nodeId") == self.node_id:
                                        blocks = n.get("assignedBlocks", [{}])[0]
                                        self.local_start_layer = blocks.get("startLayer", self.local_start_layer)
                                        self.local_end_layer = blocks.get("endLayer", self.local_end_layer)
                                        self.is_output_block = blocks.get("isOutputBlock", self.is_output_block)
                except Exception:
                    pass
        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def execute_forward_layer_math(self, activation):
        """Simulates transformer matrix multiply & self-attention for assigned layers."""
        start_time = time.time()

        # 1. Simulate Layer Compute Constraint (ARM NEON / GPU execution time)
        layer_count = max(1, self.local_end_layer - self.local_start_layer + 1)
        compute_delay_sec = (layer_count * self.layer_compute_ms) / 1000.0

        # Thermal governor throttle simulation: if battery temp > 40°C, inject 50-150ms delay
        if self.temp_c > 44.0:
            compute_delay_sec += 0.150  # Heavy thermal throttle
        elif self.temp_c > 40.0:
            compute_delay_sec += 0.050  # Light thermal throttle

        time.sleep(compute_delay_sec)

        # 2. Extract or generate activation tensor
        hidden = []
        if activation.get("hiddenStateBase64"):
            scale = activation.get("scale", 1.0)
            hidden = dequantize_int8(activation["hiddenStateBase64"], scale)
        elif activation.get("hiddenState"):
            hidden = activation["hiddenState"]
        else:
            # Initialize hidden state for token 0 embedding
            hidden = [random.uniform(-1.0, 1.0) for _ in range(self.hidden_dim)]

        # Simulate transformer layer transformation: LayerNorm + MLP projection
        for idx in range(len(hidden)):
            hidden[idx] = hidden[idx] * 0.98 + 0.02 * random.uniform(-0.5, 0.5)

        # 3. If Output Head is assigned to this node, sample the next token!
        generated_token_id = None
        generated_text_piece = None
        is_finished = False

        if self.is_output_block:
            token_idx = activation.get("tokenIndex", 0)
            if token_idx < len(VOCABULARY):
                piece = VOCABULARY[token_idx] + " "
                generated_token_id = 100 + token_idx
                generated_text_piece = piece
                is_finished = (token_idx >= len(VOCABULARY) - 1 or token_idx >= 35)
            else:
                generated_token_id = 2  # EOG
                generated_text_piece = ""
                is_finished = True

        # 4. Activation Quantization for outgoing network transmission (TENSOR-SHRINK)
        duration_ms = int((time.time() - start_time) * 1000)
        out_activation = dict(activation)
        out_activation["senderNodeId"] = self.node_id
        out_activation["executionTimeMs"] = duration_ms

        if self.enable_quant:
            b64_quant, scale = quantize_int8(hidden)
            out_activation["hiddenStateBase64"] = b64_quant
            out_activation["scale"] = scale
            out_activation.pop("hiddenState", None)
        else:
            out_activation["hiddenState"] = hidden
            out_activation.pop("hiddenStateBase64", None)

        if self.is_output_block:
            out_activation["generatedTokenId"] = generated_token_id
            out_activation["generatedTextPiece"] = generated_text_piece
            out_activation["isFinished"] = is_finished

        return out_activation

    def run_distributed_inference(self, prompt: str, max_tokens: int = 30):
        """Master runs multi-token distributed pipeline across the 5 nodes."""
        print(f"\n================================================================================")
        print(f"🚀 INITIATING 5-NODE DISTRIBUTED PIPELINE INFERENCE")
        print(f"   Model: {self.total_layers} Layers | Prompt: '{prompt}'")
        print(f"   Cluster Nodes: {len(self.registered_nodes)} active containers")
        for n in self.registered_nodes:
            print(f"   • [{n.node_id}] {n.device_name} ➔ Layers {n.assigned_start_layer}..{n.assigned_end_layer} ({n.layer_count}L){' [OUTPUT HEAD]' if n.is_output_block else ''}")
        print(f"================================================================================\n")

        full_output = ""
        total_tokens_generated = 0
        overall_start_time = time.time()
        first_token_time_ms = 0.0
        hop_latencies_matrix = []

        for token_idx in range(max_tokens):
            step_start = time.time()
            current_activation = {
                "sessionToken": self.session_token,
                "requestId": str(uuid.uuid4()),
                "promptId": prompt,
                "tokenIndex": token_idx,
                "seqPos": token_idx,
                "promptTokensCount": len(prompt.split()),
                "senderNodeId": self.node_id
            }

            token_hop_breakdown = []

            # Execute sequentially across each node in the cluster
            for i, node in enumerate(self.registered_nodes):
                hop_start = time.time()
                if node.node_id == self.node_id:
                    # Local Master Execution
                    current_activation = self.execute_forward_layer_math(current_activation)
                    duration_ms = (time.time() - hop_start) * 1000.0
                    token_hop_breakdown.append(("Master (Local)", duration_ms))
                else:
                    # Remote Container Execution over Docker Bridge Network
                    url = f"http://{node.ip_address}:{node.port}/api/pipeline/forward"
                    # Simulate configurable network latency / Wi-Fi channel delay
                    time.sleep(self.net_latency_ms / 1000.0)

                    req = urllib.request.Request(
                        url,
                        data=json.dumps(current_activation).encode("utf-8"),
                        headers={"Content-Type": "application/json", "X-Session-Token": self.session_token}
                    )
                    with urllib.request.urlopen(req, timeout=10.0) as resp:
                        current_activation = json.loads(resp.read().decode("utf-8"))

                    duration_ms = (time.time() - hop_start) * 1000.0
                    token_hop_breakdown.append((f"{node.node_id}", duration_ms))

            # The final node generated the token
            step_duration_ms = (time.time() - step_start) * 1000.0
            hop_latencies_matrix.append(token_hop_breakdown)

            if token_idx == 0:
                first_token_time_ms = step_duration_ms

            text_piece = current_activation.get("generatedTextPiece", "")
            full_output += text_piece
            total_tokens_generated += 1

            # Live token streaming output
            tok_per_sec = (token_idx + 1) / (time.time() - overall_start_time)
            print(f"Token {token_idx + 1:02d} ({step_duration_ms:5.1f} ms | {tok_per_sec:4.1f} tok/s) ➔ '{text_piece.strip()}'")

            if current_activation.get("isFinished", False):
                print("\n[EOG] End of generation token reached.")
                break

        total_elapsed = time.time() - overall_start_time
        final_tps = total_tokens_generated / total_elapsed if total_elapsed > 0 else 0.0

        print(f"\n================================================================================")
        print(f"🎉 DISTRIBUTED EXECUTION SUMMARY & BENCHMARK")
        print(f"   Generated Text: {full_output.strip()}")
        print(f"   Tokens Produced: {total_tokens_generated} tokens")
        print(f"   Total Wall-Clock Time: {total_elapsed:.3f} s")
        print(f"   Average Throughput: {final_tps:.2f} tokens/second")
        print(f"   Time To First Token (TTFT): {first_token_time_ms:.1f} ms")
        print(f"================================================================================\n")

        return {
            "prompt": prompt,
            "generatedText": full_output,
            "tokensGenerated": total_tokens_generated,
            "wallClockSec": total_elapsed,
            "tokensPerSecond": final_tps,
            "ttftMs": first_token_time_ms,
            "hopBreakdowns": hop_latencies_matrix
        }

# ----------------------------------------------------------------------
# HTTP Request Handler Generator
# ----------------------------------------------------------------------
def make_handler(node: MeshNodeServer):
    class NodeRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress default noisy access logs to keep console clean
            pass

        def send_json(self, status_code, data):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/status":
                self.send_json(200, {
                    "nodeId": node.node_id,
                    "role": node.role,
                    "socName": node.soc_name,
                    "layers": f"{node.local_start_layer}..{node.local_end_layer}",
                    "isOutput": node.is_output_block,
                    "clusterState": node.get_cluster_state() if node.role == "master" else None
                })
            else:
                self.send_json(404, {"error": "Not Found"})

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            payload = json.loads(raw_body) if raw_body else {}

            if self.path == "/api/register":
                if node.role != "master":
                    self.send_json(403, {"success": False, "message": "Only master handles registration"})
                    return
                # Register new slave
                new_node_id = payload.get("nodeId", f"worker-{len(node.registered_nodes)}")
                ip_addr = payload.get("ipAddress", self.client_address[0])
                port = payload.get("port", 8080)
                soc = payload.get("socName", "Simulated SoC")
                gflops = payload.get("gflops", 1000.0)
                ram_mb = payload.get("ramMb", 3000)

                join_order = len(node.registered_nodes)
                new_info = NodeInfo(new_node_id, payload.get("deviceName", new_node_id), ip_addr, port, join_order, soc, gflops, ram_mb)
                node.registered_nodes.append(new_info)
                node.rebalance_layers()

                # Broadcast updated layer allocations to the newly registered worker
                self.send_json(200, {
                    "success": True,
                    "message": "Registered successfully",
                    "sessionToken": node.session_token,
                    "assignedJoinOrder": join_order,
                    "assignedStartLayer": new_info.assigned_start_layer,
                    "assignedEndLayer": new_info.assigned_end_layer,
                    "isOutputBlock": new_info.is_output_block,
                    "clusterState": node.get_cluster_state()
                })

            elif self.path == "/api/heartbeat":
                node_id = payload.get("nodeId")
                for n in node.registered_nodes:
                    if n.node_id == node_id:
                        n.last_heartbeat = time.time()
                        metrics = payload.get("metrics", {})
                        n.battery_temp_c = metrics.get("batteryTemperatureC", 32.0)
                self.send_json(200, {"success": True, "clusterState": node.get_cluster_state() if node.role == "master" else None})

            elif self.path == "/api/cluster/preload":
                # Worker receives layer configuration update
                assigned = payload.get("assignedBlock", {})
                node.local_start_layer = assigned.get("startLayer", 0)
                node.local_end_layer = assigned.get("endLayer", -1)
                node.is_output_block = assigned.get("isOutputBlock", False)
                self.send_json(200, {"status": "Preloaded", "layers": f"{node.local_start_layer}..{node.local_end_layer}"})

            elif self.path == "/api/pipeline/forward":
                # Forward Pass Execution
                result_activation = node.execute_forward_layer_math(payload)
                self.send_json(200, result_activation)

            elif self.path == "/api/inference/run":
                if node.role != "master":
                    self.send_json(403, {"error": "Inference can only be initiated on master"})
                    return
                prompt = payload.get("prompt", "Compare distributed pipeline parallelism vs single phone execution")
                max_tokens = payload.get("maxTokens", 25)
                res = node.run_distributed_inference(prompt, max_tokens)
                self.send_json(200, res)

            else:
                self.send_json(404, {"error": "Unknown POST route"})

    return NodeRequestHandler

# ----------------------------------------------------------------------
# Main Entry Point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MeshLLM Distributed Node Simulator")
    parser.add_argument("--role", default=os.getenv("NODE_ROLE", "worker"), choices=["master", "worker"], help="Node Role")
    parser.add_argument("--id", default=os.getenv("NODE_ID", f"node-{uuid.uuid4().hex[:4]}"), help="Node ID")
    parser.add_argument("--port", type=int, default=int(os.getenv("NODE_PORT", "8080")), help="Port")
    parser.add_argument("--master-url", default=os.getenv("MASTER_URL", "http://mesh-master:8080"), help="Master URL")
    parser.add_argument("--total-layers", type=int, default=int(os.getenv("TOTAL_LAYERS", "30")), help="Model Total Layers")
    parser.add_argument("--soc", default=os.getenv("SIMULATED_SOC", "Snapdragon 8 Gen 2"), help="Simulated SoC")
    parser.add_argument("--compute-delay-ms", type=float, default=float(os.getenv("COMPUTE_DELAY_MS", "2.0")), help="Compute delay ms per layer")
    parser.add_argument("--net-latency-ms", type=float, default=float(os.getenv("NET_LATENCY_MS", "2.0")), help="Simulated network hop latency")
    parser.add_argument("--temp-c", type=float, default=float(os.getenv("SIMULATED_TEMP_C", "32.0")), help="Simulated Battery Temp C")
    parser.add_argument("--quant", action="store_true", default=(os.getenv("ENABLE_QUANT", "true").lower() == "true"), help="Enable INT8 Quantization")

    args = parser.parse_args()

    node = MeshNodeServer(
        role=args.role,
        node_id=args.id,
        port=args.port,
        master_url=args.master_url,
        total_layers=args.total_layers,
        soc_name=args.soc,
        layer_compute_ms=args.compute_delay_ms,
        net_latency_ms=args.net_latency_ms,
        temp_c=args.temp_c,
        enable_quant=args.quant
    )

    node.start_http_server()

    if node.role == "worker":
        # Register with master
        registered = node.register_with_master()
        if registered:
            node.start_worker_heartbeat()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print(f"[{node.node_id}] Shutting down...")

if __name__ == "__main__":
    main()
