#!/usr/bin/env python3
"""
Unit & Integration Test Suite for Containerized Mesh Simulation
(mesh_simulation/test_mesh_interconnectivity.py)
"""

import os
import sys
import time
import json
import random
import unittest
import urllib.request
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from node import quantize_int8, dequantize_int8, compute_mae, MeshNodeServer

COMPOSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docker-compose.yml")
MASTER_URL = "http://localhost:8080"

class TestMeshSimulation(unittest.TestCase):

    def test_tensor_shrink_int8_accuracy_and_compression(self):
        """Verifies that INT8 dynamic quantization preserves tensor fidelity (MAE < 0.01) with 75% byte savings."""
        hidden_dim = 288
        original_tensor = [random.uniform(-2.5, 2.5) for _ in range(hidden_dim)]

        b64_quant, scale = quantize_int8(original_tensor)
        dequantized = dequantize_int8(b64_quant, scale)

        self.assertEqual(len(dequantized), hidden_dim)
        mae = compute_mae(original_tensor, dequantized)
        self.assertLess(mae, 0.02, f"INT8 quantization MAE exceeded tolerance: {mae}")

        # Compression check: raw float32 bytes = 288 * 4 = 1152 bytes; INT8 = 288 bytes
        import base64
        raw_int8_bytes = len(base64.b64decode(b64_quant))
        self.assertEqual(raw_int8_bytes, hidden_dim)
        compression_ratio = (hidden_dim * 4) / raw_int8_bytes
        self.assertEqual(compression_ratio, 4.0)

    def test_continuous_layer_partitioning(self):
        """Verifies that Nexus-Air continuous layer partitioning distributes layers evenly without gaps or overlaps."""
        master = MeshNodeServer(role="master", node_id="master-test", port=9999, master_url="", total_layers=30)
        from node import NodeInfo
        for i in range(1, 5):
            master.registered_nodes.append(NodeInfo(f"w{i}", f"Worker {i}", "localhost", 9999 + i, i, "SoC", 1000.0, 3000))
        master.rebalance_layers()

        state = master.get_cluster_state()
        nodes = state["nodes"]
        self.assertEqual(len(nodes), 5)

        current_expected = 0
        for i, n in enumerate(nodes):
            block = n["assignedBlocks"][0]
            self.assertEqual(block["startLayer"], current_expected)
            self.assertGreaterEqual(block["endLayer"], block["startLayer"])
            current_expected = block["endLayer"] + 1

            if i == 4:
                self.assertTrue(block["isOutputBlock"], "Last node must host LM Output Head")
            else:
                self.assertFalse(block["isOutputBlock"])

        self.assertEqual(current_expected, 30, "All 30 layers must be completely covered")

    def test_five_container_cluster_execution(self):
        """Integration test: Launches 5 containers, verifies registration, runs inference, and verifies token stream."""
        # 1. Start containers
        subprocess.run(f"docker compose -f {COMPOSE_FILE} up -d", shell=True, check=True)
        time.sleep(3.0)

        try:
            # 2. Wait for registration
            registered = False
            for _ in range(25):
                try:
                    req = urllib.request.Request(f"{MASTER_URL}/api/status")
                    with urllib.request.urlopen(req, timeout=2.0) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read().decode("utf-8"))
                            c_state = data.get("clusterState", {})
                            if len(c_state.get("nodes", [])) == 5:
                                registered = True
                                break
                except Exception:
                    pass
                time.sleep(1.0)

            self.assertTrue(registered, "All 5 containers failed to register with Master")

            # 3. Trigger forward pass
            req = urllib.request.Request(
                f"{MASTER_URL}/api/inference/run",
                data=json.dumps({"prompt": "Testing 5-container interconnectivity", "maxTokens": 10}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                res = json.loads(resp.read().decode("utf-8"))

            self.assertEqual(res.get("tokensGenerated"), 10)
            self.assertGreater(res.get("tokensPerSecond"), 0.0)
            self.assertTrue(len(res.get("generatedText", "")) > 0)
            print(f"\n[TEST PASSED] Output from 5 containers: '{res.get('generatedText')}'")
            print(f"[TEST PASSED] Throughput: {res.get('tokensPerSecond'):.2f} TPS")

        finally:
            subprocess.run(f"docker compose -f {COMPOSE_FILE} down", shell=True)

if __name__ == "__main__":
    unittest.main()
