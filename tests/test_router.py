import unittest
from hybrid_ai_router import HybridCadAiRouter


class TestHybridCadAiRouter(unittest.TestCase):
    def setUp(self):
        # Typical 6.0 GB VRAM engineering laptop/workstation GPU (e.g., RTX 4050)
        self.router = HybridCadAiRouter(total_vram_gb=6.0, cad_reserved_vram_gb=2.8)

    def test_idle_workstation_allows_8b_reasoning_local(self):
        decision = self.router.select_route("math", running_processes=["explorer.exe", "code.exe"])
        self.assertFalse(decision.cad_active)
        self.assertEqual(decision.selected.model_id, "deepseek-r1:8b")

    def test_solidworks_active_caps_local_vram_to_cad_safe_4b(self):
        decision = self.router.select_route(
            "code",
            running_processes=["SLDWORKS.exe", "python.exe"],
            airgapped_opsec=True,
        )
        self.assertTrue(decision.cad_active)
        self.assertLessEqual(decision.selected.vram_gb, 3.2)
        self.assertEqual(decision.selected.model_id, "qwen3.5:4b")

    def test_cad_active_deep_reasoning_offloads_when_not_airgapped(self):
        decision = self.router.select_route(
            "math",
            running_processes=["FreeCAD.exe", "blender.exe"],
            airgapped_opsec=False,
            prefer_deep_reasoning=True,
        )
        self.assertTrue(decision.cad_active)
        self.assertEqual(decision.selected.provider, "cloud")

    def test_airgapped_opsec_never_leaks_to_cloud(self):
        with self.assertRaises(RuntimeError):
            # Requesting 'math' (4.9GB local) while CAD reserves 2.8GB/6GB AND airgapped=True must block cloud fallback
            self.router.select_route(
                "math",
                running_processes=["Inventor.exe"],
                airgapped_opsec=True,
            )


if __name__ == "__main__":
    unittest.main()
