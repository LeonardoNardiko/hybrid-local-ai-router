"""
CAD-Safe Hybrid Local/Cloud AI Router for Engineering Workstations.
Prevents GPU VRAM starvation when 3D CAD, BIM, FEA, or Cycles renderers are active
while maintaining strict OPSEC air-gapped guarantees for confidential engineering data.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
import json
import urllib.request


HEAVY_CAD_PROCESSES = frozenset({
    "sldworks.exe",
    "inventor.exe",
    "edge.exe",          # Siemens Solid Edge
    "acad.exe",
    "revit.exe",
    "freecad.exe",
    "blender.exe",
    "ccx.exe",           # CalculiX FEA solver
})


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    provider: str        # 'ollama' | 'cloud'
    model_id: str
    vram_gb: float
    capabilities: frozenset[str]


DEFAULT_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("embed", "ollama", "bge-m3:latest", 1.1, frozenset({"embedding", "rag"})),
    ModelSpec("fast-cad-safe", "ollama", "qwen3.5:4b", 3.1, frozenset({"code", "json", "chat"})),
    ModelSpec("vision-cad", "ollama", "qwen3-vl:4b", 3.1, frozenset({"vision", "ocr", "drawing"})),
    ModelSpec("coder-local", "ollama", "qwen2.5-coder:7b", 4.4, frozenset({"code", "cad-script"})),
    ModelSpec("reasoning-local", "ollama", "deepseek-r1:8b", 4.9, frozenset({"math", "reasoning", "code"})),
    ModelSpec("cloud-heavy", "cloud", "cloud-reasoning-pro", 0.0, frozenset({"math", "reasoning", "code", "vision"})),
)


@dataclass(frozen=True)
class RouteDecision:
    selected: ModelSpec
    cad_active: bool
    available_vram_budget_gb: float
    reason: str


class HybridCadAiRouter:
    def __init__(
        self,
        total_vram_gb: float = 6.0,
        cad_reserved_vram_gb: float = 2.8,
        registry: Iterable[ModelSpec] = DEFAULT_REGISTRY,
    ) -> None:
        self.total_vram_gb = total_vram_gb
        self.cad_reserved_vram_gb = cad_reserved_vram_gb
        self.registry = tuple(registry)

    def detect_cad_activity(self, running_processes: Iterable[str]) -> bool:
        normalized = {p.strip().lower() for p in running_processes}
        return bool(normalized & HEAVY_CAD_PROCESSES)

    def select_route(
        self,
        task_capability: str,
        running_processes: Iterable[str] = (),
        airgapped_opsec: bool = False,
        prefer_deep_reasoning: bool = False,
    ) -> RouteDecision:
        cad_active = self.detect_cad_activity(running_processes)
        vram_budget = (
            max(0.0, self.total_vram_gb - self.cad_reserved_vram_gb)
            if cad_active
            else self.total_vram_gb * 0.85
        )

        candidates = [
            m for m in self.registry
            if task_capability in m.capabilities
            and (not airgapped_opsec or m.provider == "ollama")
            and (m.provider == "cloud" or m.vram_gb <= vram_budget)
        ]

        if not candidates:
            raise RuntimeError(
                f"No compliant model for capability='{task_capability}' "
                f"(airgapped={airgapped_opsec}, vram_budget={vram_budget:.2f}GB)."
            )

        # When CAD is open and deep reasoning is needed without OPSEC lock, offload to cloud to protect viewport FPS
        if cad_active and prefer_deep_reasoning and not airgapped_opsec:
            cloud_opts = [m for m in candidates if m.provider == "cloud"]
            if cloud_opts:
                return RouteDecision(
                    selected=cloud_opts[0],
                    cad_active=True,
                    available_vram_budget_gb=vram_budget,
                    reason="3D CAD active: offloaded heavy reasoning to cloud to preserve GPU viewport VRAM.",
                )

        # Prefer highest-capability local model that fits safely inside the VRAM budget
        local_opts = sorted(
            [m for m in candidates if m.provider == "ollama"],
            key=lambda m: m.vram_gb,
            reverse=True,
        )
        if local_opts:
            chosen = local_opts[0]
            mode_note = "CAD-safe local budget" if cad_active else "full local GPU budget"
            return RouteDecision(
                selected=chosen,
                cad_active=cad_active,
                available_vram_budget_gb=vram_budget,
                reason=f"Selected sovereign local model '{chosen.model_id}' ({chosen.vram_gb}GB) under {mode_note}.",
            )

        return RouteDecision(
            selected=candidates[0],
            cad_active=cad_active,
            available_vram_budget_gb=vram_budget,
            reason="Fallback to cloud provider (local VRAM budget exceeded).",
        )

    @staticmethod
    def query_ollama(
        prompt: str,
        model_id: str = "qwen3.5:4b",
        endpoint: str = "http://127.0.0.1:11434/api/generate",
        timeout_s: float = 60.0,
    ) -> Optional[str]:
        payload = json.dumps({"model": model_id, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response")
