from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelMode:
    key: str
    label: str
    default_model: str
    alternatives: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class HardwareRecommendation:
    min_ram_gb: int
    max_ram_gb: int | None
    machine_label: str
    default_model: str
    better_mode: str
    note: str

    def matches(self, ram_gb: int) -> bool:
        if ram_gb < self.min_ram_gb:
            return False
        if self.max_ram_gb is None:
            return True
        return ram_gb <= self.max_ram_gb


MODEL_MODES: tuple[ModelMode, ...] = (
    ModelMode(
        key="light",
        label="Light Mode",
        default_model="qwen3:4b",
        alternatives=("phi4-mini", "qwen2.5:3b"),
        description="Basic chat, summaries, and light tool usage.",
    ),
    ModelMode(
        key="fast",
        label="Fast Mode",
        default_model="qwen3:8b",
        alternatives=(),
        description="Best default for most local assistant workflows.",
    ),
    ModelMode(
        key="balanced",
        label="Balanced Mode",
        default_model="qwen3:14b",
        alternatives=("qwen3:8b",),
        description="Better planning and summaries when slower responses are acceptable.",
    ),
    ModelMode(
        key="power",
        label="Power Mode",
        default_model="qwen3:30b",
        alternatives=("qwen3:32b",),
        description="Power-user mode for heavier agent tasks on larger machines.",
    ),
)


MAC_MINI_RECOMMENDATIONS: tuple[HardwareRecommendation, ...] = (
    HardwareRecommendation(
        min_ram_gb=0,
        max_ram_gb=8,
        machine_label="8GB RAM",
        default_model="qwen2.5:3b / qwen3:4b / phi4-mini",
        better_mode="Not recommended for 8B",
        note="Basic chat, summaries, and light tools. There is not much room for 8B.",
    ),
    HardwareRecommendation(
        min_ram_gb=9,
        max_ram_gb=16,
        machine_label="16GB RAM",
        default_model="qwen3:8b",
        better_mode="qwen3:14b if slower responses are acceptable",
        note="Best default for most Mac mini users.",
    ),
    HardwareRecommendation(
        min_ram_gb=17,
        max_ram_gb=24,
        machine_label="24GB RAM",
        default_model="qwen3:14b",
        better_mode="qwen3:8b for Fast Mode",
        note="Good balance between quality and speed.",
    ),
    HardwareRecommendation(
        min_ram_gb=25,
        max_ram_gb=32,
        machine_label="32GB RAM",
        default_model="qwen3:14b",
        better_mode="qwen3:30b / MoE option if it performs well",
        note="Better summaries, reasoning, and planning.",
    ),
    HardwareRecommendation(
        min_ram_gb=33,
        max_ram_gb=None,
        machine_label="64GB+ RAM",
        default_model="qwen3:30b",
        better_mode="Larger Qwen / DeepSeek Coder / Mixtral-style models",
        note="Power users and heavier agent tasks.",
    ),
)


def detect_memory_gb() -> int | None:
    system = platform.system().lower()
    if system == "darwin":
        return _detect_macos_memory_gb()
    if system == "linux":
        return _detect_linux_memory_gb()
    return None


def recommend_for_memory(ram_gb: int | None) -> HardwareRecommendation:
    if ram_gb is None:
        return MAC_MINI_RECOMMENDATIONS[1]
    for recommendation in MAC_MINI_RECOMMENDATIONS:
        if recommendation.matches(ram_gb):
            return recommendation
    return MAC_MINI_RECOMMENDATIONS[-1]


def _detect_macos_memory_gb() -> int | None:
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return _bytes_to_rounded_gb(int(output))
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def _detect_linux_memory_gb() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                return _bytes_to_rounded_gb(int(parts[1]) * 1024)
    return None


def _bytes_to_rounded_gb(value: int) -> int:
    gib = value / 1024 / 1024 / 1024
    return int(round(gib))
