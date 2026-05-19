from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from agent.emulator import Emulator


@dataclass
class RunningTotals:
    steps: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    wall_seconds: float = 0.0


@dataclass
class ModelUsage:
    provider: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    request_id: str | None = None
    latency_ms: int | None = None


class UsageMeter(Protocol):
    def record(self, usage: ModelUsage) -> None: ...


@dataclass
class RecordingUsageMeter:
    records: list[ModelUsage] = field(default_factory=list)

    def record(self, usage: ModelUsage) -> None:
        self.records.append(usage)


@dataclass
class StepContext:
    emulator: "Emulator"
    step_index: int
    running_totals: RunningTotals
    workdir: Path
    usage_meter: UsageMeter


@dataclass
class Action:
    kind: str
    args: dict
    buttons: list[str]
    frames_elapsed: int
    success: bool | None = None
    result_text: str | None = None


@dataclass
class StepMetrics:
    step_index: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    cost_usd: float
    wall_ms: int
    decision_count: int
    tool_call_count: int
    button_press_count: int
    emulated_frames: int
    actions: list[Action]
    usage: list[ModelUsage]
    step_ref: str
    summarization_events: int = 0
    milestone_reached: bool = False


@dataclass
class StepCounters:
    tool_call_count: int
    summarization_events: int = 0


@dataclass
class StepResult:
    actions: list[Action]
    counters: StepCounters
    text_log: str


class Harness(Protocol):
    id: str
    version: str

    def step(self, ctx: StepContext) -> StepResult: ...

    def serialize_state(self) -> bytes: ...

    def load_state(self, blob: bytes) -> None: ...

    def static_config(self) -> dict: ...
