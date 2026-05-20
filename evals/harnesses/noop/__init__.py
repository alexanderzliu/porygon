from __future__ import annotations

from dataclasses import dataclass

from agent.harness import Harness, StepContext, StepCounters, StepResult


@dataclass
class NoopHarness(Harness):
    id: str = "noop"
    version: str = "phase5-smoke"

    def step(self, ctx: StepContext) -> StepResult:
        ctx.workdir.mkdir(parents=True, exist_ok=True)
        return StepResult(
            actions=[],
            counters=StepCounters(tool_call_count=0),
            text_log="noop",
        )

    def serialize_state(self) -> bytes:
        return b"noop"

    def load_state(self, blob: bytes) -> None:
        return None

    def static_config(self) -> dict:
        return {"mode": "noop"}


def build(params: dict) -> NoopHarness:
    return NoopHarness()
