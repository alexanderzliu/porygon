from __future__ import annotations

import tempfile
import time
from dataclasses import replace
from pathlib import Path

from agent.harness import (
    Harness,
    RecordingUsageMeter,
    RunningTotals,
    StepContext,
    StepMetrics,
    StepResult,
)


def run_one_step(
    harness: Harness,
    emulator,
    *,
    step_index: int,
    running_totals: RunningTotals | None = None,
    workdir: Path | None = None,
    step_ref: str = "",
    cost_usd: float = 0.0,
    milestone_reached: bool = False,
) -> tuple[StepResult, StepMetrics]:
    if workdir is None:
        with tempfile.TemporaryDirectory(prefix=f"porygon-step-{step_index:03d}-") as tmp:
            return run_one_step(
                harness,
                emulator,
                step_index=step_index,
                running_totals=running_totals,
                workdir=Path(tmp),
                step_ref=step_ref,
                cost_usd=cost_usd,
                milestone_reached=milestone_reached,
            )

    totals = running_totals or RunningTotals()
    meter = RecordingUsageMeter()
    ctx = StepContext(
        emulator=emulator,
        step_index=step_index,
        running_totals=totals,
        workdir=workdir,
        usage_meter=meter,
    )

    start = time.monotonic()
    result = harness.step(ctx)
    wall_ms = int((time.monotonic() - start) * 1000)

    metrics = build_step_metrics(
        step_index=step_index,
        result=result,
        usage=meter.records,
        wall_ms=wall_ms,
        step_ref=step_ref,
        cost_usd=cost_usd,
        milestone_reached=milestone_reached,
    )
    return result, metrics


def build_step_metrics(
    *,
    step_index: int,
    result: StepResult,
    usage,
    wall_ms: int,
    step_ref: str,
    cost_usd: float = 0.0,
    milestone_reached: bool = False,
) -> StepMetrics:
    return StepMetrics(
        step_index=step_index,
        model_calls=len(usage),
        input_tokens=sum(record.input_tokens for record in usage),
        output_tokens=sum(record.output_tokens for record in usage),
        cache_read_tokens=sum(record.cache_read_tokens for record in usage),
        cache_creation_tokens=sum(record.cache_creation_tokens for record in usage),
        cost_usd=cost_usd,
        wall_ms=wall_ms,
        decision_count=len(result.actions),
        tool_call_count=result.counters.tool_call_count,
        button_press_count=sum(len(action.buttons) for action in result.actions),
        emulated_frames=sum(action.frames_elapsed for action in result.actions),
        actions=result.actions,
        usage=list(usage),
        step_ref=step_ref,
        summarization_events=result.counters.summarization_events,
        milestone_reached=milestone_reached,
    )


def add_step_metrics_to_totals(
    totals: RunningTotals, metrics: StepMetrics
) -> RunningTotals:
    return replace(
        totals,
        steps=totals.steps + 1,
        model_calls=totals.model_calls + metrics.model_calls,
        input_tokens=totals.input_tokens + metrics.input_tokens,
        output_tokens=totals.output_tokens + metrics.output_tokens,
        cache_read_tokens=totals.cache_read_tokens + metrics.cache_read_tokens,
        cache_creation_tokens=totals.cache_creation_tokens
        + metrics.cache_creation_tokens,
        cost_usd=totals.cost_usd + metrics.cost_usd,
        wall_seconds=totals.wall_seconds + (metrics.wall_ms / 1000),
    )
