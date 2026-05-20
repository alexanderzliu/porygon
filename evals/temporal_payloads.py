from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from agent.harness import RunningTotals
    from evals.runner import ResolvedTrialSpec


DEFAULT_TASK_QUEUE = "eval-trials"
DEFAULT_TEMPORAL_ADDRESS = "localhost:7233"
DEFAULT_TEMPORAL_NAMESPACE = "default"


@dataclass
class TrialInit:
    spec: dict[str, Any]
    resume_step_ref: dict[str, Any] | None = None
    resume_running_totals: dict[str, Any] | None = None
    harness_static_config: dict[str, Any] | None = None
    continue_as_new_every: int = 250


@dataclass
class SuiteInit:
    run_id: str
    run_dir: str
    trial_specs: list[dict[str, Any]]
    task_queue: str = DEFAULT_TASK_QUEUE
    continue_as_new_every: int = 250


@dataclass
class StepActivityInput:
    spec: dict[str, Any]
    previous_step: dict[str, Any]
    running_totals: dict[str, Any]


@dataclass
class FinalizeTrialInput:
    spec: dict[str, Any]
    outcome: str
    milestone_reached: bool
    totals: dict[str, Any]
    error: str | None = None
    harness_static_config: dict[str, Any] | None = None


@dataclass
class FinalizeSweepInput:
    run_dir: str


def trial_spec_to_payload(spec: "ResolvedTrialSpec") -> dict[str, Any]:
    return {
        "scenario_path": str(spec.scenario_path),
        "scenario_id": spec.scenario_id,
        "description": spec.description,
        "initial_state": str(spec.initial_state),
        "success": spec.success,
        "limits": spec.limits,
        "harness": {
            "id": spec.harness.id,
            "params": spec.harness.params,
        },
        "rom_path": str(spec.rom_path),
        "results_root": str(spec.results_root),
        "run_id": spec.run_id,
        "trial_index": spec.trial_index,
        "trial_id": spec.trial_id,
    }


def trial_spec_from_payload(payload: dict[str, Any]) -> ResolvedTrialSpec:
    from evals.runner import ResolvedHarnessConfig, ResolvedTrialSpec

    harness = payload.get("harness") or {}
    return ResolvedTrialSpec(
        scenario_path=Path(payload["scenario_path"]),
        scenario_id=str(payload["scenario_id"]),
        description=str(payload.get("description", "")),
        initial_state=Path(payload["initial_state"]),
        success=dict(payload.get("success") or {}),
        limits=dict(payload.get("limits") or {}),
        harness=ResolvedHarnessConfig(
            id=str(harness["id"]),
            params=dict(harness.get("params") or {}),
        ),
        rom_path=Path(payload["rom_path"]),
        results_root=Path(payload["results_root"]),
        run_id=str(payload["run_id"]),
        trial_index=int(payload.get("trial_index", 0)),
        trial_id=str(payload["trial_id"]),
    )


def step_ref_to_payload(step_ref) -> dict[str, Any]:
    return {
        "step_index": int(step_ref.step_index),
        "path": str(step_ref.path),
    }


def running_totals_from_payload(payload: dict[str, Any] | None) -> RunningTotals:
    from agent.harness import RunningTotals

    payload = payload or {}
    return RunningTotals(
        steps=int(payload.get("steps", 0)),
        model_calls=int(payload.get("model_calls", 0)),
        input_tokens=int(payload.get("input_tokens", 0)),
        output_tokens=int(payload.get("output_tokens", 0)),
        cache_read_tokens=int(payload.get("cache_read_tokens", 0)),
        cache_creation_tokens=int(payload.get("cache_creation_tokens", 0)),
        cost_usd=float(payload.get("cost_usd", 0.0)),
        wall_seconds=float(payload.get("wall_seconds", 0.0)),
    )


def running_totals_to_payload(totals: "RunningTotals") -> dict[str, Any]:
    return {
        "steps": totals.steps,
        "model_calls": totals.model_calls,
        "input_tokens": totals.input_tokens,
        "output_tokens": totals.output_tokens,
        "cache_read_tokens": totals.cache_read_tokens,
        "cache_creation_tokens": totals.cache_creation_tokens,
        "cost_usd": totals.cost_usd,
        "wall_seconds": totals.wall_seconds,
    }


def empty_running_totals_payload() -> dict[str, Any]:
    return {
        "steps": 0,
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "cost_usd": 0.0,
        "wall_seconds": 0.0,
    }
