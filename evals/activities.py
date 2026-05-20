from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # Temporal is optional for local-only eval development.
    from temporalio import activity
except ModuleNotFoundError:  # pragma: no cover - exercised implicitly by imports.
    class _ActivityFallback:
        def defn(self, *decorator_args, **_decorator_kwargs):
            if decorator_args and callable(decorator_args[0]):
                return decorator_args[0]

            def decorate(fn):
                return fn

            return decorate

        def info(self):
            class Info:
                attempt = 1

            return Info()

    activity = _ActivityFallback()  # type: ignore[assignment]

from agent.harness import RunningTotals
from evals.runner import (
    StepRef,
    _add_metrics_to_totals,
    _build_harness,
    _cap_outcome,
    _freeze_inputs,
    _jsonable,
    _read_json,
    finalize_trial,
    init_trial,
    run_agent_step,
)
from evals.temporal_payloads import (
    FinalizeSweepInput,
    FinalizeTrialInput,
    StepActivityInput,
    running_totals_from_payload,
    running_totals_to_payload,
    step_ref_to_payload,
    trial_spec_from_payload,
)


@activity.defn(name="init_trial")
def init_trial_activity(spec_payload: dict[str, Any]) -> dict[str, Any]:
    spec = trial_spec_from_payload(spec_payload)
    trial_dir = spec.results_root / spec.run_id / spec.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    _freeze_inputs(spec, trial_dir)

    harness = _build_harness(spec.harness.id, spec.harness.params)
    harness_static_config = harness.static_config()
    step_ref, milestone_reached = init_trial(spec, harness, trial_dir)

    totals = RunningTotals()
    return {
        "step_ref": step_ref_to_payload(step_ref),
        "milestone_reached": milestone_reached,
        "running_totals": running_totals_to_payload(totals),
        "harness_static_config": harness_static_config,
        "outcome": _cap_outcome(totals, spec.limits),
    }


@activity.defn(name="run_agent_step")
def run_agent_step_activity(input: StepActivityInput | dict[str, Any]) -> dict[str, Any]:
    input = _step_activity_input(input)
    spec = trial_spec_from_payload(input.spec)
    trial_dir = spec.results_root / spec.run_id / spec.trial_id
    harness = _build_harness(spec.harness.id, spec.harness.params)
    previous_step = StepRef(
        step_index=int(input.previous_step["step_index"]),
        path=Path(input.previous_step["path"]),
    )
    running_totals = running_totals_from_payload(input.running_totals)
    metrics, next_ref, milestone_reached = run_agent_step(
        spec=spec,
        harness=harness,
        previous_step=previous_step,
        running_totals=running_totals,
        trial_dir=trial_dir,
        attempt=_activity_attempt(),
    )
    next_totals = _add_metrics_to_totals(running_totals, metrics)

    return {
        "metrics": _jsonable(metrics),
        "step_ref": step_ref_to_payload(next_ref),
        "milestone_reached": milestone_reached,
        "running_totals": running_totals_to_payload(next_totals),
        "outcome": (
            "milestone_reached"
            if milestone_reached
            else _cap_outcome(next_totals, spec.limits)
        ),
    }


@activity.defn(name="finalize_trial")
def finalize_trial_activity(
    input: FinalizeTrialInput | dict[str, Any],
) -> dict[str, Any]:
    input = _finalize_trial_input(input)
    spec = trial_spec_from_payload(input.spec)
    trial_dir = spec.results_root / spec.run_id / spec.trial_id
    totals = running_totals_from_payload(input.totals)
    finalize_trial(
        trial_dir,
        spec=spec,
        outcome=input.outcome,
        milestone_reached=input.milestone_reached,
        totals=totals,
        error=input.error,
        harness_static_config=input.harness_static_config,
    )
    return _read_json(trial_dir / "trial.json")


@activity.defn(name="finalize_sweep")
def finalize_sweep_activity(
    input: FinalizeSweepInput | dict[str, Any],
) -> list[dict[str, Any]]:
    input = _finalize_sweep_input(input)
    from evals.cli import finalize_sweep

    return finalize_sweep(Path(input.run_dir))


def registered_activities() -> list[Any]:
    return [
        init_trial_activity,
        run_agent_step_activity,
        finalize_trial_activity,
        finalize_sweep_activity,
    ]


def _activity_attempt() -> int:
    try:
        return int(activity.info().attempt)
    except Exception:  # noqa: BLE001 - fallback also covers non-activity unit calls.
        return 1


def _step_activity_input(value: StepActivityInput | dict[str, Any]) -> StepActivityInput:
    if isinstance(value, StepActivityInput):
        return value
    return StepActivityInput(**value)


def _finalize_trial_input(
    value: FinalizeTrialInput | dict[str, Any],
) -> FinalizeTrialInput:
    if isinstance(value, FinalizeTrialInput):
        return value
    return FinalizeTrialInput(**value)


def _finalize_sweep_input(
    value: FinalizeSweepInput | dict[str, Any],
) -> FinalizeSweepInput:
    if isinstance(value, FinalizeSweepInput):
        return value
    return FinalizeSweepInput(**value)
