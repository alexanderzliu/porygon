from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

try:  # Temporal is optional unless the --temporal path or worker is used.
    from temporalio import workflow
except ModuleNotFoundError:  # pragma: no cover - keeps local unit imports cheap.
    class _WorkflowFallback:
        def defn(self, *decorator_args, **_decorator_kwargs):
            if decorator_args and callable(decorator_args[0]):
                return decorator_args[0]

            def decorate(cls):
                return cls

            return decorate

        def run(self, fn):
            return fn

        async def execute_activity(self, *_args, **_kwargs):
            raise RuntimeError("Temporal support requires the 'temporalio' package")

        async def execute_child_workflow(self, *_args, **_kwargs):
            raise RuntimeError("Temporal support requires the 'temporalio' package")

        def continue_as_new(self, *_args, **_kwargs):
            raise RuntimeError("Temporal support requires the 'temporalio' package")

        def info(self):
            class Info:
                def is_continue_as_new_suggested(self) -> bool:
                    return False

            return Info()

    workflow = _WorkflowFallback()  # type: ignore[assignment]

from evals.temporal_payloads import (
    DEFAULT_TASK_QUEUE,
    FinalizeSweepInput,
    FinalizeTrialInput,
    StepActivityInput,
    SuiteInit,
    TrialInit,
    empty_running_totals_payload,
)

MILESTONE_REACHED = "milestone_reached"
STEP_CAP = "step_cap"
TIME_CAP = "time_cap"
COST_CAP = "cost_cap"
ERROR = "error"

INIT_TIMEOUT = dt.timedelta(minutes=5)
STEP_TIMEOUT = dt.timedelta(minutes=30)
FINALIZE_TIMEOUT = dt.timedelta(minutes=5)


@workflow.defn(name="TrialWorkflow")
class TrialWorkflow:
    @workflow.run
    async def run(self, input: TrialInit | dict[str, Any]) -> dict[str, Any]:
        input = _trial_init(input)
        totals = input.resume_running_totals or empty_running_totals_payload()
        step_ref = input.resume_step_ref
        harness_static_config = input.harness_static_config
        milestone_reached = False
        outcome: str | None = None
        error: str | None = None

        try:
            if step_ref is None:
                init_result = await workflow.execute_activity(
                    "init_trial",
                    input.spec,
                    start_to_close_timeout=INIT_TIMEOUT,
                )
                step_ref = init_result["step_ref"]
                totals = init_result["running_totals"]
                harness_static_config = init_result.get("harness_static_config")
                milestone_reached = bool(init_result["milestone_reached"])
                outcome = (
                    MILESTONE_REACHED
                    if milestone_reached
                    else init_result.get("outcome")
                )
            else:
                outcome = _cap_outcome(totals, input.spec.get("limits") or {})

            while outcome is None:
                step_result = await workflow.execute_activity(
                    "run_agent_step",
                    StepActivityInput(
                        spec=input.spec,
                        previous_step=step_ref,
                        running_totals=totals,
                    ),
                    start_to_close_timeout=STEP_TIMEOUT,
                )
                step_ref = step_result["step_ref"]
                totals = step_result["running_totals"]
                milestone_reached = bool(step_result["milestone_reached"])
                outcome = step_result.get("outcome")

                if outcome is None and _should_continue_as_new(
                    totals, input.continue_as_new_every
                ):
                    workflow.continue_as_new(
                        TrialInit(
                            spec=input.spec,
                            resume_step_ref=step_ref,
                            resume_running_totals=totals,
                            harness_static_config=harness_static_config,
                            continue_as_new_every=input.continue_as_new_every,
                        )
                    )
        except Exception as exc:  # noqa: BLE001 - preserve failure in trial.json.
            if "ContinueAsNew" in type(exc).__name__:
                raise
            outcome = ERROR
            error = f"{type(exc).__name__}: {exc}"

        if outcome is None:
            outcome = _cap_outcome(totals, input.spec.get("limits") or {}) or ERROR

        return await workflow.execute_activity(
            "finalize_trial",
            FinalizeTrialInput(
                spec=input.spec,
                outcome=outcome,
                milestone_reached=milestone_reached,
                totals=totals,
                error=error,
                harness_static_config=harness_static_config,
            ),
            start_to_close_timeout=FINALIZE_TIMEOUT,
        )


@workflow.defn(name="SweepWorkflow")
class SweepWorkflow:
    @workflow.run
    async def run(self, input: SuiteInit | dict[str, Any]) -> dict[str, Any]:
        input = _suite_init(input)
        semaphore = asyncio.Semaphore(max(1, int(input.concurrency)))
        trials = await asyncio.gather(
            *[
                _run_trial_child(input, spec, semaphore)
                for spec in input.trial_specs
            ]
        )

        summary = await workflow.execute_activity(
            "finalize_sweep",
            FinalizeSweepInput(run_dir=input.run_dir),
            start_to_close_timeout=FINALIZE_TIMEOUT,
        )
        return {
            "run_id": input.run_id,
            "run_dir": input.run_dir,
            "trials": trials,
            "summary_rows": summary,
        }


def _cap_outcome(totals: dict[str, Any], limits: dict[str, Any]) -> str | None:
    max_steps = limits.get("max_steps")
    if max_steps is not None and int(totals.get("steps", 0)) >= int(max_steps):
        return STEP_CAP

    max_seconds = limits.get("max_seconds")
    if max_seconds is not None and float(totals.get("wall_seconds", 0.0)) >= float(
        max_seconds
    ):
        return TIME_CAP

    max_usd = limits.get("max_usd")
    if max_usd is not None and float(totals.get("cost_usd", 0.0)) >= float(max_usd):
        return COST_CAP

    return None


def _should_continue_as_new(totals: dict[str, Any], every: int) -> bool:
    steps = int(totals.get("steps", 0))
    if steps <= 0:
        return False
    if every > 0 and steps % every == 0:
        return True
    try:
        return bool(workflow.info().is_continue_as_new_suggested())
    except Exception:  # noqa: BLE001 - fallback outside a real workflow.
        return False


async def _run_trial_child(
    input: SuiteInit,
    spec: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        return await workflow.execute_child_workflow(
            TrialWorkflow.run,
            TrialInit(
                spec=spec,
                continue_as_new_every=input.continue_as_new_every,
            ),
            id=f"eval-trial-{spec['run_id']}-{spec['trial_id']}",
            task_queue=input.task_queue or DEFAULT_TASK_QUEUE,
        )


def _trial_init(value: TrialInit | dict[str, Any]) -> TrialInit:
    if isinstance(value, TrialInit):
        return value
    return TrialInit(**value)


def _suite_init(value: SuiteInit | dict[str, Any]) -> SuiteInit:
    if isinstance(value, SuiteInit):
        return value
    return SuiteInit(**value)
