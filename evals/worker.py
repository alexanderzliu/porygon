from __future__ import annotations

import argparse
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor

from evals.temporal_payloads import (
    DEFAULT_TASK_QUEUE,
    DEFAULT_TEMPORAL_ADDRESS,
    DEFAULT_TEMPORAL_NAMESPACE,
)


async def run_worker(
    *,
    address: str = DEFAULT_TEMPORAL_ADDRESS,
    namespace: str = DEFAULT_TEMPORAL_NAMESPACE,
    task_queue: str = DEFAULT_TASK_QUEUE,
    activity_workers: int | None = None,
    max_concurrent_activities: int | None = None,
    max_concurrent_workflow_tasks: int | None = None,
) -> None:
    try:
        from temporalio.client import Client
        from temporalio.worker import Worker
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Temporal support requires the 'temporalio' package. "
            "Install project requirements before running evals.worker."
        ) from exc

    from evals.activities import registered_activities
    from evals.workflows import SweepWorkflow, TrialWorkflow

    client = await Client.connect(address, namespace=namespace)
    effective_activity_limit = max_concurrent_activities or activity_workers
    worker_options = {}
    if effective_activity_limit is not None:
        worker_options["max_concurrent_activities"] = effective_activity_limit
    if max_concurrent_workflow_tasks is not None:
        worker_options["max_concurrent_workflow_tasks"] = max_concurrent_workflow_tasks

    with ThreadPoolExecutor(max_workers=activity_workers) as activity_executor:
        worker = Worker(
            client,
            task_queue=task_queue,
            workflows=[SweepWorkflow, TrialWorkflow],
            activities=registered_activities(),
            activity_executor=activity_executor,
            **worker_options,
        )
        await worker.run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.worker")
    parser.add_argument("--address", default=DEFAULT_TEMPORAL_ADDRESS)
    parser.add_argument("--namespace", default=DEFAULT_TEMPORAL_NAMESPACE)
    parser.add_argument("--task-queue", default=DEFAULT_TASK_QUEUE)
    parser.add_argument(
        "--activity-workers",
        type=_positive_arg,
        help="Max threads available for activity execution",
    )
    parser.add_argument(
        "--max-concurrent-activities",
        type=_positive_arg,
        help="Temporal worker activity poll/concurrency limit",
    )
    parser.add_argument(
        "--max-concurrent-workflow-tasks",
        type=_positive_arg,
        help="Temporal worker workflow task concurrency limit",
    )
    args = parser.parse_args(argv)
    asyncio.run(
        run_worker(
            address=args.address,
            namespace=args.namespace,
            task_queue=args.task_queue,
            activity_workers=args.activity_workers,
            max_concurrent_activities=args.max_concurrent_activities,
            max_concurrent_workflow_tasks=args.max_concurrent_workflow_tasks,
        )
    )
    return 0


def _positive_arg(value: str) -> int:
    resolved = int(value)
    if resolved < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return resolved


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
