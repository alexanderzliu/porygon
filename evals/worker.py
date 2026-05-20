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
    with ThreadPoolExecutor() as activity_executor:
        worker = Worker(
            client,
            task_queue=task_queue,
            workflows=[SweepWorkflow, TrialWorkflow],
            activities=registered_activities(),
            activity_executor=activity_executor,
        )
        await worker.run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.worker")
    parser.add_argument("--address", default=DEFAULT_TEMPORAL_ADDRESS)
    parser.add_argument("--namespace", default=DEFAULT_TEMPORAL_NAMESPACE)
    parser.add_argument("--task-queue", default=DEFAULT_TASK_QUEUE)
    args = parser.parse_args(argv)
    asyncio.run(
        run_worker(
            address=args.address,
            namespace=args.namespace,
            task_queue=args.task_queue,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
