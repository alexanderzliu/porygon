import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from evals import cli, worker, workflows
from evals.temporal_payloads import SuiteInit


class Phase6ParallelTests(unittest.TestCase):
    def write_scenario(self, root: Path) -> Path:
        (root / "start.state").write_text("PLAYERS_HOUSE_2F", encoding="utf-8")
        scenario = root / "scenario.yaml"
        scenario.write_text(
            "\n".join(
                [
                    "id: synthetic_parallel",
                    "description: Synthetic parallel eval.",
                    "initial_state: start.state",
                    "success:",
                    "  location_eq: REDS_HOUSE_1F",
                    "limits:",
                    "  max_steps: 3",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return scenario

    def test_temporal_suite_init_uses_suite_concurrency_and_cli_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario = self.write_scenario(root)
            rom = root / "pokemon.gb"
            rom.write_bytes(b"rom")
            suite = root / "suite.yaml"
            suite.write_text(
                "\n".join(
                    [
                        f"scenario: {scenario}",
                        "trials: 1",
                        "concurrency: 3",
                        "matrix:",
                        "  - harness: fake_parallel",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            suite_init = cli.resolve_suite_temporal_init(
                suite,
                run_id="parallel-run",
                rom_path=rom,
                results_root=root / "results",
            )
            overridden = cli.resolve_suite_temporal_init(
                suite,
                run_id="parallel-run-override",
                rom_path=rom,
                results_root=root / "results",
                concurrency=2,
            )

            self.assertEqual(suite_init.concurrency, 3)
            self.assertEqual(overridden.concurrency, 2)

    def test_sweep_workflow_honors_concurrency_semaphore(self):
        async def exercise() -> tuple[dict, int]:
            active = 0
            max_active = 0

            async def fake_child_workflow(
                _workflow_fn,
                trial_init,
                *,
                id=None,
                task_queue=None,
            ):
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                active -= 1
                return {
                    "trial_id": trial_init.spec["trial_id"],
                    "workflow_id": id,
                    "task_queue": task_queue,
                }

            async def fake_activity(_name, _input, **_kwargs):
                return [{"summary": True}]

            specs = [
                {"run_id": "run-p", "trial_id": f"{index:03d}_trial"}
                for index in range(5)
            ]
            suite = SuiteInit(
                run_id="run-p",
                run_dir="/tmp/run-p",
                trial_specs=specs,
                task_queue="parallel-queue",
                concurrency=2,
            )
            with (
                patch.object(
                    workflows.workflow,
                    "execute_child_workflow",
                    side_effect=fake_child_workflow,
                ),
                patch.object(
                    workflows.workflow,
                    "execute_activity",
                    side_effect=fake_activity,
                ),
            ):
                result = await workflows.SweepWorkflow().run(suite)
            return result, max_active

        result, max_active = asyncio.run(exercise())

        self.assertEqual(max_active, 2)
        self.assertEqual(len(result["trials"]), 5)
        self.assertEqual(
            [trial["task_queue"] for trial in result["trials"]],
            ["parallel-queue"] * 5,
        )

    def test_trial_continue_as_new_boundary_still_fires(self):
        self.assertFalse(workflows._should_continue_as_new({"steps": 0}, 250))
        self.assertTrue(workflows._should_continue_as_new({"steps": 250}, 250))
        self.assertFalse(workflows._should_continue_as_new({"steps": 251}, 250))

    def test_worker_cli_passes_concurrency_knobs(self):
        with patch("evals.worker.run_worker", new_callable=AsyncMock) as run_worker:
            worker.main(
                [
                    "--activity-workers",
                    "4",
                    "--max-concurrent-activities",
                    "3",
                    "--max-concurrent-workflow-tasks",
                    "2",
                ]
            )

        run_worker.assert_awaited_once()
        kwargs = run_worker.await_args.kwargs
        self.assertEqual(kwargs["activity_workers"], 4)
        self.assertEqual(kwargs["max_concurrent_activities"], 3)
        self.assertEqual(kwargs["max_concurrent_workflow_tasks"], 2)


if __name__ == "__main__":
    unittest.main()
