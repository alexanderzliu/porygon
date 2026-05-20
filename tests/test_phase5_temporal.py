import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.harness import Action, RunningTotals, StepMetrics
from evals import cli
from evals.activities import registered_activities, run_agent_step_activity
from evals.runner import StepRef, resolve_trial_spec
from evals.temporal_payloads import StepActivityInput, trial_spec_to_payload


class FakeHarness:
    def static_config(self):
        return {"fake": True}


class Phase5TemporalTests(unittest.TestCase):
    def write_scenario(self, root: Path) -> Path:
        (root / "start.state").write_text("PLAYERS_HOUSE_2F", encoding="utf-8")
        scenario = root / "scenario.yaml"
        scenario.write_text(
            "\n".join(
                [
                    "id: synthetic_temporal",
                    "description: Synthetic Temporal eval.",
                    "initial_state: start.state",
                    "success:",
                    "  location_eq: REDS_HOUSE_1F",
                    "limits:",
                    "  max_steps: 3",
                    "  max_seconds: 60",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return scenario

    def test_resolve_suite_temporal_init_freezes_trial_payloads(self):
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
                        "trials: 2",
                        "matrix:",
                        "  - harness: fake_temporal",
                        "    params_override:",
                        "      model:",
                        "        temperature: 0.25",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            suite_init = cli.resolve_suite_temporal_init(
                suite,
                run_id="temporal-run",
                rom_path=rom,
                results_root=root / "results",
                task_queue="custom-eval-queue",
            )

            self.assertEqual(suite_init.run_id, "temporal-run")
            self.assertEqual(suite_init.task_queue, "custom-eval-queue")
            self.assertEqual(len(suite_init.trial_specs), 2)
            self.assertTrue((Path(suite_init.run_dir) / "suite.yaml").exists())
            self.assertEqual(
                suite_init.trial_specs[0]["harness"]["params"]["model"][
                    "temperature"
                ],
                0.25,
            )
            self.assertEqual(
                [spec["trial_index"] for spec in suite_init.trial_specs],
                [0, 1],
            )

    def test_run_agent_step_activity_wraps_local_runner_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario = self.write_scenario(root)
            rom = root / "pokemon.gb"
            rom.write_bytes(b"rom")
            spec = resolve_trial_spec(
                scenario_path=scenario,
                harness_id="fake_temporal",
                run_id="run-a",
                trial_index=0,
                rom_path=rom,
                results_root=root / "results",
            )
            previous = StepRef(step_index=0, path=root / "step_000")
            next_ref = StepRef(step_index=1, path=root / "step_001")
            metrics = StepMetrics(
                step_index=1,
                model_calls=1,
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=2,
                cache_creation_tokens=3,
                cost_usd=0.01,
                wall_ms=250,
                decision_count=1,
                tool_call_count=1,
                button_press_count=1,
                emulated_frames=30,
                actions=[
                    Action(
                        kind="press_buttons",
                        args={"buttons": ["a"]},
                        buttons=["a"],
                        frames_elapsed=30,
                    )
                ],
                usage=[],
                step_ref=str(next_ref.path),
                milestone_reached=True,
            )

            with (
                patch("evals.activities._build_harness", return_value=FakeHarness()),
                patch(
                    "evals.activities.run_agent_step",
                    return_value=(metrics, next_ref, True),
                ) as wrapped,
            ):
                result = run_agent_step_activity(
                    StepActivityInput(
                        spec=trial_spec_to_payload(spec),
                        previous_step={
                            "step_index": previous.step_index,
                            "path": str(previous.path),
                        },
                        running_totals={
                            "steps": 0,
                            "model_calls": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "cache_read_tokens": 0,
                            "cache_creation_tokens": 0,
                            "cost_usd": 0.0,
                            "wall_seconds": 0.0,
                        },
                    )
                )

            self.assertEqual(result["outcome"], "milestone_reached")
            self.assertEqual(result["step_ref"]["step_index"], 1)
            self.assertEqual(result["running_totals"]["steps"], 1)
            self.assertEqual(result["running_totals"]["cost_usd"], 0.01)
            wrapped.assert_called_once()

    def test_registered_activities_are_importable_without_temporal_sdk(self):
        names = {activity.__name__ for activity in registered_activities()}
        self.assertEqual(
            names,
            {
                "init_trial_activity",
                "run_agent_step_activity",
                "finalize_trial_activity",
                "finalize_sweep_activity",
            },
        )


if __name__ == "__main__":
    unittest.main()
