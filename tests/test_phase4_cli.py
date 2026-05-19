import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.harness import RunningTotals
from evals import cli
from evals.runner import TrialResult


class Phase4CliTests(unittest.TestCase):
    def test_run_suite_expands_matrix_trials_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario = root / "scenario.yaml"
            scenario.write_text("id: synthetic\n", encoding="utf-8")
            suite = root / "suite.yaml"
            suite.write_text(
                "\n".join(
                    [
                        f"scenario: {scenario}",
                        "trials: 2",
                        "concurrency: 1",
                        "matrix:",
                        "  - harness: fake_a",
                        "  - harness: fake_b",
                        "    params_override:",
                        "      model:",
                        "        temperature: 0.5",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            calls = []

            def fake_run_trial(
                scenario_path,
                harness_id,
                params_path=None,
                params_override=None,
                run_id=None,
                trial_index=0,
                *,
                rom_path=None,
                results_root=None,
                emulator_factory=None,
            ):
                calls.append(
                    {
                        "scenario_path": Path(scenario_path),
                        "harness_id": harness_id,
                        "params_path": params_path,
                        "params_override": params_override,
                        "run_id": run_id,
                        "trial_index": trial_index,
                        "rom_path": rom_path,
                    }
                )
                trial_id = f"{trial_index:03d}_synthetic_{harness_id}"
                trial_dir = Path(results_root) / run_id / trial_id
                step_dir = trial_dir / "step_001"
                step_dir.mkdir(parents=True)
                (step_dir / "memory_dump.json").write_text(
                    json.dumps({"location": "REDS_HOUSE_1F"}),
                    encoding="utf-8",
                )
                trial = {
                    "run_id": run_id,
                    "trial_id": trial_id,
                    "scenario_id": "synthetic",
                    "harness_id": harness_id,
                    "outcome": "milestone_reached",
                    "milestone_reached": True,
                    "completed_steps": 1,
                    "totals": {
                        "steps": 1,
                        "model_calls": 1,
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_read_tokens": 2,
                        "cache_creation_tokens": 3,
                        "cost_usd": 0.01,
                        "wall_seconds": 0.1,
                    },
                    "error": None,
                }
                (trial_dir / "trial.json").write_text(
                    json.dumps(trial),
                    encoding="utf-8",
                )
                return TrialResult(
                    run_id=run_id,
                    trial_id=trial_id,
                    trial_dir=trial_dir,
                    outcome="milestone_reached",
                    milestone_reached=True,
                    completed_steps=1,
                    totals=RunningTotals(steps=1, model_calls=1, cost_usd=0.01),
                )

            with patch("evals.cli.run_trial", side_effect=fake_run_trial):
                run_dir = cli.run_suite(
                    suite,
                    run_id="run-x",
                    results_root=root / "results",
                )

            trial_jsons = sorted(run_dir.glob("*/trial.json"))
            summary_lines = (run_dir / "summary.jsonl").read_text().splitlines()
            summary_rows = [json.loads(line) for line in summary_lines]

            self.assertEqual(len(calls), 4)
            self.assertEqual(len(trial_jsons), 4)
            self.assertEqual(len(summary_rows), 4)
            self.assertTrue((run_dir / "suite.yaml").exists())
            self.assertEqual([call["trial_index"] for call in calls], [0, 1, 2, 3])
            self.assertEqual(calls[2]["harness_id"], "fake_b")
            self.assertEqual(
                calls[2]["params_override"],
                {"model": {"temperature": 0.5}},
            )
            self.assertEqual(summary_rows[0]["final_location"], "REDS_HOUSE_1F")
            self.assertEqual(summary_rows[0]["cost_usd"], 0.01)

    def test_inspect_prints_outcome_cost_tokens_and_final_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = Path(tmp) / "trial"
            step_dir = trial_dir / "step_002"
            step_dir.mkdir(parents=True)
            (step_dir / "memory_dump.json").write_text(
                json.dumps({"location": "REDS_HOUSE_1F"}),
                encoding="utf-8",
            )
            (trial_dir / "trial.json").write_text(
                json.dumps(
                    {
                        "trial_id": "002_synthetic_fake",
                        "outcome": "milestone_reached",
                        "completed_steps": 2,
                        "totals": {
                            "model_calls": 2,
                            "input_tokens": 100,
                            "output_tokens": 40,
                            "cache_read_tokens": 10,
                            "cache_creation_tokens": 20,
                            "cost_usd": 0.123456,
                        },
                    }
                ),
                encoding="utf-8",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                details = cli.inspect_trial(trial_dir)

            text = output.getvalue()
            self.assertEqual(details["final_location"], "REDS_HOUSE_1F")
            self.assertIn("Outcome: milestone_reached", text)
            self.assertIn("Steps: 2", text)
            self.assertIn("Cost: $0.123456", text)
            self.assertIn("input=100", text)
            self.assertIn("output=40", text)
            self.assertIn("Final location: REDS_HOUSE_1F", text)


if __name__ == "__main__":
    unittest.main()
