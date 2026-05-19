import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from agent.harness import Action, StepCounters, StepResult
from agent.memory_reader import MemoryDump
from evals.runner import TrialOutcome, finalize_trial, run_trial


def dump(location: str) -> MemoryDump:
    return MemoryDump(
        player_name="RED",
        rival_name="BLUE",
        money=0,
        location=location,
        map_id=0x25 if location == "PLAYERS_HOUSE_1F" else 0x26,
        coordinates=(1, 1),
        valid_moves=[],
        badges=[],
        inventory=[],
        dialog=None,
        party=[],
        raw={},
    )


class FakeEmulator:
    def __init__(self, rom_path, headless=True, sound=False):
        self.location = "PLAYERS_HOUSE_2F"

    def initialize(self):
        return None

    def load_state(self, path):
        text = Path(path).read_bytes().decode("utf-8")
        self.location = "PLAYERS_HOUSE_1F" if "1F" in text else "PLAYERS_HOUSE_2F"

    def save_state(self, path):
        Path(path).write_bytes(self.location.encode("utf-8"))

    def get_memory_dump(self):
        return dump(self.location)

    def stop(self):
        return None


class FakeHarness:
    id = "fake_phase3"
    version = "test"

    def __init__(self, calls):
        self.calls = calls

    def step(self, ctx):
        self.calls.append(ctx.step_index)
        ctx.emulator.location = "PLAYERS_HOUSE_1F"
        return StepResult(
            actions=[
                Action(
                    kind="press_buttons",
                    args={"buttons": ["down"]},
                    buttons=["down"],
                    frames_elapsed=130,
                    success=True,
                )
            ],
            counters=StepCounters(tool_call_count=0),
            text_log="moved",
        )

    def serialize_state(self):
        return b"fake-harness-state"

    def load_state(self, blob):
        return None

    def static_config(self):
        return {"fake": True}


class Phase3RunnerTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        module = types.ModuleType("evals.harnesses.fake_phase3")
        module.build = lambda params: FakeHarness(self.calls)
        sys.modules["evals.harnesses.fake_phase3"] = module

    def tearDown(self):
        sys.modules.pop("evals.harnesses.fake_phase3", None)

    def write_scenario(self, root: Path, initial_location: str) -> Path:
        (root / "start.state").write_text(initial_location, encoding="utf-8")
        scenario = root / "scenario.yaml"
        scenario.write_text(
            "\n".join(
                [
                    "id: synthetic_downstairs",
                    "description: Synthetic transition to 1F.",
                    "initial_state: start.state",
                    "success:",
                    "  all:",
                    "    - location_eq: REDS_HOUSE_1F",
                    "limits:",
                    "  max_steps: 5",
                    "  max_seconds: 60",
                    "  max_usd: 1.0",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return scenario

    def test_fake_harness_hits_milestone_without_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario = self.write_scenario(root, "PLAYERS_HOUSE_2F")

            result = run_trial(
                scenario,
                "fake_phase3",
                run_id="run-a",
                results_root=root / "results",
                emulator_factory=FakeEmulator,
            )

            trial_dir = result.trial_dir
            metrics = json.loads(
                (trial_dir / "step_001" / "metrics.json").read_text()
            )
            steps = (trial_dir / "steps.jsonl").read_text().splitlines()

            self.assertEqual(result.outcome, TrialOutcome.MILESTONE_REACHED)
            self.assertEqual(self.calls, [1])
            self.assertTrue((trial_dir / "trial.json").exists())
            self.assertTrue((trial_dir / "step_000" / "state.bin").exists())
            self.assertTrue((trial_dir / "step_000" / "harness_state.bin").exists())
            self.assertTrue((trial_dir / "step_000" / "memory_dump.json").exists())
            self.assertEqual(metrics["model_calls"], 0)
            self.assertEqual(metrics["cost_usd"], 0.0)
            self.assertTrue(metrics["milestone_reached"])
            self.assertEqual(len(steps), 1)

            finalize_trial(trial_dir)
            finalize_trial(trial_dir)
            self.assertEqual(len((trial_dir / "steps.jsonl").read_text().splitlines()), 1)

    def test_initial_success_finishes_without_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario = self.write_scenario(root, "PLAYERS_HOUSE_1F")

            result = run_trial(
                scenario,
                "fake_phase3",
                run_id="run-b",
                results_root=root / "results",
                emulator_factory=FakeEmulator,
            )

            self.assertEqual(result.outcome, TrialOutcome.MILESTONE_REACHED)
            self.assertEqual(result.completed_steps, 0)
            self.assertEqual(self.calls, [])
            self.assertFalse((result.trial_dir / "step_001").exists())

    def test_existing_canonical_step_short_circuits_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario = self.write_scenario(root, "PLAYERS_HOUSE_2F")

            first = run_trial(
                scenario,
                "fake_phase3",
                run_id="run-c",
                results_root=root / "results",
                emulator_factory=FakeEmulator,
            )
            second = run_trial(
                scenario,
                "fake_phase3",
                run_id="run-c",
                results_root=root / "results",
                emulator_factory=FakeEmulator,
            )

            self.assertEqual(first.trial_dir, second.trial_dir)
            self.assertEqual(self.calls, [1])
            self.assertEqual(second.completed_steps, 1)


if __name__ == "__main__":
    unittest.main()
