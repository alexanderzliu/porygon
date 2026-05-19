import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from agent.harness import Action, ModelUsage, StepCounters, StepResult
from agent.memory_strategy import SummarizeAndReplace
from agent.prompt import PromptBuilder
from agent.state_formatter import StateFormatter
from agent.step_runner import run_one_step
from evals.harnesses.baseline import build as build_baseline_harness


class FakeHarness:
    id = "fake"
    version = "test"

    def step(self, ctx):
        ctx.usage_meter.record(
            ModelUsage(
                provider="fake",
                model_id="fake-model",
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=2,
                cache_creation_tokens=1,
            )
        )
        return StepResult(
            actions=[
                Action(
                    kind="press_buttons",
                    args={"buttons": ["a", "down"]},
                    buttons=["a", "down"],
                    frames_elapsed=260,
                    success=True,
                )
            ],
            counters=StepCounters(tool_call_count=1),
            text_log="pressed",
        )

    def serialize_state(self):
        return b""

    def load_state(self, blob):
        return None

    def static_config(self):
        return {}


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="response-1",
            usage=SimpleNamespace(
                input_tokens=7,
                output_tokens=3,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            content=[SimpleNamespace(type="text", text="summary text")],
        )


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


class FakeEmulator:
    def __init__(self):
        self.pressed = []

    def get_screenshot(self):
        return Image.new("RGB", (2, 2))

    def press_buttons(self, buttons, wait=True):
        self.pressed.append((buttons, wait))
        return "\n".join(f"Pressed {button}" for button in buttons)

    def get_state_from_memory(self):
        return "Player: RED\nLocation: PLAYERS_HOUSE_2F\n"

    def get_collision_map(self):
        return "+----------+\n|....@.....|\n+----------+"


class FakeUsageMeter:
    def __init__(self):
        self.records = []

    def record(self, usage):
        self.records.append(usage)


class Phase1HarnessTests(unittest.TestCase):
    def test_step_runner_builds_metrics_from_harness_result_and_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, metrics = run_one_step(
                FakeHarness(),
                emulator=object(),
                step_index=3,
                workdir=Path(tmp),
                step_ref="step_003",
            )

        self.assertEqual(metrics.step_index, 3)
        self.assertEqual(metrics.model_calls, 1)
        self.assertEqual(metrics.input_tokens, 10)
        self.assertEqual(metrics.output_tokens, 5)
        self.assertEqual(metrics.cache_read_tokens, 2)
        self.assertEqual(metrics.cache_creation_tokens, 1)
        self.assertEqual(metrics.decision_count, 1)
        self.assertEqual(metrics.tool_call_count, 1)
        self.assertEqual(metrics.button_press_count, 2)
        self.assertEqual(metrics.emulated_frames, 260)
        self.assertEqual(metrics.step_ref, "step_003")

    def test_baseline_harness_runs_one_fake_model_tool_turn(self):
        fake_tool_call = SimpleNamespace(
            type="tool_use",
            id="tool-1",
            name="press_buttons",
            input={"buttons": ["down"], "wait": True},
        )
        fake_response = SimpleNamespace(
            id="response-1",
            usage=SimpleNamespace(
                input_tokens=11,
                output_tokens=6,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            content=[
                SimpleNamespace(type="text", text="I will move down."),
                fake_tool_call,
            ],
        )
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
        emulator = FakeEmulator()

        with patch("evals.harnesses.baseline.AnthropicBedrock", return_value=fake_client):
            harness = build_baseline_harness({"history": {"max_history": 30}})

        with tempfile.TemporaryDirectory() as tmp:
            _, metrics = run_one_step(
                harness,
                emulator=emulator,
                step_index=1,
                workdir=Path(tmp),
            )
            self.assertTrue((Path(tmp) / "prompt.json").exists())
            self.assertTrue((Path(tmp) / "response.json").exists())
            self.assertTrue((Path(tmp) / "screenshot.png").exists())
            self.assertTrue((Path(tmp) / "memory_dump.json").exists())
            self.assertTrue((Path(tmp) / "collision_map.txt").exists())

        self.assertEqual(emulator.pressed, [(["down"], True)])
        self.assertEqual(metrics.model_calls, 1)
        self.assertEqual(metrics.tool_call_count, 1)
        self.assertEqual(metrics.button_press_count, 1)
        self.assertEqual(metrics.actions[0].kind, "press_buttons")
        self.assertEqual(len(harness.message_history), 3)

    def test_summarize_and_replace_waits_for_threshold(self):
        client = FakeClient()
        meter = FakeUsageMeter()
        strategy = SummarizeAndReplace(
            max_history=3,
            model_name="fake-model",
            max_tokens=100,
            temperature=1.0,
            prompt_builder=PromptBuilder(),
            state_formatter=StateFormatter(),
        )

        history, events = strategy.maybe_summarize(
            history=[{"role": "user", "content": "hello"}],
            emulator=FakeEmulator(),
            client=client,
            usage_meter=meter,
        )
        self.assertEqual(events, 0)
        self.assertEqual(len(history), 1)
        self.assertEqual(meter.records, [])

    def test_summarize_and_replace_summarizes_at_threshold(self):
        client = FakeClient()
        meter = FakeUsageMeter()
        strategy = SummarizeAndReplace(
            max_history=2,
            model_name="fake-model",
            max_tokens=100,
            temperature=1.0,
            prompt_builder=PromptBuilder(),
            state_formatter=StateFormatter(),
        )

        history, events = strategy.maybe_summarize(
            history=[
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ],
            emulator=FakeEmulator(),
            client=client,
            usage_meter=meter,
        )

        self.assertEqual(events, 1)
        self.assertEqual(len(history), 1)
        self.assertIn("summary text", history[0]["content"][0]["text"])
        self.assertEqual(len(meter.records), 1)
        self.assertEqual(meter.records[0].input_tokens, 7)


if __name__ == "__main__":
    unittest.main()
