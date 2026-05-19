from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from agent.emulator import Emulator
from agent.harness import RunningTotals, StepContext
from agent.prompt import SYSTEM_PROMPT, SUMMARY_PROMPT, build_tool_schema
from agent.state_formatter import get_screenshot_base64
from agent.step_runner import add_step_metrics_to_totals, run_one_step
from agent.tui import NULL_TUI
from config import USE_NAVIGATOR
from evals.harnesses.baseline import build as build_baseline_harness

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

AVAILABLE_TOOLS = build_tool_schema(USE_NAVIGATOR)


class SimpleAgent:
    def __init__(
        self,
        rom_path,
        headless=True,
        sound=False,
        max_history=60,
        load_state=None,
        tui=None,
    ):
        """Initialize the simple agent.

        Args:
            rom_path: Path to the ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
            max_history: Maximum number of messages in history before summarization
            load_state: Optional PyBoy state file to load after initialization
            tui: Optional TUI instance for live rendering; defaults to a no-op stub
        """
        self.emulator = Emulator(rom_path, headless, sound)
        self.emulator.initialize()
        self.tui = tui or NULL_TUI
        self.harness = build_baseline_harness(
            {"history": {"max_history": max_history}, "tui": self.tui}
        )
        self.client = self.harness.client
        self.running = True
        self.max_history = max_history
        self.running_totals = RunningTotals()

        if load_state:
            logger.info("Loading saved state from %s", load_state)
            self.emulator.load_state(load_state)

    @property
    def message_history(self):
        return self.harness.message_history

    @message_history.setter
    def message_history(self, value):
        self.harness.message_history = value

    def process_tool_call(self, tool_call):
        with tempfile.TemporaryDirectory(prefix="porygon-simple-tool-") as tmp:
            ctx = StepContext(
                emulator=self.emulator,
                step_index=self.running_totals.steps + 1,
                running_totals=self.running_totals,
                workdir=Path(tmp),
                usage_meter=_NoopUsageMeter(),
            )
            tool_result, _ = self.harness.process_tool_call(ctx, tool_call)
            return tool_result

    def run(self, num_steps=1):
        """Main agent loop.

        Args:
            num_steps: Number of steps to run for
        """
        logger.info("Starting agent loop for %s steps", num_steps)

        steps_completed = 0
        while self.running and steps_completed < num_steps:
            try:
                self.tui.on_step(steps_completed + 1, num_steps)
                with tempfile.TemporaryDirectory(
                    prefix=f"porygon-simple-step-{steps_completed + 1:03d}-"
                ) as tmp:
                    _, metrics = run_one_step(
                        self.harness,
                        self.emulator,
                        step_index=steps_completed + 1,
                        running_totals=self.running_totals,
                        workdir=Path(tmp),
                    )

                self.running_totals = add_step_metrics_to_totals(
                    self.running_totals, metrics
                )
                steps_completed += 1
                logger.info("Completed step %s/%s", steps_completed, num_steps)

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                self.running = False
            except Exception as e:
                logger.error("Error in agent loop: %s", e)
                raise

        if not self.running:
            self.emulator.stop()

        return steps_completed

    def summarize_history(self):
        self.message_history, _ = self.harness.memory_strategy.summarize(
            history=self.message_history,
            emulator=self.emulator,
            client=self.client,
            usage_meter=_NoopUsageMeter(),
        )

    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.stop()


class _NoopUsageMeter:
    def record(self, usage):
        return None


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rom_path = os.path.join(os.path.dirname(current_dir), "pokemon.gb")

    agent = SimpleAgent(rom_path)

    try:
        steps_completed = agent.run(num_steps=10)
        logger.info("Agent completed %s steps", steps_completed)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    finally:
        agent.stop()
