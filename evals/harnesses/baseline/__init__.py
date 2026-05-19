from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass
from typing import Any

from anthropic import AnthropicBedrock

from agent.harness import Action, Harness, ModelUsage, StepContext, StepCounters, StepResult
from agent.memory_strategy import SummarizeAndReplace
from agent.prompt import PromptBuilder
from agent.state_formatter import StateFormatter
from agent.tui import NULL_TUI
from config import AWS_REGION, MAX_TOKENS, MODEL_NAME, TEMPERATURE, USE_NAVIGATOR

logger = logging.getLogger(__name__)


def _nested(params: dict, section: str, key: str, default):
    value = params.get(section, {})
    if isinstance(value, dict) and key in value:
        return value[key]
    flat_key = f"{section}_{key}"
    if flat_key in params:
        return params[flat_key]
    if key in params:
        return params[key]
    return default


def _usage_from_response(response, *, model_id: str, latency_ms: int) -> ModelUsage:
    usage = response.usage
    return ModelUsage(
        provider="bedrock",
        model_id=model_id,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        request_id=getattr(response, "id", None),
        latency_ms=latency_ms,
    )


def _usage_to_artifact(usage) -> dict:
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    try:
        return vars(usage)
    except TypeError:
        return {"repr": repr(usage)}


def _response_blocks_to_content(blocks) -> list[dict]:
    content = []
    for block in blocks:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            content.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
    return content


def _response_blocks_to_artifact(blocks) -> list[dict]:
    content = []
    for block in blocks:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            content.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
        else:
            content.append({"type": block.type, "repr": repr(block)})
    return content


@dataclass
class BaselineHarness(Harness):
    model_name: str
    aws_region: str
    temperature: float
    max_tokens: int
    max_history: int
    navigator_enabled: bool
    screenshot_upscale: int
    tui: Any | None = None

    id: str = "baseline"
    version: str = "phase1"

    def __post_init__(self) -> None:
        self.tui = self.tui or NULL_TUI
        self.client = AnthropicBedrock(aws_region=self.aws_region)
        self.prompt_builder = PromptBuilder(navigator_enabled=self.navigator_enabled)
        self.state_formatter = StateFormatter(
            screenshot_upscale=self.screenshot_upscale
        )
        self.memory_strategy = SummarizeAndReplace(
            max_history=self.max_history,
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            prompt_builder=self.prompt_builder,
            state_formatter=self.state_formatter,
            summary_observer=self._observe_summary,
        )
        self.message_history = self.prompt_builder.initial_history()
        self.tools = self.prompt_builder.tools()

    def _observe_summary(self, response, summary_text: str) -> None:
        self.tui.on_usage(response)
        self.tui.on_summary(summary_text)

    def step(self, ctx: StepContext) -> StepResult:
        ctx.workdir.mkdir(parents=True, exist_ok=True)
        messages = self.prompt_builder.messages_for_model(self.message_history)

        (ctx.workdir / "prompt.json").write_text(
            json.dumps(
                {
                    "system": self.prompt_builder.system_prompt,
                    "messages": messages,
                    "tools": self.tools,
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        start = time.monotonic()
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            system=self.prompt_builder.system_prompt,
            messages=messages,
            tools=self.tools,
            temperature=self.temperature,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        ctx.usage_meter.record(
            _usage_from_response(
                response, model_id=self.model_name, latency_ms=latency_ms
            )
        )

        logger.info("Response usage: %s", response.usage)
        (ctx.workdir / "response.json").write_text(
            json.dumps(
                {
                    "id": getattr(response, "id", None),
                    "usage": _usage_to_artifact(response.usage),
                    "content": _response_blocks_to_artifact(response.content),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        tool_calls = [block for block in response.content if block.type == "tool_use"]
        text_log_parts = []

        for block in response.content:
            if block.type == "text":
                logger.info("[Text] %s", block.text)
                text_log_parts.append(block.text)
            elif block.type == "tool_use":
                logger.info("[Tool] Using tool: %s", block.name)

        self.tui.on_response(response, "\n".join(text_log_parts))

        actions = []
        summarization_events = 0
        if tool_calls:
            self.message_history.append(
                {"role": "assistant", "content": _response_blocks_to_content(response.content)}
            )

            tool_results = []
            for tool_call in tool_calls:
                tool_result, action = self.process_tool_call(ctx, tool_call)
                tool_results.append(tool_result)
                if action is not None:
                    actions.append(action)

            self.message_history.append({"role": "user", "content": tool_results})

            self.message_history, summarization_events = (
                self.memory_strategy.maybe_summarize(
                    history=self.message_history,
                    emulator=ctx.emulator,
                    client=self.client,
                    usage_meter=ctx.usage_meter,
                )
            )

        return StepResult(
            actions=actions,
            counters=StepCounters(
                tool_call_count=len(tool_calls),
                summarization_events=summarization_events,
            ),
            text_log="\n".join(text_log_parts),
        )

    def process_tool_call(self, ctx: StepContext, tool_call) -> tuple[dict, Action | None]:
        tool_name = tool_call.name
        tool_input = tool_call.input
        logger.info("Processing tool call: %s", tool_name)

        if tool_name == "press_buttons":
            return self._press_buttons(ctx, tool_call.id, tool_input)
        if tool_name == "navigate_to":
            return self._navigate_to(ctx, tool_call.id, tool_input)

        logger.error("Unknown tool called: %s", tool_name)
        return (
            {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [
                    {"type": "text", "text": f"Error: Unknown tool '{tool_name}'"}
                ],
            },
            None,
        )

    def _press_buttons(
        self, ctx: StepContext, tool_use_id: str, tool_input: dict[str, Any]
    ) -> tuple[dict, Action]:
        buttons = tool_input["buttons"]
        wait = tool_input.get("wait", True)
        logger.info("[Buttons] Pressing: %s (wait=%s)", buttons, wait)
        self.tui.on_press(buttons)

        result_text = ctx.emulator.press_buttons(buttons, wait)
        feedback = self._capture_feedback(ctx)

        tool_result = self.state_formatter.build_tool_result(
            tool_use_id=tool_use_id,
            result_text=f"Pressed buttons: {', '.join(buttons)}",
            screenshot_intro="\nHere is a screenshot of the screen after your button presses:",
            memory_intro="\nGame state information from memory after your action:",
            feedback=feedback,
        )

        frames_elapsed = len(buttons) * (130 if wait else 20)
        action = Action(
            kind="press_buttons",
            args=tool_input,
            buttons=list(buttons),
            frames_elapsed=frames_elapsed,
            success=True,
            result_text=result_text,
        )
        return tool_result, action

    def _navigate_to(
        self, ctx: StepContext, tool_use_id: str, tool_input: dict[str, Any]
    ) -> tuple[dict, Action]:
        row = tool_input["row"]
        col = tool_input["col"]
        logger.info("[Navigation] Navigating to: (%s, %s)", row, col)
        self.tui.on_navigate(row, col)

        status, path = ctx.emulator.find_path(row, col)
        if path:
            for direction in path:
                ctx.emulator.press_buttons([direction], True)
            result = f"Navigation successful: followed path with {len(path)} steps"
        else:
            result = f"Navigation failed: {status}"

        feedback = self._capture_feedback(ctx)
        tool_result = self.state_formatter.build_tool_result(
            tool_use_id=tool_use_id,
            result_text=f"Navigation result: {result}",
            screenshot_intro="\nHere is a screenshot of the screen after navigation:",
            memory_intro="\nGame state information from memory after your action:",
            feedback=feedback,
        )

        action = Action(
            kind="navigate_to",
            args=tool_input,
            buttons=list(path),
            frames_elapsed=len(path) * 130,
            success=bool(path),
            result_text=result,
        )
        return tool_result, action

    def _capture_feedback(self, ctx: StepContext):
        feedback = self.state_formatter.capture(ctx.emulator, workdir=ctx.workdir)

        logger.info("[Memory State after action]")
        logger.info(feedback.memory_info)
        self.tui.on_game_state(feedback.memory_info)
        if feedback.collision_map:
            logger.info("[Collision Map after action]\n%s", feedback.collision_map)

        return feedback

    def serialize_state(self) -> bytes:
        return pickle.dumps({"message_history": self.message_history})

    def load_state(self, blob: bytes) -> None:
        state = pickle.loads(blob)
        self.message_history = state.get(
            "message_history", self.prompt_builder.initial_history()
        )

    def static_config(self) -> dict:
        return {
            "model_name": self.model_name,
            "aws_region": self.aws_region,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_history": self.max_history,
            "navigator_enabled": self.navigator_enabled,
            "screenshot_upscale": self.screenshot_upscale,
        }


def build(params: dict) -> Harness:
    params = params or {}
    return BaselineHarness(
        model_name=_nested(params, "model", "name", MODEL_NAME),
        aws_region=_nested(params, "aws", "region", AWS_REGION),
        temperature=float(_nested(params, "model", "temperature", TEMPERATURE)),
        max_tokens=int(_nested(params, "model", "max_tokens", MAX_TOKENS)),
        max_history=int(_nested(params, "history", "max_history", 30)),
        navigator_enabled=bool(
            _nested(params, "navigator", "enabled", USE_NAVIGATOR)
        ),
        screenshot_upscale=int(_nested(params, "screenshot", "upscale", 2)),
        tui=params.get("tui"),
    )
