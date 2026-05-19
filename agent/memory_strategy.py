from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

from agent.harness import ModelUsage, UsageMeter
from agent.prompt import PromptBuilder
from agent.state_formatter import StateFormatter

logger = logging.getLogger(__name__)


class MemoryStrategy:
    def maybe_summarize(
        self,
        *,
        history: list[dict],
        emulator,
        client,
        usage_meter: UsageMeter,
    ) -> tuple[list[dict], int]:
        raise NotImplementedError


def _usage_from_response(response, *, provider: str, model_id: str, latency_ms: int) -> ModelUsage:
    usage = response.usage
    return ModelUsage(
        provider=provider,
        model_id=model_id,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        request_id=getattr(response, "id", None),
        latency_ms=latency_ms,
    )


@dataclass
class SummarizeAndReplace(MemoryStrategy):
    max_history: int
    model_name: str
    max_tokens: int
    temperature: float
    prompt_builder: PromptBuilder
    state_formatter: StateFormatter
    provider: str = "bedrock"
    summary_observer: Callable[[object, str], None] | None = None

    def maybe_summarize(
        self,
        *,
        history: list[dict],
        emulator,
        client,
        usage_meter: UsageMeter,
    ) -> tuple[list[dict], int]:
        if len(history) < self.max_history:
            return history, 0

        return self.summarize(
            history=history,
            emulator=emulator,
            client=client,
            usage_meter=usage_meter,
        )

    def summarize(
        self,
        *,
        history: list[dict],
        emulator,
        client,
        usage_meter: UsageMeter,
    ) -> tuple[list[dict], int]:
        logger.info("[Agent] Generating conversation summary...")

        messages = self.prompt_builder.messages_for_summary(history)
        start = time.monotonic()
        response = client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            system=self.prompt_builder.system_prompt,
            messages=messages,
            temperature=self.temperature,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        usage_meter.record(
            _usage_from_response(
                response,
                provider=self.provider,
                model_id=self.model_name,
                latency_ms=latency_ms,
            )
        )

        summary_text = " ".join(
            block.text for block in response.content if block.type == "text"
        )

        logger.info("[Agent] Game Progress Summary:")
        logger.info(summary_text)
        if self.summary_observer is not None:
            self.summary_observer(response, summary_text)
        logger.info("[Agent] Message history condensed into summary.")

        return (
            self.state_formatter.build_summary_history(
                self.max_history, summary_text, emulator
            ),
            1,
        )


@dataclass
class RollingWindow(MemoryStrategy):
    max_history: int

    def maybe_summarize(
        self,
        *,
        history: list[dict],
        emulator,
        client,
        usage_meter: UsageMeter,
    ) -> tuple[list[dict], int]:
        if len(history) <= self.max_history:
            return history, 0
        return history[-self.max_history :], 0
