from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from agent.memory_reader import MemoryDump


def get_screenshot_base64(screenshot: Image.Image, upscale: int = 1) -> str:
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        screenshot = screenshot.resize(new_size)

    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()


@dataclass
class StateFeedback:
    screenshot_base64: str
    memory_info: str
    collision_map: str | None
    memory_dump: MemoryDump | None = None


class StateFormatter:
    def __init__(self, screenshot_upscale: int = 2):
        self.screenshot_upscale = screenshot_upscale

    def capture(self, emulator, workdir: Path | None = None) -> StateFeedback:
        screenshot = emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(
            screenshot, upscale=self.screenshot_upscale
        )
        memory_dump = None
        if hasattr(emulator, "get_memory_dump"):
            memory_dump = emulator.get_memory_dump()
            memory_info = memory_dump.format()
        else:
            memory_info = emulator.get_state_from_memory()
        collision_map = emulator.get_collision_map()

        if workdir is not None:
            workdir.mkdir(parents=True, exist_ok=True)
            screenshot.save(workdir / "screenshot.png")
            memory_artifact = (
                memory_dump.to_dict()
                if memory_dump is not None
                else {"text": memory_info}
            )
            (workdir / "memory_dump.json").write_text(
                json.dumps(memory_artifact, indent=2), encoding="utf-8"
            )
            if collision_map:
                (workdir / "collision_map.txt").write_text(
                    collision_map, encoding="utf-8"
                )

        return StateFeedback(
            screenshot_base64=screenshot_b64,
            memory_info=memory_info,
            collision_map=collision_map,
            memory_dump=memory_dump,
        )

    def build_tool_result(
        self,
        *,
        tool_use_id: str,
        result_text: str,
        screenshot_intro: str,
        memory_intro: str,
        feedback: StateFeedback,
    ) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": [
                {"type": "text", "text": result_text},
                {"type": "text", "text": screenshot_intro},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": feedback.screenshot_base64,
                    },
                },
                {
                    "type": "text",
                    "text": f"{memory_intro}\n{feedback.memory_info}",
                },
            ],
        }

    def build_summary_history(self, max_history: int, summary_text: str, emulator) -> list[dict]:
        screenshot = emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(
            screenshot, upscale=self.screenshot_upscale
        )

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"CONVERSATION HISTORY SUMMARY (representing {max_history} previous messages): {summary_text}",
                    },
                    {
                        "type": "text",
                        "text": "\n\nCurrent game screenshot for reference:",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "You were just asked to summarize your playthrough so far, which is the summary you see above. You may now continue playing by selecting your next action.",
                    },
                ],
            }
        ]
