from __future__ import annotations

import copy
from dataclasses import dataclass


SYSTEM_PROMPT = """You are playing Pokemon Red. You can see the game screen and control the game by executing emulator commands.

Your goal is to play through Pokemon Red and eventually defeat the Elite Four. Make decisions based on what you see on the screen.

Before each action, explain your reasoning briefly, then use the emulator tool to execute your chosen commands.

The conversation history may occasionally be summarized to save context space. If you see a message labeled "CONVERSATION HISTORY SUMMARY", this contains the key information about your progress so far. Use this information to maintain continuity in your gameplay."""

SUMMARY_PROMPT = """I need you to create a detailed summary of our conversation history up to this point. This summary will replace the full conversation history to manage the context window.

Please include:
1. Key game events and milestones you've reached
2. Important decisions you've made
3. Current objectives or goals you're working toward
4. Your current location and Pokémon team status
5. Any strategies or plans you've mentioned

The summary should be comprehensive enough that you can continue gameplay without losing important context about what has happened so far."""

INITIAL_USER_MESSAGE = {"role": "user", "content": "You may now begin playing."}


def build_tool_schema(navigator_enabled: bool = False) -> list[dict]:
    tools = [
        {
            "name": "press_buttons",
            "description": "Press a sequence of buttons on the Game Boy.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "buttons": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "a",
                                "b",
                                "start",
                                "select",
                                "up",
                                "down",
                                "left",
                                "right",
                            ],
                        },
                        "description": "List of buttons to press in sequence. Valid buttons: 'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right'",
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "Whether to wait for a brief period after pressing each button. Defaults to true.",
                    },
                },
                "required": ["buttons"],
            },
        }
    ]

    if navigator_enabled:
        tools.append(
            {
                "name": "navigate_to",
                "description": "Automatically navigate to a position on the map grid. The screen is divided into a 9x10 grid, with the top-left corner as (0, 0). This tool is only available in the overworld.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "row": {
                            "type": "integer",
                            "description": "The row coordinate to navigate to (0-8).",
                        },
                        "col": {
                            "type": "integer",
                            "description": "The column coordinate to navigate to (0-9).",
                        },
                    },
                    "required": ["row", "col"],
                },
            }
        )

    return tools


def with_recent_cache_control(messages: list[dict]) -> list[dict]:
    messages = copy.deepcopy(messages)

    if len(messages) >= 3:
        if (
            messages[-1]["role"] == "user"
            and isinstance(messages[-1]["content"], list)
            and messages[-1]["content"]
        ):
            messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        if (
            len(messages) >= 5
            and messages[-3]["role"] == "user"
            and isinstance(messages[-3]["content"], list)
            and messages[-3]["content"]
        ):
            messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}

    return messages


@dataclass
class PromptBuilder:
    system_prompt: str = SYSTEM_PROMPT
    summary_prompt: str = SUMMARY_PROMPT
    navigator_enabled: bool = False

    def initial_history(self) -> list[dict]:
        return [copy.deepcopy(INITIAL_USER_MESSAGE)]

    def tools(self) -> list[dict]:
        return build_tool_schema(self.navigator_enabled)

    def messages_for_model(self, history: list[dict]) -> list[dict]:
        return with_recent_cache_control(history)

    def messages_for_summary(self, history: list[dict]) -> list[dict]:
        messages = with_recent_cache_control(history)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.summary_prompt,
                    }
                ],
            }
        )
        return messages
