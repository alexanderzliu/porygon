from collections import deque
from dataclasses import dataclass, field

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from config import (
    PRICE_CACHE_READ_PER_MTOK,
    PRICE_CACHE_WRITE_PER_MTOK,
    PRICE_INPUT_PER_MTOK,
    PRICE_OUTPUT_PER_MTOK,
)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, u) -> None:
        self.input_tokens += u.input_tokens or 0
        self.output_tokens += u.output_tokens or 0
        self.cache_read_tokens += u.cache_read_input_tokens or 0
        self.cache_write_tokens += u.cache_creation_input_tokens or 0

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_INPUT_PER_MTOK
            + self.output_tokens * PRICE_OUTPUT_PER_MTOK
            + self.cache_read_tokens * PRICE_CACHE_READ_PER_MTOK
            + self.cache_write_tokens * PRICE_CACHE_WRITE_PER_MTOK
        ) / 1_000_000


@dataclass
class TUI:
    step: int = 0
    total_steps: int = 0
    reasoning: str = ""
    summary: str = ""
    actions: deque = field(default_factory=lambda: deque(maxlen=12))
    game_state: str = ""
    usage: Usage = field(default_factory=Usage)
    last_step_cost: float = 0.0
    _live: Live | None = None

    def start(self) -> None:
        self._live = Live(self._render(), refresh_per_second=8, screen=False)
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def on_step(self, step: int, total: int) -> None:
        self.step = step
        self.total_steps = total
        self._refresh()

    def on_response(self, response, reasoning_text: str) -> None:
        before = self.usage.cost_usd
        self.usage.add(response.usage)
        self.last_step_cost = self.usage.cost_usd - before
        self.reasoning = reasoning_text.strip() or "(no reasoning text)"
        self.summary = ""
        self._refresh()

    def on_press(self, buttons: list[str]) -> None:
        self._record_action(f"press {' '.join(buttons)}")

    def on_navigate(self, row: int, col: int) -> None:
        self._record_action(f"navigate to ({row}, {col})")

    def on_game_state(self, state: str) -> None:
        self.game_state = state
        self._refresh()

    def on_summary(self, summary_text: str) -> None:
        self.summary = summary_text
        self._refresh()

    def _record_action(self, label: str) -> None:
        self.actions.appendleft(f"[step {self.step}] {label}")
        self._refresh()

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render())

    def _render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header(), size=3),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(self._reasoning_panel(), name="left", ratio=2),
            Layout(name="right", ratio=1),
        )
        layout["body"]["right"].split_column(
            Layout(self._actions_panel(), name="actions"),
            Layout(self._game_panel(), name="game"),
        )
        return layout

    def _header(self) -> Panel:
        tokens_in = self.usage.input_tokens + self.usage.cache_read_tokens + self.usage.cache_write_tokens
        step_label = f"{self.step}/{self.total_steps}" if self.total_steps else str(self.step)
        text = Text.assemble(
            ("CLAUDE PLAYS POKEMON  ", "bold magenta"),
            (f"step {step_label}   ", "cyan"),
            (f"tokens in {tokens_in:,}  out {self.usage.output_tokens:,}   ", "dim"),
            (f"cost ${self.usage.cost_usd:.4f}  (Δ ${self.last_step_cost:.4f})", "bold green"),
        )
        return Panel(text, border_style="magenta")

    def _reasoning_panel(self) -> Panel:
        if self.summary:
            body = Text.assemble(
                ("SUMMARIZED HISTORY\n\n", "bold yellow"),
                (self.summary, ""),
            )
        else:
            body = Text(self.reasoning or "(waiting for Claude...)")
        return Panel(body, title="reasoning", border_style="cyan")

    def _actions_panel(self) -> Panel:
        body = Text("\n".join(self.actions)) if self.actions else Text("(no actions yet)", style="dim")
        return Panel(body, title="recent actions", border_style="yellow")

    def _game_panel(self) -> Panel:
        return Panel(Text(self.game_state or "(no state yet)"), title="game state", border_style="green")


class _NullTUI:
    """No-op stand-in so SimpleAgent can call self.tui.* unconditionally."""

    def start(self) -> None: pass
    def stop(self) -> None: pass
    def on_step(self, *a, **k) -> None: pass
    def on_response(self, *a, **k) -> None: pass
    def on_press(self, *a, **k) -> None: pass
    def on_navigate(self, *a, **k) -> None: pass
    def on_game_state(self, *a, **k) -> None: pass
    def on_summary(self, *a, **k) -> None: pass


NULL_TUI = _NullTUI()
