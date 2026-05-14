from collections import deque
from dataclasses import dataclass, field

from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# Sonnet 4.5 Bedrock pricing ($/MTok)
PRICE_INPUT = 3.00
PRICE_OUTPUT = 15.00
PRICE_CACHE_READ = 0.30
PRICE_CACHE_WRITE_5M = 3.75


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, u) -> None:
        self.input_tokens += getattr(u, "input_tokens", 0) or 0
        self.output_tokens += getattr(u, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_INPUT
            + self.output_tokens * PRICE_OUTPUT
            + self.cache_read_tokens * PRICE_CACHE_READ
            + self.cache_write_tokens * PRICE_CACHE_WRITE_5M
        ) / 1_000_000


@dataclass
class Display:
    step: int = 0
    total_steps: int = 0
    reasoning: str = ""
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

    def on_response(self, response, reasoning_text: str) -> None:
        before = self.usage.cost_usd
        self.usage.add(response.usage)
        self.last_step_cost = self.usage.cost_usd - before
        self.reasoning = reasoning_text.strip() or "(no reasoning text)"
        self._refresh()

    def on_action(self, label: str) -> None:
        self.actions.appendleft(f"[step {self.step}] {label}")
        self._refresh()

    def on_step(self, step: int, total: int) -> None:
        self.step = step
        self.total_steps = total
        self._refresh()

    def on_game_state(self, state: str) -> None:
        self.game_state = state
        self._refresh()

    def on_summary(self, summary: str) -> None:
        self.reasoning = f"[bold yellow]SUMMARIZED HISTORY[/bold yellow]\n\n{summary}"
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
        cost = self.usage.cost_usd
        tokens_in = self.usage.input_tokens + self.usage.cache_read_tokens + self.usage.cache_write_tokens
        tokens_out = self.usage.output_tokens
        step_label = f"{self.step}/{self.total_steps}" if self.total_steps else str(self.step)
        text = Text.assemble(
            ("CLAUDE PLAYS POKEMON  ", "bold magenta"),
            (f"step {step_label}   ", "cyan"),
            (f"tokens in {tokens_in:,}  out {tokens_out:,}   ", "dim"),
            (f"cost ${cost:.4f}  (Δ ${self.last_step_cost:.4f})", "bold green"),
        )
        return Panel(text, border_style="magenta")

    def _reasoning_panel(self) -> Panel:
        return Panel(
            Text.from_markup(self.reasoning or "(waiting for Claude...)"),
            title="reasoning",
            border_style="cyan",
        )

    def _actions_panel(self) -> Panel:
        if not self.actions:
            body = Text("(no actions yet)", style="dim")
        else:
            body = Text("\n".join(self.actions))
        return Panel(body, title="recent actions", border_style="yellow")

    def _game_panel(self) -> Panel:
        body = Text(self.game_state or "(no state yet)")
        return Panel(body, title="game state", border_style="green")
