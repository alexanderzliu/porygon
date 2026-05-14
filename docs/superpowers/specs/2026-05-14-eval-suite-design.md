# Eval Suite for the Pokemon Agent Harness

**Status:** Design approved, ready for implementation planning
**Date:** 2026-05-14
**Author:** Alex (via brainstorming session)

## Motivation

Iterating on the agent harness — prompts, game-state representation, memory strategies, tools, even harness architectures with multiple internal subagents — currently requires running the agent from the beginning of the game to observe a change's effect. This is slow, expensive, and noisy.

We want a systematic eval suite that:

1. Loads the game from a checkpointed start state (e.g., "just finished naming, standing in bedroom").
2. Runs the agent with a chosen harness configuration until it reaches a defined milestone, hits a cap, or fails.
3. Records cost, model calls, steps, wall time, and per-step artifacts so we can compare configurations.
4. Supports running N trials per (scenario × harness) cell, since the agent is stochastic (temperature 1.0).
5. Uses Temporal for orchestration — durable trials, retries, parallel fan-out across configs and trials.

The first scenario we want to run is the inspiring example: from the post-naming bedroom state, reach the first floor of Red's house. The current harness takes an unreasonable number of steps to find the stairs.

## Goals

- A first-class plugin architecture for harnesses, so architectural variations (subagents, alternative state representations, no-screenshot harnesses) are first-class and not feature flags on one mega-config.
- A scenario format that captures start state, success predicate, and caps (steps, wall-clock, USD).
- Reproducible per-trial artifact directories suitable for inspection and replay.
- Temporal workflows that orchestrate sweeps and trials with save-state-per-step durability — any worker can resume any trial at any step.
- Phased delivery: a local single-trial runner is usable well before any Temporal code lands.

## Non-goals

- Cross-machine workers, multi-tenant deployment, or a hosted Temporal cluster. Local Temporal dev server is the target.
- A web UI. CLI + filesystem artifacts only. Temporal's own Web UI is the only browser-facing surface.
- Statistical inference beyond medians and ranges across trials. No confidence-interval framework.
- A predicate DSL more expressive than what we need for early scenarios. We can extend it later.

## Architecture overview

```
SweepWorkflow                    # one per `evals run <suite.yaml>` invocation
  │  expands suite into (scenario × harness × trial_i) cells
  │  fans out child workflows, bounded by concurrency cap
  │
  ▼
TrialWorkflow(trial_id, scenario_id, harness_id, params_override, trial_index)
  │  step_ref = await init_trial(...)                  # writes step_000/
  │  loop:
  │      result, step_ref = await run_agent_step(step_ref)
  │      await record_step(trial_id, result.metrics)
  │      if await check_milestone(step_ref, scenario.success): break
  │      if exceeded caps: break
  │  await finalize_trial(trial_id, outcome, step_ref)
  │
  ▼
Activities (stateless, any worker on the `eval-trials` task queue):
  init_trial          (scenario, harness, params, trial_id)  -> StepRef
  run_agent_step      (step_ref)                              -> (StepMetrics, StepRef)
  check_milestone     (step_ref, success_spec)                -> bool
  record_step         (trial_id, step_metrics)                -> ()
  finalize_trial      (trial_id, outcome, step_ref)           -> TrialResult
```

Activities are stateless. The PyBoy emulator is created at the start of `run_agent_step`, loaded from the prior step's save state, used to apply the harness step, then serialized back to disk along with harness internal state. The next activity invocation — possibly on a different worker — picks up from that directory.

Workflow history carries only small payloads: directory path strings and a metrics dict per step. Large artifacts (screenshots, prompts, message histories, save states) live on disk under `evals/results/<run_id>/<trial_id>/step_<NNN>/`.

`run_agent_step` is intentionally coarse-grained. Splitting model invocation, button application, and memory read into separate activities would force the workflow to carry the LLM response and parsed tool calls as workflow-history state. As a single activity it is idempotent at the directory boundary: a failed step retries from the previous `step_ref` and produces a fresh `step_<NNN+1>` directory.

### Why save-state-per-step over worker-pinned sessions

We considered pinning each trial to one worker via a Session-style API, keeping the emulator alive in worker memory across activities, and checkpointing periodically. We chose save-state-per-step instead because:

- Resumability is automatic at every step boundary, regardless of which worker died.
- No worker-affinity machinery (per-trial task queues, session lifecycle, registries) needed.
- Per-step overhead — loading a PyBoy save state and re-initializing the emulator — is ~100–500 ms, small relative to a 3–10 s Claude inference.
- Activity code is trivially testable in isolation; no shared in-process state.

We can revisit if step overhead becomes meaningful at scale.

## Project layout

```
evals/
├── __init__.py
├── runner.py              # local single-trial runner (no Temporal)
├── workflows.py           # SweepWorkflow, TrialWorkflow
├── activities.py          # init_trial, run_agent_step, check_milestone, record_step, finalize_trial
├── worker.py              # `python -m evals.worker` registers workflows + activities
├── cli.py                 # `python -m evals run <suite>` | `inspect <trial>` | `replay <trial>`
├── predicates.py          # built-in predicate implementations + composition
├── scenarios/
│   └── bedroom_to_downstairs.yaml
├── harnesses/             # plugins; one directory per harness id
│   ├── baseline/
│   │   ├── __init__.py    # def build(params: dict) -> Harness
│   │   └── default.yaml
│   ├── no_screenshot/
│   │   ├── __init__.py
│   │   └── default.yaml
│   └── ...
├── suites/                # files describing matrix of (scenario × harness × N trials)
│   └── prompt_experiment_a.yaml
├── states/                # checked-in PyBoy save states used as scenario start points
│   └── after_names_bedroom.state
└── results/               # all per-run output; gitignored
    └── <run_id>/
        ├── suite.yaml             # frozen copy of inputs for reproducibility
        ├── summary.jsonl          # one row per completed trial
        └── <trial_id>/
            ├── trial.json         # trial metadata, final outcome, harness static_config()
            ├── steps.jsonl        # one row per step
            └── step_<NNN>/
                ├── state.bin             # PyBoy save state at end of this step
                ├── harness_state.bin     # harness-defined serialized internal state
                ├── running_totals.json
                ├── prompt.json           # optional, harness-emitted
                ├── response.json         # optional, harness-emitted
                ├── screenshot.png        # optional, harness-emitted
                ├── memory_dump.json
                └── collision_map.txt     # optional
```

`agent/` is reorganized to expose the harness-building primitives (prompt builder, state formatter, memory strategy, step runner) that the `baseline` harness uses. See "Refactor scope" below.

## Harness plugin protocol

Harnesses are Python packages discovered from `evals/harnesses/<id>/`. Each exposes `build(params: dict) -> Harness`. The runner imports the harness by id and calls `build(...)`; from then on it talks only to the `Harness` protocol.

```python
# Protocol and dataclass types (final module home — agent/ vs evals/ — is an open question below)

@dataclass
class RunningTotals:
    steps: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    cost_usd: float
    wall_seconds: float

@dataclass
class StepContext:
    emulator: Emulator
    step_index: int
    running_totals: RunningTotals

@dataclass
class StepMetrics:
    model_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    cost_usd: float
    wall_ms: int
    tool_call_count: int
    summarization_events: int = 0

@dataclass
class StepResult:
    actions: list[str]                      # buttons pressed; for audit and replay
    metrics: StepMetrics
    artifacts: dict[str, bytes | Path]      # harness chooses what to persist into step_NNN/
    text_log: str                           # human-readable trace for this step

class Harness(Protocol):
    id: str
    version: str

    def step(self, ctx: StepContext) -> StepResult: ...
    def serialize_state(self) -> bytes: ...
    def load_state(self, blob: bytes) -> None: ...
    def static_config(self) -> dict: ...    # frozen into trial.json metadata
```

### What "one step" means

One `Harness.step()` call equals one externally-observable interaction with the emulator: the harness decides on an action (possibly via multiple internal model calls and subagents) and applies it. This keeps `steps_to_milestone` meaningful as a comparison metric across harnesses, regardless of internal complexity. `model_calls` and `cost_usd` capture the inner cost.

### Harness internal state

`serialize_state` and `load_state` let each harness define its own state envelope. A baseline harness serializes its message history. A subagent harness serializes multiple histories plus its planner notes. The runner is agnostic.

### Token and cost accounting

Harnesses report aggregated `StepMetrics` per step. A subagent step that makes four internal model calls reports `model_calls=4` with summed token counts. Cost is computed by the harness from token counts and the model's pricing — the harness knows which model(s) it called.

## Scenario schema

```yaml
# evals/scenarios/bedroom_to_downstairs.yaml
id: bedroom_to_downstairs
description: From the bedroom on 2F, reach 1F of Red's house for the first time.
initial_state: states/after_names_bedroom.state
success:
  all:
    - location_eq: REDS_HOUSE_1F
limits:
  max_steps: 50
  max_seconds: 600
  max_usd: 2.00
```

Caps are checked between steps. Trial outcome is `milestone_reached` if the success predicate fires; otherwise it is the first cap exceeded (`step_cap`, `time_cap`, `cost_cap`); or `error` if an unrecoverable exception escaped retry.

## Suite schema

```yaml
# evals/suites/prompt_experiment_a.yaml
scenario: bedroom_to_downstairs
trials: 3
concurrency: 4                    # max concurrent TrialWorkflows in the sweep
matrix:
  - harness: baseline                                # uses baseline/default.yaml
  - harness: baseline
    params_override: { model: { temperature: 0.5 } }
  - harness: no_screenshot
  - harness: subagent_memory
    params: subagent_memory/aggressive.yaml          # alternate param file inside that dir
```

`params_override` is a deep-merge over the harness's `default.yaml`. `params: <path>` is an outright replacement (the harness still gets to define its own schema).

The sweep expands the matrix into `len(matrix) × trials` trial workflows.

## Milestone predicate DSL

YAML in a scenario's `success` field. Evaluated by `check_milestone` against a `MemoryDump` (and optionally the prior step's `MemoryDump`, for transition predicates).

| Predicate | Reads | Example |
|---|---|---|
| `location_eq: <MAP_NAME>` | `wCurMap` | `location_eq: REDS_HOUSE_1F` |
| `coords_in_box` | `wCurMap`, `wXCoord`, `wYCoord` | `{ map: REDS_HOUSE_1F, x: [0,4], y: [0,3] }` |
| `event_flag_set: <flag>` | event flags region | `event_flag_set: GOT_STARTER` |
| `badge_count_at_least: <n>` | badge byte | `badge_count_at_least: 1` |
| `party_has_pokemon: <species>` | party data | `party_has_pokemon: PIKACHU` |
| `dialog_contains: <substr>` | dialog buffer | `dialog_contains: "Welcome to"` |
| `first_time: <inner>` | requires prior `MemoryDump` | true only on the false→true transition edge |

Compound operators: `all: [...]`, `any: [...]`, `not: <inner>`.

`first_time` covers the "first visit" semantics in the inspiring scenario: a trial that re-enters the target state later doesn't trigger again.

Predicate evaluation requires structured memory reads. The current `memory_reader.py` returns a freeform string. We will add a `MemoryDump` dataclass with typed fields (location, coords, party, badges, money, inventory, event flags, dialog buffer) and keep the legacy text rendering as a `.format()` method on it.

## Metrics

Each `steps.jsonl` row contains:

- `step_index`
- `wall_ms`
- `model_calls`
- `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens`
- `cost_usd`
- `tool_call_count`
- `summarization_events`
- `actions: list[str]` — buttons pressed
- `step_ref` — path to `step_<NNN>/` for cross-referencing

Each `trial.json` records:

- `trial_id`, `run_id`, `scenario_id`, `harness_id`, `harness_version`, `trial_index`
- `params_resolved` — the final merged params dict
- `harness_static_config` — from `Harness.static_config()`
- `outcome` — `milestone_reached` | `step_cap` | `time_cap` | `cost_cap` | `error`
- `steps_to_milestone` (null if not reached)
- Aggregates across the trial: totals for tokens, cost, wall, model_calls
- `started_at`, `ended_at`

`summary.jsonl` is one row per trial flattening trial.json's key fields, for quick `jq`/Pandas aggregation across a suite.

## Refactor scope for `agent/`

The current `SimpleAgent` hardcodes prompts, formatting, summarization, and model client choices. We extract pluggable pieces that the baseline harness composes:

| New module | Responsibility | Replaces |
|---|---|---|
| `agent/harness.py` | `Harness` protocol + `RunningTotals` / `StepContext` / `StepResult` / `StepMetrics` dataclasses | — |
| `agent/prompt.py` | `PromptBuilder` — produces system prompt + per-turn user content | inline `SYSTEM_PROMPT` / `SUMMARY_PROMPT` in `simple_agent.py` |
| `agent/state_formatter.py` | `StateFormatter` — turns `MemoryDump` + screenshot + collision map into content blocks | inline formatting in `process_tool_call` |
| `agent/memory_strategy.py` | `MemoryStrategy` interface; impls: `SummarizeAndReplace` (current behavior), `RollingWindow` | `summarize_history` method |
| `agent/step_runner.py` | `run_one_step(harness, emulator, history)` — the per-step function used by both `evals/runner.py` and the Temporal activity | inline loop body in `SimpleAgent.run` |
| `agent/memory_reader.py` *(modify)* | Add `MemoryDump` dataclass; keep `.format()` for legacy text rendering | freeform string return today |

`SimpleAgent` becomes a thin facade over the baseline harness so `main.py` continues to work unchanged. `--load-state` behavior is preserved.

## Phasing

1. **Refactor harness pieces.** Extract prompt, state formatter, memory strategy, step runner. No behavior change to `main.py` — verifiable by running `python main.py --load-state pokemon.gb.state` and comparing logs.
2. **`MemoryDump` + predicate DSL.** Add `evals/predicates.py`, extend `memory_reader.py`. Unit-testable against checked-in memory dumps.
3. **Local single-trial runner.** `evals/runner.py` loads one scenario + one harness, runs until milestone or cap, writes the full `results/<run_id>/<trial_id>/` tree. **No Temporal yet.** This is the most valuable single milestone; it unblocks prompt experimentation immediately.
4. **Local sweep CLI.** `evals/cli.py` loops over the suite matrix by repeatedly calling the local runner. Sequential. Lets us run a real comparison the day after phase 3.
5. **Temporal wrapper.** `evals/activities.py`, `evals/workflows.py`, `evals/worker.py`. The `run_agent_step` activity wraps the same step runner used by the local runner. CLI gains a `--temporal` flag.
6. **Parallel fan-out.** Configure worker count and `SweepWorkflow` concurrency semaphore. Tune.

Phase 3 is the unblocking milestone. Phases 5–6 add durability and parallelism.

## Open questions

- Pricing source for cost computation. Hardcode Bedrock list prices per model in a `pricing.py` table, or pull from somewhere? Hardcoding is fine for now; the cost field is an estimate marked `estimated_cost_usd` everywhere.
- Whether `agent/harness.py` or `evals/harness_api.py` is the right home for the protocol types. Defer to implementation taste — the import graph will decide.
- Default `concurrency` when running under Temporal. Will depend on Bedrock rate limits and PyBoy CPU footprint. Pick empirically after phase 5.

## What this spec does not cover

- Visualization, dashboards, or web UI for results.
- Cross-suite trend tracking ("is `baseline` getting better over time as we tweak it?"). Possible to build on top of `summary.jsonl` later; out of scope here.
- Save-state generation tooling. We assume checked-in `.state` files are produced by hand or by a separate small script; not part of this design.
