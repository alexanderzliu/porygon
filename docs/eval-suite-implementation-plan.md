# Eval Suite Implementation Plan

**Date:** 2026-05-18
**Design spec:** `docs/superpowers/specs/2026-05-14-eval-suite-design.md`
**Purpose:** Phase-by-phase implementation guide for the Pokemon agent eval suite. This document is intentionally detailed so each phase can be picked up after clearing assistant context.

## Implementation Principles

- Keep `main.py` usable throughout. The interactive/manual agent path should keep working while eval infrastructure is added.
- Build the local runner before Temporal. Temporal should wrap the same primitives used locally, not create a second execution path.
- Treat filesystem artifacts as the durable source of truth. JSONL files are derived convenience outputs.
- Keep harnesses responsible for behavior and artifacts, while the runner owns canonical metrics, pricing, commit protocol, and finalization.
- Prefer small, testable modules over a large eval runner with many responsibilities.

## Phase 1 - Core Harness Refactor

**Goal:** Extract the current `SimpleAgent` behavior into reusable harness primitives without changing the user-facing `main.py` behavior.

### Files to Add

- `agent/harness.py`
- `agent/prompt.py`
- `agent/state_formatter.py`
- `agent/memory_strategy.py`
- `agent/step_runner.py`
- `evals/__init__.py`
- `evals/harnesses/__init__.py`
- `evals/harnesses/baseline/__init__.py`
- `evals/harnesses/baseline/default.yaml`

### Files to Modify

- `agent/simple_agent.py`
- `agent/emulator.py` if needed for cleaner step execution hooks
- `main.py` only if necessary, and only to preserve existing CLI behavior

### Required Types in `agent/harness.py`

Implement dataclasses/protocols matching the design spec:

- `RunningTotals`
- `ModelUsage`
- `UsageMeter`
- `StepContext`
- `Action`
- `StepMetrics`
- `StepCounters`
- `StepResult`
- `Harness`

Important contract:

- Harnesses return `StepResult`, not canonical `StepMetrics`.
- The runner/step-runner constructs `StepMetrics` from `StepResult`, `UsageMeter`, wall clock, predicate result, and `step_ref`.
- Harnesses must call `ctx.usage_meter.record(...)` after every model call.

### Refactor Shape

Extract current constants and behavior from `agent/simple_agent.py`:

- `SYSTEM_PROMPT` and `SUMMARY_PROMPT` move into `agent/prompt.py`.
- Screenshot/memory/collision-map content formatting moves into `agent/state_formatter.py`.
- Current summarization behavior becomes `SummarizeAndReplace` in `agent/memory_strategy.py`.
- One model-response/tool-application turn becomes a reusable function in `agent/step_runner.py`.
- `SimpleAgent` becomes a thin facade that builds the baseline harness and loops over steps.

### Baseline Harness

`evals/harnesses/baseline/__init__.py` should expose:

```python
def build(params: dict) -> Harness:
    ...
```

The baseline harness should reproduce the current `SimpleAgent` behavior:

- Anthropic Bedrock client
- existing model config defaults from `config.py`
- current prompts
- current tool schema
- current screenshot + memory + collision-map feedback
- current summarization strategy
- support for `USE_NAVIGATOR`

`default.yaml` should include configurable fields for:

- model name / inference profile
- AWS region
- temperature
- max tokens
- max history
- navigator enabled
- screenshot upscale

### Acceptance Checks

- `python main.py --steps 1` runs through one agent step.
- `python main.py --steps 1 --load-state <state>` still loads the provided save state.
- Existing CLI flags still work: `--rom`, `--steps`, `--display`, `--sound`, `--max-history`, `--load-state`.
- Baseline harness behavior is close enough to current behavior that logs show the same high-level flow: model call, tool call, emulator action, screenshot/memory feedback.

### Suggested Tests

Use fakes where possible rather than live model calls:

- fake harness can be stepped by `agent/step_runner.py`
- fake usage meter aggregates model usage
- memory strategy summarizes/replaces at the expected threshold

## Phase 2 - Structured Memory and Predicates

**Goal:** Add typed game-state reads and milestone predicates without breaking legacy text prompts.

### Files to Add

- `evals/predicates.py`
- tests for predicate evaluation, if a test framework is introduced

### Files to Modify

- `agent/memory_reader.py`
- `agent/emulator.py`
- `agent/state_formatter.py` if it should consume `MemoryDump`

### MemoryDump

Add a `MemoryDump` dataclass to `agent/memory_reader.py` with typed fields needed by early scenarios:

- `player_name`
- `rival_name`
- `money`
- `location`
- `map_id`
- `coordinates`
- `valid_moves`
- `badges`
- `inventory`
- `dialog`
- `party`
- optional raw fields needed for future predicates

Keep the legacy string representation:

- Add `MemoryDump.format() -> str`.
- Make `Emulator.get_state_from_memory()` render `MemoryDump.format()` so existing prompts continue to work.
- Add `Emulator.get_memory_dump() -> MemoryDump`.

### Predicate DSL

Implement `evals/predicates.py` with:

- `evaluate_predicate(spec: dict, current: MemoryDump, previous: MemoryDump | None = None) -> bool`

Required predicates:

- `location_eq`
- `coords_in_box`
- `dialog_contains`
- `badge_count_at_least`
- `party_has_pokemon`
- `all`
- `any`
- `not`
- `first_time`

Defer `event_flag_set` unless event flag mapping is already straightforward.

### Acceptance Checks

- Synthetic `MemoryDump` objects can be evaluated without PyBoy.
- `location_eq: REDS_HOUSE_1F` works for the bedroom-to-downstairs scenario.
- `first_time` returns true only on a false-to-true transition.
- Existing `SimpleAgent` prompt still receives readable memory text.

## Phase 3 - Local Single-Trial Runner

**Goal:** Build the first useful eval path without Temporal.

### Files to Add

- `evals/runner.py`
- `evals/pricing.py`
- `evals/scenarios/bedroom_to_downstairs.yaml`
- `evals/results/.gitignore`
- `evals/states/.gitkeep` if no state file is checked in yet

### Main Responsibilities

`evals/runner.py` should provide a callable API before adding a CLI:

```python
run_trial(
    scenario_path: Path,
    harness_id: str,
    params_path: Path | None = None,
    params_override: dict | None = None,
    run_id: str | None = None,
    trial_index: int = 0,
) -> TrialResult
```

Use narrower dataclasses if useful:

- `ResolvedTrialSpec`
- `StepRef`
- `TrialResult`
- `TrialOutcome`
- `ResolvedHarnessConfig`

### Trial Flow

1. Resolve scenario YAML and harness params.
2. Create `evals/results/<run_id>/<trial_id>/`.
3. Copy/freeze resolved inputs into the result directory.
4. Create `step_000/`:
   - load initial save state
   - initialize harness
   - write `state.bin`
   - write `harness_state.bin`
   - write `memory_dump.json`
   - evaluate initial success predicate
5. If initial predicate succeeds, finalize without charging a model call.
6. Otherwise loop until milestone, cap, or error:
   - load previous `state.bin`
   - restore previous `harness_state.bin`
   - run one harness step inside an attempt-scoped partial directory
   - runner constructs canonical `StepMetrics`
   - write `harness_state.bin`, `metrics.json`, then `state.bin`
   - atomically publish `step_<NNN>/`
   - update running totals
   - check caps
7. Finalize trial:
   - write `trial.json`
   - regenerate `steps.jsonl` from `step_<NNN>/metrics.json`

### Commit Protocol

Implement the same protocol intended for Temporal:

- canonical completed dirs are `step_<NNN>/`
- in-progress dirs are `step_<NNN>.partial.attempt_<A>/`
- local runner can use attempt `1`
- write `harness_state.bin`, then `metrics.json`, then `state.bin`
- publish with atomic rename
- if canonical step already exists, read its `metrics.json` and return it
- never append to `steps.jsonl` during step execution

### Pricing

`evals/pricing.py` should contain:

- `PRICING_VERSION`
- a hardcoded table keyed by `(provider, model_id)`
- `compute_cost(usage: list[ModelUsage]) -> float`

If exact pricing is uncertain, make unsupported models explicit:

- raise a clear error, or
- return `0` only if config says pricing is disabled

### Scenario

`evals/scenarios/bedroom_to_downstairs.yaml`:

```yaml
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

### Acceptance Checks

- Running one local trial creates:
  - `trial.json`
  - `steps.jsonl`
  - `step_000/`
  - at least one `step_001/` when initial success is false
- `step_<NNN>/metrics.json` is the canonical per-step record.
- Re-running finalization regenerates `steps.jsonl` without duplicates.
- A fake harness can hit a synthetic milestone without making a model call.

## Phase 4 - CLI and Sequential Sweeps

**Goal:** Make local evals easy to run and compare before Temporal exists.

### Files to Add

- `evals/cli.py`
- `evals/suites/prompt_experiment_a.yaml`

### Commands

Implement:

```bash
python -m evals.cli run-one evals/scenarios/bedroom_to_downstairs.yaml --harness baseline
python -m evals.cli run evals/suites/prompt_experiment_a.yaml
python -m evals.cli inspect evals/results/<run_id>/<trial_id>
```

Optional but useful:

```bash
python -m evals.cli finalize evals/results/<run_id>/<trial_id>
python -m evals.cli summarize evals/results/<run_id>
```

### Suite Schema

Support:

```yaml
scenario: bedroom_to_downstairs
trials: 3
concurrency: 1
matrix:
  - harness: baseline
  - harness: baseline
    params_override:
      model:
        temperature: 0.5
```

Sequential local sweeps can ignore concurrency except to validate it.

### Sweep Flow

1. Resolve suite.
2. Expand `matrix x trials`.
3. Call `run_trial(...)` sequentially.
4. Write or regenerate `summary.jsonl` from all `trial.json` files.

### Acceptance Checks

- A suite with two matrix entries and two trials produces four trial directories.
- `summary.jsonl` has one row per completed trial.
- `inspect` prints outcome, step count, cost, token totals, and final location.

## Phase 5 - Temporal Wrapper

**Goal:** Add durable orchestration by wrapping the local runner primitives.

### Files to Add

- `evals/activities.py`
- `evals/workflows.py`
- `evals/worker.py`

### Files to Modify

- `evals/cli.py`
- `requirements.txt`

Add Temporal Python SDK dependency when this phase starts.

### Workflow API

Implement:

- `SweepWorkflow`
- `TrialWorkflow`

Workflow rules:

- Workflow receives frozen resolved suite/trial specs from the CLI.
- Workflow does not read YAML files.
- Workflow does not import harness packages.
- Workflow does not touch PyBoy.
- Workflow history carries only small refs and metrics.
- Use Continue-As-New after `completed_steps > 0 and completed_steps % CONTINUE_AS_NEW_EVERY == 0`.

### Activities

Implement:

- `init_trial(resolved_trial_spec)`
- `run_agent_step(step_ref, success_spec)`
- `finalize_trial(trial_id, outcome, step_ref)`
- `finalize_sweep(run_id)`

Activities should call the same lower-level functions used by the local runner.

### Temporal CLI

Extend:

```bash
python -m evals.cli run evals/suites/prompt_experiment_a.yaml --temporal
python -m evals.worker
```

### Retry / Idempotence Requirements

- `run_agent_step` uses attempt-scoped partial dirs based on Temporal attempt number.
- Completed canonical step dirs short-circuit retries.
- `finalize_trial` and `finalize_sweep` overwrite derived files from canonical inputs.
- No append-only activity output is treated as source of truth.

### Acceptance Checks

- Temporal worker starts and registers workflows/activities.
- A Temporal run produces the same artifact shape as local runs.
- Killing and restarting the worker resumes from the last completed canonical step.
- Re-running a completed step activity returns persisted metrics rather than calling the model.

## Phase 6 - Hardening and Replay

**Goal:** Improve confidence, debugging, and long-term usability.

### Add Tests

- attempt-scoped partial directory collision behavior
- canonical step retry short-circuit
- finalization idempotence
- predicate evaluation
- pricing aggregation
- fake harness local trial success
- fake harness cap outcomes
- Temporal activity wrappers, if Temporal test harness is practical

### Add Result Validation

Add a validation function/command that checks:

- every canonical step has required files
- step numbers are contiguous
- each `metrics.json` step_ref matches its directory
- `trial.json` totals match summed step metrics
- `steps.jsonl` can be regenerated exactly
- `summary.jsonl` can be regenerated exactly

### Add Replay Helpers

Minimum replay:

- load a trial
- list steps and actions
- print prompt/response paths
- optionally replay button actions from `step_000` without model calls

### Acceptance Checks

- `python -m evals.cli validate evals/results/<run_id>` passes for a completed run.
- A fake trial can be replayed from action traces.
- Result metadata is sufficient to identify ROM, state, PyBoy version, harness version, params hash, pricing version, and git commit.

## Phase Handoff Template

At the end of each phase, update this document or create a short handoff note with:

- completed files
- changed public interfaces
- commands that passed
- commands that failed or were not run
- known limitations
- next recommended phase

Use this section when context is cleared:

```text
Resume from docs/eval-suite-implementation-plan.md.
Current phase: <N>
Last completed phase: <N-1>
Relevant design spec: docs/superpowers/specs/2026-05-14-eval-suite-design.md
Primary acceptance command(s): <commands>
```
