# Porygon — a Temporal eval harness for an agent that plays Pokemon Red

> An eval suite that runs a Claude-driven agent through reproducible Pokemon Red scenarios, with Temporal handling durability, retries, and parallel fan-out across (scenario × harness × trial) cells.

This repo started life from the `claude-plays-pokemon` starter — a simple agent in a loop with a Game Boy emulator. I wanted to see how proficient of a trainer I could make the agent, but with myriad ways to adjust the trainer agent's harness, I needed a way to efficiently experiment with modifications. This system, and the subject of this writeup, is the eval suite I built on top of it: an experimentation harness that lets me change a prompt or swap out a memory strategy and get back   measurements of the results and deltas (maybe less inference steps or tokens used to reach a goal) without unnecessarily having to re-playing the game each time.

The orchestration layer is Temporal. It runs each trial (a specific eval scenario e.g. "Reach Oak's lab to pick your starter after spawning in the overworld for the first time" with a specific agent harness config) step-by-step, and retries activities that fail so that any trials that fail mid-way can always pick back up and finish. A parent workflow spawns one child workflow per trial, so many experiments can run in parallel. 

The agent harness was designed with the intention of enabling different configurations & strategies (prompts, alternative state representations, different memory strategies) to be used. The rest of this document is about the design decisions in those two layers and the reasoning behind them.

---

## The problem

Iterating on an agent harness for a long-horizon game is mostly an experiment-management problem. Every change you might want to make — a new prompt, a different way of rendering game state, a memory-summarization strategy, even an entirely new architecture with internal subagents — needs to be validated against something. I needed something better than "run it from a fresh game and see what happens." because:

- **Trials are expensive.** Each step is a model call. It cost me a dollar using Sonnet 4.6 to simply get out of my bedroom the first time around. You don't want to repeat work, and you definitely don't want to repeat work *because of an infrastructure failure*.
- **Trials are long.** A single trial can be minutes to hours. Without durability, a worker crash mid-trial means all that time is wasted. 
- **The agent is stochastic.** At temperature 1.0 a single trial tells you very little. You need N trials per configuration and a way to compare distributions. The eval suite records steps to milestone, button presses, emulator frames, model calls, input/output tokens, tool calls, and dollars spent per trial, so a change that wins on one axis but loses on another can't slip past you.

This is a fan-out workload over (scenario × harness × trial) with expensive, partially-failing, long-running children. Temporal seemed like a perfect fit here.

---

## System at a glance

```
                        ┌────────────────────────────────────────────────┐
   CLI resolves all     │                  SweepWorkflow                 │
   inputs (suite YAML,  │   expands matrix → (scenario × harness × N)    │
   scenario YAML,       │   fans out child workflows with a              │
   harness params,      │   concurrency semaphore                        │
   ROM hash, save-      └────────────────────────────────────────────────┘
   state hash, git                              │
   commit, pricing                              ▼
   version) → frozen    ┌────────────────────────────────────────────────┐
   resolved spec        │                  TrialWorkflow                 │
                        │   init_trial → loop( run_agent_step ) →        │
                        │   finalize_trial                               │
                        │   Continue-As-New every N completed steps      │
                        └────────────────────────────────────────────────┘
                                                │
                                                ▼  (small payloads only:
                                                    paths + metrics dict)
                        ┌────────────────────────────────────────────────┐
                        │                    Activities                  │
                        │   init_trial, run_agent_step,                  │
                        │   finalize_trial, finalize_sweep               │
                        │   stateless — any worker can pick up any step  │
                        └────────────────────────────────────────────────┘
                                                │
                                                ▼
                        ┌────────────────────────────────────────────────┐
                        │       evals/results/<run_id>/<trial_id>/       │
                        │     step_000/  step_001/  …  step_NNN/         │
                        │     each contains: state.bin, harness_state.   │
                        │     bin, metrics.json, memory_dump.json,       │
                        │     prompt.json, response.json, screenshot.png │
                        └────────────────────────────────────────────────┘
```

Workflow history carries only directory path strings and small metrics dicts. The interesting bytes — screenshots, prompts, message histories, save states — live on disk under the canonical step directory and are referenced by path. This "claim check" pattern is employed to keep our history a manageable size

---

## Why Temporal

From our main problems, Temporal earned its place for a few reasons:

1. **Durable, resumable trials.** With each trials resource intensiveness (time & money), I wanted protection against failures i.e. for the trial to pick up at the next step on the next worker, not restart. Temporal's history-driven replay gives me that for free, *provided* I keep state on disk between steps rather than in worker memory. (More on that choice below.) I

2. **Cost-aware retries.** Each retry of run_agent_step can re-issue a paid Claude inference, so I want control over retry behavior and full visibility into it. Temporal gives me a per-activity RetryPolicy (backoff, max attempts, non-retryable error types). 

3. **Observability for free.** Every step is an activity completion event with a typed payload. The Web UI is enough to debug a stuck trial, and the `<trial_id>/step_NNN/` directories on disk are enough to replay one offline.

---

## Key Temporal design decisions

These are the choices that took the most thought. Each is a tradeoff, and each is in the code as built.

### 1. Save-state-per-step over worker-pinned sessions

The natural way to run an emulator across activities is to keep it alive in worker memory and route every step of a given trial back to the same worker. That avoids the cost of serializing/loading the emulator state on every step.

I chose the opposite: the emulator lives only inside a single activity invocation. At step end, the activity writes `state.bin` (PyBoy save state) and `harness_state.bin` (the harness's serialized internal state, e.g., the message history). The next activity invocation — possibly on a completely different worker — loads those files into a fresh emulator and continues.

This was to primary enable resumability. No worker-affinity machinery (per-trial task queues, session lifecycle hooks, registries) are needed. The overhead of loading the emulator state was a fraction of the model inference time, so it felt like a reasonable price to pay to protect against worker failure. If per-step overhead ever becomes meaningful at scale, the harness protocol already returns a `step_ref` (a path), so a future variant could pin a hot emulator to a worker via Sessions while keeping the on-disk state as the source of truth.

### 2. Step-level reentrance with attempt-scoped partial directories

Temporal retries activities on transient failures. That creates a subtle problem unique to paid stochastic workloads:

- A retry of a step that crashed *before* the model call is harmless — no money spent.
- A retry of a step that crashed *after* the model call but before persistence re-issues the (paid) call, picks a different action (temperature 1.0), and silently diverges from the abandoned attempt. Worst case you pay for two inferences for one logical step.

The protocol I built avoids this:

```
step_NNN+1.partial.attempt_A/        ← current attempt writes artifacts here
    prompt.json                       written before model call
    response.json                     written immediately after model call
    screenshot.png                    after action execution
    memory_dump.json                  after action execution
    harness_state.bin                 after harness.step() returns
    metrics.json                      includes cached milestone_reached
    state.bin                         written LAST — commit marker

step_NNN+1/                           ← canonical: atomic rename target
```

The rules:

1. **Completion check on entry.** If `step_NNN+1/` already exists, the step is done — load `metrics.json` and return. The canonical directory is only ever materialized atomically (rule 4), so its presence guarantees the full file set. No model call, no emulator load.
2. **Attempt-scoped partials.** Each attempt writes to `step_NNN+1.partial.attempt_A/` where `A` is the Temporal activity attempt number. This isolates concurrent attempts: a timed-out attempt may keep running on its original worker while Temporal schedules a retry, and an unscoped partial directory would let them stomp each other's artifacts.
3. **Incremental persistence.** The harness writes `prompt.json` before the model call and `response.json` immediately after, before any side effect. This isn't required for v1's correctness, but it's the hook for future mid-step resume: a future v2 can scan sibling partial directories and adopt their `(prompt, response)` pair to skip the model call. The plumbing is there.
4. **Atomic publish.** After the harness step completes, the activity writes `harness_state.bin`, `metrics.json`, and `state.bin` (in that order — `state.bin` last is the commit marker), then `os.rename(partial, canonical)`. POSIX rename is atomic; on `FileExistsError`/`ENOTEMPTY` (a concurrent attempt won), the loser cleans up its own partial, reads `metrics.json` from the canonical directory, and returns the winner's result.

This protocol is also what makes the eval results inspectable. Every step directory is a self-contained record of what the harness saw, what it asked the model, what the model said, what it did, and what happened.

### 3. Claim-check for large payloads

Workflow history records every activity input and output. A screenshot is 50–200 KB; a full prompt with cached context is hundreds of KB; a save state is ~150 KB. Including those in activity payloads would blow through history size limits quickly.

The activity contract is constrained to small payloads:

| Type | Size |
|------|------|
| `step_ref` (path string + step index) | < 200 bytes |
| `StepMetrics` (token counts, costs, action trace) | ~2–5 KB |
| `success_spec` (predicate definition) | ~1 KB |
| `RunningTotals` (cumulative counters) | < 500 bytes |

Everything else lives in `evals/results/<run_id>/<trial_id>/step_NNN/` and is referenced by path. The activity that writes it and the activity that reads it both see the same filesystem.

### 4. Continue-As-New for unbounded loops

Some test scenarios e.g. a full Elite 4 run may take thousands of inference steps (there are a lot of menus to get through). The workflow history may become larger than can be handled. 

`TrialWorkflow` checks `completed_steps % CONTINUE_AS_NEW_EVERY == 0` (default 250) after each step. When it fires, the workflow calls `workflow.continue_as_new` with a `TrialInit` that carries the resume `step_ref`, the current `RunningTotals`, and the step index. The fresh execution skips `init_trial` and resumes the loop from the carried step. Running totals are passed *explicitly* — so cap checks (max_steps, max_seconds, max_usd) also stay accurate across the boundary.

### 5. Resolved specs at the CLI boundary

Workflows never read mutable inputs at runtime. Before any Temporal code starts, the CLI:

- Reads the suite YAML and the referenced scenario YAML
- Resolves harness `default.yaml` + any `params_override` via deep-merge
- Hashes the initial save-state and the ROM
- Captures git commit + dirty-worktree flag, Python version, PyBoy version, pricing version
- Writes the frozen result to `evals/results/<run_id>/suite.yaml` and passes the resolved data structures to `SweepWorkflow` as immutable inputs

This matters for replay correctness: Temporal will replay a workflow from history if a worker dies, and replay must be deterministic. If the workflow reads a YAML file during execution and that file changed between the original run and the replay, history and code disagree and the workflow fails. Resolving once at the boundary makes the workflow a pure function of its input.

It also matters for reproducibility: every result directory contains the exact resolved suite, the input hashes, the pricing version, and the git commit. A trial result is replayable in principle, even months later under different pricing.

---

## The agentic side

The Temporal layer is half the story; the other half is making the agent itself swappable so that "different harness configurations" is something you can actually express and measure.

### Harness plugin protocol

Every harness implements a small Python protocol:

```python
class Harness(Protocol):
    id: str
    version: str
    def step(self, ctx: StepContext) -> StepResult: ...
    def serialize_state(self) -> bytes: ...
    def load_state(self, blob: bytes) -> None: ...
    def static_config(self) -> dict: ...
```

Harnesses live under `evals/harnesses/<id>/` with a `build(params: dict) -> Harness` entry point and a `default.yaml` declaring their parameter schema. The runner imports the harness by id and talks only to the protocol.

This was to enable flexibility with agent harness implementations. The agent harness can essentially do anything it wants, as long as the shape is respected. The internals of the agent harness are obscured to the eval suite. So a simple single-model "look at screenshot, output a button press" harness versus a more complicated multi-agent architecture are equally feasible to test. We aren't limited by a universe of predetermined configurations. This flexibility felt important because the most interesting improvements are are likely architectural vs. parametric.   

### Predicate DSL for milestones

A scenario declares success as a tree of predicates over the structured memory dump:

```yaml
# evals/scenarios/bedroom_to_downstairs.yaml
id: bedroom_to_downstairs
description: From the bedroom on 2F, find the stairs and descend to 1F.
initial_state: states/after_names_bedroom.state
success:
  all:
    - location_eq: PLAYERS_HOUSE_1F
limits:
  max_steps: 50
  max_seconds: 600
  max_usd: 2.00
```

Built-in predicates cover the things you actually want to assert across early-game scenarios: location equality, coordinate boxes, event flags, badge counts, party membership, etc. Compound operators (`all`, `any`, `not`) and a `first_time` transition predicate (true only on the false→true edge) round it out. The evaluator runs on a typed `MemoryDump` produced from the PyBoy memory map, so adding a new predicate is easy.

### Metrics

Comparing harnesses by `steps_to_milestone` alone is misleading. A navigator harness that issues a 30-button path per step has lower step counts but the same number of emulator-affecting decisions. A harness that invokes subagents to review and revise a suggested action might be one step but five internal model calls.

`StepMetrics` records different metrics in parallel that gives us a view of the bigger picture:

- `decision_count` — emulator-affecting actions taken
- `tool_call_count` — total tool invocations across all internal model calls
- `button_press_count` — raw GB buttons applied (post-expansion of navigator paths)
- `emulated_frames` — PyBoy frame-count delta over the step
- `model_calls` and `input_tokens` / `output_tokens` / `cache_*` — the cost side

Plus `actions: list[Action]` — a typed trace of every action taken, with the raw tool-call args, the expanded button sequence, and frames elapsed, and a result string for actions that can fail (`"blocked by wall"`).

When I compare two harnesses I look at all of these. A harness that's 2× faster by step count and 3× more expensive by tokens is not unambiguously better.

---

## Running it

### Local (no Temporal)

```bash
# One trial, one harness
python -m evals.cli run \
    --scenario evals/scenarios/bedroom_to_downstairs.yaml \
    --harness baseline

# A suite (sequential matrix expansion)
python -m evals.cli run-suite \
    --suite evals/suites/prompt_experiment_a.yaml
```

Each invocation produces `evals/results/<run_id>/` with one subdirectory per trial. Inspect:

```bash
python -m evals.cli inspect evals/results/<run_id>/<trial_id>
```

### With Temporal

Bring up a local Temporal dev server (`temporal server start-dev`), run a worker, then submit the suite:

```bash
# Terminal 1
.venv/bin/python -m evals.worker

# Terminal 2
.venv/bin/python -m evals.cli run-suite \
    --suite evals/suites/prompt_experiment_a.yaml \
    --temporal \
    --concurrency 4
```

The Web UI at shows the `SweepWorkflow` and its child `TrialWorkflow`s, with every activity attempt and its payload.

### Setup notes

- Models are called through AWS Bedrock
- Provide your own Pokemon Red ROM 

---

## Repository layout

```
agent/                  # the agent harness building blocks
    harness.py          # Harness protocol, StepContext, StepMetrics, RunningTotals
    prompt.py           # PromptBuilder
    state_formatter.py  # turns memory + screenshot into model content
    memory_strategy.py  # SummarizeAndReplace, RollingWindow
    memory_reader.py    # PyBoy memory map -> typed MemoryDump
    step_runner.py      # run_one_step — used by both runner and activity
    emulator.py         # PyBoy wrapper
    simple_agent.py     # thin facade preserving the original CLI entry point

evals/
    runner.py           # local single-trial runner, atomic-publish protocol
    cli.py              # `run`, `run-suite`, `inspect`, `replay`
    workflows.py        # SweepWorkflow, TrialWorkflow
    activities.py       # init_trial, run_agent_step, finalize_trial, finalize_sweep
    worker.py           # `python -m evals.worker`
    predicates.py       # predicate DSL evaluator
    pricing.py          # centralized usage -> USD
    temporal_payloads.py# small dataclasses for activity / workflow inputs
    harnesses/          # plugin directories — baseline, noop
    scenarios/          # YAML scenarios
    suites/             # YAML matrix specs
    states/             # checked-in PyBoy save states
    results/            # gitignored — per-run output trees

tests/                  
```
---