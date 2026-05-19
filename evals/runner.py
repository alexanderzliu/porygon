from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from agent.harness import Harness, RunningTotals, StepMetrics
from agent.memory_reader import InventoryItem, MemoryDump
from agent.step_runner import add_step_metrics_to_totals, run_one_step
from evals.predicates import evaluate_predicate
from evals.pricing import PRICING_VERSION, compute_cost

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = EVALS_ROOT / "results"


class TrialOutcome:
    MILESTONE_REACHED = "milestone_reached"
    STEP_CAP = "step_cap"
    TIME_CAP = "time_cap"
    COST_CAP = "cost_cap"
    ERROR = "error"


@dataclass(frozen=True)
class ResolvedHarnessConfig:
    id: str
    params: dict[str, Any]


@dataclass(frozen=True)
class ResolvedTrialSpec:
    scenario_path: Path
    scenario_id: str
    description: str
    initial_state: Path
    success: dict[str, Any]
    limits: dict[str, Any]
    harness: ResolvedHarnessConfig
    rom_path: Path
    results_root: Path
    run_id: str
    trial_index: int
    trial_id: str
    emulator_factory: Callable[..., Any] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class StepRef:
    step_index: int
    path: Path


@dataclass(frozen=True)
class TrialResult:
    run_id: str
    trial_id: str
    trial_dir: Path
    outcome: str
    milestone_reached: bool
    completed_steps: int
    totals: RunningTotals
    error: str | None = None


def run_trial(
    scenario_path: Path,
    harness_id: str,
    params_path: Path | None = None,
    params_override: dict | None = None,
    run_id: str | None = None,
    trial_index: int = 0,
    *,
    rom_path: Path | str | None = None,
    results_root: Path | str | None = None,
    emulator_factory: Callable[..., Any] | None = None,
) -> TrialResult:
    spec = resolve_trial_spec(
        scenario_path=scenario_path,
        harness_id=harness_id,
        params_path=params_path,
        params_override=params_override,
        run_id=run_id,
        trial_index=trial_index,
        rom_path=rom_path,
        results_root=results_root,
        emulator_factory=emulator_factory,
    )
    trial_dir = spec.results_root / spec.run_id / spec.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    _freeze_inputs(spec, trial_dir)

    totals = RunningTotals()
    milestone_reached = False
    outcome = TrialOutcome.ERROR
    error = None
    harness_static_config = None

    try:
        harness = _build_harness(spec.harness.id, spec.harness.params)
        harness_static_config = harness.static_config()
        step_ref, milestone_reached = init_trial(spec, harness, trial_dir)

        if milestone_reached:
            outcome = TrialOutcome.MILESTONE_REACHED
        else:
            outcome = _cap_outcome(totals, spec.limits)

        while outcome is None:
            metrics, step_ref, milestone_reached = run_agent_step(
                spec=spec,
                harness=harness,
                previous_step=step_ref,
                running_totals=totals,
                trial_dir=trial_dir,
                attempt=1,
            )
            totals = _add_metrics_to_totals(totals, metrics)

            if milestone_reached:
                outcome = TrialOutcome.MILESTONE_REACHED
            else:
                outcome = _cap_outcome(totals, spec.limits)

    except Exception as exc:  # noqa: BLE001 - trial.json should record escaped errors.
        outcome = TrialOutcome.ERROR
        error = f"{type(exc).__name__}: {exc}"

    finalize_trial(
        trial_dir,
        spec=spec,
        outcome=outcome,
        milestone_reached=milestone_reached,
        totals=totals,
        error=error,
        harness_static_config=harness_static_config,
    )
    return TrialResult(
        run_id=spec.run_id,
        trial_id=spec.trial_id,
        trial_dir=trial_dir,
        outcome=outcome,
        milestone_reached=milestone_reached,
        completed_steps=totals.steps,
        totals=totals,
        error=error,
    )


def resolve_trial_spec(
    *,
    scenario_path: Path,
    harness_id: str,
    params_path: Path | None = None,
    params_override: dict | None = None,
    run_id: str | None = None,
    trial_index: int = 0,
    rom_path: Path | str | None = None,
    results_root: Path | str | None = None,
    emulator_factory: Callable[..., Any] | None = None,
) -> ResolvedTrialSpec:
    scenario_path = Path(scenario_path).resolve()
    scenario = _load_yaml(scenario_path)
    if not isinstance(scenario, dict):
        raise ValueError(f"Scenario must be a mapping: {scenario_path}")

    scenario_id = str(scenario.get("id") or scenario_path.stem)
    initial_state = _resolve_data_path(scenario_path, scenario["initial_state"])
    harness_params = _resolve_harness_params(harness_id, params_path, params_override)
    resolved_run_id = run_id or _default_run_id()
    trial_id = f"{trial_index:03d}_{scenario_id}_{harness_id}"
    resolved_rom = _resolve_repo_path(rom_path or "pokemon.gb")

    return ResolvedTrialSpec(
        scenario_path=scenario_path,
        scenario_id=scenario_id,
        description=str(scenario.get("description", "")),
        initial_state=initial_state,
        success=dict(scenario["success"]),
        limits=dict(scenario.get("limits", {})),
        harness=ResolvedHarnessConfig(id=harness_id, params=harness_params),
        rom_path=resolved_rom,
        results_root=Path(results_root or DEFAULT_RESULTS_ROOT).resolve(),
        run_id=resolved_run_id,
        trial_index=trial_index,
        trial_id=trial_id,
        emulator_factory=emulator_factory,
    )


def init_trial(
    spec: ResolvedTrialSpec, harness: Harness, trial_dir: Path
) -> tuple[StepRef, bool]:
    step_ref = StepRef(step_index=0, path=trial_dir / "step_000")
    if step_ref.path.exists():
        memory_path = step_ref.path / "memory_dump.json"
        if memory_path.exists():
            memory_dump = _read_memory_dump(memory_path)
            return step_ref, evaluate_predicate(spec.success, memory_dump)
        shutil.rmtree(step_ref.path)

    if not spec.initial_state.exists():
        raise FileNotFoundError(f"Initial eval state not found: {spec.initial_state}")

    step_ref.path.mkdir(parents=True)
    emulator = _new_emulator(spec)
    try:
        emulator.load_state(spec.initial_state)
        _write_emulator_state(emulator, step_ref.path / "state.bin")
        (step_ref.path / "harness_state.bin").write_bytes(harness.serialize_state())
        memory_dump = _write_memory_dump(emulator, step_ref.path)
    finally:
        _stop_emulator(emulator)

    return step_ref, evaluate_predicate(spec.success, memory_dump)


def run_agent_step(
    *,
    spec: ResolvedTrialSpec,
    harness: Harness,
    previous_step: StepRef,
    running_totals: RunningTotals,
    trial_dir: Path,
    attempt: int,
) -> tuple[StepMetrics | dict[str, Any], StepRef, bool]:
    next_index = previous_step.step_index + 1
    canonical = trial_dir / f"step_{next_index:03d}"
    next_ref = StepRef(step_index=next_index, path=canonical)
    if canonical.exists():
        metrics = _read_json(canonical / "metrics.json")
        return metrics, next_ref, bool(metrics.get("milestone_reached", False))

    partial = trial_dir / f"step_{next_index:03d}.partial.attempt_{attempt}"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    emulator = _new_emulator(spec)
    try:
        emulator.load_state(previous_step.path / "state.bin")
        harness.load_state((previous_step.path / "harness_state.bin").read_bytes())

        _, metrics = run_one_step(
            harness,
            emulator,
            step_index=next_index,
            running_totals=running_totals,
            workdir=partial,
            step_ref=str(canonical),
        )
        memory_dump = _ensure_memory_dump(emulator, partial)
        previous_dump = _read_memory_dump(previous_step.path / "memory_dump.json")
        milestone_reached = evaluate_predicate(
            spec.success, memory_dump, previous_dump
        )
        pricing_enabled = _pricing_enabled(spec.harness.params)
        cost_usd = compute_cost(metrics.usage, enabled=pricing_enabled)
        metrics = replace(
            metrics,
            cost_usd=cost_usd,
            step_ref=str(canonical),
            milestone_reached=milestone_reached,
        )

        (partial / "harness_state.bin").write_bytes(harness.serialize_state())
        _write_json(partial / "metrics.json", _jsonable(metrics))
        _write_emulator_state(emulator, partial / "state.bin")
    finally:
        _stop_emulator(emulator)

    try:
        os.rename(partial, canonical)
    except FileExistsError:
        shutil.rmtree(partial, ignore_errors=True)
        metrics = _read_json(canonical / "metrics.json")
        return metrics, next_ref, bool(metrics.get("milestone_reached", False))
    except OSError:
        if canonical.exists():
            shutil.rmtree(partial, ignore_errors=True)
            metrics = _read_json(canonical / "metrics.json")
            return metrics, next_ref, bool(metrics.get("milestone_reached", False))
        raise

    return metrics, next_ref, metrics.milestone_reached


def finalize_trial(
    trial_dir: Path,
    *,
    spec: ResolvedTrialSpec | None = None,
    outcome: str | None = None,
    milestone_reached: bool | None = None,
    totals: RunningTotals | None = None,
    error: str | None = None,
    harness_static_config: dict[str, Any] | None = None,
) -> None:
    trial_dir = Path(trial_dir)
    existing = _read_existing_trial(trial_dir)
    if spec is None:
        spec = _spec_from_disk(trial_dir)
    if existing is not None:
        if outcome is None:
            outcome = existing.get("outcome")
        if error is None:
            error = existing.get("error")
        if harness_static_config is None:
            harness_static_config = existing.get("harness_static_config")

    metrics = _canonical_step_metrics(trial_dir)
    with (trial_dir / "steps.jsonl").open("w", encoding="utf-8") as steps_file:
        for step_metrics in metrics:
            steps_file.write(json.dumps(step_metrics, sort_keys=True) + "\n")

    if totals is None:
        totals = RunningTotals()
        for step_metrics in metrics:
            totals = _add_metrics_to_totals(totals, step_metrics)

    if milestone_reached is None:
        milestone_reached = bool(
            metrics and metrics[-1].get("milestone_reached", False)
        )
    if outcome is None:
        outcome = (
            TrialOutcome.MILESTONE_REACHED
            if milestone_reached
            else "unfinished"
        )

    trial = {
        "run_id": spec.run_id if spec else None,
        "trial_id": spec.trial_id if spec else trial_dir.name,
        "scenario_id": spec.scenario_id if spec else None,
        "harness_id": spec.harness.id if spec else None,
        "harness_static_config": harness_static_config,
        "outcome": outcome,
        "milestone_reached": milestone_reached,
        "completed_steps": totals.steps,
        "totals": _jsonable(totals),
        "pricing_version": PRICING_VERSION,
        "error": error,
    }
    if spec is not None:
        trial.update(_metadata(spec))
    _write_json(trial_dir / "trial.json", trial)


def _build_harness(harness_id: str, params: dict[str, Any]) -> Harness:
    module = importlib.import_module(f"evals.harnesses.{harness_id}")
    return module.build(params)


def _new_emulator(spec: ResolvedTrialSpec):
    if spec.emulator_factory is not None:
        emulator = spec.emulator_factory(spec.rom_path, headless=True, sound=False)
        initialize = getattr(emulator, "initialize", None)
        if initialize is not None:
            initialize()
        return emulator

    from agent.emulator import Emulator

    emulator = Emulator(str(spec.rom_path), headless=True, sound=False)
    emulator.initialize()
    return emulator


def _stop_emulator(emulator) -> None:
    stop = getattr(emulator, "stop", None)
    if stop is not None:
        stop()


def _write_emulator_state(emulator, path: Path) -> None:
    if hasattr(emulator, "save_state"):
        emulator.save_state(path)
        return

    pyboy = getattr(emulator, "pyboy", None)
    if pyboy is None or not hasattr(pyboy, "save_state"):
        raise TypeError("Emulator must expose save_state(path) or pyboy.save_state(file)")
    with path.open("wb") as state_file:
        pyboy.save_state(state_file)


def _write_memory_dump(emulator, step_dir: Path) -> MemoryDump:
    if not hasattr(emulator, "get_memory_dump"):
        raise TypeError("Eval predicates require emulator.get_memory_dump()")

    memory_dump = emulator.get_memory_dump()
    _write_json(step_dir / "memory_dump.json", memory_dump.to_dict())
    return memory_dump


def _ensure_memory_dump(emulator, step_dir: Path) -> MemoryDump:
    memory_path = step_dir / "memory_dump.json"
    if memory_path.exists():
        return _read_memory_dump(memory_path)
    return _write_memory_dump(emulator, step_dir)


def _read_memory_dump(path: Path) -> MemoryDump:
    data = _read_json(path)
    data.pop("text", None)
    if "coordinates" in data:
        data["coordinates"] = tuple(data["coordinates"])
    data["inventory"] = [
        item if isinstance(item, InventoryItem) else InventoryItem(**item)
        for item in data.get("inventory", [])
    ]
    return MemoryDump(**data)


def _add_metrics_to_totals(
    totals: RunningTotals, metrics: StepMetrics | dict[str, Any]
) -> RunningTotals:
    if isinstance(metrics, StepMetrics):
        return add_step_metrics_to_totals(totals, metrics)

    return RunningTotals(
        steps=totals.steps + 1,
        model_calls=totals.model_calls + int(metrics.get("model_calls", 0)),
        input_tokens=totals.input_tokens + int(metrics.get("input_tokens", 0)),
        output_tokens=totals.output_tokens + int(metrics.get("output_tokens", 0)),
        cache_read_tokens=totals.cache_read_tokens
        + int(metrics.get("cache_read_tokens", 0)),
        cache_creation_tokens=totals.cache_creation_tokens
        + int(metrics.get("cache_creation_tokens", 0)),
        cost_usd=totals.cost_usd + float(metrics.get("cost_usd", 0.0)),
        wall_seconds=totals.wall_seconds
        + (int(metrics.get("wall_ms", 0)) / 1000),
    )


def _cap_outcome(totals: RunningTotals, limits: dict[str, Any]) -> str | None:
    max_steps = limits.get("max_steps")
    if max_steps is not None and totals.steps >= int(max_steps):
        return TrialOutcome.STEP_CAP

    max_seconds = limits.get("max_seconds")
    if max_seconds is not None and totals.wall_seconds >= float(max_seconds):
        return TrialOutcome.TIME_CAP

    max_usd = limits.get("max_usd")
    if max_usd is not None and totals.cost_usd >= float(max_usd):
        return TrialOutcome.COST_CAP

    return None


def _canonical_step_metrics(trial_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for step_dir in sorted(trial_dir.glob("step_[0-9][0-9][0-9]")):
        metrics_path = step_dir / "metrics.json"
        if metrics_path.exists():
            rows.append(_read_json(metrics_path))
    return rows


def _read_existing_trial(trial_dir: Path) -> dict[str, Any] | None:
    path = trial_dir / "trial.json"
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except (json.JSONDecodeError, OSError):
        return None


def _spec_from_disk(trial_dir: Path) -> ResolvedTrialSpec | None:
    path = trial_dir / "resolved_trial.json"
    if not path.exists():
        return None
    data = _read_json(path)
    harness = data.get("harness") or {}
    return ResolvedTrialSpec(
        scenario_path=Path(data["scenario_path"]),
        scenario_id=data["scenario_id"],
        description=data["description"],
        initial_state=Path(data["initial_state"]),
        success=data.get("success") or {},
        limits=data.get("limits") or {},
        harness=ResolvedHarnessConfig(
            id=harness.get("id", ""),
            params=harness.get("params") or {},
        ),
        rom_path=Path(data["rom_path"]),
        results_root=trial_dir.parent.parent,
        run_id=data["run_id"],
        trial_index=int(data.get("trial_index", 0)),
        trial_id=data["trial_id"],
    )


def _freeze_inputs(spec: ResolvedTrialSpec, trial_dir: Path) -> None:
    shutil.copyfile(spec.scenario_path, trial_dir / "scenario.yaml")
    _write_json(trial_dir / "harness_params.json", spec.harness.params)
    _write_json(
        trial_dir / "resolved_trial.json",
        {
            "run_id": spec.run_id,
            "trial_id": spec.trial_id,
            "trial_index": spec.trial_index,
            "scenario_id": spec.scenario_id,
            "description": spec.description,
            "scenario_path": str(spec.scenario_path),
            "initial_state": str(spec.initial_state),
            "success": spec.success,
            "limits": spec.limits,
            "harness": _jsonable(spec.harness),
            "rom_path": str(spec.rom_path),
        },
    )


def _metadata(spec: ResolvedTrialSpec) -> dict[str, Any]:
    return {
        "description": spec.description,
        "scenario_path": str(spec.scenario_path),
        "scenario_sha256": _sha256(spec.scenario_path),
        "initial_state": str(spec.initial_state),
        "initial_state_sha256": _sha256(spec.initial_state),
        "rom_path": str(spec.rom_path),
        "rom_sha256": _sha256(spec.rom_path),
        "params_sha256": _hash_json(spec.harness.params),
        "python_version": platform.python_version(),
        "git": _git_metadata(),
    }


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit or None, "dirty": dirty}
    except OSError:
        return {"commit": None, "dirty": None}


def _resolve_harness_params(
    harness_id: str, params_path: Path | None, params_override: dict | None
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    default_path = EVALS_ROOT / "harnesses" / harness_id / "default.yaml"
    if default_path.exists():
        params = _deep_merge(params, _load_yaml(default_path))
    if params_path is not None:
        params = _deep_merge(params, _load_yaml(Path(params_path)))
    if params_override:
        params = _deep_merge(params, params_override)
    return params


def _pricing_enabled(params: dict[str, Any]) -> bool:
    pricing = params.get("pricing", {})
    if isinstance(pricing, dict) and "enabled" in pricing:
        return bool(pricing["enabled"])
    return True


def _resolve_data_path(scenario_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    candidates = [
        scenario_path.parent / path,
        scenario_path.parent.parent / path,
        REPO_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[1].resolve()


def _resolve_repo_path(value: Path | str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_yaml(path: Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)
    return yaml.safe_load(text) or {}


def _parse_simple_yaml(text: str) -> Any:
    lines = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        if lines[index][1].startswith("- "):
            return parse_list(index, indent)
        return parse_map(index, indent)

    def parse_map(index: int, indent: int) -> tuple[dict[str, Any], int]:
        result: dict[str, Any] = {}
        while index < len(lines):
            line_indent, content = lines[index]
            if line_indent < indent or content.startswith("- "):
                break
            if line_indent > indent:
                raise ValueError(f"Unexpected YAML indentation near: {content}")

            key, sep, raw_value = content.partition(":")
            if not sep:
                raise ValueError(f"Expected YAML mapping entry near: {content}")
            raw_value = raw_value.strip()
            index += 1
            if raw_value:
                result[key] = _parse_scalar(raw_value)
            elif index < len(lines) and lines[index][0] > line_indent:
                result[key], index = parse_block(index, lines[index][0])
            else:
                result[key] = {}
        return result, index

    def parse_list(index: int, indent: int) -> tuple[list[Any], int]:
        result = []
        while index < len(lines):
            line_indent, content = lines[index]
            if line_indent < indent or not content.startswith("- "):
                break
            if line_indent > indent:
                raise ValueError(f"Unexpected YAML indentation near: {content}")

            item = content[2:].strip()
            index += 1
            if ":" in item:
                key, _, raw_value = item.partition(":")
                raw_value = raw_value.strip()
                if raw_value:
                    value = {key: _parse_scalar(raw_value)}
                elif index < len(lines) and lines[index][0] > line_indent:
                    nested, index = parse_block(index, lines[index][0])
                    value = {key: nested}
                else:
                    value = {key: {}}

                if index < len(lines) and lines[index][0] > line_indent:
                    continuation, index = parse_block(index, lines[index][0])
                    if isinstance(value, dict) and isinstance(continuation, dict):
                        value = _deep_merge(value, continuation)
                    else:
                        raise ValueError(
                            f"Unexpected YAML list continuation near: {content}"
                        )
                result.append(value)
            else:
                result.append(_parse_scalar(item))
        return result, index

    parsed, final_index = parse_block(0, lines[0][0] if lines else 0)
    if final_index != len(lines):
        raise ValueError("Could not parse complete YAML document")
    return parsed


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "Null", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value.startswith("{") and value.endswith("}"):
        return _parse_flow_mapping(value[1:-1])
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_flow_mapping(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in _split_flow_items(text):
        key, sep, raw_value = item.partition(":")
        if not sep:
            raise ValueError(f"Expected YAML flow mapping entry near: {item}")
        result[_strip_quotes(key.strip())] = _parse_scalar(raw_value.strip())
    return result


def _split_flow_items(text: str) -> list[str]:
    items = []
    start = 0
    depth = 0
    quote: str | None = None
    for index, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
        elif char == "," and depth == 0:
            item = text[start:index].strip()
            if item:
                items.append(item)
            start = index + 1
    final_item = text[start:].strip()
    if final_item:
        items.append(final_item)
    return items


def _strip_quotes(value: str) -> str:
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            field_name: _jsonable(getattr(value, field_name))
            for field_name in value.__dataclass_fields__
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(data), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(data: Any) -> str:
    encoded = json.dumps(_jsonable(data), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _default_run_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
