from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from evals.runner import (
    DEFAULT_RESULTS_ROOT,
    EVALS_ROOT,
    REPO_ROOT,
    finalize_trial,
    run_trial,
    _load_yaml,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


def run_one_command(args: argparse.Namespace) -> None:
    params_override = _load_params_override(args.params_override)
    result = run_trial(
        Path(args.scenario),
        args.harness,
        params_path=Path(args.params).resolve() if args.params else None,
        params_override=params_override,
        run_id=args.run_id,
        trial_index=args.trial_index,
        rom_path=args.rom,
        results_root=args.results_root,
    )
    _print_trial_result(result.trial_dir)


def run_suite_command(args: argparse.Namespace) -> None:
    run_dir = run_suite(
        Path(args.suite),
        run_id=args.run_id,
        rom_path=args.rom,
        results_root=args.results_root,
    )
    rows = _read_summary_rows(run_dir)
    print(f"Run: {run_dir}")
    print(f"Trials: {len(rows)}")
    print(f"Summary: {run_dir / 'summary.jsonl'}")


def inspect_command(args: argparse.Namespace) -> None:
    inspect_trial(Path(args.trial_dir))


def finalize_command(args: argparse.Namespace) -> None:
    trial_dir = Path(args.trial_dir)
    finalize_trial(trial_dir)
    print(f"Finalized: {trial_dir}")


def summarize_command(args: argparse.Namespace) -> None:
    rows = finalize_sweep(Path(args.run_dir))
    print(f"Summary: {Path(args.run_dir) / 'summary.jsonl'}")
    print(f"Trials: {len(rows)}")


def run_suite(
    suite_path: Path,
    *,
    run_id: str | None = None,
    rom_path: str | Path | None = None,
    results_root: str | Path | None = None,
) -> Path:
    suite_path = suite_path.resolve()
    suite = _load_suite(suite_path)
    resolved_run_id = run_id or _default_run_id()
    resolved_results_root = Path(results_root or DEFAULT_RESULTS_ROOT).resolve()
    run_dir = resolved_results_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(suite_path, run_dir / "suite.yaml")

    scenario_path = _resolve_scenario_path(suite_path, suite["scenario"])
    trials = int(suite.get("trials", 1))
    matrix = suite.get("matrix") or []
    if trials < 1:
        raise ValueError("Suite 'trials' must be at least 1")
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("Suite 'matrix' must be a non-empty list")

    trial_index = 0
    for entry in matrix:
        if not isinstance(entry, dict):
            raise ValueError("Every suite matrix entry must be a mapping")
        harness_id = str(entry["harness"])
        params_path = _resolve_params_path(suite_path, harness_id, entry.get("params"))
        params_override = entry.get("params_override")
        if params_override is not None and not isinstance(params_override, dict):
            raise ValueError("'params_override' must be a mapping")

        for _ in range(trials):
            run_trial(
                scenario_path,
                harness_id,
                params_path=params_path,
                params_override=params_override,
                run_id=resolved_run_id,
                trial_index=trial_index,
                rom_path=rom_path,
                results_root=resolved_results_root,
            )
            trial_index += 1

    finalize_sweep(run_dir)
    return run_dir


def inspect_trial(trial_dir: Path) -> dict[str, Any]:
    trial_dir = trial_dir.resolve()
    trial = _read_json(trial_dir / "trial.json")
    totals = trial.get("totals") or {}
    final_memory = _read_final_memory_dump(trial_dir)
    final_location = final_memory.get("location") if final_memory else None

    print(f"Trial: {trial.get('trial_id') or trial_dir.name}")
    print(f"Outcome: {trial.get('outcome')}")
    print(f"Steps: {trial.get('completed_steps', 0)}")
    print(f"Cost: ${float(totals.get('cost_usd', 0.0)):.6f}")
    print(
        "Tokens: "
        f"input={int(totals.get('input_tokens', 0))} "
        f"output={int(totals.get('output_tokens', 0))} "
        f"cache_read={int(totals.get('cache_read_tokens', 0))} "
        f"cache_creation={int(totals.get('cache_creation_tokens', 0))}"
    )
    print(f"Model calls: {int(totals.get('model_calls', 0))}")
    print(f"Final location: {final_location or 'unknown'}")
    if trial.get("error"):
        print(f"Error: {trial['error']}")

    return {
        "trial": trial,
        "final_location": final_location,
    }


def finalize_sweep(run_dir: Path) -> list[dict[str, Any]]:
    run_dir = run_dir.resolve()
    rows = []
    for trial_json in sorted(run_dir.glob("*/trial.json")):
        trial_dir = trial_json.parent
        trial = _read_json(trial_json)
        rows.append(_summary_row(trial_dir, trial))

    summary_path = run_dir / "summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        for row in rows:
            summary_file.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evals.cli")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run_one = subcommands.add_parser("run-one", help="Run one local eval trial")
    run_one.add_argument("scenario")
    run_one.add_argument("--harness", required=True)
    run_one.add_argument("--params")
    run_one.add_argument("--params-override")
    run_one.add_argument("--run-id")
    run_one.add_argument("--trial-index", type=int, default=0)
    run_one.add_argument("--rom")
    run_one.add_argument("--results-root")
    run_one.set_defaults(func=run_one_command)

    run = subcommands.add_parser("run", help="Run a local sequential eval suite")
    run.add_argument("suite")
    run.add_argument("--run-id")
    run.add_argument("--rom")
    run.add_argument("--results-root")
    run.set_defaults(func=run_suite_command)

    inspect = subcommands.add_parser("inspect", help="Inspect a completed trial")
    inspect.add_argument("trial_dir")
    inspect.set_defaults(func=inspect_command)

    finalize = subcommands.add_parser("finalize", help="Regenerate trial artifacts")
    finalize.add_argument("trial_dir")
    finalize.set_defaults(func=finalize_command)

    summarize = subcommands.add_parser("summarize", help="Regenerate run summary")
    summarize.add_argument("run_dir")
    summarize.set_defaults(func=summarize_command)

    return parser


def _load_suite(suite_path: Path) -> dict[str, Any]:
    suite = _load_yaml(suite_path)
    if not isinstance(suite, dict):
        raise ValueError(f"Suite must be a mapping: {suite_path}")
    if "scenario" not in suite:
        raise ValueError("Suite must include 'scenario'")

    concurrency = int(suite.get("concurrency", 1))
    if concurrency < 1:
        raise ValueError("Suite 'concurrency' must be at least 1")
    return suite


def _load_params_override(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("--params-override must be a JSON object")
    return value


def _resolve_scenario_path(suite_path: Path, value: str) -> Path:
    path = Path(str(value))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                suite_path.parent / path,
                EVALS_ROOT / "scenarios" / path,
                EVALS_ROOT / "scenarios" / f"{value}.yaml",
                REPO_ROOT / path,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Eval scenario not found: {value}")


def _resolve_params_path(
    suite_path: Path, harness_id: str, value: str | None
) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                suite_path.parent / path,
                EVALS_ROOT / "harnesses" / harness_id / path,
                EVALS_ROOT / "harnesses" / path,
                REPO_ROOT / path,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Eval harness params not found: {value}")


def _summary_row(trial_dir: Path, trial: dict[str, Any]) -> dict[str, Any]:
    totals = trial.get("totals") or {}
    final_memory = _read_final_memory_dump(trial_dir)
    return {
        "run_id": trial.get("run_id"),
        "trial_id": trial.get("trial_id") or trial_dir.name,
        "scenario_id": trial.get("scenario_id"),
        "harness_id": trial.get("harness_id"),
        "outcome": trial.get("outcome"),
        "milestone_reached": bool(trial.get("milestone_reached", False)),
        "completed_steps": int(trial.get("completed_steps", 0)),
        "model_calls": int(totals.get("model_calls", 0)),
        "input_tokens": int(totals.get("input_tokens", 0)),
        "output_tokens": int(totals.get("output_tokens", 0)),
        "cache_read_tokens": int(totals.get("cache_read_tokens", 0)),
        "cache_creation_tokens": int(totals.get("cache_creation_tokens", 0)),
        "cost_usd": float(totals.get("cost_usd", 0.0)),
        "final_location": final_memory.get("location") if final_memory else None,
        "error": trial.get("error"),
    }


def _read_final_memory_dump(trial_dir: Path) -> dict[str, Any] | None:
    for step_dir in reversed(sorted(trial_dir.glob("step_[0-9][0-9][0-9]"))):
        memory_path = step_dir / "memory_dump.json"
        if memory_path.exists():
            return _read_json(memory_path)
    return None


def _print_trial_result(trial_dir: Path) -> None:
    trial = _read_json(trial_dir / "trial.json")
    totals = trial.get("totals") or {}
    print(f"Run: {trial.get('run_id')}")
    print(f"Trial: {trial.get('trial_id')}")
    print(f"Outcome: {trial.get('outcome')}")
    print(f"Steps: {trial.get('completed_steps', 0)}")
    print(f"Cost: ${float(totals.get('cost_usd', 0.0)):.6f}")
    print(f"Directory: {trial_dir}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_summary_rows(run_dir: Path) -> list[dict[str, Any]]:
    summary_path = run_dir / "summary.jsonl"
    if not summary_path.exists():
        return []
    return [
        json.loads(line)
        for line in summary_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _default_run_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
