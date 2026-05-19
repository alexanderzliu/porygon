from __future__ import annotations

import re
from typing import Any

from agent.memory_reader import MapLocation, MemoryDump


_LOCATION_ALIASES = {
    "REDSHOUSE1F": "PLAYERSHOUSE1F",
    "REDSHOUSE2F": "PLAYERSHOUSE2F",
    "REDHOUSE1F": "PLAYERSHOUSE1F",
    "REDHOUSE2F": "PLAYERSHOUSE2F",
}


def evaluate_predicate(
    spec: dict, current: MemoryDump, previous: MemoryDump | None = None
) -> bool:
    if not isinstance(spec, dict):
        raise TypeError("Predicate spec must be a dict")
    if len(spec) != 1:
        raise ValueError("Predicate spec must contain exactly one predicate")

    predicate, value = next(iter(spec.items()))

    if predicate == "location_eq":
        return _location_eq(current, value)
    if predicate == "coords_in_box":
        return _coords_in_box(current, value)
    if predicate == "dialog_contains":
        return value in (current.dialog or "")
    if predicate == "badge_count_at_least":
        return len(current.badges) >= int(value)
    if predicate == "party_has_pokemon":
        expected = _normalize_name(value)
        return any(
            _normalize_name(_party_species(member)) == expected
            for member in current.party
        )
    if predicate == "all":
        _require_list(predicate, value)
        return all(evaluate_predicate(inner, current, previous) for inner in value)
    if predicate == "any":
        _require_list(predicate, value)
        return any(evaluate_predicate(inner, current, previous) for inner in value)
    if predicate == "not":
        return not evaluate_predicate(value, current, previous)
    if predicate == "first_time":
        if previous is None:
            return False
        return (
            evaluate_predicate(value, current, previous)
            and not evaluate_predicate(value, previous, None)
        )

    raise ValueError(f"Unknown predicate: {predicate}")


def _location_eq(current: MemoryDump, expected: Any) -> bool:
    if isinstance(expected, int):
        return current.map_id == expected

    expected_key = _normalize_location(expected)
    actual_key = _normalize_location(current.location)

    if actual_key == expected_key:
        return True

    if current.map_id is not None:
        try:
            actual_from_map_id = _normalize_location(MapLocation(current.map_id).name)
        except ValueError:
            actual_from_map_id = None
        if actual_from_map_id == expected_key:
            return True

    return False


def _coords_in_box(current: MemoryDump, value: Any) -> bool:
    if not isinstance(value, dict):
        raise TypeError("coords_in_box value must be a dict")

    expected_map = value.get("map")
    if expected_map is not None and not _location_eq(current, expected_map):
        return False

    x_range = value.get("x")
    y_range = value.get("y")
    if not _is_range(x_range) or not _is_range(y_range):
        raise ValueError("coords_in_box requires x and y two-value ranges")

    x, y = current.coordinates
    return (
        int(x_range[0]) <= x <= int(x_range[1])
        and int(y_range[0]) <= y <= int(y_range[1])
    )


def _party_species(member: Any) -> str:
    if isinstance(member, dict):
        species = member.get("species_name") or member.get("species")
        return str(species or member.get("name") or "")
    return str(getattr(member, "species_name", ""))


def _normalize_location(value: Any) -> str:
    normalized = _normalize_name(value)
    return _LOCATION_ALIASES.get(normalized, normalized)


def _normalize_name(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _require_list(predicate: str, value: Any) -> None:
    if not isinstance(value, list):
        raise TypeError(f"{predicate} predicate requires a list")


def _is_range(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2
