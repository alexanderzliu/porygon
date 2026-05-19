import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from agent.memory_reader import InventoryItem, MemoryDump
from agent.state_formatter import StateFormatter
from evals.predicates import evaluate_predicate


def dump(**overrides) -> MemoryDump:
    values = {
        "player_name": "RED",
        "rival_name": "BLUE",
        "money": 3000,
        "location": "PLAYERS_HOUSE_2F",
        "map_id": 0x26,
        "coordinates": (3, 4),
        "valid_moves": ["up", "down"],
        "badges": [],
        "inventory": [],
        "dialog": None,
        "party": [],
        "raw": {},
    }
    values.update(overrides)
    return MemoryDump(**values)


class FakeStructuredEmulator:
    def get_screenshot(self):
        return Image.new("RGB", (2, 2))

    def get_memory_dump(self):
        return dump(
            location="PLAYERS_HOUSE_1F",
            map_id=0x25,
            inventory=[InventoryItem("POTION", 1)],
        )

    def get_collision_map(self):
        return None


class Phase2PredicateTests(unittest.TestCase):
    def test_location_eq_accepts_reds_house_alias(self):
        current = dump(location="PLAYERS_HOUSE_1F", map_id=0x25)

        self.assertTrue(
            evaluate_predicate({"location_eq": "REDS_HOUSE_1F"}, current)
        )

    def test_coords_in_box_checks_map_and_inclusive_ranges(self):
        current = dump(location="PLAYERS_HOUSE_1F", map_id=0x25, coordinates=(2, 3))

        self.assertTrue(
            evaluate_predicate(
                {
                    "coords_in_box": {
                        "map": "REDS_HOUSE_1F",
                        "x": [0, 4],
                        "y": [0, 3],
                    }
                },
                current,
            )
        )
        self.assertFalse(
            evaluate_predicate(
                {"coords_in_box": {"x": [0, 1], "y": [0, 3]}}, current
            )
        )

    def test_simple_predicates_and_compounds(self):
        current = dump(
            badges=["BOULDER"],
            dialog="Welcome to the world of POKEMON!",
            party=[{"species_name": "PIKACHU"}],
        )

        self.assertTrue(evaluate_predicate({"dialog_contains": "world"}, current))
        self.assertTrue(evaluate_predicate({"badge_count_at_least": 1}, current))
        self.assertTrue(evaluate_predicate({"party_has_pokemon": "PIKACHU"}, current))
        self.assertTrue(
            evaluate_predicate(
                {
                    "all": [
                        {"badge_count_at_least": 1},
                        {"party_has_pokemon": "PIKACHU"},
                    ]
                },
                current,
            )
        )
        self.assertTrue(
            evaluate_predicate(
                {
                    "any": [
                        {"party_has_pokemon": "BULBASAUR"},
                        {"dialog_contains": "Welcome"},
                    ]
                },
                current,
            )
        )
        self.assertTrue(
            evaluate_predicate({"not": {"party_has_pokemon": "BULBASAUR"}}, current)
        )

    def test_first_time_only_matches_false_to_true_transition(self):
        previous = dump(location="PLAYERS_HOUSE_2F", map_id=0x26)
        current = dump(location="PLAYERS_HOUSE_1F", map_id=0x25)
        spec = {"first_time": {"location_eq": "REDS_HOUSE_1F"}}

        self.assertFalse(evaluate_predicate(spec, current))
        self.assertTrue(evaluate_predicate(spec, current, previous))
        self.assertFalse(evaluate_predicate(spec, current, current))

    def test_state_formatter_writes_structured_memory_dump(self):
        formatter = StateFormatter(screenshot_upscale=1)

        with tempfile.TemporaryDirectory() as tmp:
            feedback = formatter.capture(FakeStructuredEmulator(), workdir=Path(tmp))
            artifact = json.loads((Path(tmp) / "memory_dump.json").read_text())

        self.assertIn("Location: PLAYERS HOUSE 1F", feedback.memory_info)
        self.assertEqual(artifact["location"], "PLAYERS_HOUSE_1F")
        self.assertEqual(artifact["inventory"], [{"name": "POTION", "quantity": 1}])
        self.assertIn("text", artifact)


if __name__ == "__main__":
    unittest.main()
