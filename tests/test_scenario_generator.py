import pytest

from voiceqa.generator.scenario_generator import generate_scenarios, validate_scenarios


def test_validate_scenarios_accepts_well_formed_list():
    raw = [
        {
            "command": "Set temperature to 22 degrees",
            "category": "happy_path",
            "expected_intent": "set_temperature",
            "expected_ui_state": "climate screen shows 22C",
        }
    ]
    cleaned = validate_scenarios(raw)
    assert cleaned[0]["id"] == "scn_000"
    assert cleaned[0]["command"] == "Set temperature to 22 degrees"


def test_validate_scenarios_rejects_missing_field():
    raw = [{"command": "x", "category": "happy_path", "expected_intent": "y"}]
    with pytest.raises(ValueError, match="missing fields"):
        validate_scenarios(raw)


def test_validate_scenarios_rejects_bad_category():
    raw = [
        {
            "command": "x",
            "category": "not_a_real_category",
            "expected_intent": "y",
            "expected_ui_state": "z",
        }
    ]
    with pytest.raises(ValueError, match="invalid category"):
        validate_scenarios(raw)


def test_validate_scenarios_rejects_empty_list():
    with pytest.raises(ValueError):
        validate_scenarios([])


def test_generate_scenarios_dry_run_covers_all_categories():
    scenarios = generate_scenarios("voice-controlled climate control", dry_run=True)
    categories = {s["category"] for s in scenarios}
    assert categories == {"happy_path", "paraphrased", "edge_case", "adversarial"}
    assert len(scenarios) >= 12
