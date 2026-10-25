from voiceqa.evaluation.evaluator_agent import evaluate_response


SCENARIO = {
    "id": "scn_000",
    "command": "Set temperature to 22 degrees",
    "category": "happy_path",
    "expected_intent": "set_temperature",
    "expected_ui_state": "climate screen shows temperature_c 22",
}


def test_matching_intent_scores_high():
    actual = {
        "ui_hierarchy": '{"temperature_c": 22, "fan_speed": 2, "ac_on": false}',
        "transcript": "Setting temperature to 22 degrees.",
        "latency_ms": 300,
        "recognized_intent": "set_temperature",
        "screenshot": "some/path.json",
    }
    result = evaluate_response(SCENARIO, actual, dry_run=True)
    assert result["intent_correctness"] == 1.0
    assert result["pass"] is True


def test_wrong_intent_scores_low():
    actual = {
        "ui_hierarchy": '{"temperature_c": 21, "fan_speed": 2, "ac_on": false}',
        "transcript": "Sorry, I didn't understand that.",
        "latency_ms": 300,
        "recognized_intent": "clarification_needed",
        "screenshot": "some/path.json",
    }
    result = evaluate_response(SCENARIO, actual, dry_run=True)
    assert result["intent_correctness"] == 0.0
    assert result["pass"] is False


def test_empty_transcript_scores_zero_naturalness():
    actual = {
        "ui_hierarchy": "{}",
        "transcript": "",
        "latency_ms": 300,
        "recognized_intent": None,
        "screenshot": "some/path.json",
    }
    result = evaluate_response(SCENARIO, actual, dry_run=True)
    assert result["response_naturalness"] == 0.0
