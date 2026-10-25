from voiceqa.mock_target.mock_assistant import MockClimateAssistant


def test_set_temperature_happy_path():
    a = MockClimateAssistant()
    result = a.handle("set temperature to 22 degrees")
    assert result["intent"] == "set_temperature"
    assert result["ui_state"]["temperature_c"] == 22


def test_paraphrased_warmer_increases_temperature():
    a = MockClimateAssistant(temperature_c=20)
    result = a.handle("make it warmer")
    assert result["intent"] == "increase_temperature"
    assert result["ui_state"]["temperature_c"] == 22


def test_out_of_range_temperature_asks_for_clarification():
    a = MockClimateAssistant()
    result = a.handle("set temperature to 222 degrees")
    assert result["intent"] == "clarification_needed"


def test_mismatched_units_asks_for_clarification():
    a = MockClimateAssistant()
    result = a.handle("set the volume to 22 degrees")
    assert result["intent"] == "clarification_needed"


def test_fan_speed_increase():
    a = MockClimateAssistant(fan_speed=2)
    result = a.handle("turn the fan up")
    assert result["intent"] == "increase_fan_speed"
    assert result["ui_state"]["fan_speed"] == 3


def test_ac_on_and_off():
    a = MockClimateAssistant()
    on_result = a.handle("turn on the air conditioning")
    assert on_result["intent"] == "ac_on"
    assert on_result["ui_state"]["ac_on"] is True

    off_result = a.handle("turn off the air conditioning")
    assert off_result["intent"] == "ac_off"
    assert off_result["ui_state"]["ac_on"] is False


def test_unknown_command_falls_back_to_clarification():
    a = MockClimateAssistant()
    result = a.handle("what's the weather like on Mars")
    assert result["intent"] == "clarification_needed"


def test_latency_is_deterministic_for_same_transcript():
    a = MockClimateAssistant()
    r1 = a.handle("set temperature to 22 degrees")
    a2 = MockClimateAssistant()
    r2 = a2.handle("set temperature to 22 degrees")
    assert r1["latency_ms"] == r2["latency_ms"]
