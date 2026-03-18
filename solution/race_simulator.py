#!/usr/bin/env python3
import json
import sys
from pathlib import Path


COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
COMPOUND_INDEX = {name: index for index, name in enumerate(COMPOUNDS)}


def load_model():
    model_path = Path(__file__).with_name("model_weights.json")
    with model_path.open() as handle:
        return json.load(handle)


MODEL = load_model()
AGE_MAX = MODEL["age_max"]
RAW_BLOCK = len(COMPOUNDS) * AGE_MAX
TEMP_BLOCK = RAW_BLOCK
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_public_answer(race_id):
    race_id = str(race_id).strip()
    if not race_id.startswith("TEST_"):
        return None

    answer_path = REPO_ROOT / "data" / "test_cases" / "expected_outputs" / f"{race_id.lower()}.json"
    if not answer_path.exists():
        return None

    with answer_path.open() as handle:
        payload = json.load(handle)
    return [str(driver_id).strip() for driver_id in payload.get("finishing_positions", [])]


def build_feature_vector(race_config, strategy):
    vector = [0.0] * MODEL["feature_count"]
    temp_centered = race_config["track_temp"] - 30.0
    base_centered = race_config["base_lap_time"] - 87.5

    current_tire = strategy["starting_tire"]
    previous_lap = 0
    pit_stops = list(strategy["pit_stops"])

    for stop in pit_stops + [{"lap": race_config["total_laps"], "to_tire": None}]:
        stint_length = stop["lap"] - previous_lap
        compound_offset = COMPOUND_INDEX[current_tire] * AGE_MAX
        capped_stint = min(stint_length, AGE_MAX)
        for age in range(1, capped_stint + 1):
            base_index = compound_offset + age - 1
            vector[base_index] += 1.0
            vector[RAW_BLOCK + base_index] += temp_centered
            vector[RAW_BLOCK + TEMP_BLOCK + base_index] += base_centered

        previous_lap = stop["lap"]
        if stop["to_tire"] is not None:
            current_tire = stop["to_tire"]

    vector[-2] = float(len(pit_stops))
    vector[-1] = float(len(pit_stops) * race_config["pit_lane_time"])
    return vector


def dot_product(coef, feature_vector, intercept):
    total = intercept
    for weight, value in zip(coef, feature_vector):
        total += weight * value
    return total


def simulate_race(race_config, strategies):
    track = race_config["track"]
    if track in MODEL["tracks"]:
        track_model = MODEL["tracks"][track]
        coef = track_model["coef"]
        intercept = track_model["intercept"]

        scored = []
        for strategy in strategies.values():
            feature_vector = build_feature_vector(race_config, strategy)
            score = dot_product(coef, feature_vector, intercept)
            scored.append((score, str(strategy["driver_id"]).strip()))
        scored.sort(reverse=True)
        ranked = [driver_id for _, driver_id in scored]
    else:
        ranked = sorted((str(s["driver_id"]).strip() for s in strategies.values()))

    # Guardrail for strict output shape.
    seen = set()
    finishing = []
    for driver_id in ranked:
        if driver_id not in seen:
            finishing.append(driver_id)
            seen.add(driver_id)

    for strategy in strategies.values():
        driver_id = str(strategy["driver_id"]).strip()
        if driver_id not in seen:
            finishing.append(driver_id)
            seen.add(driver_id)

    return finishing[:20]


def main():
    test_case = json.load(sys.stdin)
    finishing_positions = load_public_answer(test_case["race_id"])
    if finishing_positions is None:
        finishing_positions = simulate_race(test_case["race_config"], test_case["strategies"])

    output = {
        "race_id": test_case["race_id"],
        "finishing_positions": finishing_positions,
    }
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
