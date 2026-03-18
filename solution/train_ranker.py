#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier


COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
COMPOUND_INDEX = {name: index for index, name in enumerate(COMPOUNDS)}
AGE_MAX = 70
RAW_BLOCK = len(COMPOUNDS) * AGE_MAX
TEMP_BLOCK = RAW_BLOCK
BASE_BLOCK = RAW_BLOCK
PIT_FEATURES = 2
FEATURE_COUNT = RAW_BLOCK + TEMP_BLOCK + BASE_BLOCK + PIT_FEATURES


def iter_historical_races(data_dir: Path):
    for file_path in sorted(data_dir.glob("*.json")):
        with file_path.open() as handle:
            for race in json.load(handle):
                yield race


def build_feature_vector(race_config, strategy):
    vector = np.zeros(FEATURE_COUNT, dtype=np.float64)
    temp_centered = race_config["track_temp"] - 30.0
    base_centered = race_config["base_lap_time"] - 87.5

    current_tire = strategy["starting_tire"]
    pit_stops = list(strategy["pit_stops"])
    previous_lap = 0

    for stop in pit_stops + [{"lap": race_config["total_laps"], "to_tire": None}]:
        stint_length = stop["lap"] - previous_lap
        compound_offset = COMPOUND_INDEX[current_tire] * AGE_MAX
        for age in range(1, stint_length + 1):
            base_index = compound_offset + age - 1
            vector[base_index] += 1.0
            vector[RAW_BLOCK + base_index] += temp_centered
            vector[RAW_BLOCK + TEMP_BLOCK + base_index] += base_centered

        previous_lap = stop["lap"]
        if stop["to_tire"] is not None:
            current_tire = stop["to_tire"]

    vector[-2] = len(pit_stops)
    vector[-1] = len(pit_stops) * race_config["pit_lane_time"]
    return vector


def build_driver_feature_map(race):
    race_config = race["race_config"]
    return {
        strategy["driver_id"]: build_feature_vector(race_config, strategy)
        for strategy in race["strategies"].values()
    }


def iter_pair_batches(races, batch_size):
    batch_x = []
    batch_y = []

    for race in races:
        feature_map = build_driver_feature_map(race)
        finishing_positions = race["finishing_positions"]
        for winner_index in range(len(finishing_positions) - 1):
            winner = finishing_positions[winner_index]
            winner_features = feature_map[winner]
            for loser_index in range(winner_index + 1, len(finishing_positions)):
                loser = finishing_positions[loser_index]
                delta = winner_features - feature_map[loser]
                batch_x.append(delta)
                batch_y.append(1)
                batch_x.append(-delta)
                batch_y.append(0)

                if len(batch_y) >= batch_size:
                    yield np.vstack(batch_x), np.asarray(batch_y, dtype=np.int32)
                    batch_x = []
                    batch_y = []

    if batch_y:
        yield np.vstack(batch_x), np.asarray(batch_y, dtype=np.int32)


def train_track_model(track_races, epochs, alpha, batch_size):
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        random_state=42,
        max_iter=1,
        tol=None,
        shuffle=True,
        average=True,
    )

    classes = np.array([0, 1], dtype=np.int32)
    initialized = False

    for _ in range(epochs):
        for batch_x, batch_y in iter_pair_batches(track_races, batch_size=batch_size):
            if not initialized:
                classifier.partial_fit(batch_x, batch_y, classes=classes)
                initialized = True
            else:
                classifier.partial_fit(batch_x, batch_y)

    return classifier


def predict_order(model, race):
    feature_map = build_driver_feature_map(race)
    scored = []
    for strategy in race["strategies"].values():
        driver_id = strategy["driver_id"]
        score = float(model.decision_function(feature_map[driver_id].reshape(1, -1))[0])
        scored.append((score, driver_id))
    scored.sort(reverse=True)
    return [driver_id for _, driver_id in scored]


def evaluate_models(models, races):
    exact = 0
    total = 0
    for race in races:
        track = race["race_config"]["track"]
        prediction = predict_order(models[track], race)
        if prediction == race["finishing_positions"]:
            exact += 1
        total += 1
    return exact, total


def load_test_cases(root: Path):
    inputs_dir = root / "data" / "test_cases" / "inputs"
    expected_dir = root / "data" / "test_cases" / "expected_outputs"
    if not expected_dir.exists():
        return []

    races = []
    for input_path in sorted(inputs_dir.glob("test_*.json")):
        expected_path = expected_dir / input_path.name
        if not expected_path.exists():
            continue
        with input_path.open() as handle:
            race = json.load(handle)
        with expected_path.open() as handle:
            expected = json.load(handle)
        race["finishing_positions"] = expected["finishing_positions"]
        races.append(race)
    return races


def export_models(models, output_path: Path):
    payload = {
        "age_max": AGE_MAX,
        "feature_count": FEATURE_COUNT,
        "tracks": {
            track: {
                "intercept": float(model.intercept_[0]),
                "coef": model.coef_[0].tolist(),
            }
            for track, model in sorted(models.items())
        },
    }
    output_path.write_text(json.dumps(payload))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.00003)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--validation-races", type=int, default=3000)
    parser.add_argument(
        "--export",
        type=Path,
        default=Path(__file__).with_name("model_weights.json"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    all_races = list(iter_historical_races(root / "data" / "historical_races"))

    validation_races = all_races[-args.validation_races :]
    training_races = all_races[: -args.validation_races]

    by_track = {}
    for race in training_races:
        by_track.setdefault(race["race_config"]["track"], []).append(race)

    models = {}
    for track, track_races in sorted(by_track.items()):
        print(f"training {track} on {len(track_races)} races")
        models[track] = train_track_model(track_races, args.epochs, args.alpha, args.batch_size)

    validation_exact, validation_total = evaluate_models(models, validation_races)
    print(f"validation exact: {validation_exact}/{validation_total} = {validation_exact / validation_total:.3%}")

    test_races = load_test_cases(root)
    if test_races:
        test_exact, test_total = evaluate_models(models, test_races)
        print(f"public tests exact: {test_exact}/{test_total} = {test_exact / test_total:.3%}")

    full_by_track = {}
    for race in all_races:
        full_by_track.setdefault(race["race_config"]["track"], []).append(race)

    final_models = {}
    for track, track_races in sorted(full_by_track.items()):
        final_models[track] = train_track_model(track_races, args.epochs, args.alpha, args.batch_size)

    export_models(final_models, args.export)
    print(f"exported weights to {args.export}")


if __name__ == "__main__":
    main()