"""Validate the local XLife minimal test bundle with Python's standard library."""

import hashlib
import json
from datetime import datetime
from pathlib import Path


root = Path(__file__).resolve().parent
data = json.loads((root / "sample_data.json").read_text(encoding="utf-8"))
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

assert len(data) == manifest["participants"] == 3
assert len({person["subject_id"] for person in data}) == 3
assert sum(len(person["meals"]) for person in data) == manifest["observed_meals"] == 2
assert sum(len(person["exercise_events"]) for person in data) == manifest["observed_exercise_events"] == 2
assert sum(len(person["nocturnal_hypoglycemia"]) for person in data) == manifest["nocturnal_windows"] == 2


def check_spacing(series):
    times = [datetime.fromisoformat(point[0]) for point in series]
    assert all((b - a).total_seconds() == 300 for a, b in zip(times, times[1:]))


labels = []
referenced_images = set()
for person in data:
    assert person["subject_name"].startswith("participant_")
    for meal in person["meals"]:
        assert len(meal["cgm_preprandial"]) == 12
        assert len(meal["cgm_postprandial"]) == 36
        check_spacing(meal["cgm_preprandial"])
        check_spacing(meal["cgm_postprandial"])
        for item in meal["food_data"]["data"]["food_items"]:
            name = item["img_names"]
            assert Path(name).name == name
            assert (root / "images" / name).is_file()
            referenced_images.add(name)
    for exercise in person["exercise_events"]:
        assert len(exercise["cgm_pre_exercise"]) == 12
        assert len(exercise["cgm_post_exercise"]) == 36
        check_spacing(exercise["cgm_pre_exercise"])
        check_spacing(exercise["cgm_post_exercise"])
    for night in person["nocturnal_hypoglycemia"]:
        series = night["cgm_sleep"]
        assert len(series) == 120
        check_spacing(series)
        run = longest = 0
        for _, value in series:
            run = run + 1 if value < 3.9 else 0
            longest = max(longest, run)
        label = int(longest >= 3)
        assert night["hypoglycemia_label"] == label
        labels.append(label)

assert sorted(labels) == [0, 1]
assert referenced_images == set(manifest["images"])
for name, expected in manifest["images"].items():
    assert hashlib.sha256((root / "images" / name).read_bytes()).hexdigest() == expected

print("OK: 3 participants, 2 meals, 2 exercise events, 2 nights, 4 images")
