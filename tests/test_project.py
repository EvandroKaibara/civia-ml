import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import predict  # noqa: E402
from generate_data import generate  # noqa: E402

LEVELS = {"low", "medium", "high"}

GOOD = {"segment_type": "crossing", "tactile_paving": 1, "audible_signal": 1, "visual_signal": 1,
        "braille_signage": 1, "visual_alerts": 1, "sidewalk_width_m": 2.5, "obstacles_per_100m": 0,
        "slope_pct": 1.0, "ambient_noise_db": 60.0, "lighting_lux": 80.0, "traffic_vehicles_per_min": 5}
BAD = {"segment_type": "crossing", "tactile_paving": 0, "audible_signal": 0, "visual_signal": 0,
       "braille_signage": 0, "visual_alerts": 0, "sidewalk_width_m": 0.8, "obstacles_per_100m": 8,
       "slope_pct": 14.0, "ambient_noise_db": 90.0, "lighting_lux": 4.0, "traffic_vehicles_per_min": 30}


@pytest.fixture(scope="module")
def df():
    return generate()


@pytest.fixture(scope="module")
def test_split(df):
    X, y = df.drop(columns="barrier_level"), df["barrier_level"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_test, y_test


# --- data ---------------------------------------------------------------

def test_generate_is_deterministic():
    pd.testing.assert_frame_equal(generate(), generate())


def test_labels_and_profiles_are_valid(df):
    assert set(df["barrier_level"]) == LEVELS
    assert set(df["profile"]) == {"blind", "deaf"}


def test_no_class_is_nearly_empty(df):
    assert df["barrier_level"].value_counts(normalize=True).min() > 0.15


def test_only_sensor_columns_have_gaps(df):
    assert set(df.columns[df.isna().any()]) == {"ambient_noise_db", "lighting_lux"}


def test_value_ranges(df):
    assert df["sidewalk_width_m"].between(0.6, 5.0).all()
    assert df["slope_pct"].between(0, 20).all()
    assert df[["tactile_paving", "audible_signal", "visual_signal"]].isin([0, 1]).all().all()


# --- model --------------------------------------------------------------

def test_model_reproduces_reported_metric(test_split):
    X_test, y_test = test_split
    model = joblib.load(ROOT / "model.joblib")
    f1 = f1_score(y_test, model.predict(X_test), average="macro")
    reported = json.loads((ROOT / "metrics.json").read_text())
    assert f1 == pytest.approx(reported["f1_macro_test"], abs=1e-3)
    assert f1 > reported["f1_macro_baseline"] + 0.3


def test_model_never_calls_a_high_barrier_segment_low(test_split):
    X_test, y_test = test_split
    pred = joblib.load(ROOT / "model.joblib").predict(X_test)
    assert ((y_test == "high") & (pred == "low")).sum() == 0


# --- integration --------------------------------------------------------

def test_good_and_bad_segments():
    for profile in ("blind", "deaf"):
        assert predict.rate_route([GOOD], profile)["route_level"] == "low"
        assert predict.rate_route([BAD], profile)["route_level"] == "high"


def test_route_takes_the_worst_segment():
    result = predict.rate_route([GOOD, BAD, GOOD], "blind")
    assert result["segment_levels"][1] == "high"
    assert result["route_level"] == "high"


def test_tactile_paving_matters_more_for_blind_than_deaf():
    model = joblib.load(ROOT / "model.joblib")
    base = dict(BAD, obstacles_per_100m=2, sidewalk_width_m=2.0, slope_pct=4.0,
                ambient_noise_db=65.0, lighting_lux=40.0, traffic_vehicles_per_min=10)
    shift = {}
    for profile in ("blind", "deaf"):
        rows = pd.DataFrame([base, dict(base, tactile_paving=1)]).assign(profile=profile)
        p_high = model.predict_proba(rows)[:, list(model.classes_).index("high")]
        shift[profile] = p_high[0] - p_high[1]
    assert shift["blind"] > shift["deaf"]


def test_missing_sensor_values_do_not_crash():
    seg = dict(GOOD, ambient_noise_db=np.nan, lighting_lux=np.nan)
    assert predict.rate_route([seg], "deaf")["route_level"] in LEVELS
