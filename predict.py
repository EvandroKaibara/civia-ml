"""Use the trained model inside CIVIA, our project for CitizenTech.

rate_route() scores each segment of a route for one user profile and returns
a verdict a navigation screen can show: the worst segment decides the route.
"""
import pathlib

import joblib
import pandas as pd

MODEL_PATH = pathlib.Path(__file__).parent / "model.joblib"
ORDER = {"low": 0, "medium": 1, "high": 2}


def rate_route(segments: list[dict], profile: str) -> dict:
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame(segments).assign(profile=profile)
    levels = model.predict(df)
    worst = max(levels, key=ORDER.get)
    return {"profile": profile, "segment_levels": list(levels), "route_level": worst}


if __name__ == "__main__":
    route_a = [
        {"segment_type": "sidewalk", "tactile_paving": 1, "audible_signal": 0, "visual_signal": 0,
         "braille_signage": 0, "visual_alerts": 0, "sidewalk_width_m": 2.4, "obstacles_per_100m": 1,
         "slope_pct": 3.0, "ambient_noise_db": 62.0, "lighting_lux": 50.0, "traffic_vehicles_per_min": 8},
        {"segment_type": "crossing", "tactile_paving": 1, "audible_signal": 1, "visual_signal": 1,
         "braille_signage": 1, "visual_alerts": 1, "sidewalk_width_m": 2.0, "obstacles_per_100m": 0,
         "slope_pct": 2.0, "ambient_noise_db": 66.0, "lighting_lux": 60.0, "traffic_vehicles_per_min": 10},
    ]
    route_b = [
        {"segment_type": "crossing", "tactile_paving": 0, "audible_signal": 0, "visual_signal": 0,
         "braille_signage": 0, "visual_alerts": 0, "sidewalk_width_m": 1.0, "obstacles_per_100m": 6,
         "slope_pct": 12.0, "ambient_noise_db": 85.0, "lighting_lux": 8.0, "traffic_vehicles_per_min": 25},
    ]
    for name, route in [("A", route_a), ("B", route_b)]:
        for profile in ("blind", "deaf"):
            print(name, rate_route(route, profile))
