"""Simulate street-segment accessibility data for blind and deaf pedestrians.

The label is produced from an explicit scoring rule plus noise, so the dataset
is synthetic. See README.md for the rule and its limits.
"""
import numpy as np
import pandas as pd

SEED = 42
N = 3000


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "profile": rng.choice(["blind", "deaf"], n),
        "segment_type": rng.choice(
            ["sidewalk", "crossing", "transit_stop", "plaza", "stairs_ramp"],
            n, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
        "tactile_paving": rng.binomial(1, 0.45, n),
        "audible_signal": rng.binomial(1, 0.30, n),
        "visual_signal": rng.binomial(1, 0.50, n),
        "braille_signage": rng.binomial(1, 0.25, n),
        "visual_alerts": rng.binomial(1, 0.35, n),
        "sidewalk_width_m": rng.normal(2.0, 0.7, n).clip(0.6, 5.0).round(2),
        "obstacles_per_100m": rng.poisson(2.5, n),
        "slope_pct": rng.gamma(2.0, 2.5, n).clip(0, 20).round(1),
        "ambient_noise_db": rng.normal(68, 9, n).clip(40, 95).round(1),
        "lighting_lux": rng.gamma(3.0, 12.0, n).clip(2, 120).round(1),
        "traffic_vehicles_per_min": rng.poisson(12, n),
    })

    blind = df["profile"] == "blind"
    crossing = df["segment_type"] == "crossing"
    score = 70.0
    score = score + 4 * (df["sidewalk_width_m"] - 1.5).clip(upper=1.5)
    score = score - 1.2 * (df["slope_pct"] - 8).clip(lower=0)
    score = score - np.where(df["segment_type"] == "stairs_ramp", 8, 0)

    # Blind pedestrians depend on touch and sound cues.
    score = score + np.where(blind, 15 * df["tactile_paving"], 0)
    score = score + np.where(blind & crossing, 14 * df["audible_signal"], 0)
    score = score + np.where(blind, 6 * df["braille_signage"], 0)
    score = score - np.where(blind, 3.5 * df["obstacles_per_100m"], 0)
    score = score - np.where(blind, 0.5 * (df["ambient_noise_db"] - 65).clip(lower=0), 0)

    # Deaf pedestrians depend on sight, so light and visual cues matter.
    score = score + np.where(~blind & crossing, 14 * df["visual_signal"], 0)
    score = score + np.where(~blind, 9 * df["visual_alerts"], 0)
    score = score + np.where(~blind, 0.15 * (df["lighting_lux"] - 30).clip(upper=60), 0)
    score = score - np.where(~blind, 0.7 * df["traffic_vehicles_per_min"], 0)
    score = score - np.where(~blind, 1.5 * df["obstacles_per_100m"], 0)

    score = score + rng.normal(0, 6, n)
    # Cut at quantiles so no class is nearly empty: ~25% high, 45% medium, 30% low.
    lo, hi = np.quantile(score, [0.25, 0.70])
    df["barrier_level"] = pd.cut(
        score, bins=[-np.inf, lo, hi, np.inf], labels=["high", "medium", "low"]
    ).astype(str)

    # Sensors fail: leave some gaps so the pipeline has to handle them.
    for col in ["ambient_noise_db", "lighting_lux"]:
        df.loc[rng.random(n) < 0.04, col] = np.nan
    return df


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).parent / "data" / "accessibility_segments.csv"
    out.parent.mkdir(exist_ok=True)
    data = generate()
    data.to_csv(out, index=False)
    print(data["barrier_level"].value_counts(normalize=True).round(3).to_dict())
