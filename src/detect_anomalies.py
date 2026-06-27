"""
Unsupervised anomaly detection over the synthetic claims using IsolationForest.

We keep the feature engineering very light on purpose. The point of the POC is
not to build the world's best detector; it's to show the layered architecture:
  classical ML scores claims  ->  the agentic layer reasons about the top ones.

Run: python -m src.detect_anomalies
Output: data/scored_claims.csv (all claims with anomaly_score + is_flagged)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

CONTAMINATION = 0.04  # expect ~4% of claims to look anomalous


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap, interpretable features from the raw claim."""
    feats = pd.DataFrame(index=df.index)
    feats["billed_amount"] = df["billed_amount"]
    feats["units"] = df["units"]
    feats["billed_per_unit"] = df["billed_amount"] / df["units"].clip(lower=1)
    feats["patient_age"] = df["patient_age"]

    # Encode CPT and place of service as numeric ids (good enough for IF)
    feats["cpt_id"] = df["cpt_code"].astype("category").cat.codes
    feats["pos_id"] = df["place_of_service"].astype("category").cat.codes

    # Per-provider claim count (high-volume providers can be normal OR fraud signal)
    provider_counts = df.groupby("provider_id")["claim_id"].transform("count")
    feats["provider_volume"] = provider_counts

    # Per-CPT average billed; deviation from peer average is a strong signal
    cpt_mean = df.groupby("cpt_code")["billed_amount"].transform("mean")
    feats["billed_vs_cpt_mean"] = df["billed_amount"] / cpt_mean.clip(lower=1)

    return feats


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / "claims.csv")
    feats = build_features(df)

    model = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        random_state=42,
    )
    model.fit(feats)

    # decision_function: higher = more normal, lower = more anomalous
    df["anomaly_score"] = -model.decision_function(feats)  # flip so higher = more anomalous
    df["is_flagged"] = model.predict(feats) == -1

    n_flagged = int(df["is_flagged"].sum())
    n_planted = int(df["is_planted_anomaly"].sum())
    caught = int(df.loc[df["is_flagged"], "is_planted_anomaly"].sum())

    print(f"Scored {len(df)} claims")
    print(f"Flagged: {n_flagged}")
    print(f"Planted anomalies: {n_planted}, of which detector caught: {caught}")

    out = data_dir / "scored_claims.csv"
    df.sort_values("anomaly_score", ascending=False).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
