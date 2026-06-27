"""
Generate a synthetic healthcare claims dataset for the POC.

A handful of obvious anomalies are planted so the IsolationForest has signal
to catch and the reasoning agent has concrete cases to triage.

Run: python -m src.generate_data
Output: data/claims.csv
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# A small CPT code catalog with plausible price bands (USD)
CPT_CATALOG = {
    "99213": ("Office visit, established patient, 15 min", 80, 140),
    "99214": ("Office visit, established patient, 25 min", 130, 210),
    "99215": ("Office visit, established patient, 40 min", 200, 320),
    "93000": ("ECG, routine, with interpretation", 40, 90),
    "80053": ("Comprehensive metabolic panel", 25, 60),
    "70553": ("MRI, brain, with and without contrast", 1400, 2600),
    "27447": ("Total knee arthroplasty", 18000, 32000),
    "45378": ("Colonoscopy, diagnostic", 900, 1800),
    "71046": ("Chest X-ray, 2 views", 60, 140),
    "99283": ("ED visit, moderate severity", 250, 500),
}

PROVIDERS = [f"PRV-{i:04d}" for i in range(1, 41)]
MEMBERS = [f"MBR-{i:05d}" for i in range(1, 301)]

N_CLAIMS = 500


def make_claim(claim_id: int) -> dict:
    cpt, (desc, lo, hi) = random.choice(list(CPT_CATALOG.items()))
    billed = round(np.random.uniform(lo, hi), 2)
    units = 1
    return {
        "claim_id": f"CLM-{claim_id:06d}",
        "member_id": random.choice(MEMBERS),
        "provider_id": random.choice(PROVIDERS),
        "cpt_code": cpt,
        "cpt_description": desc,
        "units": units,
        "billed_amount": billed,
        "place_of_service": random.choice(["11", "11", "11", "22", "23"]),  # 11=office
        "patient_age": random.randint(18, 88),
        "is_planted_anomaly": False,
        "planted_reason": "",
    }


def main() -> None:
    rows = [make_claim(i) for i in range(1, N_CLAIMS + 1)]

    # --- Plant 10 obvious anomalies so the demo has signal ---

    # 1) Five claims with billed amounts 5x-10x the high end of the band
    for i in range(5):
        cpt, (desc, lo, hi) = random.choice(list(CPT_CATALOG.items()))
        rows.append({
            "claim_id": f"CLM-{N_CLAIMS + i + 1:06d}",
            "member_id": random.choice(MEMBERS),
            "provider_id": "PRV-9999",  # suspicious repeat provider
            "cpt_code": cpt,
            "cpt_description": desc,
            "units": 1,
            "billed_amount": round(hi * np.random.uniform(5, 10), 2),
            "place_of_service": "11",
            "patient_age": random.randint(18, 88),
            "is_planted_anomaly": True,
            "planted_reason": "billed_amount far exceeds CPT band",
        })

    # 2) Three claims with absurd unit counts (e.g., 50 ECGs in one visit)
    for i in range(3):
        cpt, (desc, lo, hi) = "93000", CPT_CATALOG["93000"]
        rows.append({
            "claim_id": f"CLM-{N_CLAIMS + 5 + i + 1:06d}",
            "member_id": random.choice(MEMBERS),
            "provider_id": random.choice(PROVIDERS),
            "cpt_code": "93000",
            "cpt_description": CPT_CATALOG["93000"][0],
            "units": random.randint(30, 80),
            "billed_amount": round(CPT_CATALOG["93000"][2] * random.randint(30, 80), 2),
            "place_of_service": "11",
            "patient_age": random.randint(18, 88),
            "is_planted_anomaly": True,
            "planted_reason": "unit count clinically implausible",
        })

    # 3) Two clinical mismatches (knee replacement on a 20-year-old, etc.)
    rows.append({
        "claim_id": f"CLM-{N_CLAIMS + 9:06d}",
        "member_id": random.choice(MEMBERS),
        "provider_id": random.choice(PROVIDERS),
        "cpt_code": "27447",
        "cpt_description": CPT_CATALOG["27447"][0],
        "units": 1,
        "billed_amount": 28000.0,
        "place_of_service": "11",  # knee replacement in an office? unusual
        "patient_age": 20,
        "is_planted_anomaly": True,
        "planted_reason": "clinical mismatch: TKA on young patient in office setting",
    })
    rows.append({
        "claim_id": f"CLM-{N_CLAIMS + 10:06d}",
        "member_id": random.choice(MEMBERS),
        "provider_id": random.choice(PROVIDERS),
        "cpt_code": "70553",
        "cpt_description": CPT_CATALOG["70553"][0],
        "units": 1,
        "billed_amount": 8200.0,
        "place_of_service": "11",
        "patient_age": 35,
        "is_planted_anomaly": True,
        "planted_reason": "MRI billed 4x typical, place of service unusual",
    })

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)

    out = Path(__file__).resolve().parent.parent / "data" / "claims.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} claims to {out}")
    print(f"Planted anomalies: {df['is_planted_anomaly'].sum()}")


if __name__ == "__main__":
    main()
