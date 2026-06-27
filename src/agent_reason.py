"""
Agentic reasoning layer (Google Gemini).

A flagged claim is handed to Gemini with:
  - the structured claim record
  - the (mock) policy document
  - any relevant member / CPT context

Gemini returns a structured JSON decision:
  {
    "decision": "investigate" | "clear" | "escalate",
    "confidence": 0.0 - 1.0,
    "rationale": "...",
    "policy_refs": ["P-101", ...],
    "suggested_next_action": "..."
  }

This is the chain-reasoning + agentic step on top of the IsolationForest score.

Run: python -m src.agent_reason
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import google.generativeai as genai
except ImportError:
    print("Install with: pip install google-generativeai", file=sys.stderr)
    raise

MODEL = "gemini-flash-latest"  # routes to the current-gen Flash model on the free tier
TOP_N = 8


def load_policies() -> str:
    path = Path(__file__).resolve().parent.parent / "policies" / "policies.md"
    return path.read_text()


def claim_to_context(row: pd.Series) -> str:
    return (
        f"claim_id: {row['claim_id']}\n"
        f"member_id: {row['member_id']}  (age {row['patient_age']})\n"
        f"provider_id: {row['provider_id']}\n"
        f"cpt_code: {row['cpt_code']}  -  {row['cpt_description']}\n"
        f"units: {row['units']}\n"
        f"billed_amount_usd: {row['billed_amount']:.2f}\n"
        f"place_of_service: {row['place_of_service']}\n"
        f"isolation_forest_anomaly_score: {row['anomaly_score']:.4f}"
    )


SYSTEM_PROMPT = """You are a payment integrity reasoning agent for a U.S. health plan.
You receive ONE claim that an unsupervised anomaly detector has flagged, plus
the plan's payment integrity policy document.

Your job is to decide what should happen to the claim. You MUST ground your
reasoning in the policy document and cite the policy IDs you rely on.

Return ONLY a single JSON object, no prose before or after, with this exact schema:

{
  "decision": "clear" | "investigate" | "escalate",
  "confidence": <float between 0 and 1>,
  "rationale": "<2-4 sentence plain-English explanation>",
  "policy_refs": ["P-101", ...],
  "suggested_next_action": "<one short sentence>"
}

Definitions:
  - "clear": the flag is likely a false positive; pay the claim.
  - "investigate": send to a human payment integrity analyst for review.
  - "escalate": refer to the Special Investigations Unit (SIU) for suspected
    fraud, waste, or abuse.

Be conservative. Prefer "investigate" over "escalate" unless the evidence is strong.
"""


def build_client(api_key: str):
    """Configure the Gemini SDK and return a model handle.

    The Streamlit app and CLI both call this so the auth path stays in one place.
    """
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            # gemini-flash-latest routes to a thinking model; thinking tokens
            # count against this budget, so give it plenty of room.
            "max_output_tokens": 4096,
        },
    )


def reason_about_claim(client, claim_block: str, policies: str) -> dict:
    user_msg = (
        f"POLICY DOCUMENT:\n```\n{policies}\n```\n\n"
        f"FLAGGED CLAIM:\n```\n{claim_block}\n```\n\n"
        "Return your JSON decision now."
    )
    resp = client.generate_content(user_msg)
    text = (resp.text or "").strip()
    # Strip a leading/trailing ```json fence if the model adds one despite JSON mode.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("ERROR: set GEMINI_API_KEY env var before running.")

    client = build_client(api_key)
    policies = load_policies()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / "scored_claims.csv")
    flagged = df[df["is_flagged"]].head(TOP_N).reset_index(drop=True)
    print(f"Reasoning about top {len(flagged)} flagged claims with {MODEL}...\n")

    results = []
    for _, row in flagged.iterrows():
        block = claim_to_context(row)
        try:
            decision = reason_about_claim(client, block, policies)
        except Exception as e:  # noqa: BLE001
            print(f"  [{row['claim_id']}] ERROR: {e}")
            decision = {
                "decision": "investigate",
                "confidence": 0.0,
                "rationale": f"Agent error: {e}",
                "policy_refs": [],
                "suggested_next_action": "manual review",
            }

        print(f"== {row['claim_id']} ==")
        print(f"  decision: {decision.get('decision')}  "
              f"(confidence {decision.get('confidence')})")
        print(f"  policy_refs: {decision.get('policy_refs')}")
        print(f"  rationale: {decision.get('rationale')}\n")

        results.append({**row.to_dict(), **{f"agent_{k}": v for k, v in decision.items()}})

    out = data_dir / "agent_decisions.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
