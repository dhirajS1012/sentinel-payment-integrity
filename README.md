# Sentinel

A two-stage payment integrity console. IsolationForest flags anomalous healthcare claims. A Gemini-backed reasoning agent grounds each flag against a policy library and emits a structured triage decision.

Built for the Cotiviti Intern Assessment by Dhiraj Salunkhe, University of Pennsylvania.

## Live demo

https://sentinel-payment-integrity.streamlit.app

The hosted app boots cold with synthetic data and is ready for the reviewer to click through. Reasoning Agent runs use a personal Gemini key with a daily quota of 20 requests.

## Submission package

| Deliverable | File |
|---|---|
| Written report | [submission/Sentinel_Report.docx](submission/Sentinel_Report.docx) |
| Slide deck | [submission/Sentinel_Slides.pptx](submission/Sentinel_Slides.pptx) |
| Video walkthrough | [submission/Sentinel_Demo.mp4](submission/Sentinel_Demo.mp4) |
| Live app | https://sentinel-payment-integrity.streamlit.app |
| Source code | this repository |

## What it does

```
synthetic claims  ->  IsolationForest  ->  Gemini reasoning agent  ->  structured decision
   (510 rows,        anomaly score and      grounded in mock           clear / investigate /
    10 planted)      is_flagged label        policy library             escalate + citations
```

Stage one is unsupervised. Stage two grounds the LLM in a policy document so every decision carries explicit policy references. The output is JSON, not a chat transcript, so it can drop straight into an analyst queue or an SIU referral.

## Example agent decision

```
CLM-000504    INVESTIGATE    confidence 95%
CPT 27447 Total knee arthroplasty | 1 unit | $199,946.88 | PRV-9999

The claim violates policy P-201 because a major elective joint replacement
(CPT 27447) was billed for a 36-year-old member, which requires documentation
of medical necessity. It also violates policy P-202 because a major surgical
procedure was billed with an office place-of-service code (POS 11), which is
a coding inconsistency.

policy_refs: [P-201, P-202]
next_action: Refer to a payment integrity analyst to request medical necessity
documentation and return the claim for place-of-service correction.
```

## Repository layout

```
.
├── data/                       generated CSVs (gitignored)
├── policies/policies.md        mock payment integrity policies
├── src/
│   ├── generate_data.py        synthetic claims with planted anomalies
│   ├── detect_anomalies.py     IsolationForest scoring
│   ├── agent_reason.py         Gemini reasoning agent (CLI)
│   └── app.py                  Sentinel console (Streamlit)
├── .streamlit/config.toml      bright theme for Streamlit Cloud
├── requirements.txt
└── README.md
```

## Run it locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=AIza...               # https://aistudio.google.com/apikey

python -m src.generate_data                 # writes data/claims.csv
python -m src.detect_anomalies              # writes data/scored_claims.csv

python -m src.agent_reason                  # CLI run, prints decisions
streamlit run src/app.py                    # full console at localhost:8501
```

The Streamlit console will auto-generate data on first launch if `data/` is empty.

## Design notes

- **The agent layer is the point.** IsolationForest with seven features catches 9 of 10 planted anomalies. Most of the engineering effort went into grounding the LLM in policy and forcing structured output, not into detector tuning.
- **Strict JSON, not chat.** The agent returns `decision`, `confidence`, `rationale`, `policy_refs`, `suggested_next_action`. No prose, no apologies, no markdown.
- **Three decisions, no auto-deny.** `clear`, `investigate`, `escalate`. The agent routes claims to a human; it never adjudicates.
- **Synthetic everything.** No PHI, no real Cotiviti policy. The mock policies in `policies/policies.md` exist so the agent has something concrete to cite.
- **Rate-aware demo.** The console paces agent calls at 13s intervals to stay inside the Gemini free-tier limit of 5 requests per minute.

## Stack

Python 3.10, scikit-learn, pandas, Streamlit, Altair, Google Generative AI SDK (`gemini-flash-latest`).

Hosted on Streamlit Community Cloud.
