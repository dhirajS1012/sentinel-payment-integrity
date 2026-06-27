# Cotiviti POC, Mock Payment Integrity Policies

These are **synthetic** policy clauses, written only for the POC demo so the agent
has something concrete to cite. They do not represent real Cotiviti policy.

---

## P-101: CPT Pricing Band Tolerance
Billed amounts that exceed the 90th percentile of historical billed amounts for
the same CPT code by more than **3x** must be reviewed by a payment integrity
analyst before payment.

## P-102: Unit Count Reasonableness
For routine diagnostic procedures (ECG, basic labs, imaging single-view), more
than **5 units** of the same CPT for the same member on the same date of service
is considered clinically implausible and must be pended for review.

## P-201: Clinical Appropriateness, Age and Procedure
Major elective surgical procedures (e.g., joint replacement, CPT 27447 / 27130)
billed for members under **40 years of age** require documentation of medical
necessity before adjudication.

## P-202: Place-of-Service Consistency
Inpatient or surgical procedures billed with an office place-of-service code
(POS 11) constitute a coding inconsistency and must be returned to the provider
for correction.

## P-301: High-Volume Provider Review
Providers in the top **1%** of monthly claim volume relative to specialty peers
are subject to enhanced 360-degree pattern review.

## P-401: AI-Generated Documentation
Where supporting documentation shows signs of synthetic generation (template
artifacts, internally inconsistent narratives), the claim is escalated to the
Special Investigations Unit (SIU).
