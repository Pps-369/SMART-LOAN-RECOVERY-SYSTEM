# Training summary

Trained 2026-09-15T21:01:59+00:00 - model **logistic_regression** (isotonic calibrated), 5 borrower segments.

## Model comparison (5-fold CV on train)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| logistic_regression | 0.8184 | 0.6565 |
| hist_gradient_boosting | 0.8105 | 0.6425 |
| random_forest | 0.8054 | 0.6383 |

## Hold-out test set

| Metric | Value |
|---|---|
| ROC-AUC | 0.8199 (days-past-due rule: 0.6566) |
| PR-AUC | 0.6611 (base loss rate: 0.2821) |
| Brier score | 0.1464 |
| Threshold (value-optimised) | 0.43 |
| Precision / Recall at threshold | 0.649 / 0.5244 |

## Business value on the test portfolio

| Policy | Net recovery value |
|---|---|
| Model-driven escalation | INR 0.83 Cr |
| Escalate everything | INR -2.64 Cr |
| Escalate if DPD >= 90 | INR -0.96 Cr |
| Perfect foresight (ceiling) | INR 2.53 Cr |

Values depend on the cost and uplift assumptions in `config.yaml`.

## Segments

| Segment | Train loss rate |
|---|---|
| S1 - Low Risk | 9.1% |
| S2 - Guarded | 20.2% |
| S3 - Moderate Risk | 26.7% |
| S4 - High Risk | 43.2% |
| S5 - Severe Risk | 50.2% |
