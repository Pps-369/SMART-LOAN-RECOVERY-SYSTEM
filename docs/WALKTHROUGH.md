# Build walkthrough: what each step does and why

Read the files in this order. Each step gives the command, what to look at, and the reasoning.

## Step 1: Understand the problem framing
A bank already knows these loans are overdue. The decision is **how much effort to spend on each**.
So the target is binary: `Written Off` (loss = 1) vs `Fully/Partially Recovered` (0). A good model
ranks loans by loss risk. A useful system also says what to do and what it is worth.

## Step 2: Data (`data_generator.py`)
```bash
python -m loan_recovery.data_generator
```
Open `data/raw/loan_recovery.csv`. Note three column groups:
- **Borrower and loan facts** (income, EMI, collateral, credit score): known at decision time.
- **Delinquency state** (days past due, missed installments): known at decision time.
- **Collection columns** (method, attempts, legal action, recovered amount): known only *afterwards*.

*Why synthetic?* Recovery data is confidential. The generator encodes realistic cause and effect, so
the modelling decisions below still matter.

## Step 3: Features (`features.py`)
Raw amounts mean little on their own. ₹3 L outstanding is serious for a ₹30k earner and minor for a ₹3 L earner.
So ratios are added: `EMI_to_Income` (affordability), `Collateral_Coverage` (what enforcement recovers),
`Debt_to_Annual_Income`, `Missed_Payment_Rate`, and `Log_Days_Past_Due` (tames skew).

`model_inputs()` is a **whitelist**. New columns added to the dataset later can't leak into the model by accident.

## Step 4: Segmentation (`clustering.py`)
- Features are log-transformed where long-tailed, then standardised. KMeans uses distances, so scale matters.
- k is tried from 3 to 6 and chosen by silhouette score. Fewer than 3 segments isn't actionable; more than 6 is hard to staff.
- Clusters are named by their **training** loss rate (S1 safest → S5 riskiest).
- `SegmentDistanceAdder` puts "distance to each segment" into the classifier pipeline. It is re-fitted
  inside each CV fold, which keeps it leakage-free.

Look at: `reports/segment_profiles.csv`, `reports/figures/segment_k_selection.png`.

## Step 5: Model selection (`models.py`, `train.py`)
```bash
python -m loan_recovery.train
```
- 60/20/20 stratified split: train / validation / test.
- 5-fold CV on train compares logistic regression, random forest and gradient boosting on **PR-AUC**.
  PR-AUC focuses on the loss class. ROC-AUC can look good even when minority-class precision is poor.
- The winner is logistic regression. Talking point: *"I chose the simpler model because it matched the
  complex ones and is easier to explain to credit and compliance teams."*

## Step 6: Calibration
`class_weight="balanced"` pushes probabilities upward. Calibration (isotonic) fixes that, so "0.40"
really means about a 40% loss rate. That matters because probabilities are multiplied by rupee amounts next.

## Step 7: Business threshold (`strategy.py`)
Escalating a loan costs ₹30,000 (assumption). If the loan would otherwise be lost, escalation saves
10% of the outstanding amount (assumption). The threshold that maximises net value on the **validation** set is 0.43.
Look at `reports/figures/threshold_value_curve.png`: too low wastes money, too high misses losses.

Talking point: *"I optimised the threshold for money, not F1, and compared against the bank's existing rule.
The 90-DPD rule lost ₹0.96 Cr on the test set, while the model made ₹0.83 Cr."*

## Step 8: The playbook
The probability sets the band. Affordability, collateral and exposure then pick the action. Example: a high-risk loan
with collateral cover goes to enforcement, while a small unsecured one goes to an agency, because intensive effort
costs more than it would recover.

## Step 9: Test evaluation and fairness
The test set is used **once**. `Gender` is excluded from the inputs, but flag rates by gender are still reported,
since excluding a column doesn't guarantee equal outcomes.

## Step 10: Serving
```bash
uvicorn api.main:app --reload       # try it at /docs
streamlit run app/dashboard.py
pytest
```
Everything scores through `LoanRecoveryPredictor`, which gives one code path for training, CLI, API and UI.
Pydantic rejects bad input with a 422 before it reaches the model.

## Step 11: Ship it
Push to GitHub. CI runs lint, tests, a fast training run, and a Docker build with a health check.
Optionally deploy the dashboard on Streamlit Community Cloud.

## Things to try (good for interviews)
1. Add the collection columns (`Collection_Attempts`, `Legal_Action_Taken`, `Collection_Method`) to the inputs. In a quick
   logistic-regression check, test ROC-AUC rose from about 0.816 to 0.825. Then explain why that "better" model is useless in production.
2. Change `escalation_cost` in `config.yaml`, retrain, and explain how the threshold moves.
3. Add `n_clusters` as a hyperparameter and see whether segment distances help the tree models.
4. Replace the random split with an out-of-time split, which needs a date column.
