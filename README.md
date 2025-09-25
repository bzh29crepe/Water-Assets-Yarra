# Water Asset Remaining Useful Life & Failure Prediction

An end-to-end, two-stage machine-learning pipeline for water infrastructure:

1. **Remaining Useful Life (RUL)** regression, then
2. **Failure Probability** classification that consumes the predicted RUL.

Results are surfaced in interactive **Streamlit dashboards** to support asset risk, maintenance prioritisation, and capital planning.

> **Data source**
> All datasets in this repository are **synthetic** and **programmatically generated** for demonstration only. They do not represent real water assets or any operational data.

---

## Repository structure

```
.
├─ data/
│  ├─ data_generation.py            # creates synthetic training & inference CSVs
│  ├─ yarra_assets.csv              # TRAIN set (with Remaining_Years, Failure)
│  └─ yarra_assets_unknown.csv      # INFERENCE set (no Remaining_Years, Failure)
│
├─ models/
│  ├─ rul_model.joblib              # trained RUL regressor
│  ├─ failure_model.joblib          # trained Failure classifier
│  ├─ rul_metrics.json              # eval metrics for RUL model
│  └─ failure_metrics.json          # eval metrics for Failure model
│
├─ streamlit_app/
│  ├─ Menu.py                       # Streamlit entry point / multipage menu
│  └─ pages/
│     ├─ 1_Remaining_Useful_Life.py # RUL dashboard
│     └─ 2_Failure_Probability.py   # Failure dashboard (uses predicted RUL)
│
├─ train_rul_model.py               # training script for RUL
├─ train_failure_model.py           # training script for Failure
├─ requirements.txt
├─ runtime.txt
├─ .gitignore
└─ README.md
```

---

## Datasets

### `data/yarra_assets.csv` (training)

Asset records including:

* **Core descriptors:** `asset_id`, `asset_name`, `asset_type`, `material`, `diameter`, `installation_year`, `status`, `network_type`, `length`, `depth`, `zone`, `pressure_rating`, `location`, `geometry`.
* **Targets:**

  * `Remaining_Years` (regression target for the RUL model).
  * `Failure` (binary classification target for the Failure model).

### `data/yarra_assets_unknown.csv` (inference)

Same core descriptors as above **but without** `Remaining_Years` and `Failure`.
Used by the app to generate predictions.

> The synthetic generator (`data_generation.py`) samples plausible distributions (age, materials, pressure, diameter, zone) and injects noise and heuristic relationships so the models learn realistic signals.

---

## Prediction pipeline

```
yarra_assets_unknown.csv
      │
      ├─ compute 'age' = current_year - installation_year
      ▼
RUL model (regression) → predict Remaining_Years
      ▼
append Remaining_Years to dataframe
      ▼
Failure model (classification) → predict Failure_Prob
      ▼
Dashboards: maps, KPIs, distributions, ranked tables
```

---

## Models

### 1) Remaining Useful Life (RUL) model

* **Type:** `RandomForestRegressor`
* **Goal:** Predict years of useful life remaining: `Remaining_Years`
* **Key features:**

  * Categorical: `asset_name`, `asset_type`, `material`, `status`, `network_type`, `zone`, `location`
  * Numeric: `diameter`, `installation_year`, `length`, `depth`, `pressure_rating`, **`age`**
* **Output:** Adds `Remaining_Years` to the dataset.
* **Why first?** Remaining life is a strong predictor of near-term failure and is fed into the failure model.

**RUL evaluation (from `models/rul_metrics.json`):**

* MAE: **1.7963**
* RMSE: **2.6472**
* R²: **0.9795**

These scores indicate high fidelity regression on the data.

---

### 2) Failure Probability model

* **Type:** `RandomForestClassifier`
* **Goal:** Probability an asset fails in the near term: `Failure_Prob ∈ [0,1]`
* **Key features:**

  * All RUL features **plus** the predicted **`Remaining_Years`**
* **Output:** Adds `Failure_Prob` to the dataset.

**Failure evaluation (from `models/failure_metrics.json`):**

* Accuracy: **0.915**
* ROC AUC: **0.9691**

**Classification report:**

|                Class | Precision | Recall |     F1 | Support |
| -------------------: | --------: | -----: | -----: | ------: |
|       0 (No failure) |    0.9080 | 0.9867 | 0.9457 |     150 |
|          1 (Failure) |    0.9459 | 0.7000 | 0.8046 |      50 |
| **Overall accuracy** | **0.915** |      — |      — |     200 |
|            Macro avg |    0.9270 | 0.8433 | 0.8751 |     200 |
|         Weighted avg |    0.9175 | 0.9150 | 0.9104 |     200 |

---

## Dashboards (Streamlit)

### Remaining Useful Life

* Filters: zone, asset type, RUL range
* KPIs: asset count, average RUL
* Map: assets coloured by `Remaining_Years`
* Top-5 shortest RUL assets
* Histogram of RUL

### Failure Probability

* Runs the **two-stage** inference (RUL → Failure)
* Filters: zone, asset type, failure-probability range
* KPIs: asset count, average failure probability, average RUL
* Map: assets coloured by `Failure_Prob`
* Top-5 highest risk assets
* Histogram of failure probability

> The pages load trained models from local `models/*.joblib` or, if configured in code, from the referenced Hugging Face URLs.

---

## Quick start

Install dependencies and run the app:

```bash
pip install -r requirements.txt
streamlit run streamlit_app/Menu.py
```

Open the URL printed by Streamlit (typically `http://localhost:8501`).

---

## (Optional) Retrain the models

RUL:

```bash
python train_rul_model.py \
  --csv data/yarra_assets.csv \
  --target Remaining_Years \
  --model_out models/rul_model.joblib \
  --metrics_out models/rul_metrics.json
```

Failure:

```bash
python train_failure_model.py \
  --csv data/yarra_assets.csv \
  --target Failure \
  --model_out models/failure_model.joblib \
  --metrics_out models/failure_metrics.json
```

> Note: If you see `InconsistentVersionWarning` when loading models, align your `scikit-learn` version between training and inference (pin the same version in `requirements.txt`) or retrain under the deployed environment.

---

## Model hosting (optional)

Models can be fetched automatically from Hugging Face (as configured in the Streamlit pages):

* RUL: `https://huggingface.co/louislb1302/failure_model.joblib/resolve/main/rul_model.joblib`
* Failure: `https://huggingface.co/louislb1302/failure_model.joblib/resolve/main/failure_model.joblib`

You may switch the pages to read directly from `./models/*.joblib` if preferred.

---

## Summary

* **Two-stage inference:** predict **RUL** then **Failure Probability**.
* **Dashboards** for exploration, ranking, and spatial context.
* **Solid metrics** on the training set, showing the approach and UX you’d bring to production with real data.
