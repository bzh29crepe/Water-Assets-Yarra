Here’s the **final README** including a description of the training dataset and the model evaluation metrics for both RUL and Failure models:

---

# Water Asset Remaining Useful Life and Failure Prediction

This project provides a **two-stage machine learning pipeline** to predict:

1. The **Remaining Useful Life (RUL)** of water infrastructure assets.
2. The **Probability of Failure** of each asset using both its physical characteristics and the predicted RUL.

The predictions are displayed in an interactive **Streamlit dashboard**, enabling asset managers and decision-makers to:

* Visualize the health of water infrastructure across different zones.
* Filter and explore assets by type, zone, and risk level.
* Prioritize inspections, maintenance, and replacement efforts efficiently.

---

## Project Overview

The project uses **two datasets**:

### 1. **Training Dataset (`yarra_assets.csv`)**

This dataset contains **historical water asset information** and serves as the foundation for model training.
It includes the following key columns:

| Column Name         | Description                                                                |
| ------------------- | -------------------------------------------------------------------------- |
| `asset_id`          | Unique identifier for each asset                                           |
| `asset_name`        | Human-readable name of the asset                                           |
| `asset_type`        | Type of asset (e.g., Pipe, Valve, Sewer)                                   |
| `material`          | Material used (e.g., Steel, Concrete, Cast Iron)                           |
| `diameter`          | Diameter of the asset (mm)                                                 |
| `installation_year` | Year the asset was installed                                               |
| `status`            | Current status (e.g., Active, Decommissioned)                              |
| `network_type`      | Network category (e.g., Water, Recycled Water)                             |
| `length`            | Length of the asset (meters)                                               |
| `depth`             | Depth at which the asset is located (meters)                               |
| `zone`              | Zone or region within the city                                             |
| `pressure_rating`   | Maximum operating pressure rating                                          |
| `location`          | Area description (e.g., Suburb A, Melbourne CBD)                           |
| `geometry`          | Geographic coordinates (latitude, longitude)                               |
| `Remaining_Years`   | Remaining useful life of the asset (target variable for RUL model)         |
| `Failure`           | Binary indicator of historical failure (target variable for Failure model) |

---

### 2. **Prediction Dataset (`yarra_assets_unknown.csv`)**

This dataset contains **current asset information** but does **not include**:

* `Remaining_Years`
* `Failure`

It is used as **input for the prediction pipeline** to generate these missing values.

---

## Prediction Pipeline

The prediction pipeline processes the raw dataset through **two sequential models**:

```
Raw Dataset (yarra_assets_unknown.csv)
      │
      ▼
Compute 'age'
      │
      ▼
RUL Model → Predict Remaining_Years
      │
      ▼
Enriched Dataset (includes Remaining_Years)
      │
      ▼
Failure Model → Predict Failure_Prob
      │
      ▼
Streamlit Dashboards (visualizations & filtering)
```

---

## Step-by-Step Process

### **Step 1: Compute Asset Age**

Before making any predictions, the system calculates the `age` of each asset:

```
age = current_year - installation_year
```

This is a critical feature for both RUL and Failure models.

---

### **Step 2: Remaining Useful Life (RUL) Model**

**Goal**: Estimate the number of years remaining before an asset reaches the end of its useful life.

**Model Type**: Regression (`RandomForestRegressor`)

**Features Used**:

* `asset_type`, `material`, `diameter`, `length`, `depth`
* `installation_year`, `age`
* `zone`, `pressure_rating`, `network_type`
* `status`, `location`

**Output**:

* Adds a new column `Remaining_Years` to the dataset.

#### RUL Model Performance:

| Metric                            | Value      |
| --------------------------------- | ---------- |
| **MAE** (Mean Absolute Error)     | **1.796**  |
| **RMSE** (Root Mean Square Error) | **2.647**  |
| **R² Score**                      | **0.9795** |

The RUL model achieves **very high accuracy**, with an R² score close to 1.0.

---

### **Step 3: Enriched Dataset**

Once RUL predictions are generated, the dataset now includes:

* Original columns from `yarra_assets_unknown.csv`.
* A new column: `Remaining_Years`.

This enriched dataset becomes the **input to the Failure model**.

---

### **Step 4: Failure Probability Model**

**Goal**: Predict the probability of failure for each asset in the near future.

**Model Type**: Classification (`RandomForestClassifier`)

**Features Used**:

* All features used in the RUL model.
* **`Remaining_Years`** predicted by the RUL model.

**Output**:

* Adds a new column `Failure_Prob` to the dataset.

#### Failure Model Performance:

| Metric       | Value     |
| ------------ | --------- |
| **Accuracy** | **0.915** |
| **ROC AUC**  | **0.969** |

**Classification Report**:

| Class                | Precision | Recall | F1-Score | Support |
| -------------------- | --------- | ------ | -------- | ------- |
| **0 (No Failure)**   | 0.908     | 0.987  | 0.946    | 150     |
| **1 (Failure)**      | 0.946     | 0.700  | 0.805    | 50      |
| **Overall Accuracy** | **0.915** | -      | -        | 200     |
| **Macro Avg**        | 0.927     | 0.843  | 0.875    | 200     |
| **Weighted Avg**     | 0.917     | 0.915  | 0.910    | 200     |

The failure model demonstrates **strong performance**, with high precision and recall for both classes and an excellent ROC AUC score of **0.969**.

---

## Streamlit Dashboards

Two interactive dashboards are provided:

### **1. Remaining Useful Life Dashboard**

**File**: `1_Remaining_Useful_Life.py`

**Purpose**:

* Visualize predicted RUL across the network.
* Filter by zone, asset type, and RUL range.
* Identify the top 5 assets closest to end of life.

**Visual Components**:

* KPIs: Total assets and average remaining life.
* Map with assets color-coded by RUL.
* Histogram showing distribution of RUL.
* Full table view of assets.

---

### **2. Failure Probability Dashboard**

**File**: `2_Failure_Probability.py`

**Purpose**:

* Predict and visualize failure probabilities.
* Combine asset characteristics with predicted RUL to assess risk.
* Identify high-risk assets for preventive maintenance.

**Visual Components**:

* KPIs: Total assets and average failure probability.
* Map with assets color-coded by failure probability.
* Histogram showing failure probability distribution.
* Detailed table sorted by risk.

---

## Data Flow

| Step               | Input                          | Model                              | Output                      |
| ------------------ | ------------------------------ | ---------------------------------- | --------------------------- |
| 1. Compute Age     | `yarra_assets_unknown.csv`     | -                                  | Adds `age` column           |
| 2. Predict RUL     | Enriched dataset with `age`    | **RUL Model** (Regression)         | Adds `Remaining_Years`      |
| 3. Predict Failure | Dataset with `Remaining_Years` | **Failure Model** (Classification) | Adds `Failure_Prob`         |
| 4. Visualization   | Final dataset                  | Streamlit Dashboards               | Interactive visual insights |

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/water-assets-yarra.git
   cd water-assets-yarra
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run streamlit_app/Menu.py
   ```

---

## Model Hosting

The trained models are hosted on Hugging Face and are automatically downloaded when running the dashboards.

| Model         | URL                                                                                         |
| ------------- | ------------------------------------------------------------------------------------------- |
| RUL Model     | `https://huggingface.co/louislb1302/failure_model.joblib/resolve/main/rul_model.joblib`     |
| Failure Model | `https://huggingface.co/louislb1302/failure_model.joblib/resolve/main/failure_model.joblib` |

---

## Summary

This project delivers a **two-step predictive maintenance solution** for water infrastructure:

1. **RUL Prediction** – Estimate how long each asset will last.
2. **Failure Prediction** – Assess failure probability based on RUL and other features.
3. **Interactive Dashboards** – Explore results visually and prioritize maintenance actions.

The system helps **optimize maintenance strategies**, reduce costs, and prevent unexpected failures by identifying **high-risk assets before issues occur**.
