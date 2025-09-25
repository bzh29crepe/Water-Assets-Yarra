<<<<<<< HEAD
<<<<<<< HEAD
# Water Asset Failure Prediction Dashboard

This project provides a **Streamlit-based dashboard** to monitor and predict failure probabilities for water infrastructure assets.
It demonstrates how machine learning can be applied to identify high-risk assets and support preventive maintenance and investment planning decisions.

> **Note:** The dataset used in this project is **simulated** and does **not** represent real-world water infrastructure data.
> It has been created purely for demonstration and educational purposes.

---

## Features

* **Machine Learning Model**

  * Predicts the probability of simulated asset failures.
  * Highlights the most important features driving asset risk.

* **Interactive Dashboard**

  * Clear, user-friendly interface with no technical knowledge required.
  * Color-coded risk levels: High, Medium, Low.
  * Automatic alerts when simulated high-risk assets are detected.

* **Data Exploration and Insights**

  * Overview of simulated asset risk distribution.
  * Filter assets by region, type, or risk category.
  * Drill-down to view individual asset details and characteristics.
  * Visual explanation of key simulated risk factors.

---

## Project Structure

```
water_asset_dashboard/
│
├── data/
│   └── infrastructure_data.csv      # Simulated dataset used for demonstration
│   └── data_generation.py           # simulate the dataset
│
├── models/
│   ├── rf_model.joblib               # Trained RandomForest model
│   ├── preprocessor.joblib           # Data preprocessing pipeline
│   └── metrics.json                   # Model performance metrics
│
├── streamlit_app/
│   ├── app.py                         # Main Streamlit dashboard
│   ├── helpers.py                      # Model loading and prediction functions
│   └── visualization.py                # Visualization and plotting utilities
│
└── train_sklearn.py                    # Training script for the model
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/water-asset-dashboard.git
cd water-asset-dashboard
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

If you want to retrain the model on the simulated data:

```bash
python train_sklearn.py --csv data/infrastructure_data.csv --target Failure --id_cols "Asset ID"
```

This will generate:

* `models/rf_model.joblib` – The trained RandomForest model.
* `models/preprocessor.joblib` – The preprocessing pipeline for consistent data transformation.
* `models/metrics.json` – Evaluation metrics and top feature importances.

---

## Running the Dashboard

To launch the dashboard:

```bash
cd streamlit_app
streamlit run app.py
```

Once running, open your browser and go to:

```
http://localhost:8501
```

---

## Using the Dashboard

### Tabs

1. **Overview**

   * Displays total simulated assets, number of high-risk assets, and percentage of high-risk assets.
   * Risk distribution and failure probability breakdown.
   * Automatic alerts when simulated high-risk assets are present.

2. **Risk Explorer**

   * Interactive table with filters for region, type, and risk level.
   * Color-coded rows for quick identification of high-risk assets.
   * Drill-down to view each simulated asset's attributes and risk factors.

3. **Insights**

   * Displays which features have the highest impact on the model predictions.
   * Provides recommendations for simulated preventive maintenance actions.

---

## Simulated Data Format

The provided dataset (`infrastructure_data.csv`) is **completely simulated**.
It is used solely to demonstrate how this tool could work with real-world data.

| Asset ID | Type  | Age (Years) | Material | Flow Rate (L/s) | Maintenance Count | Region | Failure |
| -------- | ----- | ----------- | -------- | --------------- | ----------------- | ------ | ------- |
| A-001    | Pipe  | 25          | Steel    | 500             | 2                 | East   | 1       |
| A-002    | Pump  | 10          | Plastic  | 200             | 1                 | North  | 0       |
| A-003    | Valve | 5           | Copper   | 100             | 0                 | South  | 0       |

**Columns:**

* `Asset ID`: Unique identifier for each simulated asset (not used in model training).
* `Failure`: Target variable (1 = failure, 0 = no failure, simulated).
* Remaining columns: Simulated features used for model prediction.

---

## Key Technologies

* **Python 3.9+**
* **Streamlit** – Interactive web application framework.
* **Scikit-learn** – Machine learning for model training and evaluation.
* **Pandas / NumPy** – Data manipulation and processing.
* **Plotly** – Interactive visualizations.

---

## Best Practices

* This project uses **simulated data**. In a production environment:

  * Replace the dataset with real historical data.
  * Validate the model thoroughly before use.
  * Regularly retrain and update the model as new data becomes available.
  * Use as a decision-support tool, not a fully automated decision-making system.

---

## Future Enhancements

* Integration with live sensor data or real-time monitoring systems.
* Advanced explainability with SHAP values or similar techniques.
* Secure, role-based access for different stakeholders.
* Deployment to cloud platforms like AWS, Azure, or GCP for production use.

---

## Disclaimer

This project and its dataset are entirely **simulated**.
It is intended solely for **demonstration and educational purposes**.
Any resemblance to real-world infrastructure, assets, or events is purely coincidental.

---

## License

This project is licensed under the MIT License.
=======
# Water Asset Failure Prediction Dashboard

This project provides a **Streamlit-based dashboard** to monitor and predict failure probabilities for water infrastructure assets.
It demonstrates how machine learning can be applied to identify high-risk assets and support preventive maintenance and investment planning decisions.

> **Note:** The dataset used in this project is **simulated** and does **not** represent real-world water infrastructure data.
> It has been created purely for demonstration and educational purposes.

---

## Features

* **Machine Learning Model**

  * Predicts the probability of simulated asset failures.
  * Highlights the most important features driving asset risk.

* **Interactive Dashboard**

  * Clear, user-friendly interface with no technical knowledge required.
  * Color-coded risk levels: High, Medium, Low.
  * Automatic alerts when simulated high-risk assets are detected.

* **Data Exploration and Insights**

  * Overview of simulated asset risk distribution.
  * Filter assets by region, type, or risk category.
  * Drill-down to view individual asset details and characteristics.
  * Visual explanation of key simulated risk factors.

---

## Project Structure

```
water_asset_dashboard/
│
├── data/
│   └── infrastructure_data.csv      # Simulated dataset used for demonstration
│   └── data_generation.py           # simulate the dataset
│
├── models/
│   ├── rf_model.joblib               # Trained RandomForest model
│   ├── preprocessor.joblib           # Data preprocessing pipeline
│   └── metrics.json                   # Model performance metrics
│
├── streamlit_app/
│   ├── app.py                         # Main Streamlit dashboard
│   ├── helpers.py                      # Model loading and prediction functions
│   └── visualization.py                # Visualization and plotting utilities
│
└── train_sklearn.py                    # Training script for the model
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/water-asset-dashboard.git
cd water-asset-dashboard
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

If you want to retrain the model on the simulated data:

```bash
python train_sklearn.py --csv data/infrastructure_data.csv --target Failure --id_cols "Asset ID"
```

This will generate:

* `models/rf_model.joblib` – The trained RandomForest model.
* `models/preprocessor.joblib` – The preprocessing pipeline for consistent data transformation.
* `models/metrics.json` – Evaluation metrics and top feature importances.

---

## Running the Dashboard

To launch the dashboard:

```bash
cd streamlit_app
streamlit run app.py
```

Once running, open your browser and go to:

```
http://localhost:8501
```

---

## Using the Dashboard

### Tabs

1. **Overview**

   * Displays total simulated assets, number of high-risk assets, and percentage of high-risk assets.
   * Risk distribution and failure probability breakdown.
   * Automatic alerts when simulated high-risk assets are present.

2. **Risk Explorer**

   * Interactive table with filters for region, type, and risk level.
   * Color-coded rows for quick identification of high-risk assets.
   * Drill-down to view each simulated asset's attributes and risk factors.

3. **Insights**

   * Displays which features have the highest impact on the model predictions.
   * Provides recommendations for simulated preventive maintenance actions.

---

## Simulated Data Format

The provided dataset (`infrastructure_data.csv`) is **completely simulated**.
It is used solely to demonstrate how this tool could work with real-world data.

| Asset ID | Type  | Age (Years) | Material | Flow Rate (L/s) | Maintenance Count | Region | Failure |
| -------- | ----- | ----------- | -------- | --------------- | ----------------- | ------ | ------- |
| A-001    | Pipe  | 25          | Steel    | 500             | 2                 | East   | 1       |
| A-002    | Pump  | 10          | Plastic  | 200             | 1                 | North  | 0       |
| A-003    | Valve | 5           | Copper   | 100             | 0                 | South  | 0       |

**Columns:**

* `Asset ID`: Unique identifier for each simulated asset (not used in model training).
* `Failure`: Target variable (1 = failure, 0 = no failure, simulated).
* Remaining columns: Simulated features used for model prediction.

---

## Key Technologies

* **Python 3.9+**
* **Streamlit** – Interactive web application framework.
* **Scikit-learn** – Machine learning for model training and evaluation.
* **Pandas / NumPy** – Data manipulation and processing.
* **Plotly** – Interactive visualizations.

---

## Best Practices

* This project uses **simulated data**. In a production environment:

  * Replace the dataset with real historical data.
  * Validate the model thoroughly before use.
  * Regularly retrain and update the model as new data becomes available.
  * Use as a decision-support tool, not a fully automated decision-making system.

---

## Future Enhancements

* Integration with live sensor data or real-time monitoring systems.
* Advanced explainability with SHAP values or similar techniques.
* Secure, role-based access for different stakeholders.
* Deployment to cloud platforms like AWS, Azure, or GCP for production use.

---

## Disclaimer

This project and its dataset are entirely **simulated**.
It is intended solely for **demonstration and educational purposes**.
Any resemblance to real-world infrastructure, assets, or events is purely coincidental.

---

## License

This project is licensed under the MIT License.
>>>>>>> e1b9a867411dbb77ab32dd9e3d5dd674e5971692
=======
# Water Asset Analytics Dashboard

This project demonstrates a **data science pipeline** for predicting and visualizing:

* **Remaining Useful Life (RUL)** of water infrastructure assets (Regression).
* **Failure Probability** of assets in the next 12 months (Classification).

The project includes:

1. **Synthetic Data Generation** – Creating realistic water asset datasets.
2. **Model Training** – Building predictive models using scikit-learn.
3. **Streamlit Dashboard** – Interactive visualization of results.

---

## Project Structure

```
water_asset_dashboard/
│
├── data/
│   └── yarra_assets.csv                # Synthetic dataset generated automatically
│
├── models/
│   ├── rul_model.joblib                 # Trained RUL model
│   ├── failure_model.joblib             # Trained Failure Probability model
│   └── metrics.json                     # Model evaluation metrics
│
├── streamlit_app/
│   ├── app.py                           # Streamlit main dashboard
│   └── pages/
│       ├── 1_RUL.py                      # RUL prediction page
│       └── 2_Failure_Probability.py      # Failure probability page
│
├── data_generation.py                   # Synthetic data generation script
├── train_rul_model.py                   # RUL model training script
├── train_failure_model.py               # Failure probability model training script
└── requirements.txt                     # Python dependencies
```

---

## Data Disclaimer

The dataset used in this project is **synthetic** and automatically generated for demonstration purposes only.
It **does not represent real Yarra Valley Water data** or any real-world utility dataset.

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/water-asset-dashboard.git
cd water-asset-dashboard
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Generate Synthetic Data

```bash
python data_generation.py
```

This will create `data/yarra_assets.csv` with synthetic assets, including:

* Asset attributes (type, material, diameter, etc.)
* Remaining Useful Life (RUL) values
* Failure probability labels

---

### 2. Train Models

**Train RUL Regression Model**

```bash
python train_rul_model.py --csv data/yarra_assets.csv
```

**Train Failure Probability Classification Model**

```bash
python train_failure_model.py --csv data/yarra_assets.csv
```

---

### 3. Launch the Streamlit Dashboard

```bash
cd streamlit_app
streamlit run app.py
```

The dashboard will open in your browser.

---

## Features

### **Remaining Useful Life (RUL) Page**

* Map visualization of assets by remaining life.
* KPI metrics for average remaining life.
* Top 5 assets with the lowest remaining life.
* Histogram of RUL distribution.
* Interactive slider to filter assets by remaining life range.

### **Failure Probability Page**

* Predicts probability of failure for each asset.
* Highlights high-risk assets on the map.
* Displays key metrics for preventive maintenance planning.

---

## Requirements

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
plotly==5.24.1
streamlit==1.37.1
joblib==1.4.2
```

Install with:

```bash
pip install -r requirements.txt
```
This project is for **educational and demonstration purposes only**.
It uses **synthetic data** and is not intended for real-world operational use.
>>>>>>> 22f9a3a6d5675a6a32fa46af27270238adde485c
