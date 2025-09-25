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
│   └── data_generation.py              # Synthetic dataset generation
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

The dataset used in this project is **synthetic** and generated for demonstration purposes only.
It **does not represent real Yarra water assets data** or any real-world utility dataset.

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/water-assets-yarra.git
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

---

## License

This project is for **experiments and demonstration purposes only**.
It uses **synthetic data** and is not intended for real-world operational use.