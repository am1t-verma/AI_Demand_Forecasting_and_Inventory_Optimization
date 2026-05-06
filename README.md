# 🍽️ Food Demand Forecasting

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter" />
  <img src="https://img.shields.io/badge/Model-XGBoost-green?style=flat-square" />
  <img src="https://img.shields.io/badge/App-Streamlit-red?style=flat-square&logo=streamlit" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square" />
</p>

---

## 📌 Project Overview

This project predicts **weekly food demand** (`num_orders`) for a food delivery service across multiple fulfillment centers and meal types. The goal is to understand demand patterns, engineer useful features, compare models, and expose predictions through a **Streamlit app** for AI-based forecasting and scenario analysis.

The workflow covers exploratory data analysis → feature engineering → feature selection (RFE) → baseline modeling (Linear Regression) → XGBoost tuning → a final cleaned notebook → and a Streamlit forecasting interface.

---

## ❓ Problem Statement

Food delivery businesses face a core operational challenge:

- **Too little stock** → stockouts, lost sales, unhappy customers
- **Too much stock** → overproduction, food waste, increased cost

This project builds an ML pipeline to accurately predict weekly order volumes per `(center_id, meal_id)` combination — enabling smarter procurement and kitchen planning decisions.

---

## 📊 Dataset Description

### Raw Files

| File | Description |
|---|---|
| `train.csv` | Historical weekly demand with pricing and promotion signals |
| `test.csv` | Future weeks where `num_orders` must be predicted |
| `fulfilment_center_info.csv` | Center-level metadata (city, region, type, area) |
| `meal_info.csv` | Meal-level metadata (category, cuisine) |
| `sample_submission.csv` | Required output format (`id`, `num_orders`) |

### Key Columns — `train.csv`

| Column | Description |
|---|---|
| `week` | Week number — the time axis |
| `center_id` | Fulfillment center ID |
| `meal_id` | Meal/item ID |
| `checkout_price` | Final price charged to customer |
| `base_price` | Original price before discount |
| `emailer_for_promotion` | 1 if email promotion was active, else 0 |
| `homepage_featured` | 1 if meal was featured on homepage, else 0 |
| `num_orders` | ✅ Target variable — weekly orders |

### Dataset Flow

Raw files were merged to create the main working dataset:

```
train.csv + fulfilment_center_info.csv + meal_info.csv
              ↓
    data/processed/train_merged.csv   ← used across all notebooks and the app
```

`test_merged.csv` is available but reserved for final submission generation in future work.

---

## 🌿 Branch Details

The project was developed across multiple branches, each focused on a specific stage:

### `main`
Initial work — EDA, data understanding, and feature engineering. Raw data was loaded, merged, and explored. Key patterns around demand spikes, price sensitivity, and promotion impact were identified here.

### `RFE`
Recursive Feature Elimination was applied to identify the most impactful input features and drop low-value columns. This helped reduce noise and focus the model on meaningful signals.

### `LR`
Linear Regression was implemented as the **baseline model**. Established the initial RMSE benchmark to compare against more complex models.

### `xgboost`
XGBoost was implemented with hyperparameter tuning. Best parameters were identified and saved. This branch produced the final trained model (`xgb_food_demand_center13.pkl`) used in the Streamlit app.

### Final `v2` Notebook
A second notebook (`food_demand_forecasting_v2.ipynb`) was created to consolidate the full workflow — clean, sequential, and well-documented — representing the final project state.

---

## 📁 Project Structure

```
PS_3/
├── app/
│   ├── forecasting.py          # Feature prep and future demand forecasting logic
│   ├── streamlit_app.py        # Streamlit UI — inputs, charts, downloads
│   ├── requirements.txt        # App dependencies
│   └── README.md               # App-specific notes
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── fulfilment_center_info.csv
│   │   ├── meal_info.csv
│   │   └── sample_submission.csv
│   └── processed/
│       ├── train_merged.csv          # Main working dataset
│       ├── test_merged.csv           # For future prediction use
│       └── forecast_future_weeks.csv # Output from forecasting pipeline
│
├── models/
│   ├── encoder.pkl                   # Saved LabelEncoder for categorical columns
│   └── xgb_food_demand_center13.pkl  # Trained XGBoost model
│
├── notebooks/
│   ├── eda_food_demand_forecasting_v1.ipynb   # EDA and feature engineering
│   └── food_demand_forecasting_v2.ipynb       # Final cleaned modeling pipeline
│
└── results/
    ├── correlation_heatmap.png
    ├── eda_overview.png
    ├── feature_importance.png
    ├── forecast_top_meals.png
    └── residual_analysis.png
```

---

## 🔄 Code Flow

| File | Role |
|---|---|
| `eda_food_demand_forecasting_v1.ipynb` | EDA, data exploration, feature engineering experiments |
| `food_demand_forecasting_v2.ipynb` | Final cleaned pipeline — encoding, training, evaluation, forecasting |
| `app/forecasting.py` | Loads model + encoder + `train_merged.csv`, builds features, generates future week predictions |
| `app/streamlit_app.py` | UI layer — manual inputs, forecast charts, feature importance view, CSV download |
| `models/xgb_food_demand_center13.pkl` | Trained XGBoost model served by the app |
| `models/encoder.pkl` | Categorical encoder for consistent train/test transformation |

---

## ▶️ Run the Streamlit App

**Install dependencies:**
```bash
pip install -r app/requirements.txt
```

**Run the app:**
```bash
cd app
streamlit run streamlit_app.py
```

The app loads:
- `../models/xgb_food_demand_center13.pkl`
- `../models/encoder.pkl`
- `../data/processed/train_merged.csv`

**App features:**
- Manual scenario input (change center, meal, price, promotion flags)
- AI-generated weekly demand forecast
- Feature importance chart
- Residual analysis view
- Download forecast as CSV

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `Python 3.8+` | Core language |
| `Pandas / NumPy` | Data manipulation |
| `Matplotlib / Seaborn` | EDA visualizations |
| `Scikit-learn` | RFE, Linear Regression, encoding |
| `XGBoost` | Final forecasting model |
| `Streamlit` | Interactive forecasting app |
| `Pickle` | Model and encoder serialization |
| `Jupyter Notebook` | Development environment |

---

## 🔮 Future Work

- Use `test.csv` and `test_merged.csv` for final prediction and submission output
- Train on the full dataset for a stronger generalized model
- Add a deep learning / neural network model and compare against Linear Regression and XGBoost
- Add multi-center forecasting support to the Streamlit app
- Introduce lag and rolling features for further accuracy improvement

---

## 👤 Author

**Amit Verma**
Post Graduate — Data Science & Generative AI | IIT Indore × Intellipaat

- 🔗 GitHub: [@am1t-verma](https://github.com/am1t-verma)
- 💼 LinkedIn: [linkedin.com/in/am1t-verma](https://www.linkedin.com/in/am1t-verma)
- 📂 ML Portfolio: [Mechinge-Learning](https://github.com/am1t-verma/Mechinge-Learning)
