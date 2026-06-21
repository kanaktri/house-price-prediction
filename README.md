# 🏠 House Price Prediction

A machine learning project that predicts house prices from structural and
locational features (area, bedrooms, bathrooms, amenities, etc.), with an
interactive Streamlit app for live predictions and model comparison.

## Problem Statement

Given a set of property characteristics, predict the sale price of a house.
This is a supervised regression problem evaluated on three models —
Linear Regression, Random Forest, and XGBoost — to compare a simple
baseline against ensemble tree methods.

## Dataset

`Dataset/Housing.csv` — 545 records, 13 columns:

| Column | Description |
|---|---|
| `price` | Target variable |
| `area` | Plot/floor area (sqft) |
| `bedrooms`, `bathrooms`, `stories`, `parking` | Numeric counts |
| `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` | Binary (yes/no) amenities |
| `furnishingstatus` | Categorical: furnished / semi-furnished / unfurnished |

No missing values. Source: public Kaggle "Housing Prices Dataset."
*Currency unit follows the source dataset's convention and has not been
independently verified — treat absolute price values as illustrative.*

## Methodology

1. **EDA** — price distribution, correlation heatmap, area-vs-price scatter
   (see `Documentation/`).
2. **Preprocessing** — `StandardScaler` on numeric features,
   `OneHotEncoder` on categorical features, wired together with
   `ColumnTransformer` inside an sklearn `Pipeline` (prevents train/test
   leakage during scaling/encoding).
3. **Baseline comparison** — Linear Regression, Random Forest, and XGBoost
   evaluated with **5-fold cross-validation** on the training set.
4. **Hyperparameter tuning** — `RandomizedSearchCV` over XGBoost's
   `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
   `colsample_bytree`, and `min_child_weight`.
5. **Target transform check** — tuned XGBoost is trained both on raw price
   and on `log1p(price)`; whichever generalizes better on the held-out test
   set (by R²) is kept as the final model.
6. **Final evaluation** — MAE, RMSE, R², and CV R² (mean ± std) reported for
   all three models in `Documentation/model_comparison.csv`.

Run the full pipeline:

```bash
cd Notebook
python train_model.py
```

This regenerates the EDA plots, all three model `.pkl` files, the
feature-importance chart, and the comparison report in `Documentation/`.

> Note: `eda_model.ipynb` contains the original exploratory notebook.
> `train_model.py` is the productionized version used to actually
> (re)generate the shipped models — prefer it for reproducing results.

## Results

See `Documentation/model_comparison.csv` / `.png` for the latest numbers
after running `train_model.py`. As a reference, the original (untuned,
no-CV) baseline run produced:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 682,345.67 | 912,456.78 | 0.682 |
| Random Forest | 478,912.45 | 652,789.34 | 0.821 |
| XGBoost | 412,567.89 | 589,234.56 | 0.874 |

The tuned pipeline in `train_model.py` is expected to improve on these,
particularly for XGBoost, and additionally reports cross-validated R² to
give a more reliable estimate of generalization than a single train/test
split (the dataset has only 545 rows, so a single split is noisy).

## App

An interactive Streamlit app lets you input property details, choose
between the three trained models, and see the predicted price along with
XGBoost's feature importance. It also includes:

- A light themed background with a subtle house-silhouette watermark
- One-click presets (Studio / Family Home / Luxury Villa)
- A live animated 2D house preview (SVG + CSS) that updates with your
  stories/pool/garden/AC/solar selections
- A decorative CSS-only 3D rotating house (no external 3D library needed)
- Optional lifestyle features (pool, garden, gated community, solar,
  transit proximity, locality tier) — these are **not** part of the
  trained model's feature set, so they're applied as a clearly labeled,
  transparent percentage adjustment on top of the model's base prediction
- Final price shown in **INR, USD, and AED** using static, illustrative
  FX rates (`FX_RATES` in `app.py`) — update these or wire in a live FX
  API if you need accurate real-time conversion

```bash
pip install -r requirements.txt
streamlit run Streamlit_App/app.py
```

## Project Structure

```
HousePricePrediction/
├── Dataset/
│   └── Housing.csv
├── Notebook/
│   ├── eda_model.ipynb       # original exploratory notebook
│   └── train_model.py        # CV + tuning + log-target training pipeline
├── Model/
│   ├── Linear_Regression.pkl
│   ├── Random_Forest.pkl
│   └── XGBoost.pkl
├── Documentation/
│   ├── price_distribution.png
│   ├── correlation_heatmap.png
│   ├── area_vs_price.png
│   ├── feature_importance.png
│   ├── model_comparison.csv / .xlsx / .png
├── Streamlit_App/
│   └── app.py
├── requirements.txt
└── README.md
```

## Limitations & Future Work

- Dataset is small (545 rows) — results should be treated as illustrative,
  not production-grade valuations.
- No geographic/location features beyond `prefarea` — real-world price
  prediction is highly location-driven.
- Currency/units are unverified from the source dataset.
- Could extend with: SHAP values for per-prediction explanations, a larger
  / more recent dataset, and a proper experiment-tracking setup (e.g.
  MLflow) for comparing tuning runs.
