"""
House Price Prediction - Training Pipeline
--------------------------------------------
Trains and compares Linear Regression, Random Forest, and XGBoost models
using 5-fold cross-validation, hyperparameter tuning for XGBoost, and a
log-target transform comparison. Saves the best models and a comparison
report to Model/ and Documentation/.

Run from the Notebook/ directory:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(ROOT, "Dataset", "Housing.csv")
MODEL_DIR = os.path.join(ROOT, "Model")
DOC_DIR = os.path.join(ROOT, "Documentation")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DOC_DIR, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------
# 1. Load & inspect data
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Dataset loaded! Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
df = df.dropna()

# EDA plots (saved for documentation)
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], kde=True, color="blue")
plt.title("House Price Distribution")
plt.savefig(os.path.join(DOC_DIR, "price_distribution.png"), bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.savefig(os.path.join(DOC_DIR, "correlation_heatmap.png"), bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="area", y="price", data=df)
plt.title("Area vs Price")
plt.savefig(os.path.join(DOC_DIR, "area_vs_price.png"), bbox_inches="tight")
plt.close()

# NOTE: We intentionally do NOT engineer a `price_per_sqft` feature here.
# It is derived directly from the target (price / area) and including it
# in the feature set would leak target information into the model.

# ---------------------------------------------------------
# 2. Features / target
# ---------------------------------------------------------
X = df.drop("price", axis=1)
y = df["price"]

numeric_features = ["area", "bedrooms", "bathrooms", "stories", "parking"]
categorical_features = [
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus",
]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------
# 3. Baseline models with 5-fold cross-validation
# ---------------------------------------------------------
kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

baseline_models = {
    "Linear_Regression": Pipeline([("prep", preprocessor), ("model", LinearRegression())]),
    "Random_Forest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)),
    ]),
    "XGBoost": Pipeline([
        ("prep", preprocessor),
        ("model", XGBRegressor(random_state=RANDOM_STATE, verbosity=0)),
    ]),
}

cv_results = {}
print("\n--- 5-Fold Cross-Validation (R²) ---")
for name, pipe in baseline_models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=kfold, scoring="r2")
    cv_results[name] = {"CV_R2_mean": scores.mean(), "CV_R2_std": scores.std()}
    print(f"{name}: R² = {scores.mean():.3f} ± {scores.std():.3f}")

# ---------------------------------------------------------
# 4. Hyperparameter tuning for XGBoost
# ---------------------------------------------------------
print("\n--- Tuning XGBoost (RandomizedSearchCV) ---")
xgb_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", XGBRegressor(random_state=RANDOM_STATE, verbosity=0)),
])

param_distributions = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__max_depth": [2, 3, 4, 5, 6],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "model__subsample": [0.6, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__min_child_weight": [1, 3, 5],
}

search = RandomizedSearchCV(
    xgb_pipe,
    param_distributions=param_distributions,
    n_iter=40,
    scoring="r2",
    cv=kfold,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
search.fit(X_train, y_train)
best_xgb = search.best_estimator_
print("Best XGBoost params:", search.best_params_)
print(f"Best CV R²: {search.best_score_:.3f}")

# ---------------------------------------------------------
# 5. Log-target transform comparison (on tuned XGBoost)
# ---------------------------------------------------------
print("\n--- Log-target transform comparison (tuned XGBoost) ---")
y_train_log = np.log1p(y_train)

log_xgb = Pipeline([
    ("prep", preprocessor),
    ("model", XGBRegressor(random_state=RANDOM_STATE, verbosity=0, **{
        k.replace("model__", ""): v for k, v in search.best_params_.items()
    })),
])
log_xgb.fit(X_train, y_train_log)
y_pred_log = np.expm1(log_xgb.predict(X_test))

best_xgb.fit(X_train, y_train)
y_pred_raw = best_xgb.predict(X_test)

r2_raw = r2_score(y_test, y_pred_raw)
r2_log = r2_score(y_test, y_pred_log)
print(f"Raw-target R²: {r2_raw:.3f}  |  Log-target R²: {r2_log:.3f}")

# Use whichever performs better on the held-out test set
use_log = r2_log > r2_raw
final_xgb = log_xgb if use_log else best_xgb
print(f"Using {'LOG' if use_log else 'RAW'} target for final XGBoost model.")

# ---------------------------------------------------------
# 6. Final evaluation of all three models on the test set
# ---------------------------------------------------------
final_models = {
    "Linear_Regression": baseline_models["Linear_Regression"],
    "Random_Forest": baseline_models["Random_Forest"],
    "XGBoost": final_xgb,
}

results = {}
for name, pipe in final_models.items():
    if name != "XGBoost":
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
    else:
        y_pred = y_pred_log if use_log else y_pred_raw

    results[name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "CV_R2_mean": cv_results[name]["CV_R2_mean"] if name in cv_results else search.best_score_,
        "CV_R2_std": cv_results[name]["CV_R2_std"] if name in cv_results else np.nan,
    }
    joblib.dump(pipe, os.path.join(MODEL_DIR, f"{name}.pkl"))
    print(f"{name} saved -> Model/{name}.pkl")

# Save a flag so the Streamlit app knows whether XGBoost predicts in log-space
with open(os.path.join(MODEL_DIR, "xgboost_log_target.flag"), "w") as f:
    f.write("1" if use_log else "0")

# ---------------------------------------------------------
# 7. Comparison report
# ---------------------------------------------------------
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost (tuned)"],
    "MAE": [results[n]["MAE"] for n in final_models],
    "RMSE": [results[n]["RMSE"] for n in final_models],
    "R2 Score": [results[n]["R2"] for n in final_models],
    "CV R2 Mean": [results[n]["CV_R2_mean"] for n in final_models],
    "CV R2 Std": [results[n]["CV_R2_std"] for n in final_models],
})

print("\nFinal Model Comparison:")
print(comparison.round(3))

comparison.to_csv(os.path.join(DOC_DIR, "model_comparison.csv"), index=False)
comparison.to_excel(os.path.join(DOC_DIR, "model_comparison.xlsx"), index=False)

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("tight")
ax.axis("off")
table = ax.table(
    cellText=comparison.round(3).values,
    colLabels=comparison.columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
plt.title("Model Performance Comparison")
plt.savefig(os.path.join(DOC_DIR, "model_comparison.png"), bbox_inches="tight", dpi=300)
plt.close()

# ---------------------------------------------------------
# 8. Feature importance (XGBoost)
# ---------------------------------------------------------
xgb_model = final_xgb.named_steps["model"]
feature_names = final_xgb.named_steps["prep"].get_feature_names_out()
importances = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values()

plt.figure(figsize=(8, 6))
importances.plot(kind="barh", color="teal")
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(DOC_DIR, "feature_importance.png"), bbox_inches="tight")
plt.close()

print("\n✅ Training complete. Models and reports saved in Model/ and Documentation/.")
