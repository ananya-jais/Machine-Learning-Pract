# ============================================================
# PRACTICAL 1
# CROP PRODUCTION PREDICTION USING MACHINE LEARNING
# Linear Regression vs Polynomial Regression vs
# Multivariate Regression
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("crop_production.csv")

print("Columns in Dataset:")
print(df.columns.tolist())

print("\nDataset Size:")
print(df.shape)

print("\nFirst Year:", df["Year"].min())
print("Last Year :", df["Year"].max())


# ============================================================
# 2. SELECT PRODUCTION DATA
# ============================================================

# Your FAOSTAT file contains only Production
# in the Element column.

df = df[df["Element"] == "Production"]

# Keep only Year and Value
df = df[["Year", "Value"]]

# Rename Value to Production
df = df.rename(columns={
    "Value": "Production"
})

# Convert to numeric
df["Year"] = pd.to_numeric(df["Year"])
df["Production"] = pd.to_numeric(df["Production"])

# Sort by year
df = df.sort_values("Year")

# Remove missing values
df = df.dropna()

print("\n============================================")
print("PROCESSED DATASET")
print("============================================")

print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset Size:")
print(df.shape)


# ============================================================
# 3. CREATE LAG FEATURES
# ============================================================

# These will be used for Multivariate Regression.

df["Previous_Year_Production"] = df["Production"].shift(1)

df["Production_2_Years_Ago"] = df["Production"].shift(2)

# Remove first two rows because lag values are unavailable
df = df.dropna()

print("\n============================================")
print("DATA WITH FEATURES")
print("============================================")

print(df.head())


# ============================================================
# 4. TRAINING AND TESTING DATA
# ============================================================

train = df[df["Year"] <= 2011]

test = df[df["Year"] >= 2012]

print("\n============================================")
print("TRAINING AND TESTING DATA")
print("============================================")

print(
    "Training Period:",
    train["Year"].min(),
    "-",
    train["Year"].max()
)

print(
    "Testing Period :",
    test["Year"].min(),
    "-",
    test["Year"].max()
)

print(
    "Training Observations:",
    len(train)
)

print(
    "Testing Observations :",
    len(test)
)


# ============================================================
# 5. TARGET VARIABLE
# ============================================================

y_train = train["Production"]

y_test = test["Production"]


# ============================================================
# MODEL 1: LINEAR REGRESSION
# ============================================================

X_train_linear = train[["Year"]]

X_test_linear = test[["Year"]]


linear_model = LinearRegression()

linear_model.fit(
    X_train_linear,
    y_train
)

linear_pred = linear_model.predict(
    X_test_linear
)


# ============================================================
# MODEL 2: POLYNOMIAL REGRESSION
# ============================================================

X_train_poly = train[["Year"]]

X_test_poly = test[["Year"]]


poly_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

poly_model.fit(
    X_train_poly,
    y_train
)

poly_pred = poly_model.predict(
    X_test_poly
)


# ============================================================
# MODEL 3: MULTIVARIATE LINEAR REGRESSION
# ============================================================

# We use THREE independent variables:
#
# 1. Year
# 2. Previous year's production
# 3. Production from two years ago

X_train_multi = train[
    [
        "Year",
        "Previous_Year_Production",
        "Production_2_Years_Ago"
    ]
]

X_test_multi = test[
    [
        "Year",
        "Previous_Year_Production",
        "Production_2_Years_Ago"
    ]
]


multi_model = LinearRegression()

multi_model.fit(
    X_train_multi,
    y_train
)

multi_pred = multi_model.predict(
    X_test_multi
)


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, rmse, r2


# ============================================================
# 7. CALCULATE METRICS
# ============================================================

linear_mae, linear_rmse, linear_r2 = evaluate_model(
    y_test,
    linear_pred
)

poly_mae, poly_rmse, poly_r2 = evaluate_model(
    y_test,
    poly_pred
)

multi_mae, multi_rmse, multi_r2 = evaluate_model(
    y_test,
    multi_pred
)


# ============================================================
# 8. MODEL COMPARISON TABLE
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Polynomial Regression",
        "Multivariate Regression"
    ],

    "MAE": [
        linear_mae,
        poly_mae,
        multi_mae
    ],

    "RMSE": [
        linear_rmse,
        poly_rmse,
        multi_rmse
    ],

    "R2": [
        linear_r2,
        poly_r2,
        multi_r2
    ]
})


print("\n============================================")
print("MODEL PERFORMANCE")
print("============================================")

print(
    results.to_string(index=False)
)


# ============================================================
# 9. LINEAR REGRESSION WEIGHT AND BIAS
# ============================================================

print("\n============================================")
print("LINEAR REGRESSION PARAMETERS")
print("============================================")

print(
    "Weight:",
    linear_model.coef_[0]
)

print(
    "Bias:",
    linear_model.intercept_
)

print("\nEquation:")

print(
    f"Production = "
    f"{linear_model.coef_[0]:.4f} × Year "
    f"+ {linear_model.intercept_:.4f}"
)


# ============================================================
# 10. POLYNOMIAL REGRESSION PARAMETERS
# ============================================================

poly_linear = poly_model.named_steps[
    "linearregression"
]

print("\n============================================")
print("POLYNOMIAL REGRESSION")
print("============================================")

print(
    "Polynomial Degree: 2"
)

print(
    "Coefficients:",
    poly_linear.coef_
)

print(
    "Bias:",
    poly_linear.intercept_
)


# ============================================================
# 11. MULTIVARIATE WEIGHTS AND BIAS
# ============================================================

print("\n============================================")
print("MULTIVARIATE REGRESSION PARAMETERS")
print("============================================")

features = [
    "Year",
    "Previous Year Production",
    "Production 2 Years Ago"
]

for feature, weight in zip(
    features,
    multi_model.coef_
):

    print(
        feature,
        "=",
        weight
    )

print(
    "Bias:",
    multi_model.intercept_
)


# ============================================================
# 12. FUTURE PREDICTION: 2025-2029
# ============================================================

future_years = np.array([
    [2025],
    [2026],
    [2027],
    [2028],
    [2029]
])


# ============================================================
# LINEAR REGRESSION - FUTURE PREDICTION
# ============================================================

future_linear = linear_model.predict(
    future_years
)


# ============================================================
# POLYNOMIAL REGRESSION - FUTURE PREDICTION
# ============================================================

future_poly = poly_model.predict(
    future_years
)


# ============================================================
# MULTIVARIATE REGRESSION - FUTURE PREDICTION
# ============================================================

# For multivariate regression we need:
# Year
# Previous Year Production
# Production 2 Years Ago

# Start with the last two known production values.
previous_2 = df.iloc[-2]["Production"]
previous_1 = df.iloc[-1]["Production"]

multi_future_predictions = []


for year in range(2025, 2030):

    future_input = pd.DataFrame({

        "Year": [year],

        "Previous_Year_Production": [
            previous_1
        ],

        "Production_2_Years_Ago": [
            previous_2
        ]

    })

    prediction = multi_model.predict(
        future_input
    )[0]

    multi_future_predictions.append(
        prediction
    )

    # Move predictions forward
    previous_2 = previous_1
    previous_1 = prediction


# ============================================================
# 13. CREATE FUTURE PREDICTION TABLE
# ============================================================

future_results = pd.DataFrame({

    "Year": range(2025, 2030),

    "Linear Prediction": future_linear,

    "Polynomial Prediction": future_poly,

    "Multivariate Prediction":
        multi_future_predictions

})


print("\n============================================")
print("FUTURE PRODUCTION PREDICTIONS")
print("============================================")

print(
    future_results.to_string(index=False)
)

# ============================================================
# FINAL PROFESSIONAL GRAPH
# ============================================================

plt.figure(figsize=(14, 8))

# Convert production into million tonnes for readability
actual_million = df["Production"] / 1_000_000
linear_test_million = linear_pred / 1_000_000
poly_test_million = poly_pred / 1_000_000
multi_test_million = multi_pred / 1_000_000

linear_future_million = (
    future_results["Linear Prediction"] / 1_000_000
)

poly_future_million = (
    future_results["Polynomial Prediction"] / 1_000_000
)

multi_future_million = (
    future_results["Multivariate Prediction"] / 1_000_000
)


# ============================================================
# ACTUAL HISTORICAL PRODUCTION
# ============================================================

plt.plot(
    df["Year"],
    actual_million,
    marker="o",
    markersize=4,
    linewidth=2,
    label="Actual Production"
)


# ============================================================
# TESTING PERIOD PREDICTIONS
# ============================================================

plt.plot(
    test["Year"],
    linear_test_million,
    linestyle="--",
    linewidth=1.8,
    label="Linear Regression"
)

plt.plot(
    test["Year"],
    poly_test_million,
    linestyle="--",
    linewidth=1.8,
    label="Polynomial Regression"
)

plt.plot(
    test["Year"],
    multi_test_million,
    linestyle="--",
    linewidth=1.8,
    label="Multivariate Regression"
)


# ============================================================
# FUTURE PREDICTIONS
# ============================================================

plt.plot(
    future_results["Year"],
    linear_future_million,
    linestyle=":",
    linewidth=3,
    marker="o",
    markersize=6
)

plt.plot(
    future_results["Year"],
    poly_future_million,
    linestyle=":",
    linewidth=3,
    marker="o",
    markersize=6
)

plt.plot(
    future_results["Year"],
    multi_future_million,
    linestyle=":",
    linewidth=3,
    marker="o",
    markersize=6
)


# ============================================================
# SHOW FUTURE STARTING POINT
# ============================================================

plt.axvline(
    x=2024,
    linestyle="--",
    linewidth=1.5
)

plt.text(
    2024.3,
    max(actual_million) * 0.55,
    "Future Prediction →",
    rotation=90,
    verticalalignment="center"
)


# ============================================================
# GRAPH LABELS
# ============================================================

plt.xlabel(
    "Year",
    fontsize=12
)

plt.ylabel(
    "Production (Million Tonnes)",
    fontsize=12
)

plt.title(
    "Crop Production: Actual vs Regression Predictions",
    fontsize=16
)


# ============================================================
# X-AXIS
# ============================================================

plt.xticks(
    range(1965, 2030, 5),
    rotation=45
)


# ============================================================
# GRID
# ============================================================

plt.grid(
    True,
    linestyle="--",
    alpha=0.4
)


# ============================================================
# LEGEND
# ============================================================

plt.legend(
    loc="upper left",
    fontsize=10
)


plt.tight_layout()

plt.show()