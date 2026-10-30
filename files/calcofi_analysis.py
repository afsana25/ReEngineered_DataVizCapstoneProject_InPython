"""
CALCOFI Cast and Bottle Dataset — Python Analysis
Author: Afsana Mimi

Python translation of the supplied R analysis.
The workflow imports and cleans the CALCOFI data, applies quality-code
filtering, creates a station-grouped train/test split, performs correlation
analysis, fits Linear Regression and Ridge Regression models, evaluates
predictions, and creates exploratory visualizations and a station map.

Before running:
1. Update DATA_FILE and CODEBOOK_FILE below.
2. Install the required packages:
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels openpyxl folium geopandas
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score

import statsmodels.formula.api as smf
from scipy.stats import f_oneway

import folium
from folium.plugins import FastMarkerCluster

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# 1. PATH SETUP
# ---------------------------------------------------------------------

DATA_FILE = Path(
    r"C:/Users/afsan/OneDrive/Desktop/Thesis/MERGED_CalCOFI_DATA_CAST_BOTTLE.csv"
)
CODEBOOK_FILE = Path(
    r"C:/Users/afsan/OneDrive/Desktop/Thesis/CalCofi DataSets CodeBook.xlsx"
)
OUTPUT_DIR = Path(
    r"C:/Users/afsan/OneDrive/Desktop/Thesis/calcofi_dashboards"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESPONSE = "bottle_t_deg_c"

# ---------------------------------------------------------------------
# 2. IMPORT PACKAGES / READ DATA
# ---------------------------------------------------------------------

# Read every CSV field as text first, matching the original R workflow.
df_raw = pd.read_csv(
    DATA_FILE,
    dtype="string",
    encoding="latin1",
    low_memory=False
)

# Clean column names to lowercase snake_case.
def clean_column_name(name):
    import re
    name = str(name).strip()
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()

df_raw.columns = [clean_column_name(c) for c in df_raw.columns]

print("\nRaw data preview:")
print(df_raw.head())
print("\nRaw dimensions:", df_raw.shape)

# ---------------------------------------------------------------------
# 3. READ CODEBOOK AND IDENTIFY QUALITY-CODE FIELDS
# ---------------------------------------------------------------------

df_excel = pd.read_excel(CODEBOOK_FILE, sheet_name="Sheet1")
print("\nCodebook preview:")
print(df_excel.head())
print("\nCodebook columns:", df_excel.columns.tolist())

# Find the codebook column that contains the Cast/Bottle field names.
field_column = next(
    (c for c in df_excel.columns
     if "cast" in str(c).lower() and "bottle" in str(c).lower()),
    None
)
description_column = next(
    (c for c in df_excel.columns
     if str(c).strip().lower() == "description"),
    None
)

if field_column is None or description_column is None:
    raise KeyError(
        "Could not identify the codebook columns for field names and "
        "Description. Check df_excel.columns and update field_column/"
        "description_column."
    )

quality_rows = df_excel[
    df_excel[description_column]
    .astype("string")
    .str.contains("Quality Code", case=False, na=False)
]

quality_cols = (
    quality_rows[field_column]
    .dropna()
    .astype(str)
    .str.strip()
    .str.strip("[]")
    .map(clean_column_name)
    .tolist()
)

quality_cols = [c for c in quality_cols if c in df_raw.columns]
print("\nQuality-code columns found:", quality_cols)

# ---------------------------------------------------------------------
# 4. FILTER QUALITY CODES AND REMOVE QUALITY FIELDS
# ---------------------------------------------------------------------

# Keep rows where every available quality-code field is blank, 4, 6, or 9.
allowed_codes = {"4", "6", "9"}

if quality_cols:
    quality_values = df_raw[quality_cols].astype("string")
    keep_rows = quality_values.apply(
        lambda col: col.isna() | col.str.strip().eq("") |
                    col.str.strip().isin(allowed_codes)
    ).all(axis=1)
    df_filtered = df_raw.loc[keep_rows].copy()
else:
    print(
        "WARNING: No quality-code columns were matched. "
        "No quality-code row filtering was applied."
    )
    df_filtered = df_raw.copy()

print("\nRows before quality filtering:", len(df_raw))
print("Rows after quality filtering:", len(df_filtered))

df_final = df_filtered.drop(columns=quality_cols, errors="ignore")
df_final = df_final.drop(
    columns=["bottle_rec_ind", "bottle_t_prec"],
    errors="ignore"
)

print("\nCleaned data dimensions:", df_final.shape)
print(df_final.head())

# ---------------------------------------------------------------------
# 5. GROUPED TRAIN / TEST SPLIT
# ---------------------------------------------------------------------

# Keep all records from a station in either train or test, not both.
if "cast_sta_id" not in df_final.columns:
    raise KeyError("The required grouping column 'cast_sta_id' is missing.")

splitter = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
train_idx, test_idx = next(
    splitter.split(df_final, groups=df_final["cast_sta_id"])
)

train = df_final.iloc[train_idx].copy()
test = df_final.iloc[test_idx].copy()

print("\nTraining dimensions:", train.shape)
print("Testing dimensions:", test.shape)

# ---------------------------------------------------------------------
# 6. CONVERT DATA TO NUMERIC AND KEEP NUMERIC COLUMNS
# ---------------------------------------------------------------------

train_num = train.apply(pd.to_numeric, errors="coerce")
test_num = test.apply(pd.to_numeric, errors="coerce")

train_num = train_num.select_dtypes(include=[np.number])
test_num = test_num.select_dtypes(include=[np.number])

print("\nNumeric training dimensions:", train_num.shape)
print("Numeric testing dimensions:", test_num.shape)

if RESPONSE not in train_num.columns:
    raise KeyError(
        f"Response variable '{RESPONSE}' is not numeric or is missing."
    )

print("\nTemperature summary:")
print(train_num[RESPONSE].describe())

# ---------------------------------------------------------------------
# 7. REMOVE VERY SPARSE COLUMNS AND ROWS WITH MISSING VALUES
# ---------------------------------------------------------------------

missing_rate = train_num.isna().mean()
good_cols = missing_rate[missing_rate < 0.80].index.tolist()

train_filtered = train_num[good_cols].copy()
test_filtered = test_num.reindex(columns=good_cols).copy()

train_clean = train_filtered.dropna().copy()
test_clean = test_filtered.dropna().copy()

print("\nTraining rows after complete-case filtering:", len(train_clean))
print("Testing rows after complete-case filtering:", len(test_clean))

if train_clean.empty or test_clean.empty:
    raise ValueError(
        "The complete-case datasets are empty. Consider using imputation "
        "instead of dropping every row with missing values."
    )

# ---------------------------------------------------------------------
# 8. CORRELATION ANALYSIS
# ---------------------------------------------------------------------

numeric_vars = train_clean.select_dtypes(include=[np.number])
cor_matrix = numeric_vars.corr()

cor_matrix.to_csv(OUTPUT_DIR / "calcofi_full_correlation_matrix.csv")

if RESPONSE not in cor_matrix.columns:
    raise KeyError(f"{RESPONSE} is unavailable for correlation analysis.")

temp_cor = cor_matrix[RESPONSE].dropna()
cor_table = (
    temp_cor.rename("correlation_with_temperature")
    .to_frame()
    .assign(variable=lambda x: x.index)
    .reset_index(drop=True)
)
cor_table["correlation_with_temperature"] = (
    cor_table["correlation_with_temperature"].round(3)
)
cor_table["absolute_correlation"] = (
    cor_table["correlation_with_temperature"].abs()
)
cor_table = cor_table.sort_values(
    "absolute_correlation", ascending=False
).drop(columns="absolute_correlation")

cor_table.to_csv(
    OUTPUT_DIR / "calcofi_temperature_correlations.csv",
    index=False
)

print("\nCorrelations with ocean temperature:")
print(cor_table.to_string(index=False))

# ---------------------------------------------------------------------
# 9. SELECT PREDICTOR VARIABLES
# ---------------------------------------------------------------------

selected_vars = [
    "bottle_depthm",
    "bottle_salnty",
    "bottle_po4u_m",
    "bottle_no3u_m",
    "bottle_si_o3u_m",
    "bottle_o2sat",
]

missing_predictors = [
    col for col in selected_vars
    if col not in train_clean.columns or col not in test_clean.columns
]
if missing_predictors:
    raise KeyError(
        "These selected predictors are missing after cleaning: "
        + ", ".join(missing_predictors)
    )

train_selected = train_clean[[RESPONSE] + selected_vars].dropna().copy()
test_selected = test_clean[[RESPONSE] + selected_vars].dropna().copy()

# ---------------------------------------------------------------------
# 10. LINEAR REGRESSION
# ---------------------------------------------------------------------

X_train = train_selected[selected_vars]
y_train = train_selected[RESPONSE]
X_test = test_selected[selected_vars]
y_test = test_selected[RESPONSE]

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

pred_lm = linear_model.predict(X_test)

print("\nLinear Regression coefficients:")
print(pd.Series(linear_model.coef_, index=selected_vars))
print("Linear Regression intercept:", linear_model.intercept_)

# Detailed ordinary least-squares model summary.
ols_formula = RESPONSE + " ~ " + " + ".join(selected_vars)
ols_model = smf.ols(ols_formula, data=train_selected).fit()
print("\nOLS model summary:")
print(ols_model.summary())

# ---------------------------------------------------------------------
# 11. PREDICTOR CORRELATION MATRIX AND HEATMAP
# ---------------------------------------------------------------------

predictor_corr = train_selected[selected_vars].corr()
predictor_corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

plt.figure(figsize=(9, 7))
sns.heatmap(
    predictor_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)
plt.title("Correlation Heatmap of Ocean Predictors")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "img-5.jpeg", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# 12. RIDGE REGRESSION WITH CROSS-VALIDATION
# ---------------------------------------------------------------------

# Scaling is important for Ridge because the predictors have different units.
alphas = np.logspace(-4, 4, 100)
ridge_model = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=alphas, cv=5)
)
ridge_model.fit(X_train, y_train)

best_alpha = ridge_model.named_steps["ridgecv"].alpha_
pred_ridge = ridge_model.predict(X_test)

ridge_coefficients = pd.Series(
    ridge_model.named_steps["ridgecv"].coef_,
    index=selected_vars
)

print("\nBest Ridge alpha:", best_alpha)
print("Ridge coefficients (standardized predictors):")
print(ridge_coefficients)

# ---------------------------------------------------------------------
# 13. COMPARE LINEAR AND RIDGE REGRESSION
# ---------------------------------------------------------------------

results_comparison = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Linear_Predicted": pred_lm,
    "Ridge_Predicted": pred_ridge,
    "Linear_Error": np.abs(y_test.to_numpy() - pred_lm),
    "Ridge_Error": np.abs(y_test.to_numpy() - pred_ridge),
})

summary_metrics = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression"],
    "MAE": [
        mean_absolute_error(y_test, pred_lm),
        mean_absolute_error(y_test, pred_ridge)
    ],
    "R_squared_test": [
        r2_score(y_test, pred_lm),
        r2_score(y_test, pred_ridge)
    ]
})

print("\nTest-set model comparison:")
print(summary_metrics.to_string(index=False))

results_comparison.to_csv(
    OUTPUT_DIR / "calcofi_model_predictions.csv",
    index=False
)
summary_metrics.to_csv(
    OUTPUT_DIR / "calcofi_model_metrics.csv",
    index=False
)

# Training R-squared for Ridge, for reference.
ridge_train_predictions = ridge_model.predict(X_train)
ridge_train_r2 = r2_score(y_train, ridge_train_predictions)
print("\nRidge training R-squared:", round(ridge_train_r2, 4))

# ---------------------------------------------------------------------
# 14. ACTUAL VS. PREDICTED TEMPERATURE — LINEAR REGRESSION
# ---------------------------------------------------------------------

plot_lm_final = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": pred_lm
})

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=plot_lm_final,
    x="Actual",
    y="Predicted",
    alpha=0.30,
    s=18
)
plt.plot([5, 25], [5, 25], linestyle="--", color="red",
         label="Identity Line (1:1)")
plt.xlim(5, 25)
plt.ylim(5, 25)
plt.title("Linear Regression Analysis:\nActual vs. Predicted Temperature")
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "img-6.jpeg", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# 15. DEPTH VS. TEMPERATURE
# ---------------------------------------------------------------------

ocean_data_filtered = train.copy()
ocean_data_filtered["temp"] = pd.to_numeric(
    ocean_data_filtered[RESPONSE], errors="coerce"
)
ocean_data_filtered["depth"] = pd.to_numeric(
    ocean_data_filtered["bottle_depthm"], errors="coerce"
)
ocean_data_filtered = ocean_data_filtered.dropna(
    subset=["temp", "depth"]
)
ocean_data_filtered = ocean_data_filtered[
    ocean_data_filtered["temp"] < 35
]

plt.figure(figsize=(9, 6))
sns.regplot(
    data=ocean_data_filtered,
    x="depth",
    y="temp",
    scatter_kws={"alpha": 0.20, "s": 10},
    line_kws={"color": "red"}
)
plt.title("Linear Regression: How Temperature Changes with Depth")
plt.subtitle = None
plt.xlabel("Depth (m)")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "img-7.jpeg", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# 16. TEMPERATURE VS. SALINITY BY DEPTH LAYER
# ---------------------------------------------------------------------

ocean_data_faceted = train.copy()
ocean_data_faceted["temp"] = pd.to_numeric(
    ocean_data_faceted[RESPONSE], errors="coerce"
)
ocean_data_faceted["sal"] = pd.to_numeric(
    ocean_data_faceted["bottle_salnty"], errors="coerce"
)
ocean_data_faceted["depth"] = pd.to_numeric(
    ocean_data_faceted["bottle_depthm"], errors="coerce"
)
ocean_data_faceted["depth_group"] = np.where(
    ocean_data_faceted["depth"] <= 200,
    "Shallow Water (0-200m)",
    "Deep Water (>200m)"
)
ocean_data_faceted = ocean_data_faceted.dropna(
    subset=["temp", "sal", "depth", "depth_group"]
)
ocean_data_faceted = ocean_data_faceted[
    ocean_data_faceted["temp"] < 35
]

g = sns.lmplot(
    data=ocean_data_faceted,
    x="temp",
    y="sal",
    hue="depth",
    col="depth_group",
    scatter_kws={"alpha": 0.20, "s": 8},
    height=5,
    aspect=1.0,
    fit_reg=True,
    legend=False
)
g.fig.suptitle("Temperature vs. Salinity by Depth Layer", y=1.04)
g.set_axis_labels("Temperature (°C)", "Salinity (psu)")
plt.savefig(OUTPUT_DIR / "img-8.jpeg", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# 17. CATEGORICAL ANALYSIS: MEAN TEMPERATURE BY DEPTH GROUP
# ---------------------------------------------------------------------

depth_summary_data = train.copy()
depth_summary_data["temp"] = pd.to_numeric(
    depth_summary_data[RESPONSE], errors="coerce"
)
depth_summary_data["depth"] = pd.to_numeric(
    depth_summary_data["bottle_depthm"], errors="coerce"
)
depth_summary_data["depth_group"] = np.where(
    depth_summary_data["depth"] <= 200,
    "Shallow (0-200m)",
    "Deep (>200m)"
)
depth_summary_data = depth_summary_data.dropna(
    subset=["temp", "depth_group"]
)
depth_summary_data = depth_summary_data[
    depth_summary_data["temp"] < 40
]

depth_summary = (
    depth_summary_data.groupby("depth_group")["temp"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
depth_summary["se"] = (
    depth_summary["std"] / np.sqrt(depth_summary["count"])
)

plt.figure(figsize=(7, 5))
plt.bar(
    depth_summary["depth_group"],
    depth_summary["mean"],
    yerr=depth_summary["se"],
    capsize=5,
    edgecolor="black"
)
plt.title("Categorical Analysis: Average Temperature by Depth Layer")
plt.xlabel("Depth Category")
plt.ylabel("Average Temperature (°C)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "img-9.jpeg", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# 18. ANOVA: TEMPERATURE BY DEPTH GROUP
# ---------------------------------------------------------------------

anova_data = train.copy()
anova_data["temp"] = pd.to_numeric(
    anova_data[RESPONSE], errors="coerce"
)
anova_data["depth"] = pd.to_numeric(
    anova_data["bottle_depthm"], errors="coerce"
)
anova_data["depth_group"] = np.where(
    anova_data["depth"] <= 200,
    "Shallow",
    "Deep"
)
anova_data = anova_data.dropna(
    subset=["temp", "depth", "depth_group"]
)

anova_model = smf.ols(
    "temp ~ C(depth_group)", data=anova_data
).fit()
anova_table = __import__("statsmodels.api").api.stats.anova_lm(
    anova_model, typ=2
)

print("\nANOVA results:")
print(anova_table)

# ---------------------------------------------------------------------
# 19. CHECK LATITUDE AND LONGITUDE FIELDS
# ---------------------------------------------------------------------

coordinate_columns = [
    col for col in train.columns
    if "lat" in col.lower() or "lon" in col.lower()
]
print("\nLatitude/longitude-related columns:", coordinate_columns)

# ---------------------------------------------------------------------
# 20. CREATE A STATION TEMPERATURE MAP
# ---------------------------------------------------------------------

required_map_columns = [
    "cast_sta_id", "cast_lat_dec", "cast_lon_dec", RESPONSE
]

if all(col in train.columns for col in required_map_columns):
    map_data = train[required_map_columns].copy()
    map_data["lat"] = pd.to_numeric(
        map_data["cast_lat_dec"], errors="coerce"
    )
    map_data["lon"] = pd.to_numeric(
        map_data["cast_lon_dec"], errors="coerce"
    )
    map_data["temp"] = pd.to_numeric(
        map_data[RESPONSE], errors="coerce"
    )

    map_data = map_data.dropna(subset=["lat", "lon", "temp"])
    map_data = map_data[map_data["temp"] < 40]

    map_data = (
        map_data.groupby(["cast_sta_id", "lat", "lon"], as_index=False)
        .agg(avg_temp=("temp", "mean"))
    )

    map_data.to_json(
        OUTPUT_DIR / "map_data.json",
        orient="records",
        indent=2
    )

    if not map_data.empty:
        center = [
            map_data["lat"].mean(),
            map_data["lon"].mean()
        ]
        ocean_map = folium.Map(location=center, zoom_start=5)

        min_temp = map_data["avg_temp"].min()
        max_temp = map_data["avg_temp"].max()
        color_scale = max(max_temp - min_temp, 1e-9)

        def temperature_color(value):
            # Simple blue-to-red gradient based on relative temperature.
            fraction = (value - min_temp) / color_scale
            red = int(255 * fraction)
            blue = int(255 * (1 - fraction))
            return f"#{red:02x}40{blue:02x}"

        for row in map_data.itertuples(index=False):
            folium.CircleMarker(
                location=[row.lat, row.lon],
                radius=5,
                color=temperature_color(row.avg_temp),
                fill=True,
                fill_opacity=0.8,
                popup=(
                    f"<b>Station ID:</b> {row.cast_sta_id}<br>"
                    f"<b>Average Temperature:</b> "
                    f"{row.avg_temp:.2f} °C"
                )
            ).add_to(ocean_map)

        ocean_map.save(OUTPUT_DIR / "calcofi_station_temperature_map.html")
else:
    print(
        "\nMap was not created because one or more required coordinate "
        "or station columns are missing."
    )

# ---------------------------------------------------------------------
# 21. LOAD map_data.json AND CREATE A SECOND TEST MAP
# ---------------------------------------------------------------------

map_json_path = OUTPUT_DIR / "map_data.json"

if map_json_path.exists():
    with open(map_json_path, "r", encoding="utf-8") as file:
        raw_map_data = json.load(file)

    clean_map_data = pd.DataFrame(raw_map_data)

    if {"lon", "lat"}.issubset(clean_map_data.columns):
        clean_map_data = clean_map_data.dropna(subset=["lon", "lat"])
        if not clean_map_data.empty:
            map_sf_center = [
                clean_map_data["lat"].mean(),
                clean_map_data["lon"].mean()
            ]
            map_sf = folium.Map(location=map_sf_center, zoom_start=5)

            for row in clean_map_data.itertuples(index=False):
                folium.CircleMarker(
                    location=[row.lat, row.lon],
                    radius=3,
                    color="blue",
                    weight=1,
                    fill=True
                ).add_to(map_sf)

            map_sf.save(OUTPUT_DIR / "calcofi_station_points_map.html")

print("\nAnalysis complete.")
print("Output folder:", OUTPUT_DIR.resolve())
