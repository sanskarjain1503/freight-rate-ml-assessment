from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

train = pd.read_csv(DATA / "train-test.csv", parse_dates=["date"])
validation = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
template = pd.read_csv(DATA / "validation-predictions-template.csv")
december = pd.read_csv(DATA / "december-chart-inputs.csv", parse_dates=["date"])

CITY_COORDS = (
    pd.concat([
        train[["pickup","pickup_lat","pickup_lon"]].rename(columns={"pickup":"city","pickup_lat":"lat","pickup_lon":"lon"}),
        train[["delivery","delivery_lat","delivery_lon"]].rename(columns={"delivery":"city","delivery_lat":"lat","delivery_lon":"lon"})
    ])
    .groupby("city")[["lat","lon"]].median()
)
WEIGHT_MEDIAN = float(train["weight"].median())

def features(frame, target=False):
    x = frame.copy()
    dt = pd.to_datetime(x["date"])

    if "pickup_lat" not in x.columns:
        x = x.merge(CITY_COORDS.rename(columns={"lat":"pickup_lat","lon":"pickup_lon"}),
                    left_on="pickup", right_index=True, how="left")
    if "delivery_lat" not in x.columns:
        x = x.merge(CITY_COORDS.rename(columns={"lat":"delivery_lat","lon":"delivery_lon"}),
                    left_on="delivery", right_index=True, how="left")

    x["month"] = dt.dt.month
    x["day"] = dt.dt.day
    x["dayofweek"] = dt.dt.dayofweek
    x["dayofyear"] = dt.dt.dayofyear
    x["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    x["days_since_start"] = (dt - pd.Timestamp("2025-01-01")).dt.days
    x["route"] = x["pickup"].fillna("UNKNOWN") + "__" + x["delivery"].fillna("UNKNOWN")
    x["coord_distance"] = np.sqrt((x.pickup_lat-x.delivery_lat)**2 + (x.pickup_lon-x.delivery_lon)**2)
    x["weight_missing"] = x["weight"].isna().astype(int)
    x["weight"] = x["weight"].fillna(WEIGHT_MEDIAN)
    x["log_distance"] = np.log1p(x["distance"].clip(lower=0))
    x["weight_per_mile"] = x["weight"] / x["distance"].clip(lower=1)
    x["distance_weight_interaction"] = x["distance"] * (x["weight"]/50000.0)
    x["month_sin"] = np.sin(2*np.pi*x["month"]/12)
    x["month_cos"] = np.cos(2*np.pi*x["month"]/12)
    x["dow_sin"] = np.sin(2*np.pi*x["dayofweek"]/7)
    x["dow_cos"] = np.cos(2*np.pi*x["dayofweek"]/7)

    drop = ["load_id", "market_index", "quote_signal", "predicted_rate"]
    if target:
        drop.append("posted_rate")
    return x.drop(columns=[c for c in drop if c in x.columns])

X = features(train, True)
y = train["posted_rate"].astype(float)
CAT_NAMES = ["pickup","delivery","equipment","route"]
CAT_IDX = [X.columns.get_loc(c) for c in CAT_NAMES]
FEATURE_COLUMNS = list(X.columns)

def align(frame):
    return frame.reindex(columns=FEATURE_COLUMNS)

# Chronological validation: October 2025 is the holdout.
train_mask = train.date < "2025-10-01"
val_mask = ~train_mask
m_val = CatBoostRegressor(
    iterations=1200, depth=6, learning_rate=0.05,
    loss_function="MAE", l2_leaf_reg=5,
    random_seed=42, verbose=False
)
m_val.fit(
    X[train_mask], y[train_mask],
    cat_features=CAT_IDX,
    eval_set=(X[val_mask], y[val_mask]),
    early_stopping_rounds=100
)
best = max(300, m_val.get_best_iteration() + 1)

# Final model on all labeled data.
model = CatBoostRegressor(
    iterations=best, depth=6, learning_rate=0.05,
    loss_function="MAE", l2_leaf_reg=5,
    random_seed=42, verbose=False
)
model.fit(X, y, cat_features=CAT_IDX)

val_features = align(features(validation))
p = model.predict(Pool(val_features, cat_features=CAT_IDX))
out = template[["load_id"]].copy()
out["predicted_rate"] = np.maximum(p, 0.01)
out.to_csv(OUT / "validation_predictions.csv", index=False)

dec_features = align(features(december))
p_dec = model.predict(Pool(dec_features, cat_features=CAT_IDX))
december["predicted_rate"] = np.maximum(p_dec, 0.01)
december.to_csv(OUT / "december_chart_inputs.csv", index=False)

model.save_model(OUT / "freight_rate_catboost.cbm")
print("Saved outputs/validation_predictions.csv and outputs/december_chart_inputs.csv")
