from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODEL_PATH = BASE / "catboost_rpm_model.cbm"

CATS = ["pickup", "delivery", "equipment", "route", "route_equipment"]


def make_features(df, coords, weight_median):
    x = df.copy()
    dt = pd.to_datetime(x["date"], errors="coerce")

    if "pickup_lat" not in x.columns:
        x = x.merge(
            coords.rename(columns={"lat": "pickup_lat", "lon": "pickup_lon"}),
            left_on="pickup", right_index=True, how="left"
        )
    if "delivery_lat" not in x.columns:
        x = x.merge(
            coords.rename(columns={"lat": "delivery_lat", "lon": "delivery_lon"}),
            left_on="delivery", right_index=True, how="left"
        )

    x["weight_missing"] = x["weight"].isna().astype(int)
    x.loc[x["weight"] < 0, "weight"] = np.nan
    x["weight_invalid"] = x["weight"].isna().astype(int)
    x["weight"] = x["weight"].fillna(weight_median)

    for c in ["pickup", "delivery", "equipment"]:
        x[c] = x[c].fillna("UNKNOWN").astype(str)
    x["route"] = x["pickup"] + "__" + x["delivery"]
    x["route_equipment"] = x["route"] + "__" + x["equipment"]

    lat1 = np.radians(x["pickup_lat"].astype(float))
    lat2 = np.radians(x["delivery_lat"].astype(float))
    dlat = lat2 - lat1
    dlon = np.radians(x["delivery_lon"].astype(float) - x["pickup_lon"].astype(float))
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    x["geo_distance"] = 3958.7613 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    x["lat_delta"] = x["delivery_lat"] - x["pickup_lat"]
    x["lon_delta"] = x["delivery_lon"] - x["pickup_lon"]
    x["coord_distance"] = np.sqrt(x["lat_delta"] ** 2 + x["lon_delta"] ** 2)
    x["geo_ratio"] = x["distance"] / x["geo_distance"].clip(lower=1)

    distance = x["distance"].clip(lower=0)
    x["log_distance"] = np.log1p(distance)
    x["sqrt_distance"] = np.sqrt(distance)
    x["weight_per_mile"] = x["weight"] / x["distance"].clip(lower=1)
    x["distance_weight"] = x["distance"] * x["weight"] / 50000
    x["distance_sq"] = x["distance"] ** 2 / 1e6
    x["weight_sq"] = x["weight"] ** 2 / 1e9

    x["month"] = dt.dt.month
    x["day"] = dt.dt.day
    x["dow"] = dt.dt.dayofweek
    x["doy"] = dt.dt.dayofyear
    x["week"] = dt.dt.isocalendar().week.astype(int)
    x["days_since_start"] = (dt - pd.Timestamp("2025-01-01")).dt.days
    x["month_sin"] = np.sin(2 * np.pi * x["month"] / 12)
    x["month_cos"] = np.cos(2 * np.pi * x["month"] / 12)
    x["dow_sin"] = np.sin(2 * np.pi * x["dow"] / 7)
    x["dow_cos"] = np.cos(2 * np.pi * x["dow"] / 7)
    x["doy_sin"] = np.sin(2 * np.pi * x["doy"] / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * x["doy"] / 365.25)
    x["weekend"] = (x["dow"] >= 5).astype(int)

    drop = [c for c in ["load_id", "date", "market_index", "quote_signal", "posted_rate", "predicted_rate"] if c in x.columns]
    return x.drop(columns=drop)


def main():
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    template = pd.read_csv(DATA / "validation-predictions-template.csv")
    december = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])

    coords = pd.concat([
        train[["pickup", "pickup_lat", "pickup_lon"]].rename(columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}),
        train[["delivery", "delivery_lat", "delivery_lon"]].rename(columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}),
    ]).groupby("city")[["lat", "lon"]].median()
    valid_weight = train.loc[train["weight"].notna() & (train["weight"] >= 0), "weight"]
    weight_median = float(valid_weight.median())

    X = make_features(train, coords, weight_median)
    Xv = make_features(validation, coords, weight_median).reindex(columns=X.columns)
    Xd = make_features(december, coords, weight_median).reindex(columns=X.columns)
    cat_indices = [X.columns.get_loc(c) for c in CATS]

    y = train["posted_rate"].astype(float)
    y_rpm = y / train["distance"].clip(lower=1)

    train_mask = train["date"] < "2025-10-01"
    valid_mask = ~train_mask

    # Use the selected iteration count from the validated final model.
    iterations = 170
    model = CatBoostRegressor(
        iterations=iterations,
        depth=5,
        learning_rate=0.05,
        loss_function="MAE",
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
        thread_count=4,
    )
    model.fit(X, y_rpm, cat_features=cat_indices)

    validation_pred = np.maximum(
        model.predict(Pool(Xv, cat_features=cat_indices)) * validation["distance"].to_numpy(),
        0.01,
    )
    output = template[["load_id"]].copy()
    lookup = dict(zip(validation["load_id"].astype(str), validation_pred))
    output["predicted_rate"] = output["load_id"].astype(str).map(lookup).astype(float)
    output.to_csv(BASE / "validation_predictions.csv", index=False)

    december_pred = np.maximum(
        model.predict(Pool(Xd, cat_features=cat_indices)) * december["distance"].to_numpy(),
        0.01,
    )
    december["predicted_rate"] = december_pred
    december.to_csv(BASE / "december_chart_inputs.csv", index=False)
    model.save_model(str(MODEL_PATH))

    print("Created validation_predictions.csv with", len(output), "rows")
    print("Created december_chart_inputs.csv with", len(december), "rows")
    print("Saved", MODEL_PATH.name)


if __name__ == "__main__":
    main()
