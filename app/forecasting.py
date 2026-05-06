from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
CENTER_ID = 13
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "train_merged.csv"
MODEL_PATH = ROOT_DIR / "models" / "xgb_food_demand_center13.pkl"
ENCODER_PATH = ROOT_DIR / "models" / "encoder.pkl"

CAT_COLS = ["center_type", "category", "cuisine", "discount_bin", "price_band"]
BASE_COLS = [
    "week",
    "meal_id",
    "emailer_for_promotion",
    "homepage_featured",
    "num_orders",
    "op_area",
    "discount_per",
]


@dataclass(frozen=True)
class Scenario:
    meal_id: int
    checkout_price: float
    base_price: float
    emailer_for_promotion: int
    homepage_featured: int
    center_type: str
    category: str
    cuisine: str
    op_area: float
    lag_1: float
    lag_2: float
    lag_3: float
    lag_7: float
    rolling_mean_7: float
    rolling_std_7: float
    use_manual_memory: bool = True


def clean_feature_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "", str(value))


def load_artifacts() -> tuple[Any, Any]:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


def load_training_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH)


def model_feature_names(model: Any) -> list[str]:
    return [str(col) for col in model.feature_names_in_]


def encoder_options(encoder: Any) -> dict[str, list[str]]:
    return {
        column: [str(value) for value in values]
        for column, values in zip(CAT_COLS, encoder.categories_)
    }


def discount_percent(base_price: float, checkout_price: float) -> float:
    if base_price <= 0:
        return 0.0
    discount = ((base_price - checkout_price) / base_price) * 100
    return round(max(discount, 0.0), 2)


def discount_bin(discount_per: float) -> str:
    if discount_per <= 0:
        return "0"
    if discount_per <= 5:
        return "0-5"
    if discount_per <= 10:
        return "5-10"
    if discount_per <= 20:
        return "10-20"
    if discount_per <= 30:
        return "20-30"
    if discount_per <= 40:
        return "30-40"
    if discount_per <= 50:
        return "40-50"
    return "50-100"


def price_band(checkout_price: float) -> str:
    if checkout_price <= 100:
        return "<=100"
    if checkout_price <= 150:
        return "101-150"
    if checkout_price <= 200:
        return "151-200"
    if checkout_price <= 250:
        return "201-250"
    if checkout_price <= 300:
        return "251-300"
    if checkout_price <= 400:
        return "301-400"
    if checkout_price <= 600:
        return "401-600"
    return "600+"


def _encoded_category_row(encoder: Any, scenario: Scenario) -> dict[str, float]:
    raw = pd.DataFrame(
        [
            {
                "center_type": scenario.center_type,
                "category": scenario.category,
                "cuisine": scenario.cuisine,
                "discount_bin": discount_bin(
                    discount_percent(scenario.base_price, scenario.checkout_price)
                ),
                "price_band": price_band(scenario.checkout_price),
            }
        ],
        columns=CAT_COLS,
    )
    encoded = encoder.transform(raw)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()

    names = [clean_feature_name(col) for col in encoder.get_feature_names_out(CAT_COLS)]
    return {name: float(value) for name, value in zip(names, encoded[0])}


def center_raw_data(train_df: pd.DataFrame, center_id: int = CENTER_ID) -> pd.DataFrame:
    center = train_df[train_df["center_id"] == center_id].copy()
    return center.sort_values(["meal_id", "week"]).reset_index(drop=True)


def prepare_center_frame(
    train_df: pd.DataFrame, encoder: Any, center_id: int = CENTER_ID
) -> pd.DataFrame:
    df = center_raw_data(train_df, center_id)
    df["discount"] = (df["base_price"] - df["checkout_price"]).round(2)
    df["discount_per"] = discount_percent_vector(df["base_price"], df["checkout_price"])
    df["discount_bin"] = df["discount_per"].map(discount_bin)
    df["price_band"] = df["checkout_price"].map(price_band)

    encoded = encoder.transform(df[CAT_COLS])
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()

    encoded_names = [
        clean_feature_name(col) for col in encoder.get_feature_names_out(CAT_COLS)
    ]
    encoded_df = pd.DataFrame(encoded, columns=encoded_names)
    center_df = pd.concat(
        [df[BASE_COLS].reset_index(drop=True), encoded_df.reset_index(drop=True)],
        axis=1,
    )

    group = center_df.groupby("meal_id")["num_orders"]
    center_df["lag_1"] = group.shift(1)
    center_df["lag_2"] = group.shift(2)
    center_df["lag_3"] = group.shift(3)
    center_df["lag_7"] = group.shift(7)
    center_df["rolling_mean_7"] = group.transform(
        lambda values: values.shift(1).rolling(7).mean()
    )
    center_df["rolling_std_7"] = group.transform(
        lambda values: values.shift(1).rolling(7).std()
    )
    center_df = center_df.dropna().reset_index(drop=True)
    return center_df.sort_values(["meal_id", "week"]).reset_index(drop=True)


def discount_percent_vector(base_price: pd.Series, checkout_price: pd.Series) -> pd.Series:
    discount = ((base_price - checkout_price) / base_price.replace(0, np.nan)) * 100
    return discount.fillna(0).clip(lower=0).round(2)


def meal_summary(center_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        center_df.groupby("meal_id")
        .agg(
            category=("category", "first"),
            cuisine=("cuisine", "first"),
            avg_orders=("num_orders", "mean"),
            last_orders=("num_orders", "last"),
        )
        .reset_index()
        .sort_values("avg_orders", ascending=False)
    )
    summary["label"] = summary.apply(
        lambda row: f"{int(row['meal_id'])} - {row['category']} / {row['cuisine']}",
        axis=1,
    )
    return summary


def latest_profile(center_df: pd.DataFrame, meal_id: int) -> dict[str, Any]:
    meal_df = center_df[center_df["meal_id"] == meal_id].sort_values("week")
    if meal_df.empty:
        raise ValueError(f"No history found for meal_id {meal_id}.")

    last = meal_df.iloc[-1]
    recent = meal_df["num_orders"].tail(7).astype(float).tolist()
    padded = _padded_history(recent)

    return {
        "meal_id": int(meal_id),
        "week": int(last["week"]),
        "center_type": str(last["center_type"]),
        "category": str(last["category"]),
        "cuisine": str(last["cuisine"]),
        "op_area": float(last["op_area"]),
        "checkout_price": float(last["checkout_price"]),
        "base_price": float(last["base_price"]),
        "emailer_for_promotion": int(last["emailer_for_promotion"]),
        "homepage_featured": int(last["homepage_featured"]),
        "lag_1": float(padded[-1]),
        "lag_2": float(padded[-2]),
        "lag_3": float(padded[-3]),
        "lag_7": float(padded[-7]),
        "rolling_mean_7": float(np.mean(padded[-7:])),
        "rolling_std_7": float(np.std(padded[-7:])),
        "recent_avg_4": float(meal_df["num_orders"].tail(4).mean()),
        "recent_avg_12": float(meal_df["num_orders"].tail(12).mean()),
        "historical_avg": float(meal_df["num_orders"].mean()),
        "historical_max": int(meal_df["num_orders"].max()),
    }


def _padded_history(history: list[float], size: int = 7) -> list[float]:
    if len(history) >= size:
        return history
    fill_value = float(np.mean(history)) if history else 0.0
    return [fill_value] * (size - len(history)) + history


def _apply_manual_memory(history: list[float], scenario: Scenario) -> list[float]:
    adjusted = _padded_history(history.copy())
    adjusted[-1] = float(scenario.lag_1)
    adjusted[-2] = float(scenario.lag_2)
    adjusted[-3] = float(scenario.lag_3)
    adjusted[-7] = float(scenario.lag_7)
    return adjusted


def _lags_from_history(history: list[float]) -> dict[str, float]:
    padded = _padded_history(history)
    recent = padded[-7:]
    return {
        "lag_1": float(padded[-1]),
        "lag_2": float(padded[-2]),
        "lag_3": float(padded[-3]),
        "lag_7": float(padded[-7]),
        "rolling_mean_7": float(np.mean(recent)),
        "rolling_std_7": float(np.std(recent)),
    }


def build_feature_row(
    model: Any,
    encoder: Any,
    scenario: Scenario,
    week: int,
    lags: dict[str, float],
) -> pd.DataFrame:
    features = model_feature_names(model)
    row = {feature: 0.0 for feature in features}
    row.update(
        {
            "week": float(week),
            "meal_id": float(scenario.meal_id),
            "emailer_for_promotion": float(scenario.emailer_for_promotion),
            "homepage_featured": float(scenario.homepage_featured),
            "op_area": float(scenario.op_area),
            "discount_per": float(
                discount_percent(scenario.base_price, scenario.checkout_price)
            ),
            "lag_1": float(lags["lag_1"]),
            "lag_2": float(lags["lag_2"]),
            "lag_3": float(lags["lag_3"]),
            "lag_7": float(lags["lag_7"]),
            "rolling_mean_7": float(lags["rolling_mean_7"]),
            "rolling_std_7": float(lags["rolling_std_7"]),
        }
    )

    for name, value in _encoded_category_row(encoder, scenario).items():
        if name in row:
            row[name] = value

    return pd.DataFrame([row], columns=features)


def forecast_orders(
    model: Any,
    encoder: Any,
    center_df: pd.DataFrame,
    scenario: Scenario,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meal_df = center_df[center_df["meal_id"] == scenario.meal_id].sort_values("week")
    if meal_df.empty:
        raise ValueError(f"No history found for meal_id {scenario.meal_id}.")

    history = meal_df["num_orders"].astype(float).tolist()
    if scenario.use_manual_memory:
        history = _apply_manual_memory(history, scenario)

    last_week = int(meal_df["week"].max())
    predictions: list[dict[str, float]] = []
    first_features: pd.DataFrame | None = None

    for step in range(1, horizon + 1):
        lags = _lags_from_history(history)
        if step == 1 and scenario.use_manual_memory:
            lags["rolling_mean_7"] = float(scenario.rolling_mean_7)
            lags["rolling_std_7"] = float(scenario.rolling_std_7)

        week = last_week + step
        features = build_feature_row(model, encoder, scenario, week, lags)
        raw_prediction = float(model.predict(features)[0])
        predicted_orders = int(round(max(raw_prediction, 0.0)))

        if first_features is None:
            first_features = features.copy()

        predictions.append(
            {
                "week": week,
                "meal_id": scenario.meal_id,
                "predicted_orders": predicted_orders,
                "raw_prediction": round(max(raw_prediction, 0.0), 2),
                "discount_per": discount_percent(
                    scenario.base_price, scenario.checkout_price
                ),
                "price_band": price_band(scenario.checkout_price),
                "discount_bin": discount_bin(
                    discount_percent(scenario.base_price, scenario.checkout_price)
                ),
            }
        )
        history.append(float(predicted_orders))

    return pd.DataFrame(predictions), first_features if first_features is not None else pd.DataFrame()


def feature_importance(model: Any, top_n: int = 15) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": model_feature_names(model),
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance")
    )
