"""
Flask API for crop yield prediction — mirrors Training_Code1.ipynb pipeline:
rain_temp_ratio, get_dummies(Area, Item, drop_first=True), StandardScaler, tuned DecisionTree.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

def _first_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


MODEL_PATH = _first_existing(
    MODEL_DIR / "crop_yield_prediction_model.pkl",
    BASE_DIR / "crop_yield_prediction_model.pkl",
)
SCALER_PATH = _first_existing(
    MODEL_DIR / "crop_yield_scaler.pkl",
    BASE_DIR / "crop_yield_scaler.pkl",
)
FEATURES_PATH = _first_existing(
    MODEL_DIR / "model_features.pkl",
    BASE_DIR / "model_features.pkl",
)
DATASET_PATH = BASE_DIR / "dataset" / "yield_df.csv"

_model = None
_scaler = None
_feature_names: list[str] | None = None
_areas: list[str] = []
_items: list[str] = []


def load_artifacts():
    global _model, _scaler, _feature_names
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not SCALER_PATH.is_file():
        raise FileNotFoundError(f"Missing scaler: {SCALER_PATH}")
    if not FEATURES_PATH.is_file():
        raise FileNotFoundError(f"Missing features list: {FEATURES_PATH}")
    _model = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    _feature_names = list(joblib.load(FEATURES_PATH))


def load_dropdown_options():
    global _areas, _items
    if not DATASET_PATH.is_file():
        _areas, _items = [], []
        return
    df = pd.read_csv(DATASET_PATH)
    _areas = sorted(df["Area"].dropna().unique().tolist())
    _items = sorted(df["Item"].dropna().unique().tolist())


def build_feature_row(
    area: str,
    item: str,
    year: float,
    rain: float,
    pesticides: float,
    avg_temp: float,
) -> np.ndarray:
    rain_temp_ratio = float(rain) / (float(avg_temp) + 0.1)
    row: list[float] = []
    for col in _feature_names:
        if col == "Year":
            row.append(float(year))
        elif col == "average_rain_fall_mm_per_year":
            row.append(float(rain))
        elif col == "pesticides_tonnes":
            row.append(float(pesticides))
        elif col == "avg_temp":
            row.append(float(avg_temp))
        elif col == "rain_temp_ratio":
            row.append(rain_temp_ratio)
        elif col.startswith("Area_"):
            row.append(1.0 if col[5:] == area else 0.0)
        elif col.startswith("Item_"):
            row.append(1.0 if col[5:] == item else 0.0)
        else:
            row.append(0.0)
    return np.asarray(row, dtype=np.float64).reshape(1, -1)


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            areas=_areas,
            items=_items,
        )

    @app.route("/api/meta")
    def meta():
        return jsonify(
            {
                "areas": _areas,
                "items": _items,
                "target_unit": "hg/ha",
                "target_description": "Crop yield (hectogram per hectare)",
            }
        )

    @app.route("/api/predict", methods=["POST"])
    def predict():
        data = request.get_json(silent=True) or {}
        try:
            area = str(data.get("area", "")).strip()
            item = str(data.get("item", "")).strip()
            year = float(data.get("year"))
            rain = float(data.get("average_rain_fall_mm_per_year"))
            pesticides = float(data.get("pesticides_tonnes"))
            avg_temp = float(data.get("avg_temp"))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Invalid or missing numeric fields."}), 400

        if area not in _areas or item not in _items:
            return jsonify({"ok": False, "error": "Unknown area or crop type."}), 400

        rain_temp_ratio = float(rain) / (float(avg_temp) + 0.1)
        X = build_feature_row(area, item, year, rain, pesticides, avg_temp)
        X_df = pd.DataFrame(X, columns=_feature_names)
        Xs = _scaler.transform(X_df)
        pred = float(_model.predict(Xs)[0])

        return jsonify(
            {
                "ok": True,
                "predicted_yield_hg_per_ha": round(pred, 2),
                "rain_temp_ratio": round(rain_temp_ratio, 4),
            }
        )

    return app


load_dropdown_options()
load_artifacts()
app = create_app()


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
