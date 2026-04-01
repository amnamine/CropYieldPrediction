"""
Crop Yield Prediction — Streamlit (single file).
Pipeline aligned with Training_Code1.ipynb / TP Crop Yield:
  rain_temp_ratio, get_dummies(Area, Item, drop_first=True), StandardScaler, tuned DecisionTree.
Run:  streamlit run streamlitapp.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATASET_PATH = BASE_DIR / "dataset" / "yield_df.csv"


def _first_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def _paths():
    return (
        _first_existing(MODEL_DIR / "crop_yield_prediction_model.pkl", BASE_DIR / "crop_yield_prediction_model.pkl"),
        _first_existing(MODEL_DIR / "crop_yield_scaler.pkl", BASE_DIR / "crop_yield_scaler.pkl"),
        _first_existing(MODEL_DIR / "model_features.pkl", BASE_DIR / "model_features.pkl"),
    )


@st.cache_resource
def load_model_bundle():
    model_path, scaler_path, features_path = _paths()
    if not model_path.is_file():
        raise FileNotFoundError(str(model_path))
    if not scaler_path.is_file():
        raise FileNotFoundError(str(scaler_path))
    if not features_path.is_file():
        raise FileNotFoundError(str(features_path))
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = list(joblib.load(features_path))
    return model, scaler, feature_names


@st.cache_data
def load_area_item_options():
    if not DATASET_PATH.is_file():
        return [], []
    df = pd.read_csv(DATASET_PATH)
    areas = sorted(df["Area"].dropna().unique().tolist())
    items = sorted(df["Item"].dropna().unique().tolist())
    return areas, items


def build_feature_row(
    feature_names: list[str],
    area: str,
    item: str,
    year: float,
    rain: float,
    pesticides: float,
    avg_temp: float,
) -> tuple[np.ndarray, float]:
    rain_temp_ratio = float(rain) / (float(avg_temp) + 0.1)
    row: list[float] = []
    for col in feature_names:
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
    return np.asarray(row, dtype=np.float64).reshape(1, -1), rain_temp_ratio


def main():
    st.set_page_config(
        page_title="Crop Yield Lab",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        :root { --cy-green: #2d6a4f; --cy-mint: #95d5b2; --cy-dark: #1b4332; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
            padding: 1rem 1.25rem;
            border-radius: 12px;
            border: 1px solid rgba(149, 213, 178, 0.35);
        }
        div[data-testid="stMetric"] label { color: #d8f3dc !important; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 2rem !important;
            font-weight: 700;
        }
        .cy-hero {
            font-size: 2.1rem;
            font-weight: 700;
            color: #1b4332;
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }
        .cy-sub { color: #52796f; font-size: 1rem; margin-bottom: 1.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, scaler, feature_names = load_model_bundle()
    except FileNotFoundError as e:
        st.error(
            "Missing model files. Place **crop_yield_prediction_model.pkl**, **crop_yield_scaler.pkl**, "
            "and **model_features.pkl** in the `models/` folder or project root.\n\n"
            f"Details: `{e}`"
        )
        st.stop()

    areas, items = load_area_item_options()
    if not areas or not items:
        st.warning("Could not load `dataset/yield_df.csv` — country/crop lists will be empty.")

    with st.sidebar:
        st.markdown("### 🌾 Crop Yield Lab")
        st.caption("TP Crop Yield — inference demo")
        st.markdown("---")
        st.markdown(
            "**Model stack**\n\n"
            "- Tuned `DecisionTreeRegressor` (GridSearchCV)\n"
            "- `StandardScaler` on full feature matrix\n"
            "- Target: **hg/ha** (hectogram / hectare)"
        )
        st.markdown("---")
        st.markdown(
            "**Pipeline** (notebook)\n\n"
            "1. `rain_temp_ratio = rain / (temp + 0.1)`\n"
            "2. One-hot `Area`, `Item` — `drop_first=True`\n"
            "3. Scale → predict"
        )

    col_title, _ = st.columns([2, 1])
    with col_title:
        st.markdown('<p class="cy-hero">Predict crop yield</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="cy-sub">Same preprocessing as <code>Training_Code1.ipynb</code> — test the trained model interactively.</p>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown("#### Inputs")
        c1, c2 = st.columns(2)
        with c1:
            area = st.selectbox("Country (Area)", options=areas or ["—"], index=0, help="FAO-style area from the dataset")
        with c2:
            item = st.selectbox("Crop (Item)", options=items or ["—"], index=0)

        c3, c4, c5 = st.columns(3)
        with c3:
            year = st.number_input("Year", min_value=1960, max_value=2035, value=2010, step=1)
        with c4:
            rain = st.number_input("Rainfall (mm/year)", min_value=0.0, value=1200.0, step=1.0, format="%.1f")
        with c5:
            pesticides = st.number_input("Pesticides (tonnes)", min_value=0.0, value=50.0, step=1.0, format="%.2f")

        avg_temp = st.slider("Average temperature (°C)", min_value=-5.0, max_value=45.0, value=18.0, step=0.1)

        predict = st.button("Run prediction", type="primary", use_container_width=True)

    with right:
        st.markdown("#### Result")
        placeholder = st.empty()
        if predict:
            if not areas or area == "—" or not items or item == "—":
                placeholder.error("Load the dataset or pick a valid country and crop.")
            else:
                X, rain_temp_ratio = build_feature_row(
                    feature_names, area, item, float(year), float(rain), float(pesticides), float(avg_temp)
                )
                X_df = pd.DataFrame(X, columns=feature_names)
                Xs = scaler.transform(X_df)
                pred = float(model.predict(Xs)[0])
                with placeholder.container():
                    st.metric("Predicted yield", f"{pred:,.2f} hg/ha")
                    st.caption(f"Engineered **rain_temp_ratio** = `{rain_temp_ratio:.4f}`")
        else:
            placeholder.info("Set parameters and click **Run prediction**.")

    with st.expander("About this practical (TP)", expanded=False):
        st.markdown(
            """
            This app **only runs inference**: it loads the artifacts produced after training
            (`crop_yield_prediction_model.pkl`, `crop_yield_scaler.pkl`, `model_features.pkl`).
            Feature order and encoding must match training exactly — reference categories for
            `drop_first` are the first **Area** and **Item** when sorted (e.g. Albania, Cassava on this dataset).

            **Inputs** correspond to merged yield data: country, crop type, year, annual rainfall,
            pesticide use, and mean temperature — as in `yield_df.csv`.
            """
        )

    st.divider()
    st.caption("Streamlit · single-file app · `streamlit run streamlitapp.py`")


main()
