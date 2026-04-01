#CropYield — Crop yield prediction (ML + web demos)

Academic project for **predicting agricultural crop yield** from country, crop type, climate, and pesticide use. It follows a full **scikit-learn** workflow in `Training_Code1.ipynb`, exports reusable artifacts, and ships **two optional front-ends**: a **Flask** site (HTML/CSS/JS) and a **Streamlit** app.

The target variable matches FAO-style data: **yield in hectograms per hectare (`hg/ha`)**.

---

## Repository layout

```
Chaima_CropYield/
├── Training_Code1.ipynb      # Full pipeline: EDA, features, models, tuning, saving .pkl
├── requirements.txt          # Python dependencies
├── dataset/
│   ├── yield_df.csv          # Main merged table used for training & app dropdowns
│   ├── yield.csv             # Source: crop yield (FAO-style)
│   ├── rainfall.csv          # Source: rainfall
│   ├── pesticides.csv        # Source: pesticide use
│   └── temp.csv              # Source: temperature
├── models/                   # (Create this folder) Trained artifacts — see below
│   ├── crop_yield_prediction_model.pkl
│   ├── crop_yield_scaler.pkl
│   └── model_features.pkl
├── app.py                    # Flask backend + JSON API
├── streamlitapp.py           # Streamlit UI (single file)
├── templates/
│   └── index.html            # Flask main page
└── static/
    ├── css/style.css
    └── js/app.js
```

If the `models/` directory is empty, the web apps will not run until you **train and export** the three `.pkl` files (see [Model artifacts](#model-artifacts)).

---

## Dataset

### `dataset/yield_df.csv` (primary)

Consolidated table used in the notebook and for **Area** / **Item** select lists in the apps.

| Column | Description |
|--------|-------------|
| `Area` | Country / region (categorical) |
| `Item` | Crop type (categorical) |
| `Year` | Year |
| `hg/ha_yield` | **Target** — yield (hectogram per hectare) |
| `average_rain_fall_mm_per_year` | Annual rainfall (mm) |
| `pesticides_tonnes` | Pesticide use (tonnes) |
| `avg_temp` | Average temperature (°C) |

The first column in the CSV may be an unnamed index (`Unnamed: 0`); the notebook drops it if present.

### Other files in `dataset/`

Raw or intermediate sources (`yield.csv`, `rainfall.csv`, `pesticides.csv`, `temp.csv`) aligned with typical **TP Crop Yield** / FAO-style coursework. The **merged** file used for modeling is **`yield_df.csv`**.

---

## Machine learning pipeline (`Training_Code1.ipynb`)

The notebook implements an end-to-end workflow:

1. **Load & clean** `yield_df.csv` — drop missing values, coerce rainfall to numeric.
2. **EDA** — distributions and scatter plots (yield vs rain, temperature, pesticides).
3. **Feature engineering**  
   - `rain_temp_ratio = average_rain_fall_mm_per_year / (avg_temp + 0.1)`  
   - **One-hot encoding** of `Area` and `Item` with `pandas.get_dummies(..., drop_first=True)`.
4. **Train / test split** — 80% / 20%, `random_state=42`.
5. **Scaling** — `StandardScaler` fit **only** on the training set (no leakage).
6. **Models compared** — Linear Regression, Decision Tree, Random Forest, Ridge, Lasso (metrics: MAE, RMSE, R²).
7. **Hyperparameter tuning** — `GridSearchCV` on `DecisionTreeRegressor` (the **exported** model is this **best estimator**).
8. **Persistence (deployment)** — `joblib.dump` of model, scaler, and feature name list.

Training is written as a **Kaggle-oriented** single cell (paths like `/kaggle/input/...`); for local use, point `pd.read_csv` to `dataset/yield_df.csv`.

---

## Model artifacts

Three files must be present for inference (either under **`models/`** or the **project root**):

| File | Content |
|------|---------|
| `crop_yield_prediction_model.pkl` | Best `DecisionTreeRegressor` from `GridSearchCV` |
| `crop_yield_scaler.pkl` | Fitted `StandardScaler` |
| `model_features.pkl` | Ordered list of feature column names after encoding |

**Important:** Inference builds one row in **exactly** this column order. Categorical encoding uses **`drop_first=True`**, so the **reference** levels are the **first** `Area` and **first** `Item` when sorted alphabetically on the training data (e.g. **Albania** and **Cassava** for the provided `yield_df.csv`). Choosing a reference country or crop means all corresponding dummy columns stay zero.

### Regenerating the `.pkl` files

Run the saving section of `Training_Code1.ipynb` (section *“SAVING ASSETS FOR DEPLOYMENT”*) after training, or execute equivalent code locally with paths set to `dataset/yield_df.csv` and save into `models/`.

---

## Web applications

Both apps use the **same math** as the notebook: engineered ratio, one-hot alignment with `model_features.pkl`, `scaler.transform`, then `model.predict`.

### Streamlit (`streamlitapp.py`)

- **Run:** `streamlit run streamlitapp.py`
- Single-file app: sidebar documentation, inputs (country, crop, year, rain, pesticides, temperature), **Run prediction**, metric display and `rain_temp_ratio`.
- Uses `@st.cache_resource` / `@st.cache_data` for models and dropdown data.

### Flask (`app.py`)

- **Run:** `python app.py` (default `http://127.0.0.1:5000`)
- Serves `templates/index.html` with `static/css/style.css` and `static/js/app.js`.
- **POST** `/api/predict` — JSON body:

```json
{
  "area": "France",
  "item": "Wheat",
  "year": 2010,
  "average_rain_fall_mm_per_year": 1200,
  "pesticides_tonnes": 50,
  "avg_temp": 18
}
```

Response (success): `predicted_yield_hg_per_ha`, `rain_temp_ratio`.

---

## Setup

### Requirements

- Python **3.10+** recommended (notebook metadata references 3.12).

### Install dependencies

```bash
cd Chaima_CropYield
pip install -r requirements.txt
```

Packages: `numpy`, `pandas`, `scikit-learn`, `joblib`, `flask`, `streamlit`.

### Run Streamlit

```bash
streamlit run streamlitapp.py
```

### Run Flask

```bash
python app.py
```

Open the URL shown in the terminal.

---

## Coursework context (TP Crop Yield)

This repository supports a typical **practical (TP)** on crop yield: exploratory analysis, regression models, optional regularization and tuning, and a **simple deployment** (Flask or Streamlit) to query the trained model. Align written reports and slides with your official **TP_Crop_Yield.pdf** instructions if your teacher provides one.

---

## Limitations

- Predictions are **only as good as the training data** and the chosen tree model; they are **not** a substitute for agronomic or economic forecasting.
- **Out-of-distribution** inputs (extreme years or values far from training) may behave poorly.
- Apps assume the **same** preprocessing and feature list as training; **do not** change encoding or scaler without retraining and exporting new `.pkl` files.

---

## License and credits

Use and citation according to your institution’s rules. Dataset provenance is described in coursework / Kaggle sources linked in `Training_Code1.ipynb` metadata.

If you extend the project, keep `model_features.pkl` and the training notebook in sync so deployment stays reproducible.
