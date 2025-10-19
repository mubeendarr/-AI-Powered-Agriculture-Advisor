# app.py
import os, sys, json, math, joblib
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ========= PATHS / ROOT =========
ROOT = Path(__file__).resolve().parent

# ========= SESSION STATE =========
if "rec_crops" not in st.session_state:
    st.session_state.rec_crops = None
if "yield_msg" not in st.session_state:
    st.session_state.yield_msg = None

# ========= SMALL HELPERS =========
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _pretty_label(raw: str) -> str:
    """ Make folder-like class names readable (after mapping by index). """
    s = raw
    for t in ["Tomato__", "Potato___", "Pepper__bell___", "Tomato_", "Pepper_", "Potato_"]:
        s = s.replace(t, "")
    s = s.replace("_", " ").replace("  ", " ").strip()
    s = s.replace("healthy", "Healthy")
    return s

# ========= MODEL LOADERS =========
@st.cache_resource(show_spinner=False)
def load_crop_model_bundle():
    """ RandomForest crop model + feature list + defaults. """
    model_path = ROOT / "models" / "crop_rec" / "baseline_rf.joblib"
    feat_path  = ROOT / "models" / "crop_rec" / "feature_columns.json"
    dflt_path  = ROOT / "models" / "crop_rec" / "defaults.json"
    model = joblib.load(model_path)
    features = load_json(feat_path)
    defaults = load_json(dflt_path)
    return model, features, defaults

@st.cache_resource(show_spinner=False)
def load_yield_model_bundle():
    """
    GradientBoosting (or your regressor) + feature config:
      - feature_config.json keys: num_cols, cat_cols, target
    """
    model_path = ROOT / "models" / "weather_yield" / "yield_regressor.joblib"
    fcfg_path  = ROOT / "models" / "weather_yield" / "feature_config.json"
    model = joblib.load(model_path)

    # sensible defaults (overridden by file if present)
    feat_cfg = {
        "num_cols": [
            "temp_mean_year", "rain_total_year", "humidity_mean_year", "wind_mean_year",
            "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"
        ],
        "cat_cols": ["Area", "Item"],
        "target": "hg/ha_yield",
    }
    try:
        with open(fcfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if isinstance(cfg.get("num_cols"), list) and cfg["num_cols"]:
            feat_cfg["num_cols"] = cfg["num_cols"]
        if isinstance(cfg.get("cat_cols"), list) and cfg["cat_cols"]:
            feat_cfg["cat_cols"] = cfg["cat_cols"]
        if isinstance(cfg.get("target"), str) and cfg["target"]:
            feat_cfg["target"] = cfg["target"]
    except Exception:
        pass
    return model, feat_cfg

@st.cache_resource(show_spinner=False)
def load_disease_model_and_labels():
    """
    EfficientNetB0 inference model (H5) + labels.json (exact training order).
    """
    mdl_path = ROOT / "models" / "disease" / "plant_effb0_infer.h5"
    lbl_path = ROOT / "models" / "disease" / "labels.json"
    model = tf.keras.models.load_model(mdl_path, compile=False)
    with open(lbl_path, "r", encoding="utf-8") as f:
        class_list = json.load(f)
    return model, class_list

# ========= MERGED TABLE (WEATHER+YIELD) =========
@st.cache_resource(show_spinner=False)
def load_merged_table():
    """
    Loads your merged table. Tries common filenames.
    Must include at least: Area, Year, Item, temp_mean_year, rain_total_year, humidity_mean_year,
    wind_mean_year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp.
    """
    candidates = [
        ROOT / "data" / "processed" / "weather_yield_merged.csv",  # preferred
        ROOT / "data" / "processed" / "yield_aligned.csv",         # acceptable
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            # unify rainfall name if needed
            if "rain_total_year" not in df.columns and "average_rain_fall_mm_per_year" in df.columns:
                df = df.rename(columns={"average_rain_fall_mm_per_year": "rain_total_year"})
            return df
    return pd.DataFrame([])

# ========= WEATHER UTIL =========
def _yearly_weather_from_table(table: pd.DataFrame, area: str, k_years: int, target_year: int, item: str = None):
    """
    Get/forecast yearly weather for area & target_year.
    - If exact year exists for the area, return that.
    - Else compute mean over the last k rows < target_year for (area, item),
      falling back to (area) only if needed.
    """
    if table is None or table.empty:
        return {"temperature": None, "humidity": None, "rainfall": None, "wind": None,
                "history_years_used": 0, "history_years_to": None, "method": "none", "k_years": k_years}

    # try exact area/year (area-level)
    sub_area = table[table["Area"].str.lower() == str(area).strip().lower()]
    if not sub_area.empty and (sub_area["Year"] == target_year).any():
        row = sub_area.loc[sub_area["Year"] == target_year].iloc[0]
        return {
            "temperature": float(row.get("temp_mean_year", row.get("avg_temp", np.nan))),
            "humidity": float(row.get("humidity_mean_year", np.nan)),
            "rainfall": float(row.get("rain_total_year", row.get("average_rain_fall_mm_per_year", np.nan))),
            "wind": float(row.get("wind_mean_year", np.nan)),
            "history_years_used": target_year,
            "history_years_to": target_year,
            "method": "actual",
            "k_years": 0
        }

    # else, compute mean from last k rows of (Area, Item) if item provided
    if item is not None:
        win = _history_window(table, area, item, target_year, k_years)
        if not win.empty:
            return {
                "temperature": float(win["temp_mean_year"].mean() if "temp_mean_year" in win else win["avg_temp"].mean()),
                "humidity": float(win["humidity_mean_year"].mean()) if "humidity_mean_year" in win else None,
                "rainfall": float(win["rain_total_year"].mean() if "rain_total_year" in win else win["average_rain_fall_mm_per_year"].mean()),
                "wind": float(win["wind_mean_year"].mean()) if "wind_mean_year" in win else None,
                "history_years_used": int(win["Year"].min()),
                "history_years_to": int(win["Year"].max()),
                "method": "mean_last_k_rows",
                "k_years": int(min(len(win), k_years)),
            }

    # fall back to area-only window
    sub_area = sub_area[sub_area["Year"] < int(target_year)].sort_values("Year")
    if not sub_area.empty:
        win = sub_area.tail(k_years)
        return {
            "temperature": float(win["temp_mean_year"].mean() if "temp_mean_year" in win else win["avg_temp"].mean()),
            "humidity": float(win["humidity_mean_year"].mean()) if "humidity_mean_year" in win else None,
            "rainfall": float(win["rain_total_year"].mean() if "rain_total_year" in win else win["average_rain_fall_mm_per_year"].mean()),
            "wind": float(win["wind_mean_year"].mean()) if "wind_mean_year" in win else None,
            "history_years_used": int(win["Year"].min()),
            "history_years_to": int(win["Year"].max()),
            "method": "mean_last_k_rows_area",
            "k_years": int(min(len(win), k_years)),
        }

    return {"temperature": None, "humidity": None, "rainfall": None, "wind": None,
            "history_years_used": 0, "history_years_to": None, "method": "none", "k_years": k_years}

# ========= MACRO FEATURES (FAO-LIKE) =========
import difflib
import re

def _normalize_txt(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _resolve_name_from_table(table: pd.DataFrame, col: str, user_value: str, cutoff: float = 0.8):
    """
    Resolve a user-entered name (Area or Item) to a value that exists in the table.
    - Tries case-insensitive exact match first
    - Then tries fuzzy match (difflib) with a cutoff
    Returns (resolved_value, matched_exact: bool) or (None, False) if not found.
    """
    if table is None or table.empty or col not in table.columns:
        return None, False

    vals = table[col].dropna().astype(str).unique().tolist()
    norm_map = { _normalize_txt(v): v for v in vals }
    q = _normalize_txt(user_value)

    # exact (case-insensitive) first
    if q in norm_map:
        return norm_map[q], True

    # small alias map (common FAO naming hiccups)
    aliases = {
        "united kingdom": ["uk", "u.k.", "great britain", "gb", "england (uk)", "england"],
        "maize": ["corn"],
        "rice, paddy": ["rice", "paddy rice"],
        "sweet potatoes": ["sweet potato"],
        "plantains and others": ["plantains"],
        "soybeans": ["soya", "soybean"],
        "potatoes": ["potato"],
        "wheat": ["wheats"],
    }
    for canon, alts in aliases.items():
        if q == canon or q in ( _normalize_txt(a) for a in alts ):
            if canon in norm_map:
                return norm_map[canon], False
            # also try any alt that exists
            for a in alts:
                if _normalize_txt(a) in norm_map:
                    return norm_map[_normalize_txt(a)], False

    # fuzzy
    candidates = list(norm_map.keys())
    matches = difflib.get_close_matches(q, candidates, n=1, cutoff=cutoff)
    if matches:
        return norm_map[matches[0]], False
    return None, False


def _history_window(table: pd.DataFrame, area: str, item: str, target_year: int, k_years: int = 10):
    """
    Returns a DataFrame window of up to the most recent k_years rows
    for (Area, Item) PRIOR to target_year. Robust to missing years.
    We don't assume every year exists; we just take the last k rows by Year.
    """
    if table is None or table.empty:
        return pd.DataFrame()

    # resolve names to actual table entries
    area_resolved, _ = _resolve_name_from_table(table, "Area", area)
    item_resolved, _ = _resolve_name_from_table(table, "Item", item)
    if not area_resolved or not item_resolved:
        return pd.DataFrame()

    sub = table[
        (table["Area"] == area_resolved) &
        (table["Item"] == item_resolved) &
        (table["Year"] < int(target_year))
    ].copy()

    if sub.empty:
        # fall back to ANY historical (even if all years >= target_year missing)
        sub = table[
            (table["Area"] == area_resolved) &
            (table["Item"] == item_resolved)
        ].copy()

    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values("Year")
    return sub.tail(k_years)  # last k rows, regardless of gaps

def _has_history_for_area_item(table: pd.DataFrame, area: str, item: str,
                               target_year: int, k_years: int = 10) -> dict:
    win = _history_window(table, area, item, target_year, k_years)
    if not win.empty:
        return {
            "found": True,
            "count": int(win.shape[0]),
            "min_year": int(win["Year"].min()),
            "max_year": int(win["Year"].max()),
        }
    # check any-ever
    area_resolved, _ = _resolve_name_from_table(table, "Area", area)
    item_resolved, _ = _resolve_name_from_table(table, "Item", item)
    if not area_resolved or not item_resolved:
        return {"found": False, "count": 0, "min_year": None, "max_year": None}
    sub = table[(table["Area"] == area_resolved) & (table["Item"] == item_resolved)]
    if sub.empty:
        return {"found": False, "count": 0, "min_year": None, "max_year": None}
    return {
        "found": False,  # not in last k rows before year, but exists historically
        "count": int(sub.shape[0]),
        "min_year": int(sub["Year"].min()),
        "max_year": int(sub["Year"].max()),
    }



def _item_known_to_model(y_model, item_col_name: str, item_value: str) -> bool:
    """
    Returns True if the trained pipeline's OneHotEncoder has seen this item category.
    If we cannot introspect safely, we default to True.
    """
    try:
        # Expect a pipeline: ("prep", ColumnTransformer(...)), ("rf"/"gbm", estimator)
        prep = getattr(y_model, "named_steps", {}).get("prep", None)
        if prep is None:
            return True  # no preprocessor; nothing to check

        # Find the OneHotEncoder inside the ColumnTransformer under the "cat" transformer
        cat_trans = getattr(prep, "named_transformers_", {}).get("cat", None)
        if cat_trans is None or not hasattr(cat_trans, "categories_"):
            return True

        # Find index of the 'Item' column within the cat_cols the pipeline was trained with
        # Try to read the feature config if present on the model; else skip
        trained_cat_cols = getattr(prep, "feature_names_in_", None)
        # feature_names_in_ is for whole ColumnTransformer; we need the configured columns for 'cat'
        # More robust: stored on the ColumnTransformer itself
        cat_cols = None
        try:
            for name, trans, cols in prep.transformers_:
                if name == "cat":
                    cat_cols = cols
                    break
        except Exception:
            pass

        if not cat_cols or item_col_name not in cat_cols:
            return True  # cannot locate item column -> skip strict check

        idx = list(cat_cols).index(item_col_name)
        categories_for_item = cat_trans.categories_[idx]
        # Case-insensitive compare
        return any(str(c).lower() == str(item_value).lower() for c in categories_for_item)
    except Exception:
        return True  # fail open to avoid false negatives


def _nearest_year_value(df, area, year, col):
    s = df.loc[df["Area"].eq(area) & df["Year"].le(year)].sort_values("Year", ascending=False)
    if not s.empty and col in s.columns:
        return float(s.iloc[0][col])
    s2 = df.loc[df["Area"].eq(area)].sort_values("Year", ascending=False)
    if not s2.empty and col in s2.columns:
        return float(s2.iloc[0][col])
    return float(df[col].mean())

def _macro_features_from_table(table, area, year):
    required = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
    vals = {}
    for c in required:
        if c not in table.columns:
            raise ValueError(f"Required macro column missing in merged table: {c}")
        vals[c] = _nearest_year_value(table, area, year, c)
    return vals

# ========= PREDICTORS =========
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import img_to_array

def predict_disease_name(pil_img):
    """
    Single diagnosis string (no confidence).
    Uses the SAME preprocessing as training: EfficientNetB0 preprocess_input.
    """
    model, classes = load_disease_model_and_labels()

    # 1) prepare exactly like training
    img = pil_img.convert("RGB").resize((224, 224))
    x = img_to_array(img)               # float32 [0..255]
    x = preprocess_input(x)             # EfficientNet scaling (not /255)
    x = np.expand_dims(x, axis=0)       # shape (1,224,224,3)

    # 2) predict
    preds = model.predict(x, verbose=0)[0]   # shape (num_classes,)
    idx = int(np.argmax(preds))

    # 3) map index -> original class name (must be same order as training)
    raw = classes[idx]
    return _pretty_label(raw)


def recommend_crops_from_inputs(N, P, K, pH, area, year, table, k_years=10, top_k=3):
    """
    Recommends crops using soil (N,P,K,pH) + weather (from merged table, actual or 10y trend).
    Returns (list_of_crop_names, weather_used_dict).
    """
    crop_model, crop_feats, defaults = load_crop_model_bundle()
    w = _yearly_weather_from_table(table, area, k_years, year)
    if any(x is None for x in [w["temperature"], w["humidity"], w["rainfall"]]):
        return [], w

    row = dict(defaults)
    row.update({
        "N": float(N), "P": float(P), "K": float(K), "ph": float(pH),
        "temperature": float(w["temperature"]),
        "humidity": float(w["humidity"]),
        "rainfall": float(w["rainfall"]),
    })
    X = pd.DataFrame([row])[crop_feats]

    if hasattr(crop_model, "predict_proba"):
        proba = crop_model.predict_proba(X)[0]
        classes = crop_model.classes_
        order = np.argsort(proba)[::-1]
        top_names = [str(classes[i]) for i in order[:top_k]]
    else:
        pred = crop_model.predict(X)
        top_names = [str(pred[0])]
    return top_names, w

def estimate_yield(area, item, year, table, k_years: int = 10):
    """
    Free-text Area/Item. We resolve them to table values, verify recent history,
    autofill weather/macros from last k rows, then predict with the trained pipeline.
    """
    if table is None or len(table) == 0:
        return "No yield data table available."

    # Resolve user inputs to table names (tolerant)
    area_resolved, area_exact = _resolve_name_from_table(table, "Area", area)
    item_resolved, item_exact = _resolve_name_from_table(table, "Item", item)

    if not area_resolved:
        return f"Unknown area: **{area}** — not found in the dataset."
    if not item_resolved:
        return f"Unknown crop: **{item}** — not found in the dataset."

    # Check (Area, Item) history
    hist = _has_history_for_area_item(table, area_resolved, item_resolved, target_year=int(year), k_years=k_years)
    if not hist["found"]:
        if hist["count"] == 0:
            return f"No historical records for **{item_resolved}** in **{area_resolved}** — cannot estimate yield."
        else:
            return (
                f"No recent data for **{item_resolved}** in **{area_resolved}** "
                f"(historical available {hist['min_year']}–{hist['max_year']}, but not near {int(year)})."
            )

    # Load model + feature config
    try:
        y_model, feat_cfg = load_yield_model_bundle()
        num_cols = feat_cfg["num_cols"]
        cat_cols = feat_cfg["cat_cols"]
    except Exception as e:
        return f"Yield model/config not found: {e}"

    # Ensure model knows this Item category (avoid all-zeros OHE)
    if "Item" in cat_cols:
        if not _item_known_to_model(y_model, "Item", item_resolved):
            return f"The yield model wasn’t trained on crop **{item_resolved}** — cannot estimate yield."

    # Weather from last k rows (prefer crop-specific window)
    w = _yearly_weather_from_table(table, area_resolved, k_years, int(year), item=item_resolved)
    if any(x is None for x in [w.get("temperature"), w.get("humidity"), w.get("rainfall"), w.get("wind")]):
        return "Insufficient weather data to estimate yield."

    # Macro features (nearest-year fallbacks)
    try:
        macros = _macro_features_from_table(table, area_resolved, int(year))
    except Exception as e:
        return f"Macro features unavailable: {e}"

    # Build numeric features exactly as trained
    x_num = {
        "temp_mean_year":     float(w["temperature"]),
        "rain_total_year":    float(w["rainfall"]),
        "humidity_mean_year": float(w["humidity"]),
        "wind_mean_year":     float(w["wind"]),
        "average_rain_fall_mm_per_year": float(macros["average_rain_fall_mm_per_year"]),
        "pesticides_tonnes":  float(macros["pesticides_tonnes"]),
        "avg_temp":           float(macros["avg_temp"]),
    }

    x_cat = {"Area": area_resolved, "Item": item_resolved}

    X = pd.DataFrame([{**x_num, **x_cat}])

    try:
        pred = float(y_model.predict(X[num_cols + cat_cols])[0])
        note = ""
        if not area_exact or not item_exact:
            note = " (names normalized from your input)"
        return (
            f"Estimated yield for **{item_resolved}** in **{area_resolved} {int(year)}**: "
            f"**{int(round(pred, 0)):,} hg/ha** "
            f"(using {w['method']} over {w['k_years']} year(s){note})."
        )
    except Exception:
        return "Unable to estimate yield with the current data."



# ========= UI =========
st.set_page_config(page_title="AI-Driven Agriculture Advisor", layout="wide")
st.title("AI-Driven Agriculture Advisor")
st.caption("Crop recommendation · Plant disease detection · Weather-aware yield check")

tab1, tab2 = st.tabs(["🌱 Crop Recommendation", "🩺 Plant Disease Detection"])

# ---------- TAB 1: CROP ----------
with tab1:
    st.subheader("Soil & Weather-aware recommendation")

    colA, colB = st.columns(2)
    with colA:
        area = st.text_input("Location / Area (e.g., United Kingdom)", value="United Kingdom")
        year = st.number_input("Target year", min_value=1990, max_value=2100, value=2025, step=1)
        N = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0, step=1.0)
        P = st.number_input("Phosphorus (P)", min_value=0.0, value=42.0, step=1.0)
        K = st.number_input("Potassium (K)", min_value=0.0, value=43.0, step=1.0)
        pH = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        k_win = st.slider("Weather history window (years)", 3, 15, 10)

    with colB:
        st.info(
            "Enter **N, P, K, pH, location, year**. "
            "The app pulls/forecasts **temperature, humidity, rainfall** "
            "from the merged dataset (last *k* years) and recommends crops."
        )

    table = load_merged_table()

    if st.button("Recommend Crops"):
        names, weather_used = recommend_crops_from_inputs(N, P, K, pH, area, int(year), table, k_years=int(k_win), top_k=3)
        if not names:
            st.error("Not enough weather data for that area/year. Try another location or year.")
            st.session_state.rec_crops = None
            st.session_state.yield_msg = None
        else:
            st.session_state.rec_crops = names
            st.session_state.yield_msg = None
            st.success("Recommended crop(s):")
            for n in names:
                st.markdown(f"- **{n}**")
    

    # Show current recommendations (persisted)
    if st.session_state.rec_crops:
        st.markdown("---")
        st.subheader("Optional: Check expected yield")

        st.caption("You can type any country/crop (free text). Names are resolved to the dataset.")
        area_y = st.text_input("Country / Area for yield estimate", value=area, key="area_yield")
        crop_y = st.text_input("Crop name for yield estimate", value=st.session_state.rec_crops[0], key="crop_yield")
        k_win_y = st.slider("History window (years) for yield", 3, 15, int(k_win), key="k_win_yield")

        if st.button("Estimate yield"):
            msg = estimate_yield(area_y, crop_y, int(year), table, k_years=int(k_win_y))
            st.session_state.yield_msg = msg


    if st.session_state.yield_msg:
        st.success(st.session_state.yield_msg)

# ---------- TAB 2: DISEASE ----------
with tab2:
    st.subheader("Leaf-image diagnosis")
    img_file = st.file_uploader("Upload a leaf photo (jpg/png)", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        from PIL import Image
        pil = Image.open(img_file)
        st.image(pil, caption="Uploaded image", use_column_width=True)
        if st.button("Analyze disease"):
            try:
                name = predict_disease_name(pil)
                st.success(f"Diagnosis: **{name}**")
            except Exception as e:
                st.error(f"Could not analyze the image: {e}")

st.markdown("---")
st.caption("Models: RandomForest (crop), EfficientNetB0 (disease), GradientBoosting (yield)")
