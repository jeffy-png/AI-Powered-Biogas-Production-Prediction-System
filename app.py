import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
import time
import io
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── ReportLab imports for PDF (auto-install if missing) ──────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, HRFlowable, KeepTogether)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_OK = True
except ModuleNotFoundError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab", "--quiet"])
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, HRFlowable, KeepTogether)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_OK = True

st.set_page_config(page_title="Biogas Prediction System", layout="wide", page_icon="")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');
    .stApp {
        background: linear-gradient(135deg, #040d14 0%, #0a1f2e 40%, #091a28 70%, #040d14 100%);
        color: #e0f7e9;
        font-family: 'Exo 2', sans-serif;
    }
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,255,136,0.04) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 90%, rgba(0,180,255,0.04) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
    }
    h1 {
        color: #a8ff78 !important;
        font-size: 2.4rem !important;
        font-weight: 900 !important;
        text-align: center;
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 3px !important;
        text-shadow: 0 0 40px rgba(168,255,120,0.5), 0 0 80px rgba(168,255,120,0.2);
        margin-bottom: 0 !important;
    }
    h2, h3 { color: #78ffd6 !important; font-family: 'Orbitron', sans-serif !important; letter-spacing:1px; }
    label { color: #b2dfdb !important; font-weight: 500 !important; }
    hr { border-color: rgba(120,255,214,0.15) !important; }
    p { color: #b2dfdb; }
    .glass-panel {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(120,255,214,0.12);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .section-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #78ffd6;
        opacity: 0.7;
        margin: 0 0 14px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(120,255,214,0.15);
    }

    /* ── DEFAULT CHIP BUTTONS (larger size) ── */
    div.stButton > button {
        background: linear-gradient(180deg,
            rgba(255,255,255,0.13) 0%,
            rgba(120,200,180,0.08) 50%,
            rgba(0,0,0,0.05) 100%) !important;
        color: #b2dfdb !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border: 1px solid rgba(120,255,214,0.28) !important;
        border-bottom: 6px solid rgba(0,0,0,0.6) !important;
        border-right: 2px solid rgba(0,0,0,0.35) !important;
        border-radius: 14px !important;
        padding: 20px 14px !important;
        width: 100% !important;
        min-height: 68px !important;
        margin-bottom: 8px !important;
        cursor: pointer !important;
        transition: all 0.12s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow:
            0 6px 14px rgba(0,0,0,0.55),
            0 2px 0 rgba(255,255,255,0.10) inset,
            0 -2px 0 rgba(0,0,0,0.4) inset,
            2px 0 0 rgba(255,255,255,0.06) inset !important;
        transform: perspective(150px) translateZ(0) translateY(0) !important;
        position: relative !important;
        text-shadow: 0 1px 3px rgba(0,0,0,0.7) !important;
        letter-spacing: 0.3px !important;
        line-height: 1.3 !important;
        word-break: break-word !important;
        white-space: normal !important;
    }
    div.stButton > button:hover {
        border-color: rgba(120,255,214,0.55) !important;
        background: linear-gradient(180deg,
            rgba(120,255,214,0.18) 0%,
            rgba(120,255,214,0.09) 50%,
            rgba(0,0,0,0.05) 100%) !important;
        color: #fff !important;
        box-shadow:
            0 10px 24px rgba(0,0,0,0.65),
            0 2px 0 rgba(255,255,255,0.15) inset,
            0 0 20px rgba(120,255,214,0.12),
            2px 0 0 rgba(255,255,255,0.08) inset !important;
        transform: perspective(150px) translateZ(4px) translateY(-2px) !important;
    }
    div.stButton > button:active {
        border-bottom-width: 2px !important;
        border-bottom-color: rgba(0,0,0,0.5) !important;
        box-shadow:
            0 2px 6px rgba(0,0,0,0.5) inset,
            0 1px 0 rgba(255,255,255,0.06) inset !important;
        transform: perspective(150px) translateZ(0) translateY(3px) !important;
    }

    /* ── SELECTED ANIMAL chip: blinking green glow ── */
    @keyframes greenBlink {
        0%, 100% {
            box-shadow: 0 6px 14px rgba(0,0,0,0.55),
                        0 0 0 2px #a8ff78,
                        0 0 18px rgba(168,255,120,0.6),
                        0 0 36px rgba(168,255,120,0.3),
                        0 2px 0 rgba(255,255,255,0.2) inset;
        }
        50% {
            box-shadow: 0 6px 14px rgba(0,0,0,0.55),
                        0 0 0 2px #a8ff78,
                        0 0 8px rgba(168,255,120,0.25),
                        0 0 14px rgba(168,255,120,0.12),
                        0 2px 0 rgba(255,255,255,0.2) inset;
        }
    }
    @keyframes goldBlink {
        0%, 100% {
            box-shadow: 0 6px 14px rgba(0,0,0,0.55),
                        0 0 0 2px #ffd200,
                        0 0 18px rgba(255,210,0,0.6),
                        0 0 36px rgba(255,210,0,0.3),
                        0 2px 0 rgba(255,255,255,0.2) inset;
        }
        50% {
            box-shadow: 0 6px 14px rgba(0,0,0,0.55),
                        0 0 0 2px #ffd200,
                        0 0 8px rgba(255,210,0,0.25),
                        0 0 14px rgba(255,210,0,0.12),
                        0 2px 0 rgba(255,255,255,0.2) inset;
        }
    }

    .chip-sel-animal > div.stButton > button {
        background: linear-gradient(180deg,
            rgba(168,255,120,0.28) 0%,
            rgba(120,220,80,0.16) 50%,
            rgba(60,140,20,0.10) 100%) !important;
        color: #e8ffe0 !important;
        border: 1.5px solid #a8ff78 !important;
        border-bottom: 6px solid #3a8020 !important;
        border-right: 2px solid #4a9025 !important;
        font-weight: 800 !important;
        text-shadow: 0 0 8px rgba(168,255,120,0.7), 0 1px 2px rgba(0,0,0,0.5) !important;
        animation: greenBlink 1.8s ease-in-out infinite !important;
        transform: perspective(150px) translateZ(2px) translateY(-1px) !important;
    }

    .chip-sel-food > div.stButton > button {
        background: linear-gradient(180deg,
            rgba(255,220,60,0.28) 0%,
            rgba(220,180,20,0.16) 50%,
            rgba(140,100,0,0.10) 100%) !important;
        color: #fffde0 !important;
        border: 1.5px solid #ffd200 !important;
        border-bottom: 6px solid #a07800 !important;
        border-right: 2px solid #c09000 !important;
        font-weight: 800 !important;
        text-shadow: 0 0 8px rgba(255,210,0,0.7), 0 1px 2px rgba(0,0,0,0.5) !important;
        animation: goldBlink 1.8s ease-in-out infinite !important;
        transform: perspective(150px) translateZ(2px) translateY(-1px) !important;
    }

    .cat-hdr-animal {
        background: linear-gradient(90deg, rgba(168,255,120,0.12), transparent);
        border-left: 3px solid #a8ff78;
        color: #a8ff78;
        border-radius: 0 8px 8px 0;
        padding: 8px 14px;
        font-size: 0.82rem;
        font-weight: 800;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
        margin: 12px 0 8px 0;
    }
    .cat-hdr-food {
        background: linear-gradient(90deg, rgba(255,210,0,0.12), transparent);
        border-left: 3px solid #ffd200;
        color: #ffd200;
        border-radius: 0 8px 8px 0;
        padding: 8px 14px;
        font-size: 0.82rem;
        font-weight: 800;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
        margin: 14px 0 8px 0;
    }
    .sel-summary {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(120,255,214,0.18);
        border-radius: 10px;
        padding: 10px 14px;
        margin-top: 8px;
    }
    .tag-a {
        background: rgba(168,255,120,0.12);
        color: #a8ff78;
        border: 1px solid rgba(168,255,120,0.3);
        border-radius: 20px;
        padding: 3px 11px;
        font-size: 0.75rem;
        font-weight: 700;
        margin: 2px;
        display: inline-block;
    }
    .tag-f {
        background: rgba(255,210,0,0.12);
        color: #ffd200;
        border: 1px solid rgba(255,210,0,0.3);
        border-radius: 20px;
        padding: 3px 11px;
        font-size: 0.75rem;
        font-weight: 700;
        margin: 2px;
        display: inline-block;
    }

    /* ── PREDICT button (large, strong 3D) ── */
    .predict-btn > div.stButton > button {
        background: linear-gradient(180deg,
            #d8ffaa 0%,
            #a8ff78 30%,
            #78d652 65%,
            #4a9c28 100%) !important;
        color: #061a06 !important;
        font-weight: 900 !important;
        font-size: 1.2rem !important;
        border: 1.5px solid #c8ff88 !important;
        border-bottom: 7px solid #2e6010 !important;
        border-right: 3px solid #3a7818 !important;
        border-radius: 16px !important;
        padding: 22px 40px !important;
        width: 100% !important;
        min-height: 80px !important;
        box-shadow:
            0 10px 28px rgba(168,255,120,0.5),
            0 3px 0 rgba(255,255,255,0.45) inset,
            0 -3px 0 rgba(0,0,0,0.25) inset,
            3px 0 0 rgba(255,255,255,0.2) inset !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        text-shadow: 0 1px 3px rgba(255,255,255,0.5) !important;
        transform: perspective(250px) translateZ(0) !important;
        transition: all 0.12s cubic-bezier(0.4, 0, 0.2, 1) !important;
        white-space: normal !important;
        word-break: normal !important;
    }
    .predict-btn > div.stButton > button:hover {
        box-shadow:
            0 14px 36px rgba(168,255,120,0.6),
            0 3px 0 rgba(255,255,255,0.55) inset,
            0 0 30px rgba(168,255,120,0.2) !important;
        transform: perspective(250px) translateZ(6px) translateY(-3px) !important;
        background: linear-gradient(180deg, #e8ffbe 0%, #b8ff88 30%, #88e662 65%, #5aac38 100%) !important;
    }
    .predict-btn > div.stButton > button:active {
        border-bottom-width: 3px !important;
        transform: perspective(250px) translateZ(0) translateY(4px) !important;
        box-shadow: 0 4px 10px rgba(168,255,120,0.35) inset !important;
    }

    /* ── DOWNLOAD button ── */
    .dl-btn > div.stDownloadButton > button {
        background: linear-gradient(180deg,
            rgba(120,193,243,0.32) 0%,
            rgba(80,150,220,0.18) 50%,
            rgba(40,100,180,0.12) 100%) !important;
        color: #78c1f3 !important;
        border: 1.5px solid rgba(120,193,243,0.5) !important;
        border-bottom: 5px solid rgba(40,90,160,0.7) !important;
        border-right: 2px solid rgba(60,110,180,0.5) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        min-height: 52px !important;
        padding: 12px 20px !important;
        box-shadow:
            0 6px 16px rgba(120,193,243,0.25),
            0 2px 0 rgba(255,255,255,0.1) inset,
            0 -2px 0 rgba(0,0,0,0.3) inset !important;
        width: 100% !important;
        transform: perspective(150px) translateZ(0) !important;
        transition: all 0.12s ease !important;
        text-shadow: 0 1px 3px rgba(0,0,0,0.6) !important;
    }
    .dl-btn > div.stDownloadButton > button:hover {
        background: linear-gradient(180deg,
            rgba(120,193,243,0.45) 0%,
            rgba(80,150,220,0.28) 50%,
            rgba(40,100,180,0.18) 100%) !important;
        box-shadow: 0 10px 24px rgba(120,193,243,0.4) !important;
        transform: perspective(150px) translateZ(4px) translateY(-2px) !important;
    }
    .dl-btn > div.stDownloadButton > button:active {
        border-bottom-width: 2px !important;
        transform: perspective(150px) translateZ(0) translateY(3px) !important;
    }

    .capacity-box {
        background: rgba(0,0,0,0.25);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        border: 1px solid rgba(120,255,214,0.15);
    }
    .capacity-warn { border-color: rgba(239,71,58,0.5) !important; background: rgba(239,71,58,0.06) !important; }
    .qty-label { color: #78ffd6; font-size: 0.78rem; font-weight: 700; margin-bottom: 2px; }
    .groq-answer {
        background: linear-gradient(135deg, rgba(0,12,30,0.98), rgba(0,22,44,0.95));
        border: 1.5px solid rgba(0,255,170,0.3);
        border-radius: 14px;
        padding: 22px 26px;
        margin-top: 14px;
        font-family: 'Exo 2', sans-serif;
        font-size: 0.92rem;
        line-height: 1.7;
        box-shadow: 0 0 0 1px rgba(0,255,170,0.08), 0 8px 32px rgba(0,0,0,0.5);
        color: #d4f7e8;
    }
    @media print {
        .no-print { display: none !important; }
        .stApp { background: white !important; }
        body { background: white !important; color: black !important; }
    }
    .stSlider > div > div > div > div { background: linear-gradient(90deg, #a8ff78, #78ffd6) !important; }
    .stInfo { background: rgba(120,193,243,0.08) !important; border-color: rgba(120,193,243,0.3) !important; }
    .stSuccess { background: rgba(168,255,120,0.08) !important; }
    .app-subtitle {
        text-align: center;
        color: rgba(120,255,214,0.5);
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 4px;
        margin-top: -4px;
        margin-bottom: 24px;
    }
    .hero-strip {
        background: linear-gradient(90deg, rgba(168,255,120,0.06) 0%, rgba(120,255,214,0.04) 50%, rgba(120,193,243,0.06) 100%);
        border-top: 1px solid rgba(168,255,120,0.12);
        border-bottom: 1px solid rgba(120,255,214,0.08);
        padding: 10px 0;
        margin-bottom: 20px;
        text-align: center;
    }
    .hero-kpi { display: inline-block; padding: 0 28px; border-right: 1px solid rgba(120,255,214,0.15); }
    .hero-kpi:last-child { border-right: none; }
    .hero-kpi-val { font-family:'Orbitron',sans-serif; font-size:1.1rem; font-weight:900; color:#a8ff78; }
    .hero-kpi-lbl { font-size:0.65rem; color:rgba(120,255,214,0.5); letter-spacing:2px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────
DIGESTER_VOLUME_M3        = 50.0
MAX_DAILY_INPUT_KG        = 300.0
MAX_OLR                   = 8.0
GAS_STORAGE_MAX_M3        = 200.0
DUNG_TO_GAS_MIN_M3_PER_KG = 0.04
DUNG_TO_GAS_MAX_M3_PER_KG = 0.05
RECOMMENDED_HRT_MIN_DAYS  = 12
RECOMMENDED_HRT_MAX_DAYS  = 15
TARGET_METHANE_MIN_PCT    = 55
TARGET_METHANE_MAX_PCT    = 60
MODEL_OUTPUT_TO_M3_FACTOR = 0.01
TEMP_OK_MIN = 33.0; TEMP_OK_MAX = 42.0
PH_OK_MIN   = 6.8;  PH_OK_MAX   = 7.4
CH4_OK_MIN  = 55.0; NH3_OK_MAX  = 600.0
OLR_OK_MAX  = 6.0;  CN_OK_MIN   = 15.0; CN_OK_MAX = 35.0

# ── Feedstock properties ──────────────────────────────────
FEEDSTOCK_PROPS = {
    "Cow Dung":          {"moisture": 82, "cn": 24, "nh3_factor": 0.45, "yield": 220},
    "Fish Waste":        {"moisture": 76, "cn":  5, "nh3_factor": 1.10, "yield": 450},
    "Poultry Litter":    {"moisture": 65, "cn":  8, "nh3_factor": 0.90, "yield": 310},
    "Fruit Waste":       {"moisture": 88, "cn": 35, "nh3_factor": 0.20, "yield": 560},
    "Rice Water":        {"moisture": 98, "cn": 12, "nh3_factor": 0.08, "yield": 180},
    "Rice Waste":        {"moisture": 72, "cn": 45, "nh3_factor": 0.10, "yield": 230},
    "Dry Leaf Waste":    {"moisture": 15, "cn": 55, "nh3_factor": 0.03, "yield": 160},
    "Vegetable Waste":   {"moisture": 90, "cn": 25, "nh3_factor": 0.25, "yield": 480},
    "Beans and Pulses":  {"moisture": 70, "cn": 18, "nh3_factor": 0.35, "yield": 420},
}

ANIMAL = {
    "Cow Dung":       "",
    "Fish Waste":     "",
    "Poultry Litter": "",
}
FOOD = {
    "Fruit Waste":     "",
    "Rice Water":      "",
    "Rice Waste":      "",
    "Dry Leaf Waste":  "",
    "Vegetable Waste": "",
    "Beans and Pulses": "",
}

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "biogas_model.pkl"
FEATURES_PATH = APP_DIR / "biogas_features.pkl"

@st.cache_resource(show_spinner="Loading trained biogas model...")
def load_model(model_path: str, features_path: str):
    """Load the trained model and its feature order from the app directory."""
    model_file = Path(model_path)
    features_file = Path(features_path)

    if not model_file.is_file():
        raise FileNotFoundError(
            f"Trained model file was not found: {model_file}. "
            "Place biogas_model.pkl in the same folder as app.py."
        )
    if not features_file.is_file():
        raise FileNotFoundError(
            f"Feature file was not found: {features_file}. "
            "Place biogas_features.pkl in the same folder as app.py."
        )

    loaded_model = joblib.load(model_file)
    loaded_features = joblib.load(features_file)

    if not hasattr(loaded_model, "predict"):
        raise TypeError("biogas_model.pkl does not contain a model with a predict() method.")

    if isinstance(loaded_features, (pd.Index, np.ndarray)):
        loaded_features = loaded_features.tolist()
    elif isinstance(loaded_features, tuple):
        loaded_features = list(loaded_features)

    if not isinstance(loaded_features, list) or not loaded_features:
        raise TypeError(
            "biogas_features.pkl must contain a non-empty list, tuple, NumPy array, "
            "or pandas Index of feature names."
        )
    if not all(isinstance(name, str) and name.strip() for name in loaded_features):
        raise ValueError("The feature file contains empty or non-text feature names.")
    if len(set(loaded_features)) != len(loaded_features):
        raise ValueError("The feature file contains duplicate feature names.")

    # If the estimator records its training feature names, verify their order.
    model_feature_names = getattr(loaded_model, "feature_names_in_", None)
    if model_feature_names is not None:
        model_feature_names = list(model_feature_names)
        if model_feature_names != loaded_features:
            raise ValueError(
                "The feature order in biogas_features.pkl does not match the model's "
                "training feature order. Re-create biogas_features.pkl from the same "
                "training pipeline used to save biogas_model.pkl."
            )

    return loaded_model, loaded_features

MODEL_LOAD_ERROR = None
try:
    model, features = load_model(str(MODEL_PATH), str(FEATURES_PATH))
except Exception as exc:
    model, features = None, None
    MODEL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"

# ── Session state ─────────────────────────────────────────
for k, v in [("sel", set()), ("has_predicted", False), ("pred_value", 0.0),
              ("pred_mode", "not_predicted"), ("cond_score", 50.0),
              ("agent_question", None), ("feedstock_qty", {}),
              ("custom_categories", {}),
              ("ai_answer", ""), ("ai_model_used", ""),
              ("csv_feedstocks", {}), ("print_trigger", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

ACTIVE_FEEDSTOCK_PROPS = FEEDSTOCK_PROPS.copy()
ACTIVE_FEEDSTOCK_PROPS.update(st.session_state.csv_feedstocks)

ACTIVE_ANIMAL = ANIMAL.copy()
ACTIVE_FOOD   = FOOD.copy()
for name, data in st.session_state.custom_categories.items():
    cat, emoji = data
    if cat == "Animal and Aquatic Waste": ACTIVE_ANIMAL[name] = emoji
    else:                                 ACTIVE_FOOD[name]   = emoji

ALL = {**{k: ("Animal and Aquatic Waste", v) for k, v in ACTIVE_ANIMAL.items()},
       **{k: ("Food Waste",               v) for k, v in ACTIVE_FOOD.items()}}

# ── Helpers ───────────────────────────────────────────────
def compute_from_feedstock(sel, qty_dict):
    total_kg = sum(qty_dict.get(item, 0) for item in sel)
    if total_kg == 0 or not sel: return None
    wm = wc = 0.0; tn = 0.0
    for item in sel:
        kg = qty_dict.get(item, 0)
        if kg == 0: continue
        props = ACTIVE_FEEDSTOCK_PROPS[item]
        f = kg / total_kg
        wm += props["moisture"] * f
        wc += props["cn"] * f
        tn += props["nh3_factor"] * kg
    olr = min(total_kg / DIGESTER_VOLUME_M3, MAX_OLR)
    nh3 = min(200 + tn * 3.5, 1000)
    return {
        "moisture":     round(float(np.clip(wm, 50, 95)), 1),
        "cn":           round(float(np.clip(wc, 10, 40)), 1),
        "olr":          round(float(olr), 2),
        "nh3":          round(float(nh3), 1),
        "total_kg":     round(total_kg, 1),
        "capacity_pct": round((total_kg / MAX_DAILY_INPUT_KG) * 100, 1),
        "overloaded":   total_kg > MAX_DAILY_INPUT_KG,
    }

def production_range_from_feed(total_kg):
    return (round(total_kg * DUNG_TO_GAS_MIN_M3_PER_KG, 1),
            round(total_kg * DUNG_TO_GAS_MAX_M3_PER_KG, 1))

def calibrate_prediction_m3(raw_pred, min_m3, max_m3):
    if min_m3 <= 0 and max_m3 <= 0: return 0.0, "no_feed"
    model_m3  = float(raw_pred) * MODEL_OUTPUT_TO_M3_FACTOR
    range_mid = (float(min_m3) + float(max_m3)) / 2.0
    if range_mid <= 0: return 0.0, "no_feed"
    if abs(model_m3 - range_mid) > (0.5 * range_mid):
        pred_m3, mode = range_mid, "midpoint_fallback"
    else:
        pred_m3, mode = (0.5 * model_m3) + (0.5 * range_mid), "blended"
    clipped = float(np.clip(pred_m3, float(min_m3), float(max_m3)))
    if abs(clipped - pred_m3) > 1e-9: mode = "clipped"
    return round(clipped, 2), mode

def fuzzy_score_explain(temp, ph, ch4, nh3):
    penalties = []
    if temp < TEMP_OK_MIN or temp > TEMP_OK_MAX: penalties.append(("temp", abs(temp - 37.0)))
    if ph < PH_OK_MIN   or ph > PH_OK_MAX:       penalties.append(("pH", abs(ph - 7.1) * 10))
    if ch4 < CH4_OK_MIN:                          penalties.append(("CH4", CH4_OK_MIN - ch4))
    if nh3 > NH3_OK_MAX:                          penalties.append(("NH3", nh3 - NH3_OK_MAX))
    penalties.sort(key=lambda x: x[1], reverse=True)
    top = [p[0] for p in penalties[:3]]
    return " | ".join(top) if top else "Balanced"

def apply_ai_safety_guardrails(txt):
    txt = str(txt or "").strip()
    if not txt: return "No recommendation generated. Please retry."
    if any(t in txt.lower() for t in ["kg per m3","ml per m3","dose","dosing","ca(oh)2","nahco3","chemical"]):
        txt += "\n\nSafety note: Verify any chemical dosing with a qualified engineer before application."
    return txt

def _heuristic(t, p, m, a):
    s = 50.0
    s += 15 if TEMP_OK_MIN<=t<=TEMP_OK_MAX else (-20 if t<25 or t>55 else -5)
    s += 15 if PH_OK_MIN<=p<=PH_OK_MAX     else (-20 if p<6.0 or p>8.0 else -5)
    s += 10 if m>=CH4_OK_MIN               else (-15 if m<50 else 0)
    s += 10 if a<400                        else (-20 if a>700 else 0)
    return float(np.clip(s, 0, 100))

def fuzzy_condition(t, p, m, a):
    try:
        T = ctrl.Antecedent(np.arange(20,61,1),'temp')
        P = ctrl.Antecedent(np.arange(5,9.1,0.1),'ph')
        M = ctrl.Antecedent(np.arange(40,81,1),'methane')
        A = ctrl.Antecedent(np.arange(100,1001,1),'ammonia')
        C = ctrl.Consequent(np.arange(0,101,1),'condition',defuzzify_method='centroid')
        T['low']=fuzz.trimf(T.universe,[20,20,33]);T['optimal']=fuzz.trimf(T.universe,[30,37,44]);T['high']=fuzz.trimf(T.universe,[40,60,60])
        P['acidic']=fuzz.trimf(P.universe,[5,5,6.8]);P['neutral']=fuzz.trimf(P.universe,[6.5,7,7.5]);P['alkaline']=fuzz.trimf(P.universe,[7.2,9,9])
        M['low']=fuzz.trimf(M.universe,[40,40,55]);M['good']=fuzz.trimf(M.universe,[50,62,72]);M['high']=fuzz.trimf(M.universe,[68,80,80])
        A['low']=fuzz.trimf(A.universe,[100,100,400]);A['moderate']=fuzz.trimf(A.universe,[300,500,700]);A['high']=fuzz.trimf(A.universe,[600,1000,1000])
        C['poor']=fuzz.trimf(C.universe,[0,0,40]);C['moderate']=fuzz.trimf(C.universe,[25,50,75]);C['good']=fuzz.trimf(C.universe,[60,100,100])
        rules=[
            ctrl.Rule(T['optimal']&P['neutral']&M['good']&A['low'],C['good']),
            ctrl.Rule(T['optimal']&P['neutral']&M['high']&A['low'],C['good']),
            ctrl.Rule(T['optimal']&P['neutral']&M['good']&A['moderate'],C['good']),
            ctrl.Rule(A['high'],C['poor']),
            ctrl.Rule(T['low']&P['neutral']&M['good'],C['moderate']),
            ctrl.Rule(T['high']&P['neutral']&M['good'],C['moderate']),
            ctrl.Rule(T['low']&P['acidic'],C['poor']),
            ctrl.Rule(T['high']&P['alkaline'],C['poor']),
            ctrl.Rule(P['acidic']&M['low'],C['poor']),
            ctrl.Rule(P['acidic']&M['good']&T['optimal'],C['moderate']),
            ctrl.Rule(P['alkaline']&M['low'],C['poor']),
            ctrl.Rule(P['alkaline']&M['good']&T['optimal'],C['moderate']),
            ctrl.Rule(M['low']&T['optimal']&P['neutral'],C['moderate']),
            ctrl.Rule(M['low']&A['moderate'],C['poor']),
            ctrl.Rule(T['optimal']&P['neutral'],C['moderate']),
            ctrl.Rule(T['low']|P['acidic']|M['low'],C['moderate']),
        ]
        sim=ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
        sim.input['temp']=float(np.clip(t,20,60));sim.input['ph']=float(np.clip(p,5,9))
        sim.input['methane']=float(np.clip(m,40,80));sim.input['ammonia']=float(np.clip(a,100,1000))
        sim.compute()
        sc=sim.output.get('condition',None)
        if sc is None or np.isnan(sc): sc=_heuristic(t,p,m,a)
        return round(float(sc),2)
    except:
        return _heuristic(t,p,m,a)

def build_input_df(feats,t,p,o,h,m,mo,c,ag,pr,am,fo,wt,wi):
    if not isinstance(feats, (list, tuple)) or not feats:
        raise ValueError("The model feature list is empty or invalid.")
    d={f:0 for f in feats}
    d["digester_temperature_c"]=t;d["ph_level"]=p;d["organic_loading_rate_kg_m3"]=o
    d["hydraulic_retention_time_days"]=h;d["methane_content_percent"]=m
    d["feedstock_moisture_percent"]=mo;d["carbon_nitrogen_ratio"]=c
    d["agitation_speed_rpm"]=ag;d["digester_pressure_kpa"]=pr;d["ammonia_n_mg_l"]=am
    d["foaming_incident"]=1 if fo=="Yes" else 0
    if f"waste_type_{wt}" in d: d[f"waste_type_{wt}"]=1
    if f"waste_item_{wi}" in d: d[f"waste_item_{wi}"]=1
    return pd.DataFrame([[d[f] for f in feats]],columns=feats)

def chunked_iterable(iterable, size):
    it = list(iterable)
    for i in range(0, len(it), size): yield it[i:i+size]



# ── Groq API ────────────────────────────────────

@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def call_groq(prompt, temperature=0.4, max_tokens=800, json_mode=False):

    client = get_groq_client()

    if client is None:
        return "Groq API key is missing. Check your .env file."

    try:
        response = client.chat.completions.create(
            model=os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-120b"
            ),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **(
                {"response_format": {"type": "json_object"}}
                if json_mode else {}
            )
        )

        answer = response.choices[0].message.content

        if answer and answer.strip():
            return answer.strip()

        return "Groq returned an empty response. Please try again."

    except Exception as e:
        return f"Groq API Error: {type(e).__name__}: {str(e)}"


# ══════════════════════════════════════════════════════════
# BUILD PDF REPORT (ReportLab)
# ══════════════════════════════════════════════════════════
def build_pdf_report(temp, ph, ch4, nh3, olr, moist, agit, pres, cn, hrt,
                     pred_value, has_predicted, cond_score, sel,
                     feedstock_qty, total_feed_kg, capacity_pct,
                     score_drivers, groq_answer, pred_mode):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=16*mm, bottomMargin=16*mm,
        title="Biogas Plant Daily Report"
    )

    GREEN       = colors.HexColor("#16a34a")
    DARK_GREEN  = colors.HexColor("#064e3b")
    LIGHT_GREEN = colors.HexColor("#d1fae5")
    BG_GREEN    = colors.HexColor("#f0fdf4")
    AMBER       = colors.HexColor("#d97706")
    RED         = colors.HexColor("#dc2626")
    GREY_DARK   = colors.HexColor("#374151")
    GREY_MID    = colors.HexColor("#6b7280")
    GREY_LIGHT  = colors.HexColor("#f9fafb")
    WHITE       = colors.white
    BLACK       = colors.black

    status      = "OPTIMAL"  if cond_score >= 70 else ("MODERATE" if cond_score >= 40 else "CRITICAL")
    status_col  = GREEN      if cond_score >= 70 else (AMBER      if cond_score >= 40 else RED)
    pred_txt    = f"{pred_value:.2f} m3/day" if has_predicted else "Not predicted yet"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", fontName="Helvetica-Bold",
                                 fontSize=20, textColor=DARK_GREEN,
                                 alignment=TA_CENTER, spaceAfter=14)
    subtitle_style = ParagraphStyle("Sub", fontName="Helvetica",
                                    fontSize=9, textColor=GREY_MID,
                                    alignment=TA_CENTER, spaceBefore=6, spaceAfter=12)
    section_style = ParagraphStyle("Section", fontName="Helvetica-Bold",
                                   fontSize=11, textColor=DARK_GREEN,
                                   spaceBefore=12, spaceAfter=6)
    body_style    = ParagraphStyle("Body", fontName="Helvetica",
                                   fontSize=9, textColor=GREY_DARK,
                                   leading=14, spaceAfter=4)
    small_style   = ParagraphStyle("Small", fontName="Helvetica",
                                   fontSize=8, textColor=GREY_MID, leading=12)

    story = []

    story.append(Paragraph("BIOGAS PLANT DAILY REPORT", title_style))
    story.append(Paragraph(
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}   |   "
        f"Digester Volume: {DIGESTER_VOLUME_M3:.0f} m3",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=GREEN, spaceAfter=10))

    hero_bg = colors.HexColor("#d1fae5") if cond_score >= 70 else (
              colors.HexColor("#fef3c7") if cond_score >= 40 else
              colors.HexColor("#fee2e2"))
    hero_data = [[
        Paragraph(
            f'<font size="36" color="{status_col.hexval() if hasattr(status_col,"hexval") else "#16a34a"}">'
            f'<b>{pred_txt}</b></font><br/>'
            f'<font size="10" color="#374151">Predicted Daily Biogas Output</font><br/>'
            f'<font size="9" color="{status_col.hexval() if hasattr(status_col,"hexval") else "#16a34a"}">'
            f'<b>{status}   Score: {cond_score:.0f}/100</b></font>',
            ParagraphStyle("Hero", alignment=TA_CENTER, leading=22)
        )
    ]]
    hero_table = Table(hero_data, colWidths=["100%"])
    hero_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), hero_bg),
        ("ROUNDEDCORNERS", [8]),
        ("BOX",         (0,0), (-1,-1), 2, status_col),
        ("TOPPADDING",  (0,0), (-1,-1), 16),
        ("BOTTOMPADDING",(0,0),(-1,-1), 16),
        ("LEFTPADDING", (0,0), (-1,-1), 20),
        ("RIGHTPADDING",(0,0), (-1,-1), 20),
    ]))
    story.append(hero_table)
    story.append(Spacer(1, 12))

    kpis = [
        (f"{temp:.1f} C",     "Temperature"),
        (f"{ph:.1f}",          "pH Level"),
        (f"{ch4:.1f}%",        "Methane CH4"),
        (f"{total_feed_kg:.0f} kg", "Daily Feed"),
    ]
    kpi_cells = []
    for val, lbl in kpis:
        kpi_cells.append(
            Paragraph(
                f'<font size="18"><b>{val}</b></font><br/>'
                f'<font size="8" color="#6b7280">{lbl}</font>',
                ParagraphStyle("KPI", alignment=TA_CENTER, leading=20)
            )
        )
    kpi_table = Table([kpi_cells], colWidths=["25%"]*4)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), BG_GREEN),
        ("BOX",          (0,0), (-1,-1), 1, LIGHT_GREEN),
        ("LINEAFTER",    (0,0), (2,0),   0.5, LIGHT_GREEN),
        ("ROUNDEDCORNERS", [6]),
        ("TOPPADDING",   (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0), (-1,-1), 10),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TEXTCOLOR",    (0,0), (-1,-1), DARK_GREEN),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("PROCESS PARAMETERS", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_GREEN, spaceAfter=6))

    def status_txt(ok, warn_txt="OUT OF RANGE", ok_txt="OK"):
        return ok_txt if ok else warn_txt

    param_rows = [
        ["Parameter", "Value", "Status"],
        ["Temperature",           f"{temp:.1f} C",      status_txt(TEMP_OK_MIN<=temp<=TEMP_OK_MAX)],
        ["pH Level",              f"{ph:.2f}",           status_txt(PH_OK_MIN<=ph<=PH_OK_MAX)],
        ["Methane Content",       f"{ch4:.1f} %",        status_txt(ch4>=CH4_OK_MIN, "LOW")],
        ["Ammonia (NH3)",         f"{nh3:.0f} mg/L",     status_txt(nh3<NH3_OK_MAX, "HIGH")],
        ["Organic Loading Rate",  f"{olr:.2f} kg/m3",   status_txt(olr<=OLR_OK_MAX, "HIGH")],
        ["Hydraulic Retention",   f"{hrt:.0f} days",    status_txt(RECOMMENDED_HRT_MIN_DAYS<=hrt<=RECOMMENDED_HRT_MAX_DAYS, "CHECK")],
        ["Moisture Content",      f"{moist:.1f} %",     "-"],
        ["C/N Ratio",             f"{cn:.1f}",           status_txt(CN_OK_MIN<=cn<=CN_OK_MAX, "CHECK")],
        ["Pressure",              f"{pres:.0f} kPa",    "-"],
        ["Agitation",             f"{agit} RPM",         "-"],
        ["Plant Capacity Used",   f"{capacity_pct:.1f}%", "NORMAL" if capacity_pct<70 else ("HIGH" if capacity_pct<90 else "OVERLOADED")],
        ["Prediction Mode",       pred_mode,             ""],
        ["Score Drivers",         score_drivers,         ""],
    ]

    col_w = [90*mm, 50*mm, 40*mm]
    p_table = Table(param_rows, colWidths=col_w)

    table_style = [
        ("BACKGROUND",   (0,0), (-1,0),  DARK_GREEN),
        ("TEXTCOLOR",    (0,0), (-1,0),  WHITE),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, GREY_LIGHT]),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#e5e7eb")),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("ALIGN",        (1,0), (-1,-1), "CENTER"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ]
    status_map = {
        "OUT OF RANGE": RED, "LOW": AMBER, "HIGH": RED,
        "CHECK": AMBER, "OVERLOADED": RED, "OK": GREEN
    }
    for row_idx, row in enumerate(param_rows[1:], start=1):
        st_val = row[2]
        if st_val in status_map:
            table_style.append(("TEXTCOLOR", (2, row_idx), (2, row_idx), status_map[st_val]))
            table_style.append(("FONTNAME",  (2, row_idx), (2, row_idx), "Helvetica-Bold"))

    p_table.setStyle(TableStyle(table_style))
    story.append(p_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("FEEDSTOCK COMPOSITION", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_GREEN, spaceAfter=6))

    if sel:
        feed_rows = [["Feedstock", "Quantity (kg/day)", "Category"]]
        for item in sorted(sel):
            cat = ALL[item][0] if item in ALL else "Unknown"
            feed_rows.append([item, f"{feedstock_qty.get(item, 0)} kg", cat])
        feed_rows.append(["TOTAL", f"{total_feed_kg:.0f} kg", ""])
    else:
        feed_rows = [["Feedstock", "Quantity (kg/day)", "Category"],
                     ["No feedstock selected", "-", "-"]]

    f_table = Table(feed_rows, colWidths=[90*mm, 55*mm, 35*mm])
    f_style = [
        ("BACKGROUND",   (0,0), (-1,0),   DARK_GREEN),
        ("TEXTCOLOR",    (0,0), (-1,0),   WHITE),
        ("FONTNAME",     (0,0), (-1,0),   "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1),  9),
        ("ROWBACKGROUNDS",(0,1),(-1,-2),  [WHITE, GREY_LIGHT]),
        ("GRID",         (0,0), (-1,-1),  0.4, colors.HexColor("#e5e7eb")),
        ("TOPPADDING",   (0,0), (-1,-1),  5),
        ("BOTTOMPADDING",(0,0), (-1,-1),  5),
        ("LEFTPADDING",  (0,0), (-1,-1),  8),
        ("RIGHTPADDING", (0,0), (-1,-1),  8),
        ("ALIGN",        (1,0), (-1,-1),  "CENTER"),
        ("VALIGN",       (0,0), (-1,-1),  "MIDDLE"),
    ]
    if len(feed_rows) > 1:
        f_style += [
            ("BACKGROUND", (0, len(feed_rows)-1), (-1, len(feed_rows)-1), LIGHT_GREEN),
            ("FONTNAME",   (0, len(feed_rows)-1), (-1, len(feed_rows)-1), "Helvetica-Bold"),
            ("TEXTCOLOR",  (0, len(feed_rows)-1), (-1, len(feed_rows)-1), DARK_GREEN),
        ]
    f_table.setStyle(TableStyle(f_style))
    story.append(f_table)
    story.append(Spacer(1, 16))

    story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_GREEN, spaceAfter=6))
    story.append(Paragraph(
        "Biogas Prediction System  |  For operational guidance only  |  "
        f"Report generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        ParagraphStyle("Footer", fontName="Helvetica", fontSize=7.5,
                       textColor=GREY_MID, alignment=TA_CENTER)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ══════════════════════════════════════════════════════════
# BUILD PLANT HTML
# ══════════════════════════════════════════════════════════
def build_plant_html(temp,ph,ch4,nh3,olr,moist,agit,pres,cn,hrt,
                     predicted,cond_score,has_predicted,foam,feedstock_count,
                     feedstock_qty=None,sel_items=None,total_feed_kg=0,
                     capacity_pct=0,overloaded=False,pred_mode="not_predicted",
                     score_drivers="Balanced"):
    tn=float(np.clip((temp-20)/40,0,1));cn4=float(np.clip((ch4-40)/40,0,1))
    an=float(np.clip((agit-10)/50,0,1));mn=float(np.clip((moist-50)/45,0,1))
    on_=float(np.clip((olr-1)/9,0,1));nn=float(np.clip((nh3-100)/900,0,1))
    health=cond_score/100.0
    cap_pct_norm=float(np.clip(capacity_pct/100,0,1))
    gas_store_pct=float(np.clip(cn4*0.8+cap_pct_norm*0.2,0,1))
    gas_store_m3=round(gas_store_pct*GAS_STORAGE_MAX_M3,1)
    cap_color="#a8ff78" if capacity_pct<70 else("#ffd200" if capacity_pct<90 else "#ef473a")
    cap_label="NORMAL" if capacity_pct<70 else("HIGH LOAD" if capacity_pct<90 else "OVERLOADED!")
    if health>=0.7:   lc,lg,sc_,st_="#38ef7d","#11998e","#a8ff78","OPTIMAL"
    elif health>=0.4: lc,lg,sc_,st_="#ffd200","#f7971e","#ffd200","MODERATE"
    else:             lc,lg,sc_,st_="#ef473a","#cb2d3e","#ef473a","CRITICAL"
    plant_border_color="#ef473a" if overloaded else sc_
    plant_border_w="3" if overloaded else "1"
    ph_c="#38ef7d" if 6.8<=ph<=7.4 else("#ef473a" if ph<6.2 or ph>8.0 else "#ffd200")
    nh3_c="#38ef7d" if nh3<400 else("#ef473a" if nh3>700 else "#ffd200")
    tc="#4fc3f7" if tn<0.35 else("#a8ff78" if tn<0.65 else "#ef473a")
    pres_c="#38ef7d" if 100<=pres<=135 else("#ef473a" if pres>145 else "#ffd200")
    fill_h=int(25+mn*90);fill_y=290-fill_h;br=int(24+cn4*20);th_h=int(tn*60)
    pred_txt=f"{predicted:.1f}m3" if(has_predicted and predicted is not None) else "---"
    spin=max(0.3,3-an*2.7)
    temp_pct=int(tn*100);ch4_pct=int(cn4*100);ph_pct=int((ph-5)/4*100)
    nh3_pct=int(nn*100);olr_pct=int(on_*100);moist_pct=int(mn*100)
    pres_pct=int(((pres-90)/60)*100);agit_pct=int(an*100)
    def bubbles(n=5):
        out=""
        for i in range(n):
            bx=390+(i%3)*22;by=fill_y+fill_h-12-(i//3)*16
            bd=round(max(0.8,3.5-an*2.3)+i*0.28,2);bs=2+(i%3)*2
            out+=f'<circle cx="{bx}" cy="{by}" r="{bs}" fill="{lc}" opacity="0.65" class="bub" style="animation-duration:{bd}s;animation-delay:{i*0.45}s"/>'
        return out
    foam_layer=""
    if foam=="Yes":
        foam_layer=(f'<ellipse cx="390" cy="{fill_y+8}" rx="40" ry="9" fill="rgba(255,255,255,0.22)" class="fwav"/>'
                    f'<ellipse cx="415" cy="{fill_y+4}" rx="28" ry="7" fill="rgba(255,255,255,0.14)" class="fwav2"/>')
    feed_p=""
    for i in range(min(feedstock_count,4)):
        feed_p+=f'<circle cx="{135+i*6}" cy="10" r="4" fill="{lc}" opacity="0.9" class="ffall" style="animation-delay:{i*0.4}s"/>'
    sparks=""
    if has_predicted:
        sparks=(f'<circle cx="680" cy="375" r="4" fill="#ffd200" class="spark" style="animation-delay:0s"/>'
                f'<circle cx="696" cy="365" r="3" fill="{sc_}" class="spark" style="animation-delay:.55s"/>'
                f'<circle cx="688" cy="357" r="3" fill="#78ffd6" class="spark" style="animation-delay:1.1s"/>')
    def level_marks():
        out=""
        for pct,label in [(75,"75%"),(50,"50%"),(25,"25%")]:
            y=290-int(pct/100*120)
            out+=f'<line x1="433" y1="{y}" x2="440" y2="{y}" stroke="#78ffd6" stroke-width="1.5" opacity="0.5"/>'
            out+=f'<text x="444" y="{y+4}" font-size="10" fill="#78a89a" font-family="Exo 2,sans-serif">{label}</text>'
        return out
    def bar(x,y,w,h,pct,col):
        fw=int(pct/100*w)
        return(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h//2}" fill="rgba(0,0,0,0.35)"/>'
               f'<rect x="{x}" y="{y}" width="{fw}" height="{h}" rx="{h//2}" fill="{col}" opacity="0.85"/>')
    bars_svg="".join([
        f'<text x="22" y="{75+i*52}" font-family="Exo 2,sans-serif" font-size="10" fill="{col}" font-weight="600">{lbl}</text>'
        f'{bar(22,78+i*52,148,11,pct,col)}'
        f'<text x="22" y="{97+i*52}" font-family="Orbitron,sans-serif" font-size="10" fill="{col}" font-weight="700">{val}</text>'
        for i,(lbl,pct,col,val) in enumerate([
            ("TEMPERATURE",temp_pct,tc,f"{temp:.1f} C"),
            ("pH LEVEL",ph_pct,ph_c,f"{ph:.1f}  ({'OK' if 6.8<=ph<=7.4 else 'LOW' if ph<6.8 else 'HIGH'})"),
            ("MOISTURE",moist_pct,lc,f"{moist:.1f} %"),
            ("METHANE CH4",ch4_pct,lc,f"{ch4:.1f} %"),
            ("ORG. LOADING",olr_pct,"#78ffd6",f"{olr:.1f} kg/m3"),
            ("AMMONIA NH3",nh3_pct,nh3_c,f"{nh3:.0f} mg/L"),
            ("PRESSURE",pres_pct,pres_c,f"{pres:.0f} kPa"),
            ("AGITATION",agit_pct,"#78c1f3",f"{agit} RPM"),
        ])
    ])
    pred_opacity='1' if has_predicted else '0.2'
    pred_cls='pred-val' if has_predicted else ''
    stroke_w='3.5' if has_predicted else '1'
    box_opacity='1' if has_predicted else '0.4'
    accent=(f'<rect x="12" y="468" width="756" height="142" rx="14" fill="{sc_}" opacity="0.06"/>'
            f'<rect x="12" y="468" width="756" height="4" rx="2" fill="{sc_}" opacity="0.6" filter="url(#glow)"/>') if has_predicted else ''
    arrow=f'<polygon points="418,574 432,560 432,588" fill="{sc_}" opacity="0.75" filter="url(#glow)"/>' if has_predicted else ''
    gs_fill_h=int(gas_store_pct*130);gs_fill_y=280-gs_fill_h
    overload_overlay=""
    if overloaded:
        overload_overlay=(f'<rect x="340" y="130" width="110" height="28" rx="8" fill="#ef473a" opacity="0.9"/>'
                          f'<text x="395" y="149" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="10" fill="white" font-weight="900">OVERLOADED</text>')
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Exo+2:wght@400;600;700&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0}}body{{background:#0a1a22;overflow:hidden}}svg{{display:block;width:100%;height:auto}}
  @keyframes rise{{0%{{transform:translateY(0);opacity:.7}}100%{{transform:translateY(-80px);opacity:0}}}}
  @keyframes spin{{from{{transform:rotate(0deg)}}to{{transform:rotate(360deg)}}}}
  @keyframes ffall{{0%{{transform:translateY(0);opacity:0}}35%{{opacity:.9}}100%{{transform:translateY(65px);opacity:0}}}}
  @keyframes fwav{{0%,100%{{transform:scaleX(1);opacity:.22}}50%{{transform:scaleX(1.1);opacity:.35}}}}
  @keyframes spark{{0%,100%{{opacity:0;transform:scale(0)}}50%{{opacity:1;transform:scale(1)}}}}
  @keyframes dash{{to{{stroke-dashoffset:-22}}}}@keyframes gdash{{to{{stroke-dashoffset:-20}}}}
  @keyframes throb{{0%,100%{{opacity:.4}}50%{{opacity:1}}}}
  @keyframes bpop{{0%,100%{{r:{br}}}50%{{r:{br+6}}}}}
  @keyframes countUp{{from{{opacity:0}}to{{opacity:1}}}}
  @keyframes overloadFlash{{0%,100%{{opacity:0.9}}50%{{opacity:0.3}}}}
  .bub{{animation:rise linear infinite}}.ffall{{animation:ffall 1.6s ease-in infinite}}
  .fwav{{animation:fwav 1.4s ease-in-out infinite}}.fwav2{{animation:fwav 1.1s ease-in-out infinite reverse}}
  .spark{{animation:spark 1.5s ease-in-out infinite}}.throb{{animation:throb 2s ease-in-out infinite}}
  .pipef{{stroke-dasharray:7 5;animation:dash .85s linear infinite}}
  .gasf{{stroke-dasharray:5 7;animation:gdash 1.3s linear infinite}}
  .rotor{{transform-origin:390px 255px;animation:spin {spin}s linear infinite}}
  .pump{{transform-origin:245px 230px;animation:spin 1.9s linear infinite}}
  .bpop{{animation:bpop 2.2s ease-in-out infinite}}.gen-spin{{transform-origin:679px 382px;animation:spin 1.2s linear infinite}}
  .pred-val{{animation:countUp 0.7s ease-out forwards}}.overload-flash{{animation:overloadFlash 0.8s ease-in-out infinite}}
</style></head><body>
<svg viewBox="0 0 780 680" xmlns="http://www.w3.org/2000/svg">
<defs>
  <linearGradient id="bgG" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#091520"/><stop offset="100%" stop-color="#0b1e2e"/></linearGradient>
  <linearGradient id="digG" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#183024"/><stop offset="100%" stop-color="#0c1c14"/></linearGradient>
  <linearGradient id="liqG" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{lc}" stop-opacity="0.85"/><stop offset="100%" stop-color="{lg}" stop-opacity="0.5"/></linearGradient>
  <linearGradient id="tankG" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#0e1e2e"/><stop offset="100%" stop-color="#081420"/></linearGradient>
  <linearGradient id="capG" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="{cap_color}"/><stop offset="100%" stop-color="{cap_color}" stop-opacity="0.5"/></linearGradient>
  <radialGradient id="ballG" cx="35%" cy="30%" r="65%"><stop offset="0%" stop-color="{lc}" stop-opacity="0.65"/><stop offset="100%" stop-color="{lg}" stop-opacity="0.12"/></radialGradient>
  <filter id="glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <filter id="sglow" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <clipPath id="digClip"><rect x="355" y="170" width="80" height="120"/></clipPath>
  <clipPath id="tankClip"><rect x="545" y="148" width="52" height="132"/></clipPath>
</defs>
<rect width="780" height="680" fill="url(#bgG)" rx="14"/>
<rect x="0" y="0" width="780" height="38" fill="rgba(0,0,0,0.55)" rx="14"/>
<rect x="0" y="24" width="780" height="14" fill="rgba(0,0,0,0.55)"/>
<text x="390" y="25" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="15" font-weight="900" fill="{sc_}" filter="url(#glow)">BIOGAS PLANT -- LIVE MONITOR</text>
<line x1="12" y1="37" x2="768" y2="37" stroke="{sc_}" stroke-width="0.8" opacity="0.35"/>
<rect x="12" y="44" width="756" height="415" rx="10" fill="none" stroke="{plant_border_color}" stroke-width="{plant_border_w}" opacity="0.4" class="{'overload-flash' if overloaded else ''}"/>
<rect x="600" y="48" width="160" height="22" rx="11" fill="{cap_color}" opacity="0.18" stroke="{cap_color}" stroke-width="1"/>
<text x="680" y="63" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="8.5" fill="{cap_color}" font-weight="900">{cap_label} {capacity_pct:.0f}%</text>
<rect x="12" y="45" width="160" height="408" rx="10" fill="rgba(0,0,0,0.35)" stroke="rgba(120,255,214,0.15)" stroke-width="1"/>
<text x="92" y="63" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="9" fill="#78ffd6" font-weight="700">PARAMETERS</text>
{bars_svg}
<text x="240" y="60" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="9" fill="#a8ff78" font-weight="700">FEEDSTOCK PIT</text>
<rect x="185" y="68" width="110" height="165" rx="6" fill="url(#digG)" stroke="#2a5a3a" stroke-width="1.5"/>
<rect x="189" y="{int(68+165*(1-mn*0.85)-15)}" width="102" height="{int(mn*140+15)}" fill="url(#liqG)" opacity="0.45" rx="4"/>
<rect x="205" y="115" width="70" height="16" rx="8" fill="rgba(0,0,0,0.5)" stroke="{lc}" stroke-width="1"/>
<rect x="207" y="117" width="{min(66,feedstock_count*11+2)}" height="12" rx="6" fill="{lc}" opacity="0.7"/>
<text x="240" y="127" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="8" fill="white" font-weight="700">{feedstock_count} INPUTS</text>
<text x="240" y="145" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="9" fill="{lc}">{total_feed_kg:.0f} kg/day</text>
<text x="240" y="160" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="9" fill="#78ffd6">OLR: {olr:.1f} kg/m3</text>
<text x="240" y="175" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="9" fill="#78a89a">C/N: {cn:.0f}  Moist: {moist:.0f}%</text>
<rect x="205" y="182" width="70" height="7" rx="3.5" fill="rgba(0,0,0,0.5)"/>
<rect x="205" y="182" width="{min(70,int(cap_pct_norm*70))}" height="7" rx="3.5" fill="{cap_color}" opacity="0.85"/>
<text x="240" y="199" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="8" fill="{cap_color}">Cap: {capacity_pct:.0f}%</text>
<line x1="178" y1="66" x2="302" y2="66" stroke="#2a6a3a" stroke-width="2" stroke-dasharray="4 3" opacity="0.6"/>
<g transform="translate(220,68)">{feed_p}</g>
<circle cx="320" cy="232" r="22" fill="url(#digG)" stroke="{lc}" stroke-width="1.8"/>
<g class="pump"><line x1="320" y1="215" x2="320" y2="249" stroke="{lc}" stroke-width="2.5" opacity="0.7"/><line x1="303" y1="232" x2="337" y2="232" stroke="{lc}" stroke-width="2.5" opacity="0.7"/><line x1="308" y1="220" x2="332" y2="244" stroke="{lc}" stroke-width="2" opacity="0.45"/><line x1="332" y1="220" x2="308" y2="244" stroke="{lc}" stroke-width="2" opacity="0.45"/></g>
<circle cx="320" cy="232" r="5" fill="{lc}" filter="url(#glow)"/>
<text x="320" y="268" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="9" fill="{lc}" opacity="0.85">FEED PUMP</text>
<line x1="295" y1="232" x2="298" y2="232" stroke="{lc}" stroke-width="4" class="pipef" opacity="0.7"/>
<line x1="342" y1="232" x2="355" y2="232" stroke="{lc}" stroke-width="4" class="pipef" opacity="0.7"/>
<ellipse cx="395" cy="342" rx="55" ry="10" fill="rgba(0,0,0,0.4)"/>
<polygon points="358,291 432,291 415,332 375,332" fill="#0f2218" stroke="{lc}" stroke-width="1.5" opacity="0.8"/>
<rect x="355" y="170" width="80" height="121" fill="url(#digG)" stroke="{lc}" stroke-width="2.5"/>
<g clip-path="url(#digClip)">
  <rect x="355" y="{fill_y}" width="80" height="{fill_h}" fill="url(#liqG)" opacity="0.72"/>
  <path d="M355,{fill_y} Q370,{fill_y-5} 395,{fill_y} Q420,{fill_y+5} 435,{fill_y}" fill="none" stroke="{lc}" stroke-width="2.5" opacity="0.7" class="throb"/>
  {bubbles()}{foam_layer}
</g>
<ellipse cx="395" cy="170" rx="50" ry="32" fill="#142a1c" stroke="{lc}" stroke-width="3.5"/>
<ellipse cx="395" cy="167" rx="42" ry="24" fill="rgba(120,255,214,0.05)"/>
<ellipse cx="395" cy="167" rx="{int(18+cn4*25)}" ry="{int(8+cn4*10)}" fill="{lc}" opacity="{0.06+cn4*0.18}" class="throb"/>
<rect x="380" y="144" width="30" height="26" rx="4" fill="#1a3a28" stroke="{lc}" stroke-width="2.5"/>
<text x="395" y="162" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="10" fill="{lc}" font-weight="700">GAS</text>
<rect x="390" y="170" width="10" height="140" fill="#2a5a3a" opacity="0.6"/>
<g class="rotor"><rect x="350" y="251" width="90" height="12" rx="6" fill="{lc}" opacity="0.85" filter="url(#glow)"/><rect x="385" y="232" width="20" height="60" rx="6" fill="{lc}" opacity="0.6" filter="url(#glow)"/><circle cx="395" cy="255" r="8" fill="{lc}" filter="url(#glow)"/></g>
{level_marks()}
<text x="395" y="195" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="10" fill="{sc_}" font-weight="700" opacity="0.85">ANAEROBIC DIGESTER</text>
<text x="395" y="210" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="9" fill="#78a89a" opacity="0.7">CAP: {DIGESTER_VOLUME_M3:.0f} m3</text>
<text x="395" y="320" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="11" fill="#78a89a">Level: {int(mn*100)}%</text>
<rect x="355" y="330" width="80" height="20" rx="10" fill="{sc_}" opacity="0.12" stroke="{sc_}" stroke-width="1.5"/>
<text x="395" y="345" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="10" fill="{sc_}" font-weight="900">{st_}</text>
<rect x="380" y="355" width="30" height="12" rx="6" fill="#1a3a28" stroke="{lc}" stroke-width="1.5" opacity="0.7"/>
<line x1="395" y1="367" x2="395" y2="400" stroke="{lc}" stroke-width="4.5" class="pipef"/>
<rect x="350" y="400" width="90" height="25" rx="8" fill="#0a1a10" stroke="{lc}" stroke-width="1.5" opacity="0.7"/>
<text x="395" y="418" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="10" fill="{lc}" opacity="0.85">SLURRY OUTLET</text>
{overload_overlay}
<line x1="395" y1="144" x2="395" y2="98" stroke="{lc}" stroke-width="3" opacity="0.55"/>
<line x1="395" y1="98" x2="572" y2="98" stroke="{lc}" stroke-width="3" class="gasf" opacity="0.75"/>
<polygon points="498,93 507,98 498,103" fill="{lc}" opacity="0.9"/>
<line x1="572" y1="98" x2="572" y2="148" stroke="{lc}" stroke-width="3" opacity="0.55"/>
<circle cx="572" cy="{148-br}" r="{br}" fill="url(#ballG)" stroke="{lc}" stroke-width="2.5" filter="url(#glow)" class="bpop"/>
<text x="572" y="{148-br+5}" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="11" fill="{lc}" font-weight="900">{ch4:.0f}%</text>
<text x="572" y="{148-br+18}" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="9" fill="{lc}">CH4</text>
<rect x="545" y="148" width="54" height="132" rx="7" fill="url(#tankG)" stroke="{lc}" stroke-width="3"/>
<g clip-path="url(#tankClip)"><rect x="547" y="{gs_fill_y}" width="50" height="{gs_fill_h}" fill="url(#liqG)" opacity="0.55"/></g>
<ellipse cx="572" cy="148" rx="27" ry="9" fill="#0e1e2e" stroke="{lc}" stroke-width="3"/>
<ellipse cx="572" cy="280" rx="27" ry="9" fill="#0e1e2e" stroke="{lc}" stroke-width="2"/>
<text x="572" y="205" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="12" fill="{lc}" opacity="0.9">{gas_store_m3:.0f}</text>
<text x="572" y="218" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="9" fill="#78a89a">m3 stored</text>
<text x="572" y="232" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="9" fill="#78a89a">/ {GAS_STORAGE_MAX_M3:.0f} m3 max</text>
<text x="572" y="247" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="10" fill="#78ffd6" opacity="0.7">GAS STORE</text>
<text x="572" y="260" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="10" fill="{lc}" opacity="0.7">P:{pres:.0f} kPa</text>
<line x1="572" y1="280" x2="572" y2="370" stroke="{lc}" stroke-width="3" opacity="0.5"/>
<line x1="572" y1="370" x2="640" y2="370" stroke="{lc}" stroke-width="3" class="gasf" opacity="0.7"/>
<polygon points="617,365 626,370 617,375" fill="{lc}" opacity="0.85"/>
<rect x="638" y="338" width="100" height="90" rx="10" fill="url(#digG)" stroke="{sc_}" stroke-width="2.5"/>
<text x="688" y="382" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="9" fill="{sc_}" font-weight="900">GENERATOR</text>
<circle cx="688" cy="398" r="12" fill="none" stroke="{sc_}" stroke-width="1.5" opacity="0.35"/>
<circle cx="688" cy="398" r="8" fill="none" stroke="{sc_}" stroke-width="2.5" stroke-dasharray="4 6" class="gen-spin"/>
{sparks}
<line x1="738" y1="382" x2="763" y2="382" stroke="#ffd200" stroke-width="2.5" stroke-dasharray="4 3" style="animation:dash .5s linear infinite"/>
<text x="766" y="390" font-family="Exo 2,sans-serif" font-size="8.5" fill="#ffd200" opacity="0.8">POWER</text>
<text x="718" y="60" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="9" fill="#b2dfdb" opacity="0.8">TEMP</text>
<rect x="712" y="65" width="14" height="70" rx="7" fill="rgba(0,0,0,0.5)" stroke="#3a5a6a" stroke-width="1.2"/>
<rect x="714" y="{65+70-th_h}" width="10" height="{th_h}" rx="5" fill="{tc}" opacity="0.9"/>
<circle cx="719" cy="137" r="9" fill="{tc}" filter="url(#glow)"/>
<text x="719" y="155" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="9" fill="{tc}" font-weight="700">{temp:.0f}C</text>
<text x="719" y="167" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="8" fill="{tc}">{'COLD' if tn<0.35 else 'WARM' if tn<0.65 else 'HOT'}</text>
<circle cx="719" cy="210" r="26" fill="rgba(0,0,0,0.4)" stroke="#3a5a6a" stroke-width="1.4"/>
<path d="M695,220 A26,26 0 0,1 743,220" fill="none" stroke="#1a3a2a" stroke-width="5"/>
<path d="M695,220 A26,26 0 0,1 {int(695+(pres-90)*2.55)},{int(220-abs(pres-120)*0.9)}" fill="none" stroke="{pres_c}" stroke-width="5" stroke-linecap="round"/>
<text x="719" y="208" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="9" fill="{pres_c}" font-weight="700">{pres:.0f}</text>
<text x="719" y="220" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="7" fill="#78a89a">kPa</text>
<text x="719" y="243" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="8.5" fill="{pres_c}">PRESSURE</text>
<text x="719" y="270" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="9" fill="#b2dfdb" opacity="0.8">AGITATION</text>
<rect x="695" y="275" width="48" height="9" rx="4.5" fill="rgba(0,0,0,0.4)"/>
<rect x="695" y="275" width="{int(an*48)}" height="9" rx="4.5" fill="#78c1f3" opacity="0.85"/>
<text x="719" y="295" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="9" fill="#78c1f3" font-weight="700">{agit} RPM</text>
<rect x="12" y="530" width="756" height="140" rx="16" fill="rgba(0,0,0,0.55)" stroke="{sc_}" stroke-width="{stroke_w}" opacity="{box_opacity}"/>
{accent}
<text x="30" y="552" font-family="Orbitron,sans-serif" font-size="15" fill="{sc_}" font-weight="700" opacity="0.8">DAILY BIOGAS OUTPUT</text>
<line x1="320" y1="536" x2="320" y2="665" stroke="{sc_}" stroke-width="1.5" opacity="0.2"/>
<text x="22" y="575" font-family="Exo 2,sans-serif" font-size="12" fill="{sc_}" font-weight="700">Condition: {cond_score:.0f}/100 ({st_}) | {score_drivers}</text>
<text x="22" y="598" font-family="Exo 2,sans-serif" font-size="11.5" fill="#78a89a">T:{temp:.0f}C | pH:{ph:.1f} | CH4:{ch4:.0f}% | NH3:{nh3:.0f}mg/L</text>
<text x="22" y="619" font-family="Exo 2,sans-serif" font-size="11.5" fill="#78a89a">OLR:{olr:.1f}kg/m3 | HRT:{hrt:.0f}d</text>
<text x="22" y="640" font-family="Exo 2,sans-serif" font-size="11.5" fill="#78a89a">Feed:{total_feed_kg:.0f}kg/day | Cap:{capacity_pct:.0f}% | C/N:{cn:.0f}</text>
<text x="22" y="660" font-family="Exo 2,sans-serif" font-size="10.5" fill="#78a89a">Mode: {pred_mode}</text>
<text x="545" y="612" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="52" fill="{sc_}" font-weight="900" filter="url(#sglow)" opacity="{pred_opacity}" class="{pred_cls}">{pred_txt}</text>
<text x="545" y="648" text-anchor="middle" font-family="Exo 2,sans-serif" font-size="18" fill="#78ffd6" font-weight="600" opacity="{pred_opacity}">m3/day</text>
{arrow}
</svg></body></html>"""


# ══════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════
st.markdown("""
<div style="padding: 28px 0 4px 0;">
  <h1>Biogas Prediction System</h1>
  <p class="app-subtitle">AI-POWERED · REAL-TIME · FUZZY LOGIC ENGINE</p>
</div>
""", unsafe_allow_html=True)

# ── Computed values from feedstock selection ───────────────
sel = st.session_state.sel
auto_for_plant = compute_from_feedstock(sel, st.session_state.feedstock_qty) if sel else None
olr   = auto_for_plant["olr"]      if auto_for_plant else 4.0
moist = auto_for_plant["moisture"] if auto_for_plant else 80.0
cn    = auto_for_plant["cn"]       if auto_for_plant else 25.0
nh3   = auto_for_plant["nh3"]      if auto_for_plant else 500.0
total_feed_kg  = auto_for_plant["total_kg"]     if auto_for_plant else 0.0
capacity_pct   = auto_for_plant["capacity_pct"] if auto_for_plant else 0.0
overloaded     = auto_for_plant["overloaded"]   if auto_for_plant else False

if auto_for_plant:
    temp = float(np.clip(33 + 0.05 * auto_for_plant["capacity_pct"], 20.0, 60.0))
    ph   = float(np.clip(6.8 + 0.02 * (cn - 25.0), 5.0, 9.0))
    ch4  = float(np.clip(50 + 0.2 * (moist - 50.0), 40.0, 80.0))
    agit = int(np.clip(20 + 0.2 * auto_for_plant["capacity_pct"], 10, 60))
    hrt  = int(np.clip(20 + 0.1 * auto_for_plant["capacity_pct"], 5, 50))
    pres = float(np.clip(100 + 0.2 * auto_for_plant["capacity_pct"], 90.0, 150.0))
else:
    temp, ph, ch4, agit, hrt, pres = 35.0, 7.0, 60.0, 30, 25, 120.0
foam = "No"
prod_min_m3, prod_max_m3 = production_range_from_feed(total_feed_kg)

a_sel = [i for i in sel if ALL[i][0] == "Animal and Aquatic Waste"]
f_sel = [i for i in sel if ALL[i][0] == "Food Waste"]
if len(a_sel) >= len(f_sel) and a_sel: wtype, witem = "Animal and Aquatic Waste", a_sel[0]
elif f_sel:                            wtype, witem = "Food Waste", f_sel[0]
else:                                  wtype, witem = "Animal and Aquatic Waste", "Cow Dung"

cond = st.session_state.cond_score
status_lbl = "OPTIMAL" if cond >= 70 else ("MODERATE" if cond >= 40 else "CRITICAL")
pred_display = f"{st.session_state.pred_value:.1f}" if st.session_state.has_predicted else "-"

st.markdown(f"""
<div class="hero-strip">
  <div class="hero-kpi"><div class="hero-kpi-val">{len(sel)}</div><div class="hero-kpi-lbl">Feedstocks</div></div>
  <div class="hero-kpi"><div class="hero-kpi-val">{total_feed_kg:.0f} kg</div><div class="hero-kpi-lbl">Daily Input</div></div>
  <div class="hero-kpi"><div class="hero-kpi-val">{pred_display} m3</div><div class="hero-kpi-lbl">Predicted Output</div></div>
  <div class="hero-kpi"><div class="hero-kpi-val">{cond:.0f}/100</div><div class="hero-kpi-lbl">Plant Condition · {status_lbl}</div></div>
  <div class="hero-kpi"><div class="hero-kpi-val">{capacity_pct:.0f}%</div><div class="hero-kpi-lbl">Capacity</div></div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.08, 1], gap="large")

# ════════════════════════════════════════════
# LEFT COLUMN — everything inside this block
# ════════════════════════════════════════════
with col_left:
    st.markdown('<p class="section-title">Feedstock Selection</p>', unsafe_allow_html=True)

    # ── Animal and Aquatic Waste ──────────────────────────
    st.markdown('<div class="cat-hdr-animal">Animal and Aquatic Waste</div>', unsafe_allow_html=True)
    for row in chunked_iterable(ACTIVE_ANIMAL.items(), 3):
        acols = st.columns(3)
        for i, (item, _) in enumerate(row):
            on = item in st.session_state.sel
            with acols[i]:
                st.markdown(f'<div class="{"chip-sel-animal" if on else ""}">', unsafe_allow_html=True)
                if st.button(item, key=f"b_{item}"):
                    st.session_state.sel.discard(item) if on else st.session_state.sel.add(item)
                    if item not in st.session_state.feedstock_qty:
                        st.session_state.feedstock_qty[item] = 20
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    # ── Food and Organic Waste ────────────────────────────
    st.markdown('<div class="cat-hdr-food">Food and Organic Waste</div>', unsafe_allow_html=True)
    for row in chunked_iterable(ACTIVE_FOOD.items(), 3):
        fc = st.columns(3)
        for i, (item, _) in enumerate(row):
            on = item in st.session_state.sel
            with fc[i]:
                st.markdown(f'<div class="{"chip-sel-food" if on else ""}">', unsafe_allow_html=True)
                if st.button(item, key=f"b_{item}"):
                    st.session_state.sel.discard(item) if on else st.session_state.sel.add(item)
                    if item not in st.session_state.feedstock_qty:
                        st.session_state.feedstock_qty[item] = 20
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    # ── Add Custom Waste Section ──────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">Add Custom Waste</p>', unsafe_allow_html=True)

    with st.expander("➕  My waste is not listed above — add it here", expanded=False):

        st.markdown("""
        <div style="background:rgba(120,255,214,0.05);border:1px solid rgba(120,255,214,0.2);
                    border-radius:10px;padding:12px 16px;margin-bottom:14px;">
          <p style="color:#78ffd6;font-size:0.8rem;margin:0;line-height:1.6;">
            Just type the waste name — the AI will automatically estimate all technical
            properties (moisture, C/N ratio, NH3 factor, yield). You can review and
            adjust them before adding.
          </p>
        </div>
        """, unsafe_allow_html=True)

        cw_col1, cw_col2 = st.columns([2, 1])

        with cw_col1:
            custom_waste_name = st.text_input(
                "Waste name",
                placeholder="e.g. Banana peel, Coffee grounds, Sugarcane bagasse...",
                key="custom_waste_name_input",
                label_visibility="visible"
            )

        with cw_col2:
            custom_category = st.selectbox(
                "Category",
                ["Animal and Aquatic Waste", "Food and Organic Waste"],
                key="custom_waste_category"
            )

        # ── AI Estimate button ────────────────────────────
        if st.button(
            "Ask AI to estimate properties",
            key="ai_estimate_btn",
            use_container_width=True
        ):
            if not custom_waste_name.strip():
                st.warning("Please enter a waste name first.")
            elif custom_waste_name.strip() in ACTIVE_FEEDSTOCK_PROPS:
                st.warning("This feedstock already exists in the system.")
            else:
                with st.spinner("AI is estimating properties..."):
                    estimate_prompt = f"""You are an expert biogas engineer.
Estimate realistic biogas-relevant properties for this organic waste:
{custom_waste_name}

Return ONLY one valid JSON object with exactly these keys:
{{
  "moisture": 75,
  "cn": 25,
  "nh3_factor": 0.3,
  "yield": 300,
  "confidence": 5,
  "notes": "Brief explanation of the estimate."
}}

Constraints:
- moisture: number from 5 to 98 (percent)
- cn: number from 5 to 60 (C/N ratio)
- nh3_factor: number from 0.01 to 1.5
- yield: whole number from 100 to 700 (L/kg VS)
- confidence: whole number from 1 to 10
- Use plausible estimates based on the named material.
- Do not include Markdown fences or text outside the JSON.
"""

                    raw = call_groq(
                        estimate_prompt,
                        temperature=0.2,
                        max_tokens=300,
                        json_mode=True
                    )

                    
                    try:
                        if raw.startswith("Groq API Error:") or raw.startswith(
                            "Groq API key is missing"
                        ):
                            st.error(raw)
                            st.session_state.pop("ai_estimated_props", None)
                            st.session_state.pop("ai_estimate_for", None)

                        else:
                            cleaned = raw.strip()

                            cleaned = re.sub(
                                r"^```(?:json)?\s*",
                                "",
                                cleaned,
                                flags=re.IGNORECASE
                            )

                            cleaned = re.sub(r"\s*```$", "", cleaned)

                            props = json.loads(cleaned)

                            if not isinstance(props, dict):
                                raise ValueError(
                                    "AI response must be a JSON object."
                                )

                            required_keys = {
                                "moisture",
                                "cn",
                                "nh3_factor",
                                "yield",
                                "confidence",
                                "notes"
                            }

                            missing = required_keys.difference(props)

                            if missing:
                                raise ValueError(
                                    "Missing JSON properties: "
                                    + ", ".join(sorted(missing))
                                )

                            estimated = {
                                "moisture": float(np.clip(
                                    float(props["moisture"]), 5, 98
                                )),
                                "cn": float(np.clip(
                                    float(props["cn"]), 5, 60
                                )),
                                "nh3_factor": float(np.clip(
                                    float(props["nh3_factor"]), 0.01, 1.5
                                )),
                                "yield": int(np.clip(
                                    int(float(props["yield"])), 100, 700
                                )),
                                "confidence": int(np.clip(
                                    int(float(props["confidence"])), 1, 10
                                )),
                                "notes": str(props["notes"])[:500]
                            }

                            st.session_state["ai_estimated_props"] = estimated
                            st.session_state["ai_estimate_for"] = (
                                custom_waste_name.strip()
                            )

                            for widget_key in (
                                "adj_moisture",
                                "adj_cn",
                                "adj_nh3",
                                "adj_yield"
                            ):
                                st.session_state.pop(widget_key, None)

                            st.success("AI properties estimated successfully.")

                    except (
                        ValueError,
                        TypeError,
                        KeyError,
                        json.JSONDecodeError
                    ) as e:
                        st.error(f"Could not process the AI response: {e}")
                        st.session_state.pop("ai_estimated_props", None)
                        st.session_state.pop("ai_estimate_for", None)


        # ── Show estimated properties + allow override ────
        if (
            "ai_estimated_props" in st.session_state
            and st.session_state.get("ai_estimate_for")
            == custom_waste_name.strip()
        ):
            ep = st.session_state["ai_estimated_props"]
            conf = ep["confidence"]
            conf_color = (
                "#a8ff78" if conf >= 7
                else "#ffd200" if conf >= 4
                else "#ef473a"
            )

            st.markdown(f"""
            <div style="margin:14px 0 6px 0;">
              <span style="font-family:'Orbitron',sans-serif;font-size:0.72rem;color:#78ffd6;
                           font-weight:700;letter-spacing:1px;">AI ESTIMATED PROPERTIES</span>
              <span style="margin-left:10px;background:rgba(120,255,214,0.08);
                           border:1px solid {conf_color};border-radius:12px;
                           padding:2px 10px;font-size:0.72rem;color:{conf_color};font-weight:700;">
                Confidence: {conf}/10
              </span>
            </div>
            <div style="color:#78a89a;font-size:0.78rem;margin-bottom:10px;
                        font-style:italic;">{ep["notes"]}</div>
            """, unsafe_allow_html=True)

            st.markdown("**Review and adjust if needed:**")

            prop_col1, prop_col2 = st.columns(2)
            prop_col3, prop_col4 = st.columns(2)

            with prop_col1:
                st.markdown(
                    '<div class="qty-label">Moisture Content (%)</div>',
                    unsafe_allow_html=True
                )
                adj_moisture = st.slider(
                    "moisture_adj", 5, 98, int(ep["moisture"]),
                    step=1, label_visibility="collapsed", key="adj_moisture"
                )

            with prop_col2:
                st.markdown(
                    '<div class="qty-label">C/N Ratio</div>',
                    unsafe_allow_html=True
                )
                adj_cn = st.slider(
                    "cn_adj", 5, 60, int(ep["cn"]),
                    step=1, label_visibility="collapsed", key="adj_cn"
                )

            with prop_col3:
                st.markdown(
                    '<div class="qty-label">NH3 Factor (0.01–1.5)</div>',
                    unsafe_allow_html=True
                )
                adj_nh3 = st.slider(
                    "nh3_adj", 0.01, 1.5, float(ep["nh3_factor"]),
                    step=0.01, label_visibility="collapsed", key="adj_nh3"
                )

            with prop_col4:
                st.markdown(
                    '<div class="qty-label">Biogas Yield (L/kg VS)</div>',
                    unsafe_allow_html=True
                )
                adj_yield = st.slider(
                    "yield_adj", 100, 700, int(ep["yield"]),
                    step=10, label_visibility="collapsed", key="adj_yield"
                )

            st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
            if st.button(
                f"✅ Add '{custom_waste_name.strip()}' to the system",
                key="confirm_add_waste",
                use_container_width=True
            ):
                name = custom_waste_name.strip()
                category = custom_category

                if name in ACTIVE_FEEDSTOCK_PROPS:
                    st.warning("This feedstock already exists in the system.")
                else:
                    new_props = {
                        "moisture": float(adj_moisture),
                        "cn": float(adj_cn),
                        "nh3_factor": float(adj_nh3),
                        "yield": int(adj_yield)
                    }

                    st.session_state.csv_feedstocks[name] = new_props.copy()
                    emoji = "🆕"

                    if category == "Animal and Aquatic Waste":
                        st.session_state.custom_categories[name] = (
                            "Animal and Aquatic Waste", emoji
                        )
                    else:
                        st.session_state.custom_categories[name] = (
                            "Food Waste", emoji
                        )

                    st.session_state.sel.add(name)
                    st.session_state.feedstock_qty[name] = 20
                    st.session_state.pop("ai_estimated_props", None)
                    st.session_state.pop("ai_estimate_for", None)

                    st.success(
                        f"'{name}' added successfully and selected. "
                        "Adjust its quantity below."
                    )
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Show already-added custom wastes ──────────────
        if st.session_state.custom_categories:
            st.markdown("---")
            st.markdown(
                '<p class="section-title">Your Custom Wastes</p>',
                unsafe_allow_html=True
            )

            for cname, (ccat, _) in list(
                st.session_state.custom_categories.items()
            ):
                ccol1, ccol2 = st.columns([4, 1])

                with ccol1:
                    props = st.session_state.csv_feedstocks.get(cname, {})
                    st.markdown(
                        f'<span style="color:#a8ff78;font-weight:700;">{cname}</span> '
                        f'<span style="color:#78a89a;font-size:0.78rem;">— {ccat} | '
                        f'Moisture: {props.get("moisture", "?")}% | '
                        f'C/N: {props.get("cn", "?")} | '
                        f'Yield: {props.get("yield", "?")} L/kg</span>',
                        unsafe_allow_html=True
                    )

                with ccol2:
                    if st.button("Remove", key=f"rm_{cname}"):
                        st.session_state.custom_categories.pop(cname, None)
                        st.session_state.csv_feedstocks.pop(cname, None)
                        st.session_state.sel.discard(cname)
                        st.session_state.feedstock_qty.pop(cname, None)
                        st.session_state.pop("ai_estimated_props", None)
                        st.session_state.pop("ai_estimate_for", None)
                        st.rerun()

    # ── Quantity sliders ──────────────────────────────────
    if sel:
        st.markdown("---")
        st.markdown('<p class="section-title">Feedstock Quantities (kg/day)</p>', unsafe_allow_html=True)
        total_input = 0.0
        for item in sorted(sel):
            if item not in st.session_state.feedstock_qty:
                st.session_state.feedstock_qty[item] = 20
            st.markdown(f'<div class="qty-label">{item}</div>', unsafe_allow_html=True)
            new_val = st.slider(
                label=f"qty_slider_{item}",
                min_value=0,
                max_value=150,
                value=st.session_state.feedstock_qty.get(item, 20),
                step=5,
                label_visibility="collapsed",
                key=f"qty_{item}"
            )
            st.session_state.feedstock_qty[item] = new_val
            total_input += new_val

        cap_pct  = min((total_input / MAX_DAILY_INPUT_KG) * 100, 100)
        cap_col  = "#a8ff78" if cap_pct < 70 else ("#ffd200" if cap_pct < 90 else "#ef473a")
        cap_stat = "NORMAL" if cap_pct < 70 else ("HIGH" if cap_pct < 90 else "OVERLOADED")
        warn_cls = "capacity-warn" if total_input > MAX_DAILY_INPUT_KG else ""
        st.markdown(f"""
        <div class="capacity-box {warn_cls}">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="color:{cap_col};font-weight:700;font-size:0.82rem;">Plant Capacity</span>
            <span style="color:{cap_col};font-weight:800;font-size:0.82rem;">{cap_stat}</span>
          </div>
          <div style="background:rgba(0,0,0,0.4);border-radius:6px;height:10px;margin-bottom:6px;">
            <div style="width:{cap_pct:.1f}%;background:{cap_col};height:10px;border-radius:6px;transition:width 0.3s;"></div>
          </div>
          <div style="color:#78a89a;font-size:0.76rem;">
            Total: <b style="color:{cap_col};">{total_input:.0f} kg/day</b> /
            Max: <b style="color:#78ffd6;">{MAX_DAILY_INPUT_KG:.0f} kg/day</b> |
            Digester: <b style="color:#78ffd6;">{DIGESTER_VOLUME_M3:.0f} m3</b>
          </div>
        </div>""", unsafe_allow_html=True)

        tags = "".join([
            f'<span class="tag-a">{i} ({st.session_state.feedstock_qty.get(i,0)}kg)</span>'
            if ALL[i][0] == "Animal and Aquatic Waste"
            else f'<span class="tag-f">{i} ({st.session_state.feedstock_qty.get(i,0)}kg)</span>'
            for i in sorted(sel)
        ])
        st.markdown(f'<div class="sel-summary"><b style="color:#78ffd6;">{len(sel)} selected:</b> {tags}</div>',
                    unsafe_allow_html=True)

# ════════════════════════════════════════════
# RIGHT COLUMN
# ════════════════════════════════════════════
with col_right:
    st.markdown('<p class="section-title">Live Plant Monitor</p>', unsafe_allow_html=True)

    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    predict_btn = st.button("Predict Biogas Output", key="predict_main", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        if model is None or features is None:
            st.error("The trained prediction model could not be loaded.")
            if MODEL_LOAD_ERROR:
                st.code(MODEL_LOAD_ERROR)
            st.info(
                f"Expected both files in this folder: {APP_DIR}\n"
                "biogas_model.pkl\nbiogas_features.pkl"
            )
            st.session_state.pred_mode = "model_missing"
        else:
            st.session_state.has_predicted = True

    if st.session_state.has_predicted and model is not None and features is not None:
        try:
            inp = build_input_df(features, temp, ph, olr, hrt, ch4, moist, cn, agit, pres, nh3, foam, wtype, witem)
            if inp.isna().any().any():
                st.error("Invalid input detected.")
                st.session_state.pred_mode = "invalid_input"
            else:
                if "biogas_flow_rate_m3_day" in inp.columns:
                    inp.loc[:, "biogas_flow_rate_m3_day"] = (prod_min_m3 + prod_max_m3) / 2.0
                raw_pred = float(model.predict(inp)[0])
                pred_m3, pred_mode = calibrate_prediction_m3(raw_pred, prod_min_m3, prod_max_m3)
                st.session_state.pred_value = pred_m3
                st.session_state.pred_mode  = pred_mode
                st.session_state.cond_score = fuzzy_condition(temp, ph, ch4, nh3)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.session_state.pred_mode = "prediction_error"

    if total_feed_kg <= 0:
        st.session_state.has_predicted = False
        st.session_state.pred_value    = 0.0
        st.session_state.pred_mode     = "no_feed"

    score_drivers = fuzzy_score_explain(temp, ph, ch4, nh3)

    html_content = build_plant_html(
        temp=temp, ph=ph, ch4=ch4, nh3=nh3, olr=olr, moist=moist,
        agit=agit, pres=pres, cn=cn, hrt=hrt,
        predicted       = st.session_state.pred_value if st.session_state.has_predicted else None,
        cond_score      = st.session_state.cond_score,
        has_predicted   = st.session_state.has_predicted,
        foam            = foam,
        feedstock_count = len(sel),
        feedstock_qty   = st.session_state.feedstock_qty,
        sel_items       = sel,
        total_feed_kg   = total_feed_kg,
        capacity_pct    = capacity_pct,
        overloaded      = overloaded,
        pred_mode       = st.session_state.pred_mode,
        score_drivers   = score_drivers,
    )
    components.html(html_content, height=720, scrolling=False)

    # ── Daily Report — Download Only ──────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">Daily Report</p>', unsafe_allow_html=True)

    if st.session_state.has_predicted or sel:
        try:
            pdf_bytes = build_pdf_report(
                temp, ph, ch4, nh3, olr, moist, agit, pres, cn, hrt,
                st.session_state.pred_value, st.session_state.has_predicted,
                st.session_state.cond_score, sel,
                st.session_state.feedstock_qty, total_feed_kg, capacity_pct,
                score_drivers,
                st.session_state.get("ai_answer", ""),
                st.session_state.pred_mode,
            )

            st.markdown('<div class="dl-btn">', unsafe_allow_html=True)
            st.download_button(
                label="⬇  Download Daily Report (PDF)",
                data=pdf_bytes,
                file_name=f"biogas_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Report generation failed: {e}")

# ══════════════════════════════════════════════════════════
# AI ENGINEER SECTION
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:10px 0 18px 0;">
  <span style="font-family:'Orbitron',sans-serif;font-size:1.25rem;font-weight:900;
               background:linear-gradient(90deg,#a8ff78,#78ffd6,#78c1f3);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               letter-spacing:2px;">
    AI BIOGAS ENGINEER
  </span>
</div>
""", unsafe_allow_html=True)

if sel:
    pred_disp  = f"{st.session_state.pred_value:.1f} m3/day" if st.session_state.has_predicted else "click Predict first"
    score_disp = f"{st.session_state.cond_score:.0f}/100"
    stat_disp  = "OPTIMAL" if st.session_state.cond_score >= 70 else ("MODERATE" if st.session_state.cond_score >= 40 else "CRITICAL")

    center_col = st.columns([1, 3, 1])
    with center_col[1]:
        st.markdown("<p style='color:#78ffd6;font-size:0.78rem;font-weight:700;letter-spacing:1px;margin-bottom:8px;'>QUICK QUESTIONS</p>", unsafe_allow_html=True)
        quick_qs = [
            "How can I increase biogas output?",
            "Why is my condition score low?",
            "What feedstock mix is best?",
            "How to fix pH imbalance?",
            "Explain my current parameters",
            "What is my plant capacity status?",
        ]
        qcols = st.columns(3)
        for i, q in enumerate(quick_qs):
            with qcols[i % 3]:
                if st.button(q, key=f"quick_{i}"):
                    st.session_state.agent_question = q

        st.markdown("<p style='color:#78ffd6;font-size:0.78rem;font-weight:700;letter-spacing:1px;margin:12px 0 6px 0;'>ASK YOUR QUESTION</p>", unsafe_allow_html=True)
        typed = st.text_input(
            "Ask your question",
            placeholder="e.g. How do I fix high ammonia? What C/N ratio should I target?",
            label_visibility="collapsed",
            key="agent_typed"
        )
        if typed.strip():
            st.session_state.agent_question = typed.strip()

    if st.session_state.get("agent_question"):
        question = st.session_state.agent_question
        full_prompt = f"""You are an expert biogas plant engineer with 20 years of experience.
Rules
- Use only the live data provided.
- Be practical, clear and actionable.
- Answer any biogas-related doubt (parameters, troubleshooting, feedstock, safety, output improvement, etc.).

Live Plant Data:
Temperature: {temp:.1f}C | pH: {ph:.1f} | Methane: {ch4:.1f}% | Ammonia: {nh3:.0f} mg/L
OLR: {olr:.1f} kg/m3 | HRT: {hrt} days | Moisture: {moist:.1f}% | C/N: {cn:.1f}
Pressure: {pres:.0f} kPa | Agitation: {agit} RPM | Foaming: {foam}
Total Feedstock: {total_feed_kg:.0f} kg/day | Capacity: {capacity_pct:.0f}% | Digester: {DIGESTER_VOLUME_M3:.0f} m3
Feedstock mix: {', '.join(sorted(sel)) if sel else "Cow Dung"}
Condition Score: {score_disp} ({stat_disp}) | Predicted Output: {pred_disp}

User Question: {question}

Answer in short bullet points + one final 'Immediate Next Step'."""

        if st.session_state.get("agent_answered_question") != question:
            with st.spinner("AI is processing your plant data..."):
                answer = call_groq(full_prompt)
                answer = apply_ai_safety_guardrails(answer)
                st.session_state.ai_answer = answer
                st.session_state.agent_answered_question = question

        st.markdown(
            '<div class="ai-answer"><div style="margin-bottom:12px;">'
            '<span style="color:#78ffd6;font-family:Space Mono,monospace;'
            'font-size:0.72rem;font-weight:700;letter-spacing:1px;">'
            'AI ENGINEER RESPONSE</span></div></div>',
            unsafe_allow_html=True
        )
        st.write(st.session_state.get("ai_answer", ""))

        if st.button("Clear Response", use_container_width=True, key="clear_resp"):
            st.session_state.agent_question = None
            st.session_state.ai_answer = ""
            st.session_state.agent_answered_question = None
            st.rerun()
else:
    st.markdown("""
    <div style="text-align:center;padding:28px;background:rgba(120,255,214,0.04);border:1px dashed rgba(120,255,214,0.2);border-radius:12px;">
      <p style="color:rgba(120,255,214,0.5);margin-top:8px;font-family:'Space Mono',monospace;font-size:0.8rem;letter-spacing:1px;">
        SELECT FEEDSTOCK TO ACTIVATE AI ENGINEER
      </p>
    </div>
    """, unsafe_allow_html=True)