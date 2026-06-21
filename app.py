"""
House Price Prediction — Streamlit front end.

Layout of this file:
  1. Page config & constants
  2. Data: model files, field bounds, currency rates, property presets
  3. Asset loading (property photos -> base64, cached)
  4. Theme: a single injected CSS block (palette, type, bento layout,
     sharp buttons, dynamic background)
  5. Signature component: floating isometric house (pure CSS, static markup)
  6. Model loading helpers
  7. Page sections: header, presets, form, prediction, footer
"""

import base64
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ===========================================================
# 1. PAGE CONFIG & PATHS
# ===========================================================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "Model")
ASSETS_DIR = os.path.join(APP_DIR, "assets")

# ===========================================================
# 2. DATA
# ===========================================================
MODEL_FILES = {
    "XGBoost (recommended)": "XGBoost.pkl",
    "Random Forest": "Random_Forest.pkl",
    "Linear Regression": "Linear_Regression.pkl",
}

FIELD_BOUNDS = {
    "area": (1650, 16200, 3000, 50),
    "bedrooms": (1, 6, 3, 1),
    "bathrooms": (1, 4, 2, 1),
    "stories": (1, 4, 2, 1),
    "parking": (0, 3, 1, 1),
}

# Approximate, illustrative FX rates — NOT live rates. Update before
# relying on these for anything real.
FX_RATES = {"INR": 1.0, "USD": 0.012, "AED": 0.044}
CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "AED": "AED "}

# Each preset maps to: form field values, the background image to show,
# and the matching dataset filter range used for "comparable listings".
PROPERTY_PRESETS = {
    "studio": {
        "label": "Studio",
        "icon": "▢",
        "image": "studio.jpg",
        "fields": dict(area=1700, bedrooms=1, bathrooms=1, stories=1, parking=0),
        "area_range": (1650, 2400),
    },
    "family_home": {
        "label": "Family House",
        "icon": "⌂",
        "image": "family_home.jpg",
        "fields": dict(area=3500, bedrooms=3, bathrooms=2, stories=2, parking=2),
        "area_range": (2800, 5500),
    },
    "luxury_villa": {
        "label": "Luxury Villa",
        "icon": "◆",
        "image": "luxury_villa.jpg",
        "fields": dict(area=9000, bedrooms=5, bathrooms=4, stories=3, parking=3),
        "area_range": (7000, 16200),
    },
}
DEFAULT_BACKGROUND = "hero_home.jpg"

# ===========================================================
# 3. ASSET LOADING
# ===========================================================
@st.cache_data(show_spinner=False)
def load_image_b64(filename: str) -> str:
    """Read an image from /assets and return a base64 data URI string."""
    path = os.path.join(ASSETS_DIR, filename)
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = "jpeg" if filename.lower().endswith((".jpg", ".jpeg")) else "png"
    return f"data:image/{ext};base64,{encoded}"


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    csv_path = os.path.join(PROJECT_ROOT, "Dataset", "Housing.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


# ===========================================================
# 4. THEME
# ===========================================================
def inject_theme(background_filename: str) -> None:
    """Single CSS block: design tokens, layout, components, and the
    currently active background image (blurred + scrimmed for legibility).
    """
    bg_data_uri = load_image_b64(background_filename)

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap');

        :root {{
            --ink: #15140f;
            --paper: #faf7f2;
            --oak: #a9805c;
            --sage: #5b6b5e;
            --stone: #d8cfbf;
            --white: #ffffff;
            --line: rgba(21, 20, 15, 0.14);
            --font-display: 'Fraunces', Georgia, serif;
            --font-body: 'Inter', -apple-system, sans-serif;
        }}

        .stApp {{
            background-color: var(--ink);
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            z-index: 0;
            background-image:
                linear-gradient(180deg, rgba(21,20,15,0.74) 0%, rgba(21,20,15,0.88) 100%),
                url("{bg_data_uri}");
            background-size: cover;
            background-position: center;
            filter: blur(7px);
            transform: scale(1.04);
            transition: background-image 0.5s ease;
        }}
        .stApp [data-testid="stHeader"] {{
            background: transparent;
        }}
        .block-container {{
            position: relative;
            z-index: 1;
            max-width: 1440px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }}

        section[data-testid="stSidebar"] {{
            background: var(--paper);
            border-right: 1px solid var(--line);
        }}

        h1, h2, h3, h4 {{
            font-family: var(--font-display);
            color: var(--paper);
            letter-spacing: -0.01em;
        }}
        p, span, label, div, li {{
            font-family: var(--font-body);
        }}
        .stApp, .stMarkdown, .stCaption {{
            color: var(--paper);
        }}

        .bento-hero {{
            display: grid;
            grid-template-columns: 1.3fr 1fr;
            gap: 0;
            border: 1px solid rgba(250, 247, 242, 0.22);
            background: rgba(21, 20, 15, 0.32);
            backdrop-filter: blur(2px);
            margin-bottom: 0;
        }}
        .bento-hero__intro {{
            padding: 3.8rem 3.2rem 3.6rem 2.8rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            border-right: 1px solid rgba(250, 247, 242, 0.22);
        }}
        .bento-hero__eyebrow {{
            font-family: var(--font-body);
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--stone);
            margin-bottom: 1.3rem;
        }}
        .bento-hero__title {{
            font-family: var(--font-display);
            font-weight: 500;
            font-size: 2.7rem;
            line-height: 1.15;
            color: var(--paper);
            margin: 0 0 1.4rem 0;
        }}
        .bento-hero__title em {{
            font-style: italic;
            color: var(--oak);
        }}
        .bento-hero__desc {{
            font-family: var(--font-body);
            font-size: 1rem;
            line-height: 1.75;
            color: rgba(250, 247, 242, 0.78);
            max-width: 40ch;
            margin: 0;
        }}
        .bento-hero__house {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2.4rem 1rem 2rem;
            gap: 1.4rem;
        }}
        .bento-hero__house-label {{
            font-family: var(--font-display);
            font-style: italic;
            font-weight: 500;
            font-size: 1.3rem;
            color: var(--paper);
            letter-spacing: 0.01em;
        }}

        .home-icon-wrap {{
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .home-icon {{
            position: relative;
            width: 120px;
            height: 104px;
            animation: home-icon-float 4.5s ease-in-out infinite;
        }}
        @keyframes home-icon-float {{
            0%, 100% {{ transform: translateY(0px); }}
            50%      {{ transform: translateY(-10px); }}
        }}
        .home-icon__roof {{
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 0;
            border-left: 60px solid transparent;
            border-right: 60px solid transparent;
            border-bottom: 46px solid var(--ink);
        }}
        .home-icon__roof::after {{
            content: "";
            position: absolute;
            top: 6px;
            left: -53px;
            width: 0;
            height: 0;
            border-left: 53px solid transparent;
            border-right: 53px solid transparent;
            border-bottom: 40px solid var(--oak);
        }}
        .home-icon__body {{
            position: absolute;
            top: 42px;
            left: 14px;
            width: 92px;
            height: 62px;
            background: var(--paper);
            border: 2px solid var(--ink);
            border-top: none;
        }}
        .home-icon__door {{
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 22px;
            height: 34px;
            background: var(--ink);
        }}
        .home-icon__window {{
            position: absolute;
            top: 13px;
            width: 18px;
            height: 18px;
            background: var(--stone);
            border: 2px solid var(--ink);
        }}
        .home-icon__window--left {{ left: 12px; }}
        .home-icon__window--right {{ right: 12px; }}
        .home-icon__shadow {{
            width: 92px;
            height: 16px;
            margin: 10px auto 0;
            background: radial-gradient(ellipse at center, rgba(0,0,0,0.42) 0%, rgba(0,0,0,0) 72%);
            animation: home-icon-shadow 4.5s ease-in-out infinite;
        }}
        @keyframes home-icon-shadow {{
            0%, 100% {{ transform: scale(1); opacity: 0.85; }}
            50%      {{ transform: scale(0.78); opacity: 0.5; }}
        }}

        div.stButton > button {{
            background: transparent;
            color: var(--paper);
            border: 1.5px solid rgba(250, 247, 242, 0.55);
            border-radius: 0;
            padding: 0.85em 1em;
            font-family: var(--font-body);
            font-weight: 600;
            font-size: 0.92rem;
            letter-spacing: 0.02em;
            transition: background 0.18s ease, color 0.18s ease, border-color 0.18s ease;
            box-shadow: none;
        }}
        div.stButton > button:hover {{
            background: var(--paper);
            color: var(--ink);
            border-color: var(--paper);
        }}
        div.stButton > button:focus-visible {{
            outline: 2px solid var(--oak);
            outline-offset: 2px;
        }}
        div.stButton > button p {{
            font-weight: 600;
            font-size: 0.92rem;
        }}

        .predict-btn-marker + div div.stButton > button {{
            background: var(--oak);
            border-color: var(--oak);
            color: var(--ink);
        }}
        .predict-btn-marker + div div.stButton > button:hover {{
            background: var(--paper);
            color: var(--ink);
            border-color: var(--paper);
        }}

        .property-active div.stButton > button {{
            background: var(--paper);
            color: var(--ink);
            border-color: var(--paper);
        }}

        .section-heading {{
            display: flex;
            align-items: baseline;
            gap: 0.7rem;
            margin: 3rem 0 1.1rem 0;
            padding-top: 1.8rem;
            border-top: 1px solid rgba(250, 247, 242, 0.22);
        }}
        .section-heading__tag {{
            font-family: var(--font-body);
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--ink);
            background: var(--oak);
            padding: 0.3rem 0.6rem;
        }}
        .section-heading__title {{
            font-family: var(--font-display);
            font-weight: 600;
            font-size: 1.5rem;
            color: var(--paper);
            letter-spacing: -0.01em;
        }}
        .section-sub {{
            font-family: var(--font-body);
            font-size: 0.92rem;
            color: rgba(250, 247, 242, 0.65);
            margin: -0.6rem 0 1.3rem 0;
            max-width: 64ch;
        }}
        .panel {{
            background: rgba(250, 247, 242, 0.06);
            border: 1px solid rgba(250, 247, 242, 0.22);
            padding: 1.8rem 2rem;
        }}
        .panel + .panel {{ margin-top: 1.6rem; }}
        .panel__title {{
            font-family: var(--font-body);
            font-weight: 700;
            font-size: 1.05rem;
            color: var(--paper);
            letter-spacing: 0.01em;
            margin: 0 0 0.3rem 0;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.95rem;
        }}
        .panel__caption {{
            font-size: 0.85rem;
            color: rgba(250, 247, 242, 0.6);
            margin-bottom: 1.2rem;
        }}

        .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
            border-radius: 0 !important;
            border-color: var(--line) !important;
        }}
        .stSelectbox label, .stNumberInput label, .stCheckbox label p {{
            color: var(--paper) !important;
            font-weight: 500;
            font-size: 0.85rem;
        }}

        .price-card {{
            background: var(--paper);
            color: var(--ink);
            border-radius: 0;
            padding: 1.4rem 1.7rem;
            border: 1px solid var(--line);
        }}
        .price-card b {{ color: var(--ink); }}

        [data-testid="stMetric"] {{
            background: rgba(250, 247, 242, 0.08);
            border: 1px solid rgba(250, 247, 242, 0.22);
            padding: 0.9rem 1rem;
        }}
        [data-testid="stMetricLabel"] {{ color: var(--stone) !important; }}
        [data-testid="stMetricValue"] {{ color: var(--paper) !important; font-family: var(--font-display); }}

        hr {{ border-color: rgba(250, 247, 242, 0.18) !important; }}

        @media (max-width: 900px) {{
            .bento-hero {{ grid-template-columns: 1fr; }}
            .bento-hero__intro {{
                border-right: none;
                border-bottom: 1px solid rgba(250, 247, 242, 0.22);
                padding: 2rem 1.4rem 1.6rem;
            }}
            .bento-hero__title {{ font-size: 2rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================
# 5. SIGNATURE COMPONENT — floating house icon
# ===========================================================
def render_home_icon() -> str:
    # Single-line HTML: Streamlit's markdown pass can mis-handle multi-line
    # HTML blocks spliced into other HTML, rendering tags as literal text.
    return (
        '<div class="home-icon-wrap"><div class="home-icon">'
        '<div class="home-icon__roof"></div>'
        '<div class="home-icon__body">'
        '<div class="home-icon__window home-icon__window--left"></div>'
        '<div class="home-icon__window home-icon__window--right"></div>'
        '<div class="home-icon__door"></div>'
        '</div></div></div>'
        '<div class="home-icon__shadow"></div>'
    )


# ===========================================================
# 5b. SMALL HTML HELPERS (kept single-line — see render_home_icon note)
# ===========================================================
def render_section_heading(tag: str, title: str) -> str:
    return (
        '<div class="section-heading">'
        f'<span class="section-heading__tag">{tag}</span>'
        f'<span class="section-heading__title">{title}</span>'
        '</div>'
    )
@st.cache_resource
def load_model(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        st.stop()
    return joblib.load(path)


@st.cache_resource
def xgboost_uses_log_target() -> bool:
    flag_path = os.path.join(MODEL_DIR, "xgboost_log_target.flag")
    if os.path.exists(flag_path):
        with open(flag_path) as f:
            return f.read().strip() == "1"
    return False


# ===========================================================
# 7. SESSION STATE DEFAULTS
# ===========================================================
if "active_property" not in st.session_state:
    st.session_state["active_property"] = None  # None = default hero image

# ===========================================================
# THEME (must run before any visible markup, after state is known)
# ===========================================================
active_key = st.session_state["active_property"]
background_file = (
    PROPERTY_PRESETS[active_key]["image"] if active_key else DEFAULT_BACKGROUND
)
inject_theme(background_file)

# ===========================================================
# HEADER — bento hero: intro text + floating house signature
# ===========================================================
st.markdown(
    '<div class="bento-hero">'
    '<div class="bento-hero__intro">'
    '<div class="bento-hero__eyebrow">House Price Prediction</div>'
    '<h1 class="bento-hero__title">Know what a home<br>is really <em>worth</em>.</h1>'
    '<p class="bento-hero__desc">Estimate prices across studios, family houses, '
    'and villas using trained machine learning models — with results in '
    'INR, USD, and AED.</p>'
    '</div>'
    '<div class="bento-hero__house">'
    + render_home_icon() +
    '<div class="bento-hero__house-label">Find Your Home</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ===========================================================
# PROPERTY SELECTOR — sharp, angular preset buttons
# ===========================================================
st.markdown(render_section_heading("01", "Choose a Property Type"), unsafe_allow_html=True)
st.markdown(
    '<p class="section-sub">Pick the profile closest to what you\'re searching for. '
    'It pre-fills typical specs and unlocks pricing, comparables, and a full '
    'breakdown below.</p>',
    unsafe_allow_html=True,
)

p1, p2, p3 = st.columns(3)
preset_cols = list(zip(PROPERTY_PRESETS.items(), [p1, p2, p3]))

for (key, preset), col in preset_cols:
    with col:
        is_active = st.session_state["active_property"] == key
        wrapper_class = "property-active" if is_active else ""
        st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
        button_label = f"{preset['icon']}  {preset['label'].upper()}"
        if st.button(button_label, use_container_width=True, key=f"preset_{key}"):
            st.session_state["active_property"] = key
            for field_name, field_value in preset["fields"].items():
                st.session_state[field_name] = field_value
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# Everything below stays hidden until a property type is chosen, so the
# landing screen shows only the hero and the three buttons.
if active_key is None:
    st.markdown(
        '<p class="section-sub" style="margin-top:1.6rem;">'
        'Select Studio, Family House, or Luxury Villa above to continue.</p>',
        unsafe_allow_html=True,
    )
else:
    preset = PROPERTY_PRESETS[active_key]

    # -------------------------------------------------------
    # MODEL SELECTOR
    # -------------------------------------------------------
    st.markdown(render_section_heading("02", "Prediction Model"), unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel__title">Choose a model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel__caption">XGBoost is the most accurate on this '
        'dataset; Random Forest and Linear Regression are included for comparison.</div>',
        unsafe_allow_html=True,
    )
    model_choice = st.selectbox(
        "Choose a model", list(MODEL_FILES.keys()), label_visibility="collapsed"
    )
    model = load_model(MODEL_FILES[model_choice])
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------
    # PROPERTY DETAILS — full width, two clear sub-sections
    # -------------------------------------------------------
    st.markdown(render_section_heading("03", f"Property Details — {preset['label']}"), unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel__title">Specifications</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        a_min, a_max, a_val, a_step = FIELD_BOUNDS["area"]
        area = st.number_input(
            "Area (sqft)", min_value=a_min, max_value=a_max,
            value=st.session_state.get("area", a_val), step=a_step, key="area",
        )
        b_min, b_max, b_val, b_step = FIELD_BOUNDS["bedrooms"]
        bedrooms = st.number_input(
            "Bedrooms", min_value=b_min, max_value=b_max,
            value=st.session_state.get("bedrooms", b_val), step=b_step, key="bedrooms",
        )

    with c2:
        ba_min, ba_max, ba_val, ba_step = FIELD_BOUNDS["bathrooms"]
        bathrooms = st.number_input(
            "Bathrooms", min_value=ba_min, max_value=ba_max,
            value=st.session_state.get("bathrooms", ba_val), step=ba_step, key="bathrooms",
        )
        s_min, s_max, s_val, s_step = FIELD_BOUNDS["stories"]
        stories = st.number_input(
            "Stories", min_value=s_min, max_value=s_max,
            value=st.session_state.get("stories", s_val), step=s_step, key="stories",
        )

    with c3:
        p_min, p_max, p_val, p_step = FIELD_BOUNDS["parking"]
        parking = st.number_input(
            "Parking Spaces", min_value=p_min, max_value=p_max,
            value=st.session_state.get("parking", p_val), step=p_step, key="parking",
        )
        furnishingstatus = st.selectbox(
            "Furnishing Status", ["furnished", "semi-furnished", "unfurnished"]
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel__title">Access &amp; Amenities</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        mainroad = st.selectbox("Main Road Access", ["yes", "no"])
        guestroom = st.selectbox("Guest Room", ["yes", "no"])
    with d2:
        basement = st.selectbox("Basement", ["yes", "no"])
        hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
    with d3:
        airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
        prefarea = st.selectbox("Preferred Area", ["yes", "no"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel__title">Additional Lifestyle Features</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel__caption">These aren\'t part of the trained ML model\'s '
        'original features, so they\'re applied as a transparent percentage adjustment '
        'on top of the model\'s base prediction — not learned from data.</div>',
        unsafe_allow_html=True,
    )
    f1, f2, f3 = st.columns(3)
    with f1:
        pool = st.checkbox("Swimming Pool")
        garden = st.checkbox("Garden / Lawn")
    with f2:
        gated = st.checkbox("Gated Community")
        solar = st.checkbox("Solar Panels")
    with f3:
        metro = st.checkbox("Near Metro / Transit")
        locality = st.selectbox("Locality Tier", ["Standard", "Premium", "Ultra-Premium"])

    st.markdown('<span class="predict-btn-marker"></span>', unsafe_allow_html=True)
    predict_clicked = st.button("Predict Price", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------
    # COMPARABLE LISTINGS — full width
    # -------------------------------------------------------
    st.markdown(render_section_heading("04", "Comparable Listings"), unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    dataset = load_dataset()
    if not dataset.empty:
        lo, hi = preset["area_range"]
        comparable = dataset[(dataset["area"] >= lo) & (dataset["area"] <= hi)]
        st.markdown(
            f'<div class="panel__title">{len(comparable)} matching listings</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="panel__caption">From the training dataset, '
            f'{lo:,}–{hi:,} sqft, matching the {preset["label"]} profile.</div>',
            unsafe_allow_html=True,
        )
        if not comparable.empty:
            display_cols = [
                c for c in ["area", "bedrooms", "bathrooms", "stories", "price"]
                if c in comparable.columns
            ]
            st.dataframe(
                comparable[display_cols].sort_values("area").head(10),
                use_container_width=True, hide_index=True,
            )
    else:
        st.markdown('<div class="panel__title">Dataset not found</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel__caption">Comparable listings are unavailable.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------
    # PREDICTION RESULT
    # -------------------------------------------------------
    if predict_clicked:
        input_data = pd.DataFrame({
            "area": [area], "bedrooms": [bedrooms], "bathrooms": [bathrooms],
            "stories": [stories], "parking": [parking], "mainroad": [mainroad],
            "guestroom": [guestroom], "basement": [basement],
            "hotwaterheating": [hotwaterheating], "airconditioning": [airconditioning],
            "prefarea": [prefarea], "furnishingstatus": [furnishingstatus],
        })

        try:
            raw_prediction = model.predict(input_data)[0]
            if model_choice.startswith("XGBoost") and xgboost_uses_log_target():
                base_price = np.expm1(raw_prediction)
            else:
                base_price = raw_prediction

            # Transparent heuristic adjustment for lifestyle features
            adjustment = 0.0
            if pool: adjustment += 0.05
            if garden: adjustment += 0.02
            if gated: adjustment += 0.03
            if solar: adjustment += 0.02
            if metro: adjustment += 0.04
            adjustment += {"Standard": 0.0, "Premium": 0.06, "Ultra-Premium": 0.12}[locality]

            final_price_inr = base_price * (1 + adjustment)

            st.markdown(render_section_heading("05", "Prediction Result"), unsafe_allow_html=True)
            st.markdown('<div class="panel">', unsafe_allow_html=True)

            st.markdown(
                '<div class="price-card">'
                f'<b>Base model prediction:</b> ₹ {base_price:,.0f} &nbsp;|&nbsp; '
                f'<b>Lifestyle adjustment:</b> +{adjustment * 100:.1f}%'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="panel__title" style="margin-top:1.4rem;">Final Estimated Price</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            for col, cur in zip([m1, m2, m3], ["INR", "USD", "AED"]):
                converted = final_price_inr * FX_RATES[cur]
                with col:
                    st.metric(label=cur, value=f"{CURRENCY_SYMBOLS[cur]}{converted:,.2f}")

            st.caption(
                "FX rates are static/illustrative, not live — for indicative comparison only."
            )

            with st.expander("View input details"):
                st.dataframe(input_data, use_container_width=True)

            if model_choice.startswith("XGBoost"):
                try:
                    xgb_model = model.named_steps["model"]
                    feature_names = model.named_steps["prep"].get_feature_names_out()
                    importances = pd.Series(
                        xgb_model.feature_importances_, index=feature_names
                    ).sort_values(ascending=True)

                    st.markdown('<div class="panel__title" style="margin-top:1.4rem;">Why this price? (Feature Importance)</div>', unsafe_allow_html=True)
                    st.caption("Global importance from the XGBoost model, not specific to this single prediction.")
                    st.bar_chart(importances)
                except Exception:
                    pass

            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")

# ===========================================================
# FOOTER
# ===========================================================
st.markdown(
    '<div class="section-heading"><span class="section-heading__tag">Info</span>'
    '<span class="section-heading__title">About</span></div>'
    '<p style="color:rgba(250,247,242,0.6); font-size:0.85rem;">'
    'Built with Streamlit, scikit-learn &amp; XGBoost &middot; '
    'Predictions are estimates, not valuations.</p>',
    unsafe_allow_html=True,
)
