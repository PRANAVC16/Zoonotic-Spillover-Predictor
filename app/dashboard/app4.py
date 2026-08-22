# ============================================================
# SpilloverAI
# Zoonotic Spillover Intelligence Platform
# Final Editorial UI
# ============================================================

from pathlib import Path
import base64
import json

import joblib
import numpy as np
import pandas as pd

import streamlit as st

import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SpilloverAI | KFD",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROJECT ROOT
# ============================================================

def find_project_root():

    current = Path(__file__).resolve().parent

    for folder in [current, *current.parents]:

        if (
            (folder / "data").exists()
            and
            (folder / "notebooks").exists()
        ):
            return folder

    return Path(__file__).resolve().parents[2]


ROOT = find_project_root()

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "outputs" / "models"

ASSETS_DIR = ROOT / "app" / "assets"


# ============================================================
# PATHS
# ============================================================

DASHBOARD_DATA_PATH = (
    DATA_DIR / "dashboard_environment_data.csv"
)

MASTER_FEATURES_PATH = (
    DATA_DIR / "master_features.csv"
)

MODEL_PATH = (
    MODEL_DIR / "kfd_outbreak_model_v2.pkl"
)

METADATA_PATH = (
    MODEL_DIR / "kfd_outbreak_model_v2_metadata.json"
)

LOCAL_HERO_IMAGE = (
    ASSETS_DIR / "western_ghats_hero.jpg"
)


# ============================================================
# IMAGE HANDLING
# ============================================================

def image_to_data_uri(path):

    if not path.exists():
        return None

    suffix = path.suffix.lower()

    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }.get(
        suffix,
        "image/jpeg"
    )

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode()

    return (
        f"data:{mime};base64,{encoded}"
    )


local_hero = image_to_data_uri(
    LOCAL_HERO_IMAGE
)


if local_hero:

    HERO_IMAGE = local_hero

else:

    # Fallback only.
    # Replace with your preferred licensed
    # Western Ghats photograph later.

    HERO_IMAGE = (
        "https://images.unsplash.com/"
        "photo-1500530855697-b586d89ba3ee"
        "?auto=format&fit=crop&w=1800&q=88"
    )


# ============================================================
# DISTRICT COORDINATES
# ============================================================

DISTRICT_COORDINATES = {

    "Shivamogga": (
        13.9299,
        75.5681
    ),

    "Uttara Kannada": (
        14.7937,
        74.6869
    ),

    "Chikkamagaluru": (
        13.3161,
        75.7720
    ),

    "Kodagu": (
        12.4244,
        75.7382
    ),

    "Wayanad": (
        11.6854,
        76.1320
    ),

    "Kannur": (
        11.8745,
        75.3704
    ),

    "North Goa": (
        15.4909,
        73.8278
    ),

    "South Goa": (
        15.1170,
        74.1240
    ),

    "Sindhudurg": (
        16.3492,
        73.5594
    )
}


# ============================================================
# GLOBAL CSS
# ============================================================

st.html(
    """
<style>

/* =========================================================
   DESIGN TOKENS
========================================================= */

:root {

    --ink: #0D141B;
    --paper: #F2EBE4;
    --taupe: #AEA198;
    --grey: #969696;

    --forest: #304E43;
    --forest-soft: #71877C;

    --red: #9F453B;

    --line: rgba(13, 20, 27, 0.13);

}


/* =========================================================
   BASE
========================================================= */

html,
body,
[class*="css"] {

    font-family:
        "Helvetica Neue",
        "Inter",
        "Segoe UI",
        Arial,
        sans-serif;

}


.stApp {

    background:
        var(--taupe);

    color:
        var(--ink);

}


.block-container {

    max-width:
        1420px;

    padding-top:
        2rem;

    padding-bottom:
        5rem;

    padding-left:
        2.5rem;

    padding-right:
        2.5rem;

}


/* =========================================================
   HIDE STREAMLIT CHROME
========================================================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {

    background:
        transparent;

}


/* =========================================================
   STREAMLIT BUTTONS
========================================================= */

.stButton > button {

    border:
        1px solid
        rgba(13,20,27,0.18);

    background:
        var(--paper);

    color:
        var(--ink);

    border-radius:
        999px;

    padding:
        0.55rem
        1.15rem;

    min-height:
        42px;

    font-size:
        13px;

    transition:
        all 0.2s ease;

}


.stButton > button:hover {

    background:
        var(--ink);

    color:
        var(--paper);

    border-color:
        var(--ink);

}


/* =========================================================
   SELECT BOXES
========================================================= */

div[data-baseweb="select"] > div {

    background:
        rgba(242,235,228,0.92);

    border:
        1px solid
        rgba(13,20,27,0.12);

    border-radius:
        12px;

    min-height:
        46px;

}


/* =========================================================
   BRAND BAR
========================================================= */

.brandbar {

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

    padding:
        5px
        3px
        18px
        3px;

}


.brand-left {

    display:
        flex;

    gap:
        11px;

    align-items:
        center;

}


.logo-mark {

    width:
        28px;

    height:
        28px;

    border-radius:
        50%;

    background:
        var(--ink);

    color:
        var(--paper);

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    font-size:
        15px;

}


.brand-name {

    font-size:
        15px;

    font-weight:
        650;

    letter-spacing:
        -0.025em;

}


.brand-meta {

    color:
        rgba(13,20,27,0.60);

    font-size:
        11px;

    letter-spacing:
        0.12em;

    text-transform:
        uppercase;

}


/* =========================================================
   MENU
========================================================= */

.menu-panel {

    background:
        var(--ink);

    color:
        var(--paper);

    border-radius:
        26px;

    padding:
        42px;

    margin-bottom:
        22px;

}


.menu-kicker {

    color:
        #A9B0B3;

    font-size:
        11px;

    text-transform:
        uppercase;

    letter-spacing:
        0.13em;

}


.menu-title {

    font-size:
        clamp(
            3.7rem,
            7vw,
            7.2rem
        );

    line-height:
        0.88;

    letter-spacing:
        -0.065em;

    font-weight:
        500;

    margin-top:
        22px;

    max-width:
        800px;

}


/* =========================================================
   HERO
========================================================= */

.hero {

    position:
        relative;

    display:
        grid;

    grid-template-columns:
        1fr
        1fr;

    min-height:
        650px;

    border-radius:
        28px;

    overflow:
        hidden;

    background:
        var(--paper);

}


.hero-left {

    min-height:
        650px;

    padding:
        46px;

    display:
        flex;

    flex-direction:
        column;

    justify-content:
        space-between;

}


.hero-right {

    min-height:
        650px;

    background-size:
        cover;

    background-position:
        center;

    position:
        relative;

}


.hero-right::after {

    content:
        "";

    position:
        absolute;

    inset:
        0;

    background:
        linear-gradient(
            180deg,
            rgba(0,0,0,0.02),
            rgba(0,0,0,0.20)
        );

}


.hero-logo {

    font-size:
        13px;

    font-weight:
        600;

}


.hero-eyebrow {

    font-size:
        11px;

    text-transform:
        uppercase;

    letter-spacing:
        0.15em;

    color:
        #74716D;

}


.hero-headline {

    position:
        absolute;

    z-index:
        3;

    left:
        50%;

    top:
        49%;

    transform:
        translate(
            -50%,
            -50%
        );

    width:
        min(
            960px,
            82%
        );

    font-size:
        clamp(
            3.5rem,
            6vw,
            6.7rem
        );

    line-height:
        0.89;

    letter-spacing:
        -0.065em;

    text-align:
        center;

    font-weight:
        400;

    color:
        var(--ink);

}


.hero-headline .light-on-image {

    color:
        var(--paper);

}


.hero-copy {

    font-size:
        12px;

    color:
        #555652;

    max-width:
        310px;

    line-height:
        1.65;

}


.hero-footer {

    font-size:
        12px;

    line-height:
        1.8;

}


/* =========================================================
   FLOATING MODEL CARD
========================================================= */

.model-card {

    position:
        absolute;

    z-index:
        5;

    right:
        34px;

    bottom:
        34px;

    width:
        340px;

    border-radius:
        22px;

    background:
        rgba(
            242,
            235,
            228,
            0.96
        );

    backdrop-filter:
        blur(16px);

    padding:
        24px;

    box-shadow:
        0 18px 55px
        rgba(0,0,0,0.23);

}


.card-overline {

    font-size:
        10px;

    letter-spacing:
        0.14em;

    text-transform:
        uppercase;

    color:
        #6F706D;

}


.model-status {

    font-size:
        34px;

    letter-spacing:
        -0.05em;

    margin:
        9px
        0
        15px
        0;

}


.signal-high {

    color:
        var(--red);

}


.signal-low {

    color:
        var(--forest);

}


.model-line {

    height:
        1px;

    background:
        rgba(13,20,27,0.14);

    margin:
        14px
        0;

}


.model-row {

    display:
        flex;

    justify-content:
        space-between;

    font-size:
        11px;

    line-height:
        1.9;

}


/* =========================================================
   FILTER BAR
========================================================= */

.filter-shell {

    margin-top:
        18px;

    margin-bottom:
        22px;

}


.filter-caption {

    font-size:
        10px;

    text-transform:
        uppercase;

    letter-spacing:
        0.13em;

    color:
        rgba(13,20,27,0.60);

}


/* =========================================================
   EDITORIAL SURFACE
========================================================= */

.paper {

    background:
        var(--paper);

    border-radius:
        28px;

    padding:
        50px;

    margin-top:
        22px;

}


/* =========================================================
   SECTIONS
========================================================= */

.section-index {

    font-size:
        10px;

    text-transform:
        uppercase;

    letter-spacing:
        0.15em;

    color:
        #72716D;

    margin-bottom:
        22px;

}


.section-title {

    font-size:
        clamp(
            3.2rem,
            5.2vw,
            5.7rem
        );

    line-height:
        0.91;

    letter-spacing:
        -0.062em;

    font-weight:
        400;

    max-width:
        920px;

    margin-bottom:
        22px;

}


.section-text {

    max-width:
        570px;

    font-size:
        13px;

    line-height:
        1.75;

    color:
        #65645F;

    margin-bottom:
        35px;

}


/* =========================================================
   METRIC STRIP
========================================================= */

.metric-strip {

    display:
        grid;

    grid-template-columns:
        repeat(
            4,
            1fr
        );

    border-top:
        1px solid
        var(--line);

    border-bottom:
        1px solid
        var(--line);

    margin:
        35px
        0;

}


.metric-item {

    padding:
        28px
        24px;

    border-right:
        1px solid
        var(--line);

}


.metric-item:last-child {

    border-right:
        none;

}


.metric-label {

    font-size:
        10px;

    letter-spacing:
        0.13em;

    text-transform:
        uppercase;

    color:
        #77746F;

}


.metric-value {

    font-size:
        37px;

    letter-spacing:
        -0.045em;

    margin-top:
        12px;

}


.metric-unit {

    font-size:
        11px;

    color:
        #83807B;

    margin-top:
        5px;

}


/* =========================================================
   EDITORIAL CARD
========================================================= */

.editorial-card {

    border:
        1px solid
        var(--line);

    border-radius:
        20px;

    padding:
        27px;

    min-height:
        100%;

}


.editorial-card.dark {

    background:
        var(--ink);

    color:
        var(--paper);

    border:
        none;

}


.big-value {

    font-size:
        50px;

    letter-spacing:
        -0.055em;

    margin:
        10px
        0;

}


.small-copy {

    font-size:
        12px;

    line-height:
        1.7;

    color:
        #6D6B67;

}


.dark .small-copy {

    color:
        #A9B0B3;

}


/* =========================================================
   DARK MODEL AREA
========================================================= */

.dark-section {

    background:
        var(--ink);

    color:
        var(--paper);

    border-radius:
        28px;

    padding:
        50px;

    margin-top:
        22px;

}


.dark-section .section-index {

    color:
        #9EA5A8;

}


.dark-section .section-text {

    color:
        #A9B0B3;

}


/* =========================================================
   RESEARCH METHOD
========================================================= */

.method-row {

    display:
        grid;

    grid-template-columns:
        70px
        1fr
        2fr;

    gap:
        20px;

    padding:
        24px
        0;

    border-top:
        1px solid
        var(--line);

}


.method-number {

    font-size:
        11px;

    color:
        #77736E;

}


.method-title {

    font-size:
        23px;

    letter-spacing:
        -0.035em;

}


.method-text {

    font-size:
        12px;

    line-height:
        1.65;

    color:
        #686762;

}


/* =========================================================
   FOOTER
========================================================= */

.site-footer {

    display:
        flex;

    justify-content:
        space-between;

    padding:
        24px
        3px
        4px
        3px;

    font-size:
        10px;

    text-transform:
        uppercase;

    letter-spacing:
        0.11em;

    color:
        rgba(13,20,27,0.65);

}


/* =========================================================
   RESPONSIVE
========================================================= */

@media (
    max-width: 950px
) {

    .hero {

        grid-template-columns:
            1fr;

    }


    .hero-left,
    .hero-right {

        min-height:
            520px;

    }


    .hero-headline {

        width:
            90%;

    }


    .metric-strip {

        grid-template-columns:
            repeat(
                2,
                1fr
            );

    }


    .method-row {

        grid-template-columns:
            50px
            1fr;

    }


    .method-text {

        grid-column:
            2;

    }

}


@media (
    max-width: 600px
) {

    .block-container {

        padding-left:
            1rem;

        padding-right:
            1rem;

    }


    .hero-left {

        padding:
            30px;

    }


    .model-card {

        right:
            20px;

        bottom:
            20px;

        width:
            calc(
                100%
                -
                40px
            );

    }


    .metric-strip {

        grid-template-columns:
            1fr;

    }


    .metric-item {

        border-right:
            none;

        border-bottom:
            1px solid
            var(--line);

    }


    .paper,
    .dark-section {

        padding:
            30px;

    }

}

</style>
"""
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_environment_data():

    if DASHBOARD_DATA_PATH.exists():

        df = pd.read_csv(
            DASHBOARD_DATA_PATH
        )

    elif MASTER_FEATURES_PATH.exists():

        df = pd.read_csv(
            MASTER_FEATURES_PATH
        )

    else:

        raise FileNotFoundError(
            "No processed environmental dataset found."
        )


    df["Year"] = pd.to_numeric(
        df["Year"],
        errors="coerce"
    )


    df = df.dropna(
        subset=["Year"]
    )


    df["Year"] = (
        df["Year"]
        .astype(int)
    )


    # --------------------------------------------------------
    # TEMPERATURE RANGE
    # --------------------------------------------------------

    if (
        "Temperature_Range"
        not in df.columns
        and
        "Max_Temperature"
        in df.columns
        and
        "Min_Temperature"
        in df.columns
    ):

        df["Temperature_Range"] = (

            df["Max_Temperature"]

            -

            df["Min_Temperature"]
        )


    # --------------------------------------------------------
    # ENVIRONMENTAL INDEX
    # --------------------------------------------------------

    if (
        "Environmental_Risk_Score"
        not in df.columns
    ):

        required = [

            "Temperature",
            "Rainfall",
            "NDVI",
            "Forest_Loss"
        ]


        if all(
            col in df.columns
            for col in required
        ):

            df[
                "Environmental_Risk_Score"
            ] = (

                0.30
                *
                df["Temperature"].rank(
                    pct=True
                )

                +

                0.25
                *
                df["Rainfall"].rank(
                    pct=True
                )

                +

                0.20
                *
                (
                    1
                    -
                    df["NDVI"].rank(
                        pct=True
                    )
                )

                +

                0.25
                *
                df["Forest_Loss"].rank(
                    pct=True
                )

            ) * 100


    if (
        "Environmental_Risk_Level"
        not in df.columns
    ):

        def risk_level(score):

            if pd.isna(score):
                return "Unavailable"

            if score < 33:
                return "Low"

            if score < 66:
                return "Moderate"

            return "High"


        df[
            "Environmental_Risk_Level"
        ] = (

            df[
                "Environmental_Risk_Score"
            ].apply(
                risk_level
            )
        )


    # --------------------------------------------------------
    # COORDINATES
    # --------------------------------------------------------

    df["Latitude"] = (

        df["District"]
        .map(
            lambda district:
                DISTRICT_COORDINATES
                .get(
                    district,
                    (
                        np.nan,
                        np.nan
                    )
                )[0]
        )
    )


    df["Longitude"] = (

        df["District"]
        .map(
            lambda district:
                DISTRICT_COORDINATES
                .get(
                    district,
                    (
                        np.nan,
                        np.nan
                    )
                )[1]
        )
    )


    return df


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if MODEL_PATH.exists():

        return joblib.load(
            MODEL_PATH
        )

    return None


@st.cache_data
def load_metadata():

    if METADATA_PATH.exists():

        with open(
            METADATA_PATH,
            "r"
        ) as file:

            return json.load(
                file
            )

    return {}


try:

    environment_df = (
        load_environment_data()
    )

except Exception as error:

    st.error(
        "Unable to load environmental data."
    )

    st.exception(
        error
    )

    st.stop()


kfd_model = load_model()

model_metadata = load_metadata()


# ============================================================
# HELPERS
# ============================================================

def fmt(
    value,
    decimals=2
):

    if pd.isna(value):

        return "N/A"

    return (
        f"{float(value):.{decimals}f}"
    )


def predict_kfd(row):

    if kfd_model is None:

        return (
            None,
            None
        )


    features = (
        model_metadata.get(
            "features",
            [
                "Temperature",
                "Rainfall",
                "NDVI",
                "Temperature_Range"
            ]
        )
    )


    missing = [

        feature

        for feature in features

        if feature
        not in row.index
    ]


    if missing:

        return (
            None,
            None
        )


    X_input = pd.DataFrame(
        [
            {
                feature:
                    row[feature]

                for feature
                in features
            }
        ]
    )


    pred = int(

        kfd_model
        .predict(
            X_input
        )[0]

    )


    score = None


    if hasattr(
        kfd_model,
        "predict_proba"
    ):

        score = float(

            kfd_model
            .predict_proba(
                X_input
            )[0, 1]

        )


    return (
        pred,
        score
    )


def style_chart(
    fig,
    height=440,
    dark=False
):

    font_color = (
        "#F2EBE4"
        if dark
        else "#0D141B"
    )


    grid_color = (
        "rgba(242,235,228,0.12)"
        if dark
        else "rgba(13,20,27,0.08)"
    )


    fig.update_layout(

        height=height,

        paper_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        plot_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        font=dict(
            color=font_color,
            size=12
        ),

        title=dict(
            font=dict(
                size=17
            ),
            x=0
        ),

        margin=dict(
            l=15,
            r=15,
            t=55,
            b=25
        ),

        legend_title_text=""
    )


    fig.update_xaxes(

        showgrid=False,

        zeroline=False
    )


    fig.update_yaxes(

        gridcolor=grid_color,

        zeroline=False
    )


    return fig


def section_intro(
    index,
    title,
    text,
    dark=False
):

    wrapper = (
        "dark-section"
        if dark
        else "paper"
    )


    st.html(
        f"""
<div class="{wrapper}">

    <div class="section-index">
        {index}
    </div>

    <div class="section-title">
        {title}
    </div>

    <div class="section-text">
        {text}
    </div>

</div>
        """
    )


# ============================================================
# APP STATE
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = (
        "Overview"
    )


if "menu_open" not in st.session_state:

    st.session_state.menu_open = (
        False
    )


def go_to(page_name):

    st.session_state.page = (
        page_name
    )

    st.session_state.menu_open = (
        False
    )


# ============================================================
# BRAND BAR
# ============================================================

brand_col, button_col = (

    st.columns(
        [8, 1]
    )

)


with brand_col:

    st.html(
        """
<div class="brandbar">

    <div class="brand-left">

        <div class="logo-mark">
            ◐
        </div>

        <div class="brand-name">
            SpilloverAI
        </div>

    </div>

    <div class="brand-meta">

        Zoonotic Environmental
        Intelligence

    </div>

</div>
        """
    )


with button_col:

    if st.button(
        "☰",
        use_container_width=True,
        key="menu_toggle"
    ):

        st.session_state.menu_open = (
            not
            st.session_state.menu_open
        )

        st.rerun()


# ============================================================
# MENU PANEL
# ============================================================

if st.session_state.menu_open:

    st.html(
        """
<div class="menu-panel">

    <div class="menu-kicker">
        Navigate SpilloverAI
    </div>

    <div class="menu-title">
        Explore the
        research.
    </div>

</div>
        """
    )


    m1, m2, m3 = (
        st.columns(3)
    )


    with m1:

        if st.button(
            "Overview",
            use_container_width=True
        ):

            go_to(
                "Overview"
            )

            st.rerun()


        if st.button(
            "Climate",
            use_container_width=True
        ):

            go_to(
                "Climate"
            )

            st.rerun()


    with m2:

        if st.button(
            "Vegetation",
            use_container_width=True
        ):

            go_to(
                "Vegetation"
            )

            st.rerun()


        if st.button(
            "Forest & Terrain",
            use_container_width=True
        ):

            go_to(
                "Forest & Terrain"
            )

            st.rerun()


    with m3:

        if st.button(
            "KFD Model",
            use_container_width=True
        ):

            go_to(
                "KFD Model"
            )

            st.rerun()


        if st.button(
            "Research",
            use_container_width=True
        ):

            go_to(
                "Research"
            )

            st.rerun()


    st.markdown(
        "<div style='height:25px'></div>",
        unsafe_allow_html=True
    )


# ============================================================
# FILTERS
# ============================================================

districts = sorted(

    environment_df[
        "District"
    ]
    .dropna()
    .unique()

)


filter_1, filter_2, filter_space = (

    st.columns(
        [
            1.4,
            1,
            4
        ]
    )

)


with filter_1:

    selected_district = (

        st.selectbox(
            "District",
            districts
        )

    )


district_df = (

    environment_df[
        environment_df[
            "District"
        ]
        ==
        selected_district
    ]

    .sort_values(
        "Year"
    )

    .copy()

)


available_years = sorted(

    district_df[
        "Year"
    ]
    .unique(),

    reverse=True

)


with filter_2:

    selected_year = (

        st.selectbox(
            "Environmental year",
            available_years
        )

    )


selected_df = (

    district_df[
        district_df[
            "Year"
        ]
        ==
        selected_year
    ]

)


if selected_df.empty:

    st.error(
        "No record exists for the selected district-year."
    )

    st.stop()


row = selected_df.iloc[0]


# ============================================================
# VALUES
# ============================================================

temperature = row.get(
    "Temperature",
    np.nan
)

rainfall = row.get(
    "Rainfall",
    np.nan
)

ndvi = row.get(
    "NDVI",
    np.nan
)

forest_loss = row.get(
    "Forest_Loss",
    np.nan
)

elevation = row.get(
    "Elevation",
    np.nan
)

temperature_range = row.get(
    "Temperature_Range",
    np.nan
)

environment_score = row.get(
    "Environmental_Risk_Score",
    np.nan
)

environment_level = row.get(
    "Environmental_Risk_Level",
    "Unavailable"
)


prediction, model_score = (
    predict_kfd(
        row
    )
)


if prediction == 1:

    signal = "Elevated"

    signal_css = (
        "signal-high"
    )


elif prediction == 0:

    signal = "Lower"

    signal_css = (
        "signal-low"
    )


else:

    signal = "Unavailable"

    signal_css = ""


score_text = (

    fmt(
        model_score,
        3
    )

    if model_score
    is not None

    else "N/A"
)


# ============================================================
# HERO
# ============================================================

def render_hero():

    st.html(
        f"""
<div class="hero">

    <div class="hero-left">

        <div>

            <div class="hero-logo">
                ◐ SpilloverAI
            </div>

        </div>


        <div>

            <div class="hero-eyebrow">
                KFD /
                Western Ghats
            </div>

            <div class="hero-copy">

                Satellite,
                climate and machine-learning
                intelligence for understanding
                environmental conditions
                associated with Kyasanur
                Forest Disease.

            </div>

        </div>


        <div class="hero-footer">

            {selected_district}<br>

            {selected_year}
            environmental observation

        </div>

    </div>


    <div
        class="hero-right"
        style="
            background-image:
            url('{HERO_IMAGE}');
        "
    >

        <div class="model-card">

            <div class="card-overline">
                AI outbreak signal
            </div>


            <div
                class="
                    model-status
                    {signal_css}
                "
            >

                {signal}

            </div>


            <div class="model-line">
            </div>


            <div class="model-row">

                <span>
                    Model score
                </span>

                <strong>
                    {score_text}
                </strong>

            </div>


            <div class="model-row">

                <span>
                    District
                </span>

                <strong>
                    {selected_district}
                </strong>

            </div>


            <div class="model-row">

                <span>
                    Model
                </span>

                <strong>
                    Dynamic RF V2
                </strong>

            </div>

        </div>

    </div>


    <div class="hero-headline">

        Reading the
        <span class="light-on-image">
            environment
        </span>

        before disease
        emerges.

    </div>

</div>
        """
    )


# ============================================================
# PAGE
# ============================================================

page = st.session_state.page


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    render_hero()


    st.html(
        f"""
<div class="paper">

    <div class="section-index">
        01 / Environmental overview
    </div>

    <div class="section-title">

        Four signals.
        One ecological
        picture.

    </div>

    <div class="section-text">

        SpilloverAI integrates
        dynamic climate and satellite
        observations to describe
        environmental conditions
        across KFD-relevant
        Western Ghats districts.

    </div>


    <div class="metric-strip">

        <div class="metric-item">

            <div class="metric-label">
                Temperature
            </div>

            <div class="metric-value">
                {fmt(temperature,1)}°
            </div>

            <div class="metric-unit">
                Mean annual °C
            </div>

        </div>


        <div class="metric-item">

            <div class="metric-label">
                Rainfall
            </div>

            <div class="metric-value">
                {fmt(rainfall,0)}
            </div>

            <div class="metric-unit">
                Annual mm
            </div>

        </div>


        <div class="metric-item">

            <div class="metric-label">
                Vegetation
            </div>

            <div class="metric-value">
                {fmt(ndvi,3)}
            </div>

            <div class="metric-unit">
                NDVI
            </div>

        </div>


        <div class="metric-item">

            <div class="metric-label">
                Environmental index
            </div>

            <div class="metric-value">
                {fmt(environment_score,0)}
            </div>

            <div class="metric-unit">
                {environment_level}
            </div>

        </div>

    </div>

</div>
        """
    )


    left, right = (

        st.columns(
            [
                1,
                1.8
            ]
        )

    )


    with left:

        st.html(
            f"""
<div class="editorial-card">

    <div class="card-overline">
        Current KFD classifier
    </div>

    <div
        class="
            big-value
            {signal_css}
        "
    >
        {signal}
    </div>

    <div class="small-copy">

        The Dynamic Random Forest V2
        evaluates temperature,
        rainfall, NDVI and
        temperature range.

        <br><br>

        The resulting score is
        an experimental classifier
        output and not a calibrated
        epidemiological probability.

    </div>

</div>
            """
        )


    with right:

        trajectory = (

            district_df[
                [
                    "Year",
                    "Temperature",
                    "NDVI"
                ]
            ]

            .melt(

                id_vars="Year",

                var_name="Indicator",

                value_name="Value"
            )

        )


        fig = px.line(

            trajectory,

            x="Year",

            y="Value",

            color="Indicator",

            markers=True,

            title=(
                f"Environmental trajectory — "
                f"{selected_district}"
            )
        )


        fig = style_chart(
            fig,
            405
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar":
                    False
            }
        )


    # ========================================================
    # MAP
    # ========================================================

    st.html(
        """
<div class="paper">

    <div class="section-index">
        02 / Geography
    </div>

    <div class="section-title">

        Nine districts.
        One connected
        landscape.

    </div>

    <div class="section-text">

        The study spans KFD-relevant
        Western Ghats districts across
        Karnataka, Kerala, Goa and
        Maharashtra.

    </div>

</div>
        """
    )


    map_df = (

        environment_df[
            environment_df["Year"]
            ==
            selected_year
        ]

        [
            [
                "District",
                "State",
                "Latitude",
                "Longitude",
                "Temperature",
                "NDVI"
            ]
        ]

        .drop_duplicates(
            "District"
        )

        .dropna(
            subset=[
                "Latitude",
                "Longitude"
            ]
        )

    )


    map_fig = px.scatter_geo(

        map_df,

        lat="Latitude",

        lon="Longitude",

        hover_name="District",

        hover_data={
            "State": True,
            "Temperature": ":.1f",
            "NDVI": ":.3f",
            "Latitude": False,
            "Longitude": False
        }

    )


    map_fig.update_traces(

        marker=dict(
            size=14,
            color="#304E43"
        )

    )


    map_fig.update_geos(

        projection_type="mercator",

        showland=True,

        landcolor="#E6DED6",

        showocean=True,

        oceancolor="#D1D9D5",

        showcountries=True,

        countrycolor="#7D817E",

        center=dict(
            lat=14,
            lon=75
        ),

        lataxis_range=[
            10.5,
            17.5
        ],

        lonaxis_range=[
            72.5,
            77.5
        ]

    )


    map_fig.update_layout(

        height=540,

        paper_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        )

    )


    st.plotly_chart(
        map_fig,
        use_container_width=True,
        config={
            "displayModeBar":
                False
        }
    )


# ============================================================
# CLIMATE
# ============================================================

elif page == "Climate":

    render_hero()


    st.html(
        """
<div class="paper">

    <div class="section-index">
        02 / Climate
    </div>

    <div class="section-title">

        Temperature
        tells only
        part of the story.

    </div>

    <div class="section-text">

        Annual climate observations help
        describe the changing environmental
        conditions experienced by each
        Western Ghats study region.

    </div>

</div>
        """
    )


    temperature_columns = [

        column

        for column in [

            "Temperature",
            "Min_Temperature",
            "Max_Temperature"

        ]

        if column
        in district_df.columns

    ]


    temp_long = (

        district_df[
            [
                "Year",
                *temperature_columns
            ]
        ]

        .melt(

            id_vars="Year",

            var_name="Temperature_Type",

            value_name="Temperature_Value"

        )

    )


    climate_left, climate_right = (

        st.columns(
            [
                1.8,
                1
            ]
        )

    )


    with climate_left:

        fig = px.line(

            temp_long,

            x="Year",

            y="Temperature_Value",

            color="Temperature_Type",

            markers=True,

            title="Temperature profile"

        )


        fig = style_chart(
            fig,
            450
        )


        fig.update_yaxes(
            title="Temperature °C"
        )


        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                "displayModeBar":
                    False
            }

        )


    with climate_right:

        st.html(
            f"""
<div class="editorial-card">

    <div class="card-overline">
        Current climate
    </div>

    <div class="big-value">
        {fmt(temperature,1)}°
    </div>

    <div class="small-copy">

        Annual rainfall

        <br>

        <strong style="
            color:#0D141B;
            font-size:24px;
        ">
            {fmt(rainfall,0)} mm
        </strong>

        <br><br>

        Temperature range

        <br>

        <strong style="
            color:#0D141B;
            font-size:24px;
        ">
            {fmt(temperature_range,1)} °C
        </strong>

    </div>

</div>
            """
        )


    rain_fig = px.bar(

        district_df,

        x="Year",

        y="Rainfall",

        title="Annual rainfall"

    )


    rain_fig = style_chart(
        rain_fig,
        430
    )


    rain_fig.update_yaxes(
        title="Rainfall mm"
    )


    st.plotly_chart(

        rain_fig,

        use_container_width=True,

        config={
            "displayModeBar":
                False
        }

    )


# ============================================================
# VEGETATION
# ============================================================

elif page == "Vegetation":

    render_hero()


    st.html(
        """
<div class="paper">

    <div class="section-index">
        03 / Satellite vegetation
    </div>

    <div class="section-title">

        Reading forests
        from space.

    </div>

    <div class="section-text">

        Sentinel-derived NDVI provides a
        consistent remote-sensing measure
        of vegetation condition across
        time and geography.

    </div>

</div>
        """
    )


    veg_left, veg_right = (

        st.columns(
            [
                1.8,
                1
            ]
        )

    )


    with veg_left:

        fig = px.line(

            district_df,

            x="Year",

            y="NDVI",

            markers=True,

            title=(
                f"NDVI trajectory — "
                f"{selected_district}"
            )

        )


        fig = style_chart(
            fig,
            430
        )


        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                "displayModeBar":
                    False
            }

        )


    with veg_right:

        first_ndvi = (

            district_df
            .sort_values("Year")
            .iloc[0]["NDVI"]

        )


        ndvi_change = (

            ndvi
            -
            first_ndvi

        )


        st.html(
            f"""
<div class="editorial-card">

    <div class="card-overline">
        Selected year
    </div>

    <div class="big-value">
        {fmt(ndvi,3)}
    </div>

    <div class="small-copy">

        Mean NDVI

        <br><br>

        Change from first
        available observation

        <br>

        <strong style="
            color:#0D141B;
            font-size:24px;
        ">
            {fmt(ndvi_change,3)}
        </strong>

    </div>

</div>
            """
        )


    comparison = (

        environment_df[
            environment_df["Year"]
            ==
            selected_year
        ]

        [
            [
                "District",
                "NDVI"
            ]
        ]

        .sort_values(
            "NDVI",
            ascending=False
        )

    )


    fig = px.bar(

        comparison,

        x="District",

        y="NDVI",

        title=(
            f"Vegetation across districts — "
            f"{selected_year}"
        )

    )


    fig = style_chart(
        fig,
        430
    )


    st.plotly_chart(

        fig,

        use_container_width=True,

        config={
            "displayModeBar":
                False
        }

    )


# ============================================================
# FOREST & TERRAIN
# ============================================================

elif page == "Forest & Terrain":

    render_hero()


    st.html(
        """
<div class="paper">

    <div class="section-index">
        04 / Landscape
    </div>

    <div class="section-title">

        Ecology has
        a geography.

    </div>

    <div class="section-text">

        Forest disturbance and terrain
        help characterize each district,
        although the ablation experiment
        showed that static features did
        not transfer strongly to completely
        unseen districts.

    </div>

</div>
        """
    )


    landscape = (

        environment_df

        .groupby(
            "District",
            as_index=False
        )

        .agg(

            Forest_Loss=(
                "Forest_Loss",
                "mean"
            ),

            Elevation=(
                "Elevation",
                "mean"
            )

        )

    )


    c1, c2 = st.columns(2)


    with c1:

        fig = px.bar(

            landscape
            .sort_values(
                "Forest_Loss",
                ascending=False
            ),

            x="District",

            y="Forest_Loss",

            title="Forest loss indicator"

        )


        fig = style_chart(
            fig,
            430
        )


        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                "displayModeBar":
                    False
            }

        )


    with c2:

        fig = px.bar(

            landscape
            .sort_values(
                "Elevation",
                ascending=False
            ),

            x="District",

            y="Elevation",

            title="Mean elevation"

        )


        fig = style_chart(
            fig,
            430
        )


        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                "displayModeBar":
                    False
            }

        )


# ============================================================
# KFD MODEL
# ============================================================

elif page == "KFD Model":

    render_hero()


    st.html(
        f"""
<div class="dark-section">

    <div class="section-index">
        05 / Machine intelligence
    </div>

    <div class="section-title">

        Can environmental
        signals travel
        across geography?

    </div>

    <div class="section-text">

        The primary research model is a
        Dynamic Random Forest trained
        using temperature, rainfall,
        NDVI and temperature range.

        Its spatial validation performance
        remains modest, while temporal
        performance in represented regions
        is more promising.

    </div>


    <div style="
        display:grid;
        grid-template-columns:
            1.2fr 1fr;
        gap:30px;
        margin-top:35px;
    ">

        <div>

            <div class="card-overline"
                 style="color:#9EA5A8;">
                Current AI signal
            </div>

            <div
                style="
                    font-size:72px;
                    letter-spacing:-0.06em;
                    margin-top:8px;
                "
                class="{signal_css}"
            >

                {signal}

            </div>

        </div>


        <div>

            <div class="model-row">
                <span>Model score</span>
                <strong>{score_text}</strong>
            </div>

            <div class="model-row">
                <span>District</span>
                <strong>{selected_district}</strong>
            </div>

            <div class="model-row">
                <span>Year</span>
                <strong>{selected_year}</strong>
            </div>

            <div class="model-row">
                <span>Model</span>
                <strong>Dynamic Random Forest V2</strong>
            </div>

        </div>

    </div>

</div>
        """
    )


    if kfd_model is None:

        st.error(
            "The trained KFD model could not be loaded."
        )


    else:

        features = (

            model_metadata.get(
                "features",
                [
                    "Temperature",
                    "Rainfall",
                    "NDVI",
                    "Temperature_Range"
                ]
            )

        )


        input_df = pd.DataFrame(
            {

                "Feature":
                    features,

                "Observed Value":
                    [
                        row.get(
                            feature,
                            np.nan
                        )

                        for feature
                        in features
                    ]

            }
        )


        model_left, model_right = (

            st.columns(
                [
                    1,
                    1.8
                ]
            )

        )


        with model_left:

            st.markdown(
                "#### Environmental inputs"
            )


            st.dataframe(

                input_df,

                use_container_width=True,

                hide_index=True

            )


        with model_right:

            estimator = (

                kfd_model
                .named_steps["model"]

            )


            if hasattr(
                estimator,
                "feature_importances_"
            ):

                importance_df = (

                    pd.DataFrame(
                        {

                            "Feature":
                                features,

                            "Importance":
                                estimator
                                .feature_importances_

                        }
                    )

                    .sort_values(
                        "Importance"
                    )

                )


                fig = px.bar(

                    importance_df,

                    x="Importance",

                    y="Feature",

                    orientation="h",

                    title="Model feature importance"

                )


                fig = style_chart(
                    fig,
                    380
                )


                st.plotly_chart(

                    fig,

                    use_container_width=True,

                    config={
                        "displayModeBar":
                            False
                    }

                )


    # ========================================================
    # VALIDATION
    # ========================================================

    st.html(
        """
<div class="paper">

    <div class="section-index">
        06 / Validation
    </div>

    <div class="section-title">

        Promising through
        time. Limited
        across space.

    </div>

    <div class="section-text">

        Leave-One-District-Out validation
        tested geographic transfer, while
        later-year validation assessed
        temporal performance in regions
        already represented during training.

    </div>

</div>
        """
    )


    validation_df = pd.DataFrame(
        {

            "Experiment": [

                "Dynamic RF · Spatial",

                "Combined Decision Tree · Spatial",

                "Decision Tree · Temporal"

            ],

            "Balanced Accuracy": [

                0.565,
                0.543,
                0.833

            ],

            "F1": [

                0.545,
                0.618,
                0.909

            ],

            "ROC-AUC": [

                0.544,
                0.354,
                0.800

            ]

        }
    )


    st.dataframe(

        validation_df,

        use_container_width=True,

        hide_index=True

    )


    st.warning(
        "Research prototype only. "
        "The model score is not a calibrated "
        "epidemiological probability and must "
        "not be interpreted as an operational "
        "public-health forecast."
    )


# ============================================================
# RESEARCH
# ============================================================

elif page == "Research":

    render_hero()


    st.html(
        """
<div class="paper">

    <div class="section-index">
        07 / Research framework
    </div>

    <div class="section-title">

        From satellite
        pixels to outbreak
        intelligence.

    </div>

    <div class="section-text">

        SpilloverAI combines remote sensing,
        climate observations, landscape
        variables and verified KFD occurrence
        records within a reproducible
        environmental machine-learning
        pipeline.

    </div>

</div>
        """
    )


    methods = [

        (
            "01",
            "Climate",
            (
                "Annual temperature and rainfall "
                "observations were assembled for "
                "Western Ghats study districts."
            )
        ),

        (
            "02",
            "Satellite vegetation",
            (
                "Sentinel-2 imagery was processed "
                "through Google Earth Engine to "
                "derive annual NDVI indicators."
            )
        ),

        (
            "03",
            "Forest & terrain",
            (
                "Hansen Global Forest Change "
                "and SRTM elevation provide "
                "landscape context."
            )
        ),

        (
            "04",
            "Disease evidence",
            (
                "Historical KFD occurrence labels "
                "were compiled from source-backed "
                "government and surveillance records."
            )
        ),

        (
            "05",
            "Machine learning",
            (
                "Logistic Regression, Decision Tree, "
                "Random Forest and XGBoost were "
                "evaluated against verified outbreak "
                "occurrence."
            )
        ),

        (
            "06",
            "Spatial validation",
            (
                "Leave-One-District-Out validation "
                "tested whether environmental patterns "
                "could generalize to completely unseen "
                "districts."
            )
        ),

        (
            "07",
            "Feature ablation",
            (
                "Dynamic environmental features "
                "generalized better spatially than "
                "static elevation and forest-loss "
                "features."
            )
        )

    ]


    st.html(
        '<div class="paper">'
    )


    for number, title, text in methods:

        st.html(
            f"""
<div class="method-row">

    <div class="method-number">
        {number}
    </div>

    <div class="method-title">
        {title}
    </div>

    <div class="method-text">
        {text}
    </div>

</div>
            """
        )


    st.html(
        "</div>"
    )


    # ========================================================
    # LIMITATIONS
    # ========================================================

    st.html(
        """
<div class="dark-section">

    <div class="section-index">
        08 / Scientific boundaries
    </div>

    <div class="section-title">

        What SpilloverAI
        cannot claim.

    </div>

    <div class="section-text">

        Scientific limitations are part
        of the system rather than hidden
        behind the interface.

    </div>

</div>
        """
    )


    st.markdown(
        """
- The verified KFD machine-learning dataset remains small.
- Historical climate coverage is uneven across districts.
- Surveillance periods are not perfectly harmonized across all historical sources.
- Remote-sensing extraction currently uses representative district regions rather than full administrative polygons.
- The current forest-loss representation has limited temporal variation.
- The Random Forest score is not a calibrated disease probability.
- Feature importance represents model usage, not causal relationships.
- Cross-district generalization remains modest.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
<div class="site-footer">

    <span>
        ◐ SpilloverAI
    </span>

    <span>
        KFD /
        Western Ghats /
        2015–2024
    </span>

</div>
    """
)