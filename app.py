import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import requests
from streamlit_lottie import st_lottie
import plotly.graph_objects as go

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="🏡 India House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------
# LOAD BACKGROUND IMAGE
# ---------------------------
def set_bg(image_file):
    encoded = open(image_file, "rb").read()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded.hex()}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("bg.jpg")   # <<—— ADD YOUR HOME IMAGE HERE

# ---------------------------
# CUSTOM CSS
# ---------------------------
st.markdown("""
<style>

.title {
    font-size: 45px;
    font-weight: 800;
    color: #ffffff;
    text-align: center;
    margin-top: -20px;
    text-shadow: 2px 2px 4px #000000;
}

.subtitle {
    font-size: 18px;
    color: #f0f0f0;
    text-align: center;
    margin-top: -10px;
}

.stCard {
    background: rgba(255,255,255,0.80);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}

.pred-box {
    background: rgba(255,255,255,0.90);
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    font-size: 32px;
    font-weight: 700;
    color: #004aad;
    margin-top: 20px;
    border: 2px solid #ffffff;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOTTIE ANIMATION LOADER
# ---------------------------
def load_lottie(url):
    r = requests.get(url)
    return r.json()

lottie_house = load_lottie(
    "https://assets10.lottiefiles.com/packages/lf20_t9gkkhz4.json"
)

# ---------------------------
# LOAD MODELS
# ---------------------------
model_lr = joblib.load("model.pkl")
rf_model = joblib.load("rf_model_small.pkl")
scaler = joblib.load("scaler.pkl")

models = {
    "Linear Regression (Fast)": model_lr,
    "Random Forest (Accurate)": rf_model
}

# ---------------------------
# HEADER
# ---------------------------
st.markdown("<h1 class='title'>🏡 Premium India House Price Prediction</h1>",
            unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered home valuation — enhanced with animations, charts & modern UI.</p>",
            unsafe_allow_html=True)

# ---------------------------
# LOTTIE ANIMATION
# ---------------------------
st_lottie(lottie_house, height=200, key="houseAnim")

# ---------------------------
# MODEL SELECTOR
# ---------------------------
with st.sidebar:
    st.header("⚙️ Choose ML Model")
    model_option = st.selectbox("Select Algorithm", list(models.keys()))
    selected_model = models[model_option]

    st.info("💡 Random Forest gives more reliable predictions.")

# ---------------------------
# USER INPUT
# ---------------------------
st.markdown("<div class='stCard'>", unsafe_allow_html=True)
st.subheader("🏘 Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1, 10, 2)
    sqft_living = st.number_input("Living Area (sqft)", 300, 10000, 1200)
    sqft_lot = st.number_input("Plot Area (sqft)", 400, 100000, 2000)

with col2:
    floors = st.number_input("Floors", 1, 4, 1)
    waterfront = st.selectbox("Waterfront View", [0, 1])
    view = st.number_input("View Rating", 0, 4, 1)
    condition = st.number_input("Condition (1-5)", 1, 5, 3)

with col3:
    grade = st.number_input("Grade (1–13)", 1, 13, 7)
    sqft_above = st.number_input("Area Above Ground", 300, 7000, 1000)
    sqft_basement = st.number_input("Basement Area", 0, 5000, 250)
    year = st.number_input("Year Built", 1900, 2024, 2010)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# PREDICTION
# ---------------------------
input_data = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                        waterfront, view, condition, grade, sqft_above,
                        sqft_basement, year]])

scaled = scaler.transform(input_data)

if st.button("🔮 Predict Price", use_container_width=True):

    prediction = selected_model.predict(scaled)[0]

    st.markdown(
        f"<div class='pred-box'>🏷 Price Estimate: <br> ₹ {prediction:,.2f} </div>",
        unsafe_allow_html=True
    )

    # ---------------------------
    # MODEL COMPARISON CHART
    # ---------------------------
    st.subheader("📊 Model Comparison")

    values = []
    names = []

    for name, model in models.items():
        pred = model.predict(scaled)[0]
        values.append(pred)
        names.append(name)

    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=values,
            marker=dict(color=["#004aad", "#ff9800"])
        )
    ])
    fig.update_layout(
        title="Prediction Comparison Between Models",
        xaxis_title="Model",
        yaxis_title="Predicted Price (INR)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("""
---
🔧 *Built with ❤️ by Mohammed Faiyaz — Streamlit + Machine Learning*
""")
