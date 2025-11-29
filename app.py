import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="🏡 India House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------
# CUSTOM CSS STYLING
# ---------------------------

st.markdown("""
<style>

body {
    background: linear-gradient(to right, #ECE9E6, #FFFFFF);
}

/* Card style */
.stCard {
    background: rgba(255, 255, 255, 0.7);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    backdrop-filter: blur(8px);
}

/* Title */
.title {
    font-size: 45px;
    font-weight: 800;
    color: #333;
    text-align: center;
    padding-bottom: 10px;
    font-family: 'Helvetica Neue', sans-serif;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    color: #555;
    text-align: center;
    margin-top: -15px;
}

/* Prediction box */
.pred-box {
    background: #ffffff;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    font-size: 28px;
    font-weight: 700;
    color: #0a5796;
    border: 2px solid #e1e1e1;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD MODELS
# ---------------------------
model_lr = joblib.load("model.pkl")
rf_model = joblib.load("rf_model_small.pkl")
scaler = joblib.load("scaler.pkl")

models = {
    "Linear Regression (Fast & Simple)": model_lr,
    "Random Forest (More Accurate)": rf_model
}

# ---------------------------
# HEADER
# ---------------------------
st.markdown("<h1 class='title'>🏡 India House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A modern ML-powered tool to estimate home prices based on popular Indian housing features.</p>", unsafe_allow_html=True)

st.write("")

# ---------------------------
# SIDEBAR (MODEL SELECTOR)
# ---------------------------
with st.sidebar:
    st.header("⚙️ Choose Model")
    model_option = st.selectbox(
        "Select ML Model",
        list(models.keys())
    )
    selected_model = models[model_option]

    st.info("🔍 Tip: Random Forest gives more stable predictions.")

# ---------------------------
# INPUT FORM
# ---------------------------
st.markdown("<div class='stCard'>", unsafe_allow_html=True)
st.subheader("🏘 Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1, 10, 2)
    sqft_living = st.number_input("Living Area (sqft)", 300, 10000, 1200)
    sqft_lot = st.number_input("Plot Area (sqft)", 500, 100000, 2000)

with col2:
    floors = st.number_input("Floors", 1, 4, 1)
    waterfront = st.selectbox("Waterfront View", [0, 1])
    view = st.number_input("View Rating (0-4)", 0, 4, 1)
    condition = st.number_input("Condition (1-5)", 1, 5, 3)

with col3:
    grade = st.number_input("Grade (1-13)", 1, 13, 7)
    sqft_above = st.number_input("Area Above Ground (sqft)", 300, 7000, 1000)
    sqft_basement = st.number_input("Basement Area (sqft)", 0, 5000, 300)
    year = st.number_input("Year (Construction)", 1900, 2024, 2010)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# PREDICTION
# ---------------------------
input_data = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                        waterfront, view, condition, grade, sqft_above,
                        sqft_basement, year]])

scaled_data = scaler.transform(input_data)

if st.button("🔮 Predict House Price", use_container_width=True):
    prediction = selected_model.predict(scaled_data)[0]
    prediction = round(prediction, 2)

    st.markdown(f"""
        <div class='pred-box'>
            🏷️ **Estimated Price:** ₹ {prediction:,.2f} INR
        </div>
    """, unsafe_allow_html=True)

# ---------------------------
# FOOTER
# ---------------------------
st.write("")
st.markdown("""
---
💡 *Developed by Mohammed Faiyaz — Powered by Machine Learning & Streamlit*
""")
