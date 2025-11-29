# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Indian House Price Estimator", layout="wide", initial_sidebar_state="expanded")

# -----------------------
# Helper functions
# -----------------------
def format_in_inr(x):
    try:
        x = int(round(x))
    except:
        x = int(x)
    s = f"₹ {x:,}"
    return s

def load_model_safe(path):
    try:
        return joblib.load(path)
    except Exception as e:
        return None

def add_bg(image_url=None):
    # If you want to use a local image in assets/, use path and base64 encoding instead.
    if image_url:
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("{image_url}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: local;
            }}
            .block-container {{
                background-color: rgba(255,255,255,0.85);
                border-radius: 12px;
                padding: 1.2rem;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# -----------------------
# Background + Header
# -----------------------
# You can replace the URL with your own hosted image or an assets/certificate.jpg path
bg_url = "https://images.unsplash.com/photo-1592595896551-12c1e1f7b4be?auto=format&fit=crop&w=1350&q=80"
add_bg(bg_url)

st.markdown("""
    <div style='text-align:center; padding-top:18px;'>
        <h1 style='color:#0f172a; margin-bottom:0;'>🏠 Indian House Price Estimator</h1>
        <p style='color:#334155; margin-top:4px;'>Multi-model predictions • Modern UI • Tailored for Indian buyers 🇮🇳</p>
    </div>
""", unsafe_allow_html=True)

# -----------------------
# Load models & scaler (flexible)
# -----------------------
MODEL_FILES = {
    "Linear Regression (lr_model.pkl)": "lr_model.pkl",
    "Random Forest (rf_model.pkl)": "rf_model.pkl",
    "XGBoost (xgb_model.pkl)": "xgb_model.pkl",
    "Single Model (model.pkl)": "model.pkl"
}

loaded_models = {}
for label, fname in MODEL_FILES.items():
    if os.path.exists(fname):
        m = load_model_safe(fname)
        if m is not None:
            loaded_models[label.split(" ")[0]] = (label, m)

# also try load scaler
scaler = None
if os.path.exists("scaler.pkl"):
    try:
        scaler = joblib.load("scaler.pkl")
    except:
        scaler = None

# If nothing loaded, show message and try to fallback to model.pkl
if not loaded_models:
    st.warning("No ML model files found in the app root. Upload `model.pkl` or `lr_model.pkl` / `rf_model.pkl` / `xgb_model.pkl` and redeploy.")
    st.info("Expected files: lr_model.pkl, rf_model.pkl, xgb_model.pkl, scaler.pkl (optional).")
    st.stop()

# -----------------------
# Sidebar: quick settings & show files
# -----------------------
with st.sidebar:
    st.header("⚙️ App Controls")
    st.write("Available models:")
    for k, v in loaded_models.items():
        st.write(f"- **{v[0]}**")
    st.write("---")
    st.write("Model files in repo root:")
    for f in os.listdir("."):
        if f.endswith(".pkl"):
            st.write(f"- {f}")
    st.write("---")
    show_logs = st.checkbox("Show debug logs", value=False)

# -----------------------
# Input form (Indian features)
# -----------------------
with st.container():
    st.subheader("🏘 Property Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.selectbox("City", ["Bengaluru", "Mumbai", "Hyderabad", "Chennai", "Delhi", "Pune"])
        area = st.text_input("Locality (e.g., Koramangala, Andheri West)", value="")
        bhk = st.selectbox("BHK", [1,2,3,4,5], index=2)
        bathrooms = st.selectbox("Bathrooms", [1,2,3,4], index=1)
        total_sqft = st.number_input("Total Carpet Area (sqft)", min_value=250, max_value=10000, value=1200, step=50)

    with col2:
        prop_type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa"])
        facing = st.selectbox("Facing", ["East","West","North","South"])
        age = st.slider("Age of property (years)", 0, 60, 8)
        balconies = st.selectbox("Balconies", [0,1,2,3], index=1)
        parking = st.selectbox("Parking Available?", ["Yes","No"])

    with col3:
        floor = st.selectbox("Floor (for apartment)", ["Ground","1","2","3","4","5+"], index=2)
        carpet_to_built_ratio = st.slider("Carpet/Built % (est.)", 40, 95, 70)
        txn_month = st.selectbox("Transaction Month", list(range(1,13)), index=5)
        txn_year = st.number_input("Transaction Year", min_value=2000, max_value=2030, value=2023)
        extra_notes = st.text_area("Notes (optional)", height=70)

# -----------------------
# Simple encoding to numeric vector (consistent with training)
# -----------------------
# NOTE: The training script must follow same feature engineering.
city_map = {"Bengaluru":1,"Mumbai":2,"Hyderabad":3,"Chennai":4,"Delhi":5,"Pune":6}
ptype_map = {"Apartment":1,"Independent House":2,"Villa":3}
facing_map = {"East":1,"West":2,"North":3,"South":4}
park_map = {"Yes":1,"No":0}
floor_map = {"Ground":0,"1":1,"2":2,"3":3,"4":4,"5+":5}

# Build feature vector in consistent order — adapt to your trained model's expected order!
# Default ordering used here (make sure matches training): [bhk, bathrooms, total_sqft, city_code, prop_type, facing_code, age, parking, balconies, floor_num, carpet_ratio, txn_year, txn_month]
features = [
    bhk,
    bathrooms,
    total_sqft,
    city_map.get(city, 0),
    ptype_map.get(prop_type, 0),
    facing_map.get(facing, 0),
    age,
    park_map.get(parking, 0),
    balconies,
    floor_map.get(floor, 0),
    carpet_to_built_ratio,
    txn_year,
    txn_month
]

X_input = np.array([features], dtype=float)

# -----------------------
# Make predictions — for each loaded model
# -----------------------
st.subheader("🤖 Model Predictions")

preds = {}
for mkey, (label, model) in loaded_models.items():
    try:
        # If model is linear regression and scaler exists, scale input
        name = mkey
        if "Linear" in label or "lr_model" in label:
            if scaler is not None:
                Xproc = scaler.transform(X_input)
            else:
                Xproc = X_input
        else:
            # RF and XGBoost typically do not need scaling
            Xproc = X_input

        p = model.predict(Xproc)[0]
        preds[name] = p
    except Exception as e:
        preds[name] = None
        if show_logs:
            st.error(f"Error predicting with {label}: {e}")

# Show results side-by-side
if preds:
    cols = st.columns(len(preds))
    for (c, (name, value)) in zip(cols, preds.items()):
        with c:
            if value is None:
                st.write(f"**{name}**")
                st.write("❌ Error")
            else:
                st.write(f"**{name}**")
                st.markdown(f"<h3 style='color:green'>{format_in_inr(value)}</h3>", unsafe_allow_html=True)

    # Combined bar chart
    st.subheader("📊 Model Comparison")
    valid_preds = {k:v for k,v in preds.items() if v is not None}
    if valid_preds:
        fig, ax = plt.subplots()
        ax.bar(valid_preds.keys(), list(valid_preds.values()), color=['#0ea5a4','#fb7185','#60a5fa'][:len(valid_preds)])
        ax.set_ylabel("Price (INR)")
        ax.set_title("Predicted Price by Model")
        st.pyplot(fig)
else:
    st.info("No predictions made. Ensure models exist and feature order matches training.")

# -----------------------
# Show RF Feature Importance (if available)
# -----------------------
if "Random" in loaded_models:
    rf_label, rf_model = loaded_models["Random"]
    try:
        fi = None
        if hasattr(rf_model, "feature_importances_"):
            fi = rf_model.feature_importances_
        if fi is not None:
            st.subheader("🔎 Feature Importance (Random Forest)")
            feat_names = ["bhk","bathrooms","total_sqft","city","ptype","facing","age","parking","balconies","floor","carpet_ratio","txn_year","txn_month"]
            fi_df = pd.DataFrame({"feature": feat_names, "importance": fi})
            fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
            st.table(fi_df.head(10))
            fig2, ax2 = plt.subplots(figsize=(6,3))
            ax2.barh(fi_df['feature'].head(10), fi_df['importance'].head(10))
            ax2.invert_yaxis()
            ax2.set_xlabel("Importance")
            st.pyplot(fig2)
    except Exception as e:
        if show_logs:
            st.error(f"RF importance error: {e}")

# -----------------------
# Model Info / Notes
# -----------------------
st.sidebar.markdown("## Model Info")
for name in loaded_models:
    # Placeholder metrics: replace with real ones from your training script
    st.sidebar.write(f"**{name}**")
    st.sidebar.write("- R²: (replace with real value)")
    st.sidebar.write("- MAE: (replace)")

st.sidebar.write("---")
st.sidebar.write("Upload new model files via GitHub and redeploy.")
st.sidebar.write("Expected model names: lr_model.pkl, rf_model.pkl, xgb_model.pkl, scaler.pkl")

# -----------------------
# Optional: EMI Calculator (small utility)
# -----------------------
st.subheader("🏦 EMI Calculator (Optional)")
col1, col2, col3 = st.columns(3)
with col1:
    loan_amt = st.number_input("Loan amount (₹)", min_value=100000, value=2000000, step=50000)
with col2:
    rate = st.slider("Annual interest rate (%)", 5.0, 12.0, 7.5)
with col3:
    tenure_years = st.slider("Tenure (years)", 1, 30, 10)

if st.button("Calculate EMI"):
    r = rate/(12*100)
    n = tenure_years*12
    emi = (loan_amt*r*(1+r)**n)/((1+r)**n - 1)
    st.success(f"EMI: {format_in_inr(emi)}/month  — Total payment: {format_in_inr(emi*n)}")

# -----------------------
# Footer
# -----------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Built by **Mohammed Faiyaz** • Add live demo link to your resume • Models & preprocessing must match training.")
