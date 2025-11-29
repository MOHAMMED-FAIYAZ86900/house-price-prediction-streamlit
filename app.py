# app.py — Full Premium Multi-page Streamlit App
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import requests
import base64
import io
from fpdf import FPDF
import plotly.graph_objects as go

# map
import folium
from streamlit_folium import st_folium

# lottie helper (wrap http/json and local file)
from streamlit_lottie import st_lottie

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="🏡 India House Price Suite",
                   page_icon="🏠",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------
# Utility helpers
# ---------------------------
def add_bg_local(image_file: str):
    """Set local image as app background (base64 embed)."""
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{b64}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .block-container {{
                background-color: rgba(255,255,255,0.88);
                border-radius: 12px;
                padding: 1.0rem;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        pass

def load_lottie_file(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def format_in_inr(x):
    try:
        x = int(round(x))
    except Exception:
        x = int(x)
    return f"₹ {x:,}"

def create_pdf_report(prediction, features_dict, model_name, filename="prediction_report.pdf"):
    """Return bytes of PDF created in-memory using FPDF"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "House Price Prediction Report", ln=1, align='C')
    pdf.ln(6)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Model: {model_name}", ln=1)
    pdf.cell(0, 8, f"Estimated Price: {format_in_inr(prediction)}", ln=1)
    pdf.ln(6)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Input Features:", ln=1)
    pdf.set_font("Arial", '', 11)
    for k, v in features_dict.items():
        pdf.cell(0, 7, f"- {k}: {v}", ln=1)
    # return bytes
    return pdf.output(dest='S').encode('latin-1')

# ---------------------------
# Assets (local then fallback remote)
# ---------------------------
add_bg_local("assets/bg.jpg")
sidebar_lottie = load_lottie_file("assets/sidebar.json") or load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_6HFXXE.json")
hero_lottie    = load_lottie_file("assets/house.json")   or load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_t9gkkhz4.json")

# ---------------------------
# Dark mode toggle (basic)
# ---------------------------
if "dark" not in st.session_state:
    st.session_state.dark = False
if st.sidebar.checkbox("🌗 Dark Mode", value=False):
    st.session_state.dark = True
    # add dark CSS (basic)
    st.markdown(
        """
        <style>
        .block-container { background-color: rgba(10,10,10,0.75) !important; color: #eee; }
        body { background-color: #000; color: #ddd; }
        </style>
        """, unsafe_allow_html=True
    )
else:
    st.session_state.dark = False

# ---------------------------
# Load available models / scaler (graceful)
# ---------------------------
models = {}
for name, fname in [
    ("Linear Regression", "lr_model.pkl"),
    ("Random Forest", "rf_model_small.pkl"),
    ("XGBoost", "xgb_model.pkl"),
    ("Single Model", "model.pkl")
]:
    try:
        m = joblib.load(fname)
        models[name] = (fname, m)
    except Exception:
        pass

scaler = None
try:
    scaler = joblib.load("scaler.pkl")
except Exception:
    scaler = None

# if no models found, prompt and stop
if not models:
    st.sidebar.warning("No model files found in repo root. Upload lr_model.pkl or rf_model_small.pkl or model.pkl and redeploy.")
    st.stop()

# ---------------------------
# Sidebar navigation & lottie
# ---------------------------
with st.sidebar:
    if sidebar_lottie:
        st_lottie(sidebar_lottie, height=140)
    st.title("🏡 India House App")
    page = st.radio("Navigate", ["Home", "Price Prediction", "EMI Calculator", "Map Explorer", "About"])
    st.markdown("---")
    st.write("Available models:")
    for k in models.keys():
        st.write(f"- {k}")
    st.markdown("---")
    st.write("Files in repo:")
    for f in sorted([f for f in [] if False]):  # placeholder if you want to list files via os.listdir
        st.write(f"- {f}")

# ---------------------------
# HOME page
# ---------------------------
if page == "Home":
    col1, col2 = st.columns([2,1])
    with col1:
        st.title("🏡 India House Price Suite")
        st.write("Welcome! Use the sidebar to navigate. This app supports multiple models, PDF export, map visualization, EMI calculator and more.")
        st.markdown("**Quick tips:** Ensure that the model(s) and `scaler.pkl` (if using Linear Regression) are uploaded to the repository root.")
    with col2:
        if hero_lottie:
            st_lottie(hero_lottie, height=260)
    st.markdown("---")
    st.subheader("Live Model Comparison (quick demo)")
    # quick demo: show sample predictions for each model on a default sample vector if possible
    sample = np.array([[3,2,1200,2000,1,0,0,3,7,900,300,2010]])  # shape (1,12) — modify as per your actual trained features
    preds_demo = {}
    for name, (fname, mdl) in models.items():
        try:
            if "Linear" in name and scaler is not None:
                p = mdl.predict(scaler.transform(sample))[0]
            else:
                p = mdl.predict(sample)[0]
            preds_demo[name] = p
        except Exception:
            preds_demo[name] = None

    names = []
    vals = []
    for k,v in preds_demo.items():
        if v is not None:
            names.append(k)
            vals.append(v)
    if vals:
        fig = go.Figure([go.Bar(x=names, y=vals, marker_color=['#0ea5a4','#fb7185','#60a5fa'][:len(vals)])])
        fig.update_layout(title="Demo Predictions by Model", yaxis_title="Price (INR)")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# PRICE PREDICTION page
# ---------------------------
elif page == "Price Prediction":
    st.header("🏘 Price Prediction")
    # choose model
    model_keys = list(models.keys())
    chosen_model_name = st.selectbox("Choose model for prediction", model_keys, index=0)
    model_file, model_obj = models[chosen_model_name]

    # Input UI (you MUST keep feature order consistent with training)
    st.subheader("Enter Property Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
        sqft_living = st.number_input("Sqft Living", min_value=200, max_value=10000, value=1200)
        sqft_lot = st.number_input("Sqft Lot", min_value=0, max_value=100000, value=5000)
    with c2:
        floors = st.number_input("Floors", min_value=1, max_value=10, value=1)
        waterfront = st.selectbox("Waterfront (0/1)", [0,1])
        view = st.number_input("View (0-4)", min_value=0, max_value=4, value=0)
        condition = st.number_input("Condition (1-5)", min_value=1, max_value=5, value=3)
    with c3:
        grade = st.number_input("Grade (1-13)", min_value=1, max_value=13, value=7)
        sqft_above = st.number_input("Sqft Above", min_value=0, max_value=10000, value=900)
        sqft_basement = st.number_input("Sqft Basement", min_value=0, max_value=5000, value=300)
        year = st.number_input("Year Built", min_value=1900, max_value=2024, value=2010)

    feature_vec = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                             waterfront, view, condition, grade, sqft_above,
                             sqft_basement, year]], dtype=float)

    st.markdown("----")
    colp1, colp2 = st.columns([2,1])
    with colp1:
        if st.button("🔮 Predict"):
            # apply scaler if Linear Regression and scaler exists
            try:
                if "Linear" in chosen_model_name and scaler is not None:
                    inp = scaler.transform(feature_vec)
                else:
                    inp = feature_vec
                pred = model_obj.predict(inp)[0]
                st.success(f"Estimated Price: **{format_in_inr(pred)}**")
                low = int(pred * 0.93)
                high = int(pred * 1.07)
                st.info(f"Recommended price range: **{format_in_inr(low)} - {format_in_inr(high)}** (±7%)")

                # comparison chart across available models
                comp_names = []
                comp_vals = []
                for n, (fname, m) in models.items():
                    try:
                        if "Linear" in n and scaler is not None:
                            v = m.predict(scaler.transform(feature_vec))[0]
                        else:
                            v = m.predict(feature_vec)[0]
                        comp_names.append(n)
                        comp_vals.append(v)
                    except Exception:
                        pass
                if comp_vals:
                    fig = go.Figure([go.Bar(x=comp_names, y=comp_vals, marker_color=["#0f766e","#f97316","#2563eb"])])
                    fig.update_layout(title="Model Predictions Comparison", yaxis_title="Price (INR)")
                    st.plotly_chart(fig, use_container_width=True)

                # create PDF report and show download
                report_bytes = create_pdf_report(pred, {
                    "Bedrooms": bedrooms,
                    "Bathrooms": bathrooms,
                    "Sqft living": sqft_living,
                    "Sqft lot": sqft_lot,
                    "Year": year
                }, chosen_model_name)
                st.download_button("📄 Download Prediction Report (PDF)", data=report_bytes, file_name="prediction_report.pdf", mime="application/pdf")
            except Exception as e:
                st.error("Prediction failed. Check model & feature order. Error: " + str(e))
    with colp2:
        # quick details card
        st.markdown("### Model Info")
        try:
            st.write(f"**Model file:** {model_file}")
            st.write(f"**Model type:** {chosen_model_name}")
            if hasattr(model_obj, "feature_importances_"):
                st.write("Has feature importances (RF/XGB).")
            if chosen_model_name.startswith("Linear"):
                st.write("Requires scaling — scaler.pkl applied if present.")
        except Exception:
            st.write("Model meta not available.")

# ---------------------------
# EMI Calculator page
# ---------------------------
elif page == "EMI Calculator":
    st.header("💰 EMI Calculator")
    loan_amt = st.number_input("Loan Amount (₹)", min_value=10000, max_value=100000000, value=2000000, step=50000)
    rate = st.number_input("Annual Interest Rate (%)", min_value=1.0, max_value=20.0, value=7.5, step=0.1)
    years = st.number_input("Tenure (years)", min_value=1, max_value=40, value=10)
    if st.button("Calculate EMI"):
        r = rate / (12 * 100)
        n = years * 12
        emi = (loan_amt * r * (1 + r)**n) / ((1 + r)**n - 1)
        total = emi * n
        st.success(f"Monthly EMI: {format_in_inr(emi)}")
        st.write(f"Total Payment over {years} years: {format_in_inr(total)}")

# ---------------------------
# MAP Explorer page
# ---------------------------
elif page == "Map Explorer":
    st.header("🗺 Map Explorer")
    st.markdown("Use latitude & longitude to preview location. For Indian cities, try Bengaluru: (12.9716, 77.5946)")
    lat = st.number_input("Latitude", value=12.9716, format="%.6f")
    lon = st.number_input("Longitude", value=77.5946, format="%.6f")
    zoom = st.slider("Zoom level", 8, 16, 12)
    m = folium.Map(location=[lat, lon], zoom_start=zoom)
    folium.Marker([lat, lon], popup="Selected property").add_to(m)
    st_folium(m, width=900, height=500)

# ---------------------------
# ABOUT page
# ---------------------------
elif page == "About":
    st.header("ℹ️ About")
    st.markdown("""
    **Developer:** Mohammed Faiyaz  
    **Project:** India House Price Suite — an end-to-end ML app for price estimation, EMI calculation and visualization.  

    _Notes:_  
    - Ensure your model & scaler files are uploaded to repo root.  
    - Feature order in this app must match order used during training.  
    - If you used a different training feature set, update `feature_vec` building accordingly.
    """)
    st.markdown("Contact: faiyaz562000@gmail.com")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Built with ❤️ — Streamlit • Deploy on Streamlit Cloud • Add live demo link to your resume")
