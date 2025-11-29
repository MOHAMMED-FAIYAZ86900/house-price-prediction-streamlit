import streamlit as st
import numpy as np
import joblib
from fpdf import FPDF

st.title("🏡 House Price Prediction")

model_lr = joblib.load("model.pkl")
rf_model = joblib.load("rf_model_small.pkl")
scaler = joblib.load("scaler.pkl")

models = {
    "Linear Regression (Fast)": model_lr,
    "Random Forest (Accurate)": rf_model,
}

# Sidebar model selection
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
selected_model = models[model_choice]

# Inputs
bed = st.number_input("Bedrooms", 1, 10, 3)
bath = st.number_input("Bathrooms", 1, 10, 2)
sqft = st.number_input("Sqft Living", 300, 10000, 1200)
lot = st.number_input("Sqft Lot", 300, 100000, 2000)
floor = st.number_input("Floors", 1, 4, 1)

water = st.selectbox("Waterfront", [0, 1])
view = st.number_input("View", 0, 4, 1)
condi = st.number_input("Condition", 1, 5, 3)
grade = st.number_input("Grade", 1, 13, 7)
above = st.number_input("Sqft Above", 300, 7000, 1000)
base = st.number_input("Basement", 0, 4000, 300)
year = st.number_input("Year Built", 1900, 2024, 2010)

# Prediction Logic
data = np.array([[bed, bath, sqft, lot, floor, water, view, condi, grade, above, base, year]])
scaled = scaler.transform(data)

if st.button("Predict Price"):
    price = selected_model.predict(scaled)[0]
    st.success(f"💰 Estimated Price: ₹ {price:,.2f}")

    # Recommended Range
    st.info(f"📌 Recommended Range: ₹ {price*0.93:,.2f} - ₹ {price*1.06:,.2f}")

    # PDF Export
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="House Price Prediction Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Estimated Price: ₹ {price:,.2f}", ln=True)
    pdf.output("prediction_report.pdf")

    with open("prediction_report.pdf", "rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name="prediction.pdf")
