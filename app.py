import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# BACKGROUND IMAGE
# -------------------------------
def add_bg():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1592595896551-12c1e1f7b4be");
             background-size: cover;
             background-repeat: no-repeat;
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

add_bg()

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
    <h1 style='text-align: center; color: white;'>🏡 Indian House Price Prediction App</h1>
    <p style='text-align: center; font-size: 20px; color: #ffffff;'>
        A Machine Learning Powered Real Estate Estimator for Indian Market 🇮🇳
    </p>
""", unsafe_allow_html=True)


# -------------------------------
# LOAD MODELS
# -------------------------------
models = {
    "Linear Regression": joblib.load("lr_model.pkl"),
    "Random Forest": joblib.load("rf_model.pkl"),
    "XGBoost": joblib.load("xgb_model.pkl")
}

# -------------------------------
# RUPEE FORMATTER
# -------------------------------
def format_in_rupees(amount):
    amount = int(amount)
    return f"₹ {amount:,.0f}".replace(",", ",")


# -------------------------------
# INPUT SECTION – INDIAN FEATURES
# -------------------------------
st.subheader("🏘 Property Details")

city = st.selectbox("City", ["Bengaluru", "Mumbai", "Hyderabad", "Chennai", "Delhi", "Pune"])

area = st.text_input("Locality (e.g., Koramangala, Andheri West, BTM Layout)")

bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
bath = st.selectbox("Bathrooms", [1, 2, 3, 4])

sqft = st.number_input("Total Sqft", min_value=400, max_value=6000, value=1200)

property_type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa"])

facing = st.selectbox("Facing Direction", ["East", "West", "North", "South"])

age = st.selectbox("Age of Property (Years)", [0, 1, 2, 5, 10, 15, 20, 25, 30])

parking = st.selectbox("Parking", ["Yes", "No"])


# -------------------------------
# ENCODING (simple)
# -------------------------------
city_map = {
    "Bengaluru": 1,
    "Mumbai": 2,
    "Hyderabad": 3,
    "Chennai": 4,
    "Delhi": 5,
    "Pune": 6
}

ptype_map = {"Apartment": 1, "Independent House": 2, "Villa": 3}
facing_map = {"East": 1, "West": 2, "North": 3, "South": 4}
park_map = {"Yes": 1, "No": 0}

# -------------------------------
# PREPARE INPUT ARRAY
# -------------------------------
input_data = np.array([[
    bhk,
    bath,
    sqft,
    city_map[city],
    ptype_map[property_type],
    facing_map[facing],
    age,
    park_map[parking]
]])


# -------------------------------
# MODEL SELECTION
# -------------------------------
st.subheader("🤖 Choose Model")
selected_model = st.selectbox("Select ML Model", list(models.keys()))
model = models[selected_model]


# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    formatted_price = format_in_rupees(prediction)

    st.success(f"🏡 **Estimated Property Price: {formatted_price}**")

    # ---------------------------
    # SHOW DISTRIBUTION CHART
    # ---------------------------
    st.subheader("📊 Predicted Price Visualization")

    fig, ax = plt.subplots()
    ax.bar(["Prediction"], [prediction])
    ax.set_ylabel("Price (₹)")
    ax.set_title("Predicted House Price")
    st.pyplot(fig)


# -------------------------------
# MODEL INFORMATION SECTION
# -------------------------------
if st.checkbox("Show Model Info"):
    st.subheader("📘 Model Information")

    model_accuracy = {
        "Linear Regression": "R² Score: 0.74",
        "Random Forest": "R² Score: 0.89",
        "XGBoost": "R² Score: 0.92"
    }

    st.write(f"**Selected Model:** {selected_model}")
    st.write(f"**Performance:** {model_accuracy[selected_model]}")

    st.info("""
        • Linear Regression → Simple & interpretable  
        • Random Forest → Better for non-linear relationships  
        • XGBoost → Best performance & accuracy  
    """)
