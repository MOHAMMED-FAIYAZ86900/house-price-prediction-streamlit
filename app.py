import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------ BACKGROUND IMAGE ------------------
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1507089947368-19c1da9775ae");
             background-size: cover;
             background-attachment: fixed;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
add_bg_from_url()

# ------------------ CUSTOM HEADER ------------------
st.markdown("""
    <h1 style='text-align: center; color: white; text-shadow: 1px 1px 3px black;'>
        🏡 Indian House Price Prediction App 🇮🇳
    </h1>
    <p style='text-align: center; font-size: 20px; color: white; text-shadow: 1px 1px 2px black;'>
        Powered by Machine Learning & Real Estate Insights
    </p>
""", unsafe_allow_html=True)

# ------------------ LOAD MODELS ------------------
models = {
    "Linear Regression": joblib.load("lr_model.pkl"),
    "Random Forest": joblib.load("rf_model.pkl"),
    "XGBoost": joblib.load("xgb_model.pkl")
}

model_choice = st.selectbox("Select Model", list(models.keys()))
model = models[model_choice]

# ------------------ INDIAN HOUSING INPUTS ------------------
st.subheader("Fill Property Details:")

city = st.selectbox("City", ["Bengaluru", "Mumbai", "Hyderabad", "Chennai", "Delhi", "Pune"])
area = st.selectbox("Area", ["BTM", "Indiranagar", "Whitefield", "HSR Layout", "Koramangala", "JP Nagar"])

bhk = st.number_input("BHK", 1, 10, 2)
bath = st.number_input("Bathrooms", 1, 5, 2)
sqft = st.number_input("Total Sqft", 300, 10000, 1000)

facing = st.selectbox("Facing Direction", ["East", "West", "North", "South"])
age = st.number_input("Age of Property (Years)", 0, 30, 5)

parking = st.selectbox("Car Parking", ["No", "Yes"])
balcony = st.selectbox("Balcony", [0, 1, 2])

# ------------------ INPUT ENCODING ------------------
facing_map = {"East": 0, "West": 1, "North": 2, "South": 3}
parking_map = {"No": 0, "Yes": 1}

input_data = np.array([[bhk, bath, sqft, age, balcony, parking_map[parking], facing_map[facing]]])

# ------------------ RUPEE FORMAT ------------------
def indian_format(n):
    return "₹ {:,}".format(int(n)).replace(",", ",")

# ------------------ PREDICTION ------------------
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"🏠 Estimated Price: **{indian_format(prediction)}**")

    # Bar chart visualization
    fig, ax = plt.subplots()
    ax.bar(["Predicted Price"], [prediction])
    ax.set_ylabel("Price (₹)")
    ax.set_title("Predicted House Price")
    st.pyplot(fig)

# ------------------ MODEL INFO SECTION ------------------
if st.checkbox("Show Model Information"):
    st.write("### 📊 Model Performance:")
    st.write({
        "Linear Regression": {"R² Score": 0.72, "Best For": "Simple relationships"},
        "Random Forest":
