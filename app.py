import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction App")
st.write("Enter the details below to predict the house price.")

# Input fields
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0)
sqft_living = st.number_input("Sqft Living Area", min_value=0, max_value=15000, value=2000)
sqft_lot = st.number_input("Sqft Lot Area", min_value=0, max_value=100000, value=5000)
floors = st.number_input("Floors", min_value=1.0, max_value=4.0, value=1.0)
waterfront = st.selectbox("Waterfront View (0 = No, 1 = Yes)", [0, 1])
view = st.number_input("View Rating (0–4)", min_value=0, max_value=4, value=0)
condition = st.number_input("Condition (1–5)", min_value=1, max_value=5, value=3)
grade = st.number_input("Grade (1–13)", min_value=1, max_value=13, value=7)
sqft_above = st.number_input("Sqft Above Ground", min_value=0, max_value=10000, value=1500)
sqft_basement = st.number_input("Sqft Basement", min_value=0, max_value=5000, value=500)
yr_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=2000)
yr_renovated = st.number_input("Year Renovated (0 = never)", min_value=0, max_value=2024, value=0)
zipcode = st.number_input("Zipcode", min_value=98000, max_value=99999, value=98178)
lat = st.number_input("Latitude", min_value=47.0, max_value=48.0, value=47.5)
long = st.number_input("Longitude", min_value=-122.5, max_value=-121.0, value=-122.2)
sqft_living15 = st.number_input("Living Area of Neighbors", min_value=0, max_value=10000, value=1500)
sqft_lot15 = st.number_input("Lot Area of Neighbors", min_value=0, max_value=100000, value=5000)

# Prepare input data
input_data = np.array([[
    bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition,
    grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, lat, long,
    sqft_living15, sqft_lot15
]])

# Prediction
if st.button("Predict House Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"🏡 Estimated House Price: **${prediction:,.2f}**")
