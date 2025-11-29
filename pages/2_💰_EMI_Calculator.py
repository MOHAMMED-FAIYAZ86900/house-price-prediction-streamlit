import streamlit as st
import math

st.title("💰 EMI Calculator")

loan = st.number_input("Loan Amount (₹)", 50000, 500000000, 5000000)
rate = st.number_input("Interest Rate (%)", 5.0, 18.0, 8.5)
years = st.number_input("Loan Tenure (Years)", 1, 40, 20)

monthly_rate = rate / (12 * 100)
months = years * 12

emi = loan * monthly_rate * ((1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)

st.success(f"📌 EMI per month: ₹ {emi:,.2f}")
