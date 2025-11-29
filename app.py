import streamlit as st
from streamlit_lottie import st_lottie
import json
import requests

st.set_page_config(
    page_title="🏡 India House Price App",
    page_icon="🏠",
    layout="wide"
)

# ---------------- DARK MODE ---------------- #
if "dark" not in st.session_state:
    st.session_state.dark = False

mode = st.sidebar.toggle("🌗 Dark Mode")

if mode:
    st.session_state.dark = True
    st.markdown("""
        <style>
        body {
            background-color: #0a0a0a;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.session_state.dark = False

# ---------------- LOTTIE LOADER ---------------- #
def load_lottie(url):
    return requests.get(url).json()

lottie_sidebar = load_lottie("https://assets7.lottiefiles.com/packages/lf20_6HFXXE.json")

st.sidebar.markdown("### ⚡ Powered by AI")
st_lottie(lottie_sidebar, height=150, key="sidebarAnim")

# ---------------- MAIN PAGE ---------------- #
st.title("🏡 India Real Estate Intelligence")
st.write("Navigate using the left-side menu to explore prediction, EMI, maps & more.")

