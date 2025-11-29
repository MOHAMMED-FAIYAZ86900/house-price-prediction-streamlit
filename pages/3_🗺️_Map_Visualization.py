import streamlit as st
import folium
from streamlit_folium import st_folium

st.title("🗺️ Plot Property on Map")

lat = st.number_input("Latitude", 8.0, 38.0, 12.9716)
lon = st.number_input("Longitude", 68.0, 97.0, 77.5946)

m = folium.Map(location=[lat, lon], zoom_start=12)
folium.Marker([lat, lon], popup="Your Property").add_to(m)

st_folium(m, width=700)
