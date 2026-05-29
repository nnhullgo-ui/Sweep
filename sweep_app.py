
import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import os

st.set_page_config(page_title="Sweep", page_icon="🗺️", layout="wide")

json_path = "C:/Users/naila/detections.json"

if os.path.exists(json_path):
    with open(json_path) as f:
        detections = json.load(f)
else:
    detections = []
    st.warning("No detections yet — run Program 1 first!")

st.title("🗺️ Sweep")
st.caption("AI-powered litter detection from above")

col1, col2, col3 = st.columns(3)
col1.metric("Total Detections", len(detections))
col2.metric("Avg Confidence", f"{sum(d['confidence'] for d in detections)/len(detections)*100:.0f}%")
col3.metric("Area Scanned", "1 park")

st.divider()
st.subheader("Litter Map")
m = folium.Map(location=[43.0481, -76.1474], zoom_start=16)

for detection in detections:
    folium.Marker(
        location=[detection["lat"], detection["lon"]],
        popup=f"{detection['label']} ({detection['confidence']*100:.0f}% confidence)",
        tooltip=detection["label"],
        icon=folium.Icon(color="red", icon="trash", prefix="fa")
    ).add_to(m)

st_folium(m, width=900, height=500)

st.divider()
st.subheader("All Detections")
for i, d in enumerate(detections):
    with st.expander(f"{i+1}. {d['label']} — {d['confidence']*100:.0f}% confidence"):
        st.write(f"📍 Location: {d['lat']}, {d['lon']}")
        st.write(f"🕐 Time: {d['timestamp']}")