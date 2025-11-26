# dashboard.py (updated to persist predictions & history)
import streamlit as st
import requests
from PIL import Image
import io
import folium
from streamlit_folium import st_folium
import time
import base64
import os
from datetime import datetime

API_URL = "http://localhost:8000/predict"

st.set_page_config(layout="wide")
st.title("Disaster Image Classification Dashboard")

# initialize session state for persistence
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []

# helpful: path to a sample image in the environment (your uploaded screenshot/image)
SAMPLE_IMAGE_PATH = "/mnt/data/c318b3db-fb62-433d-a276-5614bc6b76c6.png"
use_sample = st.sidebar.button("Use sample image")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("Upload Disaster Image", type=["jpg", "jpeg", "png"])
    # allow using the sample image
    if use_sample and os.path.exists(SAMPLE_IMAGE_PATH):
        try:
            with open(SAMPLE_IMAGE_PATH, "rb") as f:
                sample_bytes = f.read()
            uploaded = st.file_uploader("Upload Disaster Image", type=["jpg", "jpeg", "png"], key="sample_uploader")
            # display sample image (we will send it later)
            st.image(Image.open(io.BytesIO(sample_bytes)).convert("RGB"), caption="Sample image (loaded)", use_column_width=True)
            # store sample bytes in session to be used when Predict is clicked
            st.session_state["_sample_bytes"] = sample_bytes
        except Exception as e:
            st.warning(f"Could not load sample image: {e}")

    # show uploaded image preview (if a user uploaded)
    if uploaded and hasattr(uploaded, "getvalue"):
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True)

with col2:
    lat = st.number_input("Latitude", value=0.0, format="%.6f")
    lon = st.number_input("Longitude", value=0.0, format="%.6f")
    source = st.text_input("Source (operator / CCTV / drone)", value="operator")
    caption = st.text_area("Caption / notes")

    # Predict button
    if st.button("Predict"):
        # select image bytes:
        if uploaded and hasattr(uploaded, "getvalue") and uploaded.getvalue():
            image_bytes = uploaded.getvalue()
            image_name = getattr(uploaded, "name", f"upload_{int(time.time())}.jpg")
            image_type = getattr(uploaded, "type", "image/jpeg")
        elif st.session_state.get("_sample_bytes") is not None:
            image_bytes = st.session_state["_sample_bytes"]
            image_name = os.path.basename(SAMPLE_IMAGE_PATH)
            image_type = "image/png"
        else:
            st.error("No image provided. Upload an image or click 'Use sample image' in the sidebar.")
            image_bytes = None

        if image_bytes is not None:
            files = {"image": (image_name, image_bytes, image_type)}
            data = {"lat": lat, "lon": lon, "source": source, "caption": caption}

            try:
                resp = requests.post(API_URL, files=files, data=data, timeout=10)
                if resp.status_code == 200:
                    result = resp.json()
                    # persist the last result and append to history
                    result["received_at"] = datetime.utcnow().isoformat() + "Z"
                    st.session_state["last_result"] = result
                    st.session_state["history"].insert(0, {  # newest first
                        "time": result["received_at"],
                        "label": result["label"],
                        "confidence": float(result["confidence"])
                    })
                    # keep history size reasonable
                    if len(st.session_state["history"]) > 50:
                        st.session_state["history"] = st.session_state["history"][:50]
                    st.success("Prediction saved and persisted in this session.")
                else:
                    st.error(f"API returned status {resp.status_code}: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request error: {e}")

# Right below the main columns show the last result (persisted)
st.markdown("---")
st.header("Last Prediction (persisted)")

if st.session_state["last_result"]:
    r = st.session_state["last_result"]
    st.subheader(f"Label: {r['label']}  —  Confidence: {r['confidence']:.3f}")
    st.write("Meta:", r.get("meta", {}))
    st.write("Timestamp:", r.get("timestamp"))
    st.write("Received at (saved):", r.get("received_at"))

    # show map if coords present and non-zero
    meta = r.get("meta", {})
    lat_m = meta.get("lat")
    lon_m = meta.get("lon")
    # Accept 0.0 coords if user actually provided them; otherwise check not None
    if lat_m is not None and lon_m is not None:
        try:
            m = folium.Map(location=[float(lat_m), float(lon_m)], zoom_start=12)
            folium.Marker([float(lat_m), float(lon_m)], popup=f"{r['label']} ({r['confidence']:.2f})").add_to(m)
            st_folium(m, width=700, height=400)
        except Exception:
            st.info("Could not render map for given coordinates.")
else:
    st.info("No prediction yet. Upload an image and press Predict (or use the sample image).")

# Show a compact history for quick review
st.markdown("---")
st.subheader("Recent Predictions (session)")

if st.session_state["history"]:
    for h in st.session_state["history"][:20]:
        st.write(f"{h['time']} — **{h['label']}** — confidence: {h['confidence']:.3f}")
else:
    st.write("No history yet in this session.")
