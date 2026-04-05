import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import requests

# ── Page Config ─────────────────────────────
st.set_page_config(
    page_title="🦅 Bird vs Drone Detection",
    page_icon="🚁",
    layout="wide"
)

# ── Title UI ───────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>🦅 Bird vs Drone AI Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>YOLOv8 • Real-time Detection • Clean UI</p>",
    unsafe_allow_html=True
)

# ── Download Model from GitHub ─────────────
MODEL_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/best.pt"

def download_model():
    if not os.path.exists("best.pt"):
        with open("best.pt", "wb") as f:
            f.write(requests.get(MODEL_URL).content)

download_model()

# ── Load Model ─────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ── Sidebar ────────────────────────────────
st.sidebar.title("⚙️ Settings")
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25)

# ── Upload Image ───────────────────────────
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("### 🔍 Detection Result")

        img_array = np.array(image)

        results = model.predict(
            source=img_array,
            conf=confidence,
            save=False
        )

        result_img = results[0].plot()
        st.image(result_img, caption="Detected Output", use_column_width=True)

        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            st.success(f"✅ {len(boxes)} Object(s) Detected")

            for box in boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                label = model.names[cls_id]

                st.write(f"🔹 **{label}** ({conf_score*100:.2f}%)")
        else:
            st.warning("⚠️ No object detected")

# ── Footer ────────────────────────────────
st.markdown("---")
st.markdown("🚀 Built using YOLOv8 + Streamlit")
