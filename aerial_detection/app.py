import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# ── Page Config ─────────────────────────────
st.set_page_config(
    page_title="🦅 Bird vs Drone Detection",
    page_icon="🚁",
    layout="wide"
)

# ── Custom CSS (Modern UI) ──────────────────
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
    }
    .sub {
        text-align: center;
        font-size: 18px;
        color: #cccccc;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🦅 Bird vs Drone AI System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">YOLO Detection • Real-time Prediction • Clean UI</div>', unsafe_allow_html=True)

# ── Model Load ─────────────────────────────
MODEL_PATH = "best.pt"   # <-- same folder me hona chahiye

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

yolo_model = load_model()

# ── Sidebar ────────────────────────────────
st.sidebar.title("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)

# ── File Upload ────────────────────────────
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("### 🔍 Detection Result")

        # Convert image to numpy
        img_array = np.array(image)

        # Prediction
        results = yolo_model.predict(
            source=img_array,
            conf=confidence,
            save=False
        )

        # Plot result
        result_img = results[0].plot()

        st.image(result_img, caption="Detected Output", use_column_width=True)

        # ── Show detections ─────────────────
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            st.success(f"✅ {len(boxes)} Object(s) Detected")

            for box in boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])

                label = yolo_model.names[cls_id]

                st.write(f"🔹 **{label}** ({conf_score*100:.2f}%)")
        else:
            st.warning("⚠️ No object detected")

# ── Footer ────────────────────────────────
st.markdown("---")
st.markdown("💡 Built with YOLOv8 • Streamlit • Deep Learning")
