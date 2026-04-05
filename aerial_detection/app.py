import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# ── PAGE CONFIG ─────────────────────────────
st.set_page_config(
    page_title="🦅 Bird vs Drone AI",
    layout="wide"
)

# ── CUSTOM CSS (MODERN UI) ──────────────────
st.markdown("""
<style>
.big-title {
    font-size:42px !important;
    font-weight:700;
}
.card {
    padding:20px;
    border-radius:15px;
    background-color:#111;
    box-shadow: 0 0 15px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🦅 Bird vs Drone Detection System</p>', unsafe_allow_html=True)
st.caption("YOLOv8 • Real-time AI Detection • Hackathon Ready 🚀")

# ── LOAD MODEL ─────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # ⚠️ same folder me hona chahiye

model = load_model()

# ── SIDEBAR SETTINGS ───────────────────────
st.sidebar.header("⚙️ Settings")

CONF_THRESHOLD = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.6
)

# ── FILE UPLOAD ────────────────────────────
uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="🖼 Original Image", use_column_width=True)

    # ── PREDICTION ─────────────────────────
    results = model.predict(img_np, conf=0.25)

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:

        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)

        box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        cls = int(boxes.cls[best_idx].cpu().numpy())
        conf = confidences[best_idx]

        class_names = model.names
        label = class_names[cls]

        if conf >= CONF_THRESHOLD:
            # ── DRAW BOX ───────────────────
            x1, y1, x2, y2 = box
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                img_np,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            with col2:
                st.image(img_np, caption="🎯 Detection Result", use_column_width=True)

                st.markdown("### 🔍 Final Prediction")

                if label.lower() == "bird":
                    st.success(f"🟢 Bird Detected")
                else:
                    st.error(f"🔴 Drone Detected")

                st.metric("Confidence", f"{conf*100:.2f}%")

                # Download button
                result_img = Image.fromarray(img_np)
                st.download_button(
                    label="⬇️ Download Result",
                    data=result_img.tobytes(),
                    file_name="result.png",
                    mime="image/png"
                )

        else:
            with col2:
                st.image(img, caption="⚠️ Low Confidence", use_column_width=True)
                st.warning("Detection found but confidence is too low")
                st.info("Try another image or reduce threshold")

    else:
        with col2:
            st.image(img, caption="❌ No Detection", use_column_width=True)
            st.error("No object detected")

# ── FOOTER ────────────────────────────────
st.markdown("---")
st.caption("Built with ❤️ using YOLOv8 + Streamlit")