import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Digit + Alphabet Recognizer",
    page_icon="🔥",
    layout="centered"
)

# ── Load Model ────────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    model_path = 'digit_alpha_model.keras'
    if not os.path.exists(model_path):
        st.error("Model file not found! Make sure digit_alpha_model.keras is in the folder.")
        st.stop()
    return load_model(model_path)

model = load_cnn_model()

# ── Class Labels ──────────────────────────────────────────
classes_list = ['0','1','2','3','4','5','6','7','8','9',
                'A','B','C','D','E','F','G','H','I','J',
                'K','L','M','N','O','P','Q','R','S','T',
                'U','V','W','X','Y','Z']

# ── Preprocessing ─────────────────────────────────────────
def preprocess(img_array):
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    img_array = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)

    if np.mean(img_array) > 127:
        img_array = cv2.bitwise_not(img_array)

    _, img_array = cv2.threshold(img_array, 100, 255, cv2.THRESH_BINARY)
    img_array = img_array.astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# ── UI ────────────────────────────────────────────────────
st.title("🔥 Digit + Alphabet Recognizer")
st.markdown("### Recognizes handwritten **0–9** and **A–Z** using a CNN model")
st.markdown("---")

# ── Tabs: Upload OR Camera ────────────────────────────────
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Use Camera"])

def show_results(img_array, image_display):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image_display, width=200)

    x    = preprocess(img_array)
    pred = model.predict(x, verbose=0)[0]
    top5_idx = np.argsort(pred)[-5:][::-1]

    top_label = classes_list[top5_idx[0]]
    top_prob  = pred[top5_idx[0]] * 100

    with col2:
        st.subheader("Prediction")
        st.markdown(f"""
        <div style='text-align:center; padding:20px;
                    background:#1e1e2e; border-radius:12px;'>
            <h1 style='color:#a6e3a1; font-size:80px; margin:0'>{top_label}</h1>
            <p style='color:#cdd6f4; font-size:18px'>{top_prob:.1f}% confident</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Top 5 Predictions")

    for rank, idx in enumerate(top5_idx):
        label = classes_list[idx]
        prob  = float(pred[idx]) * 100
        col_rank, col_label, col_bar, col_pct = st.columns([0.5, 0.5, 5, 1.5])
        with col_rank:
            st.markdown(f"**#{rank+1}**")
        with col_label:
            st.markdown(f"**{label}**")
        with col_bar:
            st.progress(min(int(prob), 100))
        with col_pct:
            st.markdown(f"`{prob:.1f}%`")

    with st.expander("🔍 What the model sees (28×28)"):
        processed = preprocess(img_array)[0, :, :, 0]
        st.image(processed, width=140,
                 caption="Preprocessed input fed to CNN")

# Tab 1 — Upload
with tab1:
    uploaded = st.file_uploader(
        "Upload a clear image of ONE digit or letter",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded:
        image   = Image.open(uploaded).convert("RGB")
        img_arr = np.array(image)
        show_results(img_arr, image)

# Tab 2 — Camera
with tab2:
    camera_img = st.camera_input("Take a photo of a digit or letter")
    if camera_img:
        image   = Image.open(camera_img).convert("RGB")
        img_arr = np.array(image)
        show_results(img_arr, image)

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray'>Built with TensorFlow + Streamlit "
    "| Trained on EMNIST Balanced Dataset</p>",
    unsafe_allow_html=True
)