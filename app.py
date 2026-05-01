import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os
import sys

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Digit + Alphabet Recognizer",
    page_icon="🔥",
    layout="wide"
)

with st.sidebar:
    st.header("App settings")
    dark_mode = st.toggle("Dark mode", value=False)
    st.divider()
    st.header("How to use")
    st.markdown(
        """
        1. Upload a clear image **or** take a photo.
        2. Make sure there’s only **one** character.
        3. Dark ink on a light background works best.
        """
    )
    st.divider()
    st.subheader("Model info")
    st.caption("Pretrained LeNet‑5 CNN on EMNIST Balanced (36 classes).")
    st.divider()
    st.subheader("Runtime")
    st.write(f"Python: {sys.version.split()[0]}")

bg = "#0b1220" if dark_mode else "#f7f9fc"
panel = "#111827" if dark_mode else "#ffffff"
text = "#f8fafc" if dark_mode else "#101828"
muted = "#cbd5e1" if dark_mode else "#475467"
border = "#1f2937" if dark_mode else "#eaecf0"
badge_bg = "#1f2937" if dark_mode else "#f2f4f7"
badge_text = "#e2e8f0" if dark_mode else "#344054"

st.markdown(
    f"""
    <style>
    .main {{
        background: {bg};
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }}
    h1, h2, h3, h4 {{
        color: {text};
    }}
    .hero {{
        background: {panel};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.08);
        margin-bottom: 1.2rem;
    }}
    .hero-title {{
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }}
    .hero-subtitle {{
        color: {muted};
        font-size: 1.05rem;
        margin-bottom: 0.8rem;
    }}
    .badge {{
        display: inline-block;
        padding: 0.25rem 0.6rem;
        margin-right: 0.4rem;
        border-radius: 999px;
        background: {badge_bg};
        color: {badge_text};
        font-size: 0.85rem;
        font-weight: 600;
    }}
    .prediction-card {{
        text-align: center;
        padding: 1.3rem 1rem;
        background: linear-gradient(135deg, #0b1220, #1f2937);
        border-radius: 16px;
        color: #f8fafc;
    }}
    .prediction-label {{
        font-size: 4rem;
        font-weight: 800;
        margin: 0;
        color: #a6e3a1;
    }}
    .prediction-confidence {{
        margin: 0.3rem 0 0 0;
        color: #cbd5f5;
        font-size: 1rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ── Load Model ────────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    model_path = 'digit_alpha_model.keras'
    if not os.path.exists(model_path):
        st.error("Model file not found! Make sure digit_alpha_model.keras is in the folder.")
        st.stop()
    return load_model(model_path, compile=False)

model = load_cnn_model()

# ── Class Labels ──────────────────────────────────────────
classes_list = ['0','1','2','3','4','5','6','7','8','9',
                'A','B','C','D','E','F','G','H','I','J',
                'K','L','M','N','O','P','Q','R','S','T',
                'U','V','W','X','Y','Z']

SIMILAR_PAIRS = {
    ('B', '8'), ('O', '0'), ('S', '5'), ('Z', '2'), ('I', '1'),
    ('G', '6'), ('Q', 'O'), ('D', '0'), ('C', 'G'), ('T', '7')
}

# ── Preprocessing ─────────────────────────────────────────
def preprocess(img_array):
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()

    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        pad = int(0.15 * max(w, h))
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, cleaned.shape[1])
        y1 = min(y + h + pad, cleaned.shape[0])
        cropped = cleaned[y0:y1, x0:x1]
    else:
        cropped = cleaned

    h, w = cropped.shape
    size = max(h, w) + 10
    square = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = cropped

    moments = cv2.moments(square)
    if moments["m00"] != 0:
        c_x = moments["m10"] / moments["m00"]
        c_y = moments["m01"] / moments["m00"]
        shift_x = (size / 2) - c_x
        shift_y = (size / 2) - c_y
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        square = cv2.warpAffine(square, matrix, (size, size), borderValue=0)

    final = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    input_tensor = final.astype("float32") / 255.0
    input_tensor = input_tensor.reshape(1, 28, 28, 1)

    return {
        "gray": gray,
        "cleaned": cleaned,
        "centered": square,
        "final": final,
        "input": input_tensor
    }

@st.cache_data(show_spinner=False)
def generate_sample_image(char, size=200):
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 4.2 if char.isalpha() else 4.5
    thickness = 8
    text_size = cv2.getTextSize(char, font, scale, thickness)[0]
    text_x = max((size - text_size[0]) // 2, 5)
    text_y = (size + text_size[1]) // 2
    cv2.putText(canvas, char, (text_x, text_y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return canvas

# ── UI ────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🔥 Digit + Alphabet Recognizer</div>
        <div class="hero-subtitle">Recognizes handwritten <b>0–9</b> and <b>A–Z</b> using a CNN model.</div>
        <div>
            <span class="badge">EMNIST Balanced</span>
            <span class="badge">36 Classes</span>
            <span class="badge">CNN Inference</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "This is a **pretrained LeNet‑5** model. Predictions are based on what the model learned during training."
)

# ── Tabs: Upload OR Camera ────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "📷 Use Camera", "✨ Samples"])

def show_results(img_array, image_display):
    with st.spinner("Analyzing image..."):
        processed = preprocess(img_array)
        pred = model.predict(processed["input"], verbose=0)[0]

    preview_cols = st.columns(3)
    with preview_cols[0]:
        st.image(image_display, caption="Original", use_container_width=True)
    with preview_cols[1]:
        st.image(processed["cleaned"], caption="Cleaned (threshold)", use_container_width=True)
    with preview_cols[2]:
        st.image(processed["final"], caption="Final 28×28", use_container_width=True)

    col1, col2 = st.columns(2)

    top3_idx = np.argsort(pred)[-3:][::-1]
    top_labels = [classes_list[idx] for idx in top3_idx]
    top_probs = [float(pred[idx]) * 100 for idx in top3_idx]

    with col1:
        with st.container(border=True):
            st.subheader("Prediction")
            st.markdown(
                f"""
                <div class='prediction-card'>
                    <div class='prediction-label'>{top_labels[0]}</div>
                    <div class='prediction-confidence'>{top_probs[0]:.1f}% confident</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if top_probs[0] < 60:
                st.warning("Low confidence prediction. Try a clearer image.")

            if len(top_labels) > 1:
                close = abs(top_probs[0] - top_probs[1]) <= 10
                similar = (top_labels[0], top_labels[1]) in SIMILAR_PAIRS or (top_labels[1], top_labels[0]) in SIMILAR_PAIRS
                if close and similar:
                    st.info(f"Possible result: {top_labels[0]} or {top_labels[1]}")

    with col2:
        with st.container(border=True):
            st.subheader("Top 3 Predictions")
            for rank, (label, prob) in enumerate(zip(top_labels, top_probs), start=1):
                st.markdown(f"**{rank}) {label} — {prob:.1f}%**")

            chart_data = [
                {"label": label, "prob": prob}
                for label, prob in zip(top_labels, top_probs)
            ]

            st.subheader("📈 Confidence chart")
            st.vega_lite_chart(
                {
                    "mark": {"type": "bar", "cornerRadiusEnd": 4},
                    "data": {"values": chart_data},
                    "encoding": {
                        "x": {"field": "label", "type": "ordinal", "sort": None},
                        "y": {"field": "prob", "type": "quantitative", "title": "Confidence (%)"},
                        "tooltip": [
                            {"field": "label", "type": "ordinal", "title": "Class"},
                            {"field": "prob", "type": "quantitative", "title": "Confidence (%)", "format": ".1f"}
                        ]
                    }
                },
                use_container_width=True
            )

# Tab 1 — Upload
with tab1:
    uploaded = st.file_uploader(
        "Drag & drop or browse a clear image of ONE digit or letter",
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

# Tab 3 — Samples
with tab3:
    st.subheader("Try a few examples")
    st.caption("These are synthetic samples. Real handwriting may look different.")
    sample_chars = ["0", "2", "5", "8", "A", "M", "Z"]
    cols = st.columns(3)
    for i, char in enumerate(sample_chars):
        sample_img = generate_sample_image(char)
        sample_pil = Image.fromarray(sample_img)
        with cols[i % 3]:
            st.image(sample_pil, width=160, caption=f"Sample {char}")
            if st.button(f"Use {char}", key=f"sample_{char}_{i}"):
                show_results(sample_img, sample_pil)

# ── Footer ────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray'>Built with TensorFlow + Streamlit "
    "| Trained on EMNIST Balanced Dataset</p>",
    unsafe_allow_html=True
)
