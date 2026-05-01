# 🔥 Digit + Alphabet Recognizer

A deep learning app that recognizes handwritten digits (0–9)
and uppercase letters (A–Z) using a CNN trained on EMNIST Balanced.

## 🚀 Live Demo
[Click here to try it](https://your-app-url.streamlit.app)

## 🧠 Model
- Architecture: Custom CNN (3 Conv blocks + Dense layers)
- Dataset: EMNIST Balanced (36 classes)
- Accuracy: ~93%

## 🛠️ Run Locally
pip install -r requirements.txt
streamlit run app.py

## 📁 Project Structure
digit-alpha-recognizer/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── digit_alpha_model.keras