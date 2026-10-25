import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

st.set_page_config(page_title="Wild Animal Detector", page_icon="🐾", layout="centered")

# Beautiful Background & UI Styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #f8ffae, #43c6ac);
        background-attachment: fixed;
    }
    .main-title {
        color: #1f4068;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-title'>🐾 Wild Animal Detection AI</h1>", unsafe_allow_html=True)
st.write("Upload an image of a Cat, Elephant, Lion, Panda, or Tiger!")

# Google Drive se model automatically download karne ka function
@st.cache_resource
def load_my_model():
    model_path = 'animal_model.h5'
    if not os.path.exists(model_path):
        # Yahan apnay Google Drive file ki ID dalein (e.g., 1A2B3C4D...)
        file_id = 'https://drive.google.com/file/d/1itvBWD5p0-STTjcm9tc_UIPjyghanRAn/view?usp=sharing'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    
    model = tf.keras.models.load_model(model_path)
    return model

with st.spinner("Downloading & Loading AI Model... Please wait."):
    model = load_my_model()

class_names = ['cats', 'elephants', 'lions', 'pandas', 'tigers']

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100

    st.write("---")
    
    # Error Handling (Non-animal ya kam confidence par error)
    if confidence < 60.0:
        st.error("❌ **Error: This is not a recognized animal!** Please upload a valid animal photo.")
    else:
        st.success(f"✅ Prediction: **{predicted_class.upper()}**")
        st.info(f"🔍 Confidence Score: **{confidence:.2f}%**")
