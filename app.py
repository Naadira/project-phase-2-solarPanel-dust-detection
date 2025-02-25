import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import subprocess
import os

# Download model if not present
if not os.path.isfile('solarpanelimageclassifier.h5'):
    subprocess.run(['curl --output solarpanelimageclassifier.h5 "https://media.githubusercontent.com/media/MuhammadAqhariNasrin/Optimizing-Solar-Panel-Efficiency-Computer-Vision-Approach-to-Dust-Detection-on-Solar-Panel/main/solarpanelimageclassifier.h5"'], shell=True)

# Load model
model = tf.keras.models.load_model('solarpanelimageclassifier.h5', compile=False)

# Streamlit UI
st.set_page_config(page_title="Solar Panel Classifier")
st.markdown("""
    <style>
        .title { text-align: center; font-size: 30px; font-weight: bold; color: #ffffff; background: linear-gradient(90deg, #ff9800, #f44336); padding: 10px; border-radius: 10px; }
        .upload-box { text-align: center; padding: 15px; border: 2px dashed #4CAF50; border-radius: 10px; background-color: #f1f8e9; }
        .result { text-align: center; font-size: 20px; font-weight: bold; padding: 15px; border-radius: 8px; color: white; }
        .button { display: flex; justify-content: center; padding-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='title'>Solar Panel Dust Classifier</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], help="Choose a solar panel image to classify")

def make_prediction(image):
    resize = tf.image.resize(image, (256, 256))
    yhat = model.predict(np.expand_dims(resize/255, 0))
    return yhat[0][0]

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img)
    prediction = make_prediction(img_array)
    result_text = "✅ Clean Panel" if prediction <= 0.5 else "🚨 Dusty Panel"
    result_color = "#4CAF50" if prediction <= 0.5 else "#D32F2F"
    st.markdown(f"<div class='result' style='background-color: {result_color};'>{result_text}</div>", unsafe_allow_html=True)

st.markdown("""
<div class='button'>
    <a href='https://drive.google.com/drive/folders/12Q3MBI8SPw0vHsO_kkS5izkxw0F7tXx4' target='_blank'>
        <button style='background-color:#1976D2; color:white; padding:10px 20px; border:none; border-radius:5px; cursor:pointer;'>📥 Download Sample Images</button>
    </a>
</div>
""", unsafe_allow_html=True)
