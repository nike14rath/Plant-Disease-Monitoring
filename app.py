import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load trained model
model = tf.keras.models.load_model("Trained_model.keras")

# Class names
class_name = [
'Apple___Apple_scab',
'Apple___Black_rot',
'Apple___Cedar_apple_rust',
'Apple___healthy',
'Blueberry___healthy',
'Cherry_(including_sour)___Powdery_mildew',
'Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
'Corn_(maize)___Common_rust_',
'Corn_(maize)___Northern_Leaf_Blight',
'Corn_(maize)___healthy',
'Grape___Black_rot',
'Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
'Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)',
'Peach___Bacterial_spot',
'Peach___healthy',
'Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy',
'Potato___Early_blight',
'Potato___Late_blight',
'Potato___healthy',
'Raspberry___healthy',
'Soybean___healthy',
'Squash___Powdery_mildew',
'Strawberry___Leaf_scorch',
'Strawberry___healthy',
'Tomato___Bacterial_spot',
'Tomato___Early_blight',
'Tomato___Late_blight',
'Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot',
'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
'Tomato___Tomato_mosaic_virus',
'Tomato___healthy'
]

# Title
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to detect disease")

# Image uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Convert BGR to RGB if needed
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # Display image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize for model
    image_resized = cv2.resize(img, (128,128))

    # Convert to array
    input_arr = np.array(image_resized)
    input_arr = np.expand_dims(input_arr, axis=0)

    # Prediction
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)

    # Get disease name
    model_prediction = class_name[result_index]

    # Display result
    st.subheader("Prediction Result")
    st.success(f"Disease Name: {model_prediction}")