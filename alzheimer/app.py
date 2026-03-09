import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

IMG_SIZE = 224

# Load Model
model = tf.keras.models.load_model("model/alzheimer_model.h5")

# Load class json
with open("F:\\DL\\alzheimer\\class_indices.json") as f:
    class_indices = json.load(f)

class_names = {v:k for k,v in class_indices.items()}

st.title("Alzheimer MRI Detection")

st.write("Upload MRI Image")

# Upload image
uploaded_file = st.file_uploader("Choose MRI Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    pred_index = np.argmax(prediction)

    pred_class = class_names[pred_index]

    confidence = np.max(prediction)

    st.subheader("Prediction")

    st.write("Class:", pred_class)

    st.write("Confidence:", round(confidence*100,2),"%")