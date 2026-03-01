import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
from tensorflow.keras.models import load_model
from PIL import Image


# Page Config

st.set_page_config(
    page_title="LaLiga Logo Classifier",
    layout="centered"
)

st.title("⚽ LaLiga Team Logo Classifier")
st.markdown("Upload a LaLiga team logo and the AI will predict the team.")


# Load Model

model = load_model("F:\DL\Laliga\model\laliga_logo_model.h5")


# Load Correct Class Order

with open("class_indices.json") as f:
    class_indices = json.load(f)

# IMPORTANT: sort by index value
class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]


# Upload Image

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    
    # Preprocess
  
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

  
    # Prediction
  
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    st.success(f"🏆 Predicted Team: {class_names[predicted_index]}")
    st.info(f"Confidence: {confidence*100:.2f}%")

   
    # Top 3 Predictions
   
    st.subheader("Top 3 Predictions")

    top3 = prediction[0].argsort()[-3:][::-1]

    for i in top3:
        st.write(f"{class_names[i]} → {prediction[0][i]*100:.2f}%")

  
    # Probability Bar Chart
  
    st.subheader("Prediction Probabilities")

    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(class_names, prediction[0])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.invert_yaxis()

    st.pyplot(fig)