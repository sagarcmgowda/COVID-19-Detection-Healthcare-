import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="X-ray Classification App",
    layout="centered"
)

st.title("🩻 X-ray Image Classification")
st.write("Upload or capture an X-ray image to get a prediction")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("efficientnet_covid_model.keras")

model = load_model()

# -----------------------------
# Class Names (MUST match training order)
# -----------------------------
class_names = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]

IMG_SIZE = (224, 224)

# -----------------------------
# Image Input Options
# -----------------------------
st.subheader("📥 Choose Image Input Method")

option = st.radio(
    "Select input method:",
    ("Upload Image", "Use Camera")
)

image = None

# -----------------------------
# Upload Image Option
# -----------------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload X-ray Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# -----------------------------
# Camera Capture Option
# -----------------------------
elif option == "Use Camera":
    camera_image = st.camera_input("Capture X-ray Image")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# -----------------------------
# Prediction
# -----------------------------
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess (NO preprocess_input here – already in model)
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Output
    st.subheader("Prediction Result")
    st.success(f"**Class:** {class_names[predicted_class]}")
    st.info(f"**Confidence:** {confidence:.2f}")

    # Probability breakdown
    st.subheader("Class Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i]:.4f}")
