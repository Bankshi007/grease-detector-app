import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Path to your trained model
MODEL_PATH = "D:/Engineering Support/ML/grease_model.h5"

# Load model for inference only
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Streamlit page config
st.set_page_config(page_title="Image Detection App", layout="centered")
st.title("🔍 Image Detection App")
st.write("Upload an image and let the model detect its class.")

# Class names must match training labels
class_names = ["Alarm", "Normal", "Warning"]
target_size = (224, 224)

# Preprocessing function (CLAHE + resize + normalize)
def preprocess_image(pil_img):
    # Convert PIL to NumPy array (RGB)
    img = np.array(pil_img.convert("RGB"))
    # Convert to LAB and apply CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab2 = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    # Resize and normalize
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Detecting...")

    # Preprocess
    img_array = preprocess_image(image)

    # Predict
    try:
        preds = model.predict(img_array)
        # Handle list/tuple output
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds = np.array(preds)
        # Flatten if needed
        if preds.ndim > 2:
            preds = preds.reshape(preds.shape[0], -1)
        scores = preds[0]
        class_idx = int(np.argmax(scores))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Display probabilities
    st.write("**Class probabilities:**")
    for name, s in zip(class_names, scores):
        st.write(f"- {name}: {s * 100:.2f}%")

    # Display final prediction
    confidence = scores[class_idx] * 100
    st.success(f"Prediction: {class_names[class_idx]} ({confidence:.2f}%)")

# Dependencies:
# pip install streamlit tensorflow pillow numpy opencv-python
