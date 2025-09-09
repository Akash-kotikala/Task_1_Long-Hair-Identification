import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
from PIL import Image
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
import os

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load models with custom objects for compatibility
hair_model_path = r"C:\Users\Mahendra\Desktop\NullClass\LONG Hair Identification\hair_length_model.h5"
age_gender_model_path = r"C:\Users\Mahendra\Desktop\NullClass\LONG Hair Identification\utkface_model.h5"

try:
    hair_model = load_model(hair_model_path, custom_objects={'mse': MeanSquaredError})
    age_gender_model = load_model(age_gender_model_path, custom_objects={'mse': MeanSquaredError})
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Mock age scaler (replace with actual mean/std if available)
age_scaler = StandardScaler()
age_scaler.mean_ = np.array([50.0])  # Approximate mean age in UTKFace
age_scaler.scale_ = np.array([20.0])  # Approximate std dev

# Hair mask function (from previous code)
def create_hair_mask(image):
    try:
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Multiple HSV ranges for hair colors
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 50])
        mask_dark = cv2.inRange(img_hsv, lower_dark, upper_dark)
        
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([25, 200, 200])
        mask_brown = cv2.inRange(img_hsv, lower_brown, upper_brown)
        
        lower_blonde = np.array([20, 30, 100])
        upper_blonde = np.array([35, 150, 255])
        mask_blonde = cv2.inRange(img_hsv, lower_blonde, upper_blonde)
        
        mask = mask_dark | mask_brown | mask_blonde
        
        # Face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            y_top = max(0, y - int(h * 0.5))
            y_bottom = min(img_bgr.shape[0], y + h)
            x_left = max(0, x - int(w * 0.2))
            x_right = min(img_bgr.shape[1], x + w + int(w * 0.2))
            mask[0:y_top, :] = 0
            mask[y_bottom:, :] = 0
            mask[:, 0:x_left] = 0
            mask[:, x_right:] = 0
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        hair_region = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        hair_region_rgb = cv2.cvtColor(hair_region, cv2.COLOR_BGR2RGB)
        mask_pil = Image.fromarray(mask)
        hair_pil = Image.fromarray(hair_region_rgb)
        return hair_pil, mask_pil
    except Exception as e:
        st.warning(f"Error creating hair mask: {e}")
        return None, None

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_hair_length(image):
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = hair_model.predict(img_array, verbose=0)
    hair_length = 1 if prediction[0][0] > 0.5 else 0
    return hair_length, prediction[0][0]

def predict_age_gender(image_array):
    age_pred_scaled, gender_pred = age_gender_model.predict(image_array, verbose=0)
    age_pred = age_scaler.inverse_transform(age_pred_scaled)[0][0]
    actual_gender = 'Female' if gender_pred[0][0] >= 0.5 else 'Male'
    return int(age_pred), actual_gender

# Streamlit GUI
st.title("Long Hair Identification System")
st.write("Upload a face image to predict gender with age (20-30) and hair length rules.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Processing..."):
        # Generate hair mask
        hair_region, hair_mask = create_hair_mask(image)
        
        # Display hair mask and region
        if hair_region is not None and hair_mask is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(hair_mask, caption="Hair Mask", use_column_width=True)
            with col2:
                st.image(hair_region, caption="Hair Region", use_column_width=True)
        else:
            st.warning("Could not generate hair mask.")
        
        # Predict age and gender
        img_array = preprocess_image(image)
        age, actual_gender = predict_age_gender(img_array)
        
        # Predict hair length
        hair_length, confidence = predict_hair_length(image)
        hair_str = "Long" if hair_length == 1 else "Short"
        
        # Apply rules
        predicted_gender = actual_gender
        if 20 <= age <= 30:
            if hair_length == 1:
                predicted_gender = 'Female'
            elif hair_length == 0 and actual_gender == 'Female':
                predicted_gender = 'Male'
        
        # Display results
        st.subheader("Predictions:")
        st.write(f"**Estimated Age:** {age}")
        st.write(f"**Hair Length:** {hair_str} (Confidence: {confidence:.2f})")
        st.write(f"**Actual Predicted Gender:** {actual_gender}")
        st.write(f"**Final Gender (with rules):** {predicted_gender}")