import streamlit as st
import pickle
import numpy as np
import os

# Load the trained model using a relative path (compatible with deployment)
MODEL_PATH = "crop_recommendation_model.pkl"

# Check if the model file exists before loading
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
else:
    st.error(f"❌ Model file not found! Please upload 'crop_recommendation_model.pkl'.")
    st.stop()

# Set page configuration
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="centered")

# Title of the app
st.markdown("<h1 style='text-align: center;'>Crop Recommendation System 🌱</h1>", unsafe_allow_html=True)

# Input fields
st.markdown("### Enter the following details:")
col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, format="%.2f", help="Enter Nitrogen content in the soil")
with col2:
    P = st.number_input("Phosphorus (P)", min_value=0.0, format="%.2f", help="Enter Phosphorus content in the soil")
with col3:
    K = st.number_input("Potassium (K)", min_value=0.0, format="%.2f", help="Enter Potassium content in the soil")

col4, col5, col6 = st.columns(3)

with col4:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, format="%.2f", help="Enter temperature in Celsius")
with col5:
    humidity = st.number_input("Humidity (%)", min_value=0.0, format="%.2f", help="Enter humidity percentage")
with col6:
    ph = st.number_input("pH", min_value=0.0, format="%.2f", help="Enter the pH value of the soil")

rainfall = st.number_input("Rainfall (cm)", min_value=0.0, format="%.2f", help="Enter the rainfall in cm")

# Predict button
if st.button("Predict"):
    try:
        # Create input array for prediction
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict the crop
        prediction = model.predict_proba(input_data)[0]
        top_3_indices = prediction.argsort()[-3:][::-1]
        top_3_crops = [model.classes_[i] for i in top_3_indices]

        # Display the recommended crops
        st.markdown("## 🌾 Recommended Crops:")
        for i, crop in enumerate(top_3_crops, 1):
            st.markdown(f"**{i}) {crop}**")
    except ValueError:
        st.error("❌ Please enter valid numerical values.")
