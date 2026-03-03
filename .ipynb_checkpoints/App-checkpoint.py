import streamlit as st
import joblib
import numpy as np

model = joblib.load("models/revenue_model.pkl")
features = joblib.load("models/model_features.pkl")

st.title("Amazon Revenue Prediction")

input_data = []

for feature in features:
    value = st.number_input(f"Enter {feature}")
    input_data.append(value)

if st.button("Predict"):
    prediction = model.predict([input_data])
    st.success(f"Predicted Revenue: {prediction[0]}")