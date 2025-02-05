import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import pickle

model = load_model("battery_life_model.h5") 
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)



def predict_battery_life(type_discharge, Capacity, Re, Rct, label_encoder, scaler, model):
    type_discharge_encoded = label_encoder.transform([type_discharge])[0]
    X_input = np.array([[type_discharge_encoded,Capacity, Re, Rct]])
    X_input_scaled = scaler.transform(X_input)

    predicted_battery_life = model.predict(X_input_scaled)

    return predicted_battery_life[0]

# streamlit for frontend
st.title("Battery Life Prediction using ANN")

# User input fields
type_discharge = st.selectbox("Select Discharge Type", ['charge', 'discharge', 'impedance'])
Capacity = st.number_input("Enter Capacity", min_value=0.0)
Re = st.number_input("Enter Re", min_value=-1e12, max_value=1e12)
Rct = st.number_input("Enter Rct", min_value=-1e12, max_value=1e12)

# Button to make prediction
if st.button('Predict Battery Life'):
    predicted_battery_life = predict_battery_life(type_discharge, Capacity, Re, Rct, label_encoder, scaler, model)
    st.write(f"The predicted battery life is: {predicted_battery_life} units")