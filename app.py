import streamlit as st
import numpy as np
import joblib
model = joblib.load('ridge_model.pkl')
st.title("Cancer Detection App")
st.markdown("Predict if the cancer is **Benign (0)** or **Malignant(1)** based on clinical data.")

# Input fields
age = st.number_input("AGE")
smoking = st.number_input("SMOKING")
coughing = st.number_input("COUGHING")

# Predict button
if st.button("Predict"):
  input_data = np.array([[age, smoking, coughing]])
  prediction = model.predict(input_data)[0]
  st.write("Prediction: Malignant" if prediction == 1 else "Prediction: Benign")
