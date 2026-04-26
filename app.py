import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction App")
st.write("Enter house details:")

# Inputs (match dataset EXACTLY)
sqft = st.number_input("Square Footage", min_value=300, value=2000)
bed = st.number_input("Bedrooms", min_value=1, value=3)
bath = st.number_input("Bathrooms", min_value=1, value=2)
year = st.number_input("Year Built", min_value=1900, value=2010)
lot = st.number_input("Lot Size", min_value=500.0, value=4000.0)
garage = st.number_input("Garage Size", min_value=0, value=1)
neigh = st.slider("Neighborhood Quality", 1, 10, 5)

# Predict
if st.button("Predict Price"):
    
    data = np.array([[sqft, bed, bath, year, lot, garage, neigh]])
    
    pred = model.predict(data)
    price = pred[0]

    # Basic sanity check
    if price <= 0:
        st.warning("⚠️ Invalid prediction. Try different inputs.")
    else:
        st.success(f"💰 Estimated Price: ₹ {price:,.2f}")

    # Debug (optional)
    with st.expander("Debug Info"):
        st.write("Input Data:", data)
        st.write("Prediction:", price)