import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved components
columns = pickle.load(open("columns.pkl", "rb"))
model = pickle.load(open("svr_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏠 Co-Living Rent Price Estimator")

# Inputs
location = st.selectbox("Location", ["Mumbai", "Pune", "Bangalore"])
room_type = st.radio("Room Type", ["Shared", "Private"])
ac = st.checkbox("AC")
wifi = st.checkbox("WiFi")
food = st.checkbox("Food Provided")
distance = st.slider("Distance from college (in km)", 0.1, 10.0, 1.0)
season = st.selectbox("Season", ["Winter", "Summer", "Monsoon"])

if st.button("Predict Rent"):
    # Convert inputs into proper format
    input_dict = {
        "Distance": distance,
        "AC": int(ac),
        "WiFi": int(wifi),
        "Food": int(food),
        "RoomType_Private": int(room_type == "Private"),
        "Location_Mumbai": int(location == "Mumbai"),
        "Location_Pune": int(location == "Pune"),
        "Season_Summer": int(season == "Summer"),
        "Season_Winter": int(season == "Winter")
    }

    input_df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure correct column order
    input_df = input_df[columns]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"💰 Estimated Monthly Rent: ₹{int(prediction[0])}")
