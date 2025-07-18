import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("house_price_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_names = joblib.load("feature_names.joblib")

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Prediction App")
st.markdown("Fill in the details below to predict the house price.")

bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
sqft_living = st.number_input("Living Area (sqft)", min_value=300, max_value=10000, value=1500)
floors = st.selectbox("Floors", options=[1, 2, 3])
waterfront = st.selectbox("Waterfront View", options=[0, 1])
view = st.slider("View Rating", 0, 4, 0)
condition = st.slider("Condition Rating", 1, 5, 3)
sqft_basement = st.number_input("Basement Area (sqft)", min_value=0, max_value=5000, value=0)
yr_built = st.number_input("Year Built", min_value=1900, max_value=2023, value=2000)
yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=2023, value=0)
year = st.number_input("Sale Year", min_value=2000, max_value=2025, value=2023)
month = st.slider("Sale Month", 1, 12, 7)

total_rooms = bedrooms + bathrooms
house_age = year - yr_built
is_renovated = 1 if yr_renovated > 0 else 0
luxury = 1 if sqft_living > 3000 else 0
price_per_sqft = 0

input_dict = {
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'sqft_living': sqft_living,
    'floors': floors,
    'waterfront': waterfront,
    'view': view,
    'condition': condition,
    'sqft_basement': sqft_basement,
    'yr_built': yr_built,
    'yr_renovated': yr_renovated,
    'year': year,
    'month': month,
    'total_rooms': total_rooms,
    'house_age': house_age,
    'is_renovated': is_renovated,
    'luxury': luxury,
    'price_per_sqft': price_per_sqft
}

input_df = pd.DataFrame([input_dict])
input_df = input_df[feature_names]
scaled_input = scaler.transform(input_df.values)

if st.button("Predict Price"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"Estimated House Price: ₹ {np.round(prediction, 2):,}")