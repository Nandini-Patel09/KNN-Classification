import streamlit as st
import pickle
import numpy as np

# Load files
model = pickle.load(open("./models/knn_model.pkl", "rb"))
scaler = pickle.load(open("./models/scaler.pkl", "rb"))
encoder = pickle.load(open("./models/encoder.pkl", "rb"))

st.title("Iris Flower Classification using KNN")

# Inputs
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Predict
if st.button("Predict"):

    features = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)

    flower_name = encoder.inverse_transform(prediction)

    st.success(f"Predicted Flower: {flower_name[0]}")