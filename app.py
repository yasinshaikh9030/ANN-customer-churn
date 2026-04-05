import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

model = load_model("model.h5")
with open("labelencoder_gender.pkl", "rb") as file:
    labelencoder_gender = pickle.load(file)
with open("onehotEncoder_geo.pkl", "rb") as file:
    onehotencoder_geo = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# steamlit app
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox("Geography", onehotencoder_geo.categories_[0])
gender = st.selectbox("Gender", labelencoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

input_data = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        "Gender": [labelencoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
    }
)
geo_encoded = onehotencoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, columns=onehotencoder_geo.get_feature_names_out(["Geography"])
)

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
# Fix column order
# input_data = input_data.reindex(columns=scaler.feature_names_in_)
input_data_scaled = scaler.transform(input_data)

# prediction
predection = model.predict(input_data_scaled)
predection_prob = predection[0][0]

st.write("churn probability:", predection_prob)
if predection_prob > 0.5:
    st.write(f"Customer is likely to churn with a probability of {predection_prob:.2f}")
else:
    st.write(
        f"Customer is unlikely to churn with a probability of {predection_prob:.2f}"
    )
