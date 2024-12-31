import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
try:
    model = tf.keras.models.load_model('regression.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Load the encoders and scaler
try:
    with open('lb_encoder_gender.pkl', 'rb') as file:
        lb_encoder_gender = pickle.load(file)

    with open('onehot_encode_geo.pkl', 'rb') as file:
        onehot_encode_geo = pickle.load(file)

    with open('stscaler.pkl', 'rb') as file:
        stscaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading preprocessing files: {e}")

# Streamlit app
st.title('Estimated Salary Prediction')

# User Inputs
geography = st.selectbox('Geography', onehot_encode_geo.categories_[0])
gender = st.selectbox('Gender', lb_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0, 1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
try:
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [lb_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'Exited': [exited]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encode_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encode_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = stscaler.transform(input_data)

    # Predict salary
    prediction = model.predict(input_data_scaled)
    predicted_salary = prediction[0][0]

    st.write(f"Predicted Estimated Salary: ${predicted_salary:.2f}")

except Exception as e:
    st.error(f"Error during prediction: {e}")
