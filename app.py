import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender1 = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo1 = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler1 = pickle.load(file)

# ----------------- Web App ----------------- #

# Custom Styling
st.set_page_config(page_title="Churn Prediction", layout="centered")
st.markdown("""
    <style>
        .title {
            font-size:40px;
            color:#0c4b8e;
            text-align:center;
            font-weight:600;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            color: #999;
            font-size: 13px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">💼 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown("---")

# Layout with Columns
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo1.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender1.classes_)
    age = st.slider('🎂 Age', 18, 92)
    credit_score = st.number_input('💳 Credit Score', min_value=300.0, max_value=900.0, step=1.0)
    balance = st.number_input('💰 Balance', step=100.0)

with col2:
    estimated_salary = st.number_input('💼 Estimated Salary', step=1000.0)
    tenure = st.slider('⏳ Tenure (Years)', 0, 10)
    num_of_products = st.slider('📦 Number of Products', 1, 4)
    has_cr_card = st.selectbox('💳 Has Credit Card?', ['No', 'Yes'])
    is_active_member = st.selectbox('✅ Active Member?', ['No', 'Yes'])

# Convert string inputs to numerical
has_cr_card = 1 if has_cr_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender1.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo1.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo1.get_feature_names_out(['Geography']))

# Merge everything
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale
input_data_scaled = scaler1.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Show result
st.markdown("---")
st.subheader("🔍 Prediction Result")
st.write(f"**Churn Probability:** `{prediction_proba:.2f}`")

if prediction_proba > 0.5:
    st.error("⚠️ The customer is **likely to churn.**")
else:
    st.success("✅ The customer is **not likely to churn.**")

# Footer
st.markdown('<div class="footer">Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)
