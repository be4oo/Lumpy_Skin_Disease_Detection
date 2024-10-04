import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# Load the model
with open('best_automl_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define your features
features = [
    'Longitude', 'Latitude', 'Cloud_Cover_Percentage', 'Diurnal_Temp_Range',
    'Frost_Days', 'Evapotranspiration', 'Precipitation_Amount', 'Mean_Temp',
    'Wet_Days_Count', 'elevation', 'Human_Population', 'Cattle_Population', 'vpd'
]

def process_input(input_data):
    # Add any preprocessing steps here if needed
    return input_data

# Streamlit app
st.title('Lumpy Skin Disease Prediction')

# Input fields for user
st.header('Enter Environmental and Population Data')

# Input fields
input_data = {}
for feature in features:
    if feature in ['Longitude', 'Latitude']:
        input_data[feature] = st.number_input(f'Enter {feature}', format='%.6f')
    elif feature in ['Human_Population', 'Cattle_Population']:
        input_data[feature] = st.number_input(f'Enter {feature}', min_value=0.0, format='%.6f')
    else:
        input_data[feature] = st.number_input(f'Enter {feature}', format='%.6f')

# Predict button
if st.button('Predict'):
    # Create a DataFrame from input
    input_df = pd.DataFrame([input_data])
    
    # Process the input data
    processed_input = process_input(input_df)
    
    # Make prediction
    prediction_proba = model.predict_proba(processed_input)[0]
    prediction = model.predict(processed_input)[0]
    
    # Display the prediction
    if prediction == 1:
        st.success(f'The predicted class is: Positive')
        st.info(f'Probability of Positive: {prediction_proba[1]:.4f}')
        st.info(f'Probability of Negative: {prediction_proba[0]:.4f}')
    else:
        st.success(f'The predicted class is: Negative')
        st.info(f'Probability of Negative: {prediction_proba[0]:.4f}')
        st.info(f'Probability of Positive: {prediction_proba[1]:.4f}')
    
    # Display the processed input (optional, for debugging)
    st.write("Processed Input:")
    st.write(processed_input)

# Instructions
st.sidebar.header('Instructions')
st.sidebar.info(
    'Fill in the environmental and population data in the input fields on the left. '
    'Click the "Predict" button to get the Lumpy Skin Disease prediction.'
)

# About
st.sidebar.header('About')
st.sidebar.info(
    'This app predicts the likelihood of Lumpy Skin Disease based on various environmental '
    'and population factors. It uses an Ensemble model trained on historical data.'
)