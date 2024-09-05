import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained LightGBM model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the selected features
selected_features = [
    ' ROA(C) before interest and depreciation before interest',
 ' Net Value Per Share (B)',
 ' Operating Profit Per Share (Yuan ¥)',
 ' Per Share Net profit before tax (Yuan ¥)',
 ' Net worth/Assets',
 ' Retained Earnings to Total Assets',
 ' Total expense/Assets',
 ' Current Liability to Current Assets',
 ' Liability-Assets Flag',
 " Net Income to Stockholder's Equity"
]

# Set up Streamlit app
st.set_page_config(page_title="Bankruptcy Prediction App", layout="wide")
st.title('🏦 Bankruptcy Prediction App')
st.write("""
This app predicts the likelihood of bankruptcy based on financial metrics.
Please enter the values for the selected features in the sidebar.
""")

# Sidebar for user input
st.sidebar.header('Input Features')
inputs = []
for feature in selected_features:
    value = st.sidebar.number_input(f'{feature}', value=0.0)
    inputs.append(value)

# Convert input to numpy array
input_data = np.array(inputs).reshape(1, -1)

# Standardize the input data
input_data_scaled = scaler.transform(input_data)

# Main panel
st.header('Input Features Overview')
input_df = pd.DataFrame(input_data, columns=selected_features)
st.write(input_df)

#  Predict button
if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0][1]
    
    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.write('Prediction: 🛑 Bankrupt')
    else:
        st.write('Prediction: ✅ Not Bankrupt')
    st.write(f'Probability of Bankruptcy: {probability:.2f}')
# Add a footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    color: black;
    text-align: center;
    padding: 10px;
}
</style>
<div class="footer">
    <p>Developed by Jiyashan Pathan</p>
</div>
""", unsafe_allow_html=True)
