
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

numerical_col = ['carat', 'depth', 'table', 'x', 'y', 'z']
categorical_col = ['cut', 'color', 'clarity']

# Loading the pre-trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

# Defining the lists for dropdowns
cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

# Streamlit app title
st.title('Diamond Price Prediction')

# User input selection for categorical features
selected_cut = st.selectbox('Select a cut:', cut_categories)
selected_color = st.selectbox('Select a color:', color_categories)
selected_clarity = st.selectbox('Select a clarity:', clarity_categories)

# User input for numerical features
carat = st.number_input('Carat:', min_value=0.0)
depth = st.number_input('Depth:', min_value=0.0)
table = st.number_input('Table:', min_value=0.0)
x = st.number_input('x:', min_value=0.0)
y = st.number_input('y:', min_value=0.0)
z = st.number_input('z:', min_value=0.0)

# Encode the categorical variables
encoded_features = encoder.transform([[selected_cut, selected_color, selected_clarity]])

# Create the input DataFrame with both numerical and categorical (encoded) features
input_data = pd.DataFrame({
    'carat': [carat],
    'cut': [encoded_features[0][0]],   # Cut is encoded
    'color': [encoded_features[0][1]], # Color is encoded
    'clarity': [encoded_features[0][2]], # Clarity is encoded
    'depth': [depth],
    'table': [table],
    'x': [x],
    'y': [y],
    'z': [z]
})

# Scale the numerical features
input_data[numerical_col] = scaler.transform(input_data[numerical_col])

# Prediction button
if st.button('Predict'):
    # Make the prediction using the pipeline
    prediction = model.predict(input_data)
    if prediction <0:
        st.write("Please Enter Valid Value")
    else :
        st.write(f'The predicted price of the diamond is: ${prediction[0]:,.2f}')


