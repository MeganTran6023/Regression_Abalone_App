#imports

## core
import streamlit as st
import pandas as pd
import numpy as np
## sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# Kaggle dataset:
# https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset
# code referenced:
# https://github.com/dataprofessor/code/blob/master/streamlit/part2/iris-ml-app.py

#Title 
st.write("""

# Simple Abalone Tree Prediction App

This app predicts the **Abalone tree** age!
""")

st.write("""

## Abalone Dataset
""")

# Load Abalone dataset
data_ab = pd.read_csv("abalone_genderDict.csv")
# Replace with your correct file path

# Check if 'Rings' column exists before dropping
if 'Rings' in data_ab.columns:
    st.write(data_ab)
    X = data_ab.drop(columns=['Rings'])  # Separate features
    y = data_ab["Rings"]  # Target variable
else:
    st.write("The 'Rings' column is not found in the dataset. Please check your data source.")
    X = data_ab  # No target variable to process further

st.sidebar.header('User Input Parameters')


# Function for user input features
def user_input_features():
    sex = st.sidebar.selectbox('Sex', data_ab['Sex'].unique())
    length = st.sidebar.slider('Length', float(data_ab['Length'].min()),
                               float(data_ab['Length'].max()), 1.0)
    diameter = st.sidebar.slider('Diameter', float(data_ab['Diameter'].min()),
                                 float(data_ab['Diameter'].max()), 1.0)
    height = st.sidebar.slider('Height', float(data_ab['Height'].min()),
                               float(data_ab['Height'].max()), 2.0)
    whole_weight = st.sidebar.slider('Whole weight', float(data_ab['Whole weight'].min()),
                                    float(data_ab['Whole weight'].max()), 3.0)
    shucked_weight = st.sidebar.slider('Shucked weight', float(data_ab['Shucked weight'].min()),
                                     float(data_ab['Shucked weight'].max()), 2.0)
    viscera_weight = st.sidebar.slider('Viscera weight', float(data_ab['Viscera weight'].min()),
                                      float(data_ab['Viscera weight'].max()), 1.0)
    shell_weight = st.sidebar.slider('Shell weight', float(data_ab['Shell weight'].min()),
                                     float(data_ab['Shell weight'].max()), 1.0)

    user_data = {'Sex': sex,
                 'Length': length, 'Diameter': diameter, 'Height': height,
                 'Whole weight': whole_weight, 'Shucked weight': shucked_weight,
                 'Viscera weight': viscera_weight, 'Shell weight': shell_weight}

    df = pd.DataFrame(user_data, index=[0])



    return df


# Get user input
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

# Only proceed with training and prediction if 'Rings' exists
if 'Rings' in data_ab.columns:
    # Split data into training and testing sets (assuming 'Rings' is the target variable)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

    # Create and train the model (Linear Regression)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on user input data
    prediction = model.predict(df)

    # Round the predicted rings to the nearest whole number, with:
    #   - Positive values rounded up
    #   - Negative values rounded to zero (floor)
    rounded_prediction = np.where(prediction >= 0, np.round(prediction), 0)

    # Display the predicted rings
    st.subheader('Predicted Rings')
    st.write(rounded_prediction)  # Assuming a single prediction