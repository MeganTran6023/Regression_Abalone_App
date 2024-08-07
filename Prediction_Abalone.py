# # Running
#conda activate dp
# python -m streamlit run your_script.py

#imports
import streamlit as st
import pandas as pd
import numpy as np

 
# File uploader
# upload_file = st.file_uploader('Upload CSV file.')
# dataf = pd.read_csv(upload_file)
# st.dataframe(df,width=1800, height = 1200)


#Tutorial based off of: https://github.com/dataprofessor/code/blob/master/streamlit/part2/Abalone-ml-app.py

st.write("""

# Simple Abalone Flower Prediction App

This app predicts the **Abalone tree** age!
""")

st.write("""

## Abalone Dataset
""")

#View Abalone csv file
data_ab = pd.read_csv("abalone.csv") #path folder of the data file
# one hot encoder with gender

st.write(data_ab) #displays the table of data

st.sidebar.header('User Input Parameters')
#function for input parameters
def user_input_features():
    length = st.sidebar.slider('Length', 0.0 , 1.0, 0.5)
    diameter = st.sidebar.slider('Diameter', 0.0 , 1.0, 0.5)
    height = st.sidebar.slider('Height', 0.0 , 2.0, 1.0)
    

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)


# Show table of features and value user input:


#### Dataset

#1 - initializing Variables

X = data_ab.drop(columns=["Rings"])
y = data_ab["Rings"]

#2 - Logistic Regression Model

#imports
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X, y)