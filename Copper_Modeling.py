# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data Preprocessing
def preprocess_data(data):
    # Handle missing values
    data = data.dropna()

    # Handle categorical variables
    data = pd.get_dummies(data, columns=['category'])

    # Normalize or scale numerical features
    data['feature'] = (data['feature'] - data['feature'].mean()) / data['feature'].std()

    return data

# Exploratory Data Analysis (EDA)
def perform_eda(data):
    # Descriptive statistics
    st.write("Descriptive Statistics")
    st.write(data.describe())

    # Visualization
    st.write("Data Visualization")
    st.bar_chart(data['column'])

    # Correlation matrix
    st.write("Correlation Matrix")
    sns.heatmap(data.corr(), annot=True)

# Model Development
def train_model(data):
    # Split the data into training and testing sets
    X = data.drop('target_variable', axis=1)
    y = data['target_variable']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

# Streamlit Application
def main():
    # Load the data
    data = pd.read_csv("Copper_Set.csv")

    # Preprocess the data
    data = preprocess_data(data)

    # Perform EDA
    perform_eda(data)

    # Train the model
    model, mse, r2 = train_model(data)

    # Streamlit UI
    st.title("Industrial Copper Modeling")
    st.write("Mean Squared Error:", mse)
    st.write("R-squared Score:", r2)

    # User input and prediction
    st.sidebar.title("Make a Prediction")
    # Example: input_feature = st.sidebar.slider('Feature', min_value, max_value)
    # Example: prediction = model.predict([[input_feature]])

    # Display the prediction
    st.write("Prediction:", prediction)

if __name__ == '__main__':
    main()
