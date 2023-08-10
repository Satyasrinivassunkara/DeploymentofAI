#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load California Housing Prices dataset
def load_data():
    data = fetch_california_housing(as_frame=True)
    return data

# Split data into features (X) and target (y)
def split_data(data):
    X = data.data
    y = data.target
    return X, y

# Train a Linear Regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Main function
def main():
    st.title('California Housing Prices: Regression Analysis')

    # Load data
    data = load_data()

    # Split data
    X, y = split_data(data)

    # User input for test size
    test_size = st.sidebar.slider('Test Size', 0.1, 0.5, 0.2, 0.05)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write('## Model Performance')
    st.write(f'Mean Squared Error: {mse:.2f}')
    st.write(f'R-squared: {r2:.2f}')

    # Plot predictions vs. actual
    st.write('## Predictions vs. Actual')
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Predictions vs. Actual')
    st.pyplot(fig)

if __name__ == '__main__':
    main()


# In[ ]:




