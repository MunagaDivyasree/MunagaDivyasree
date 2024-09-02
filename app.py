import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load and prepare the model
@st.cache_data
def load_model():
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('minmax', MinMaxScaler()),
        ('selector', SelectKBest(score_func=f_classif, k=10)),
        ('classifier', RandomForestClassifier())
    ])

    # Train the model
    pipeline.fit(X_train, y_train)
    return pipeline, X.columns  # Returning the model and original feature names

model, feature_names = load_model()

# Streamlit app
st.title('Breast Cancer Prediction')
st.write('Enter the values for the following features:')

# Create input fields for each feature
feature_inputs = {}
for feature in feature_names:
    feature_value = st.number_input(f'{feature}', value=0.0)
    feature_inputs[feature] = feature_value

# Convert input data to DataFrame
input_df = pd.DataFrame([feature_inputs])

# Apply the same transformations as during training
input_transformed = model[:-1].transform(input_df)  # Exclude the classifier step

# Predict using the model
if st.button('Predict'):
    prediction = model.named_steps['classifier'].predict(input_transformed)
    prediction_proba = model.named_steps['classifier'].predict_proba(input_transformed)
    st.write(f'Prediction: {"Malignant" if prediction[0] == 1 else "Benign"}')
    st.write(f'Prediction Probabilities: {prediction_proba}')
