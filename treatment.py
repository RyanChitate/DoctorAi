import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Title
st.title('Personalized Treatment Recommendation')

# Load synthetic dataset (Replace with your own dataset)
data = pd.DataFrame({
    'Age': [35, 45, 55, 40, 50, 60, 30, 55, 65, 50],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male'],
    'Smoker': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
    'Treatment': ['A', 'B', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B']  # Example treatment plan
})

# Display dataset
st.write("Example Dataset:")
st.write(data)

# Feature engineering
X = data.drop(columns=['Treatment'])
X = pd.get_dummies(X)  # Convert categorical variables to dummy variables
y = data['Treatment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# User input for prediction
st.write('Enter patient characteristics:')
age = st.number_input('Age', min_value=0, max_value=120, value=30)
gender = st.selectbox('Gender', ['Male', 'Female'])
smoker = st.selectbox('Smoker', ['Yes', 'No'])

# Make prediction
input_data = {'Age': age, 'Gender_Male': 1 if gender == 'Male' else 0, 'Gender_Female': 1 if gender == 'Female' else 0, 'Smoker_Yes': 1 if smoker == 'Yes' else 0, 'Smoker_No': 1 if smoker == 'No' else 0}
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)

# Display recommendation
st.write(f'Recommended Treatment Plan: {prediction[0]}')
