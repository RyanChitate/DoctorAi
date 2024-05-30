import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run():
    # Title
    st.title('Personalized Treatment Recommendation')

    # Load synthetic dataset (Replace with your own dataset)
    data = pd.DataFrame({
        'Age': [35, 45, 55, 40, 50, 60, 30, 55, 65, 50, 42, 38, 47, 53, 58],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male',
                   'Male', 'Female', 'Male', 'Female', 'Female'],
        'Smoker': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No',
                   'Yes', 'No', 'No', 'Yes', 'No'],
        'Blood Pressure': ['High', 'Normal', 'High', 'Normal', 'High', 'Normal', 'High', 'Normal', 'High', 'Normal',
                           'High', 'Normal', 'High', 'Normal', 'Normal'],
        'Cholesterol': ['High', 'Normal', 'Normal', 'High', 'Normal', 'High', 'Normal', 'High', 'Normal', 'High',
                        'High', 'Normal', 'Normal', 'High', 'Normal'],
        'Symptoms': ['Pain', 'Fatigue', 'Dizziness', 'Nausea', 'Pain', 'Fatigue', 'Dizziness', 'Nausea', 'Pain',
                     'Fatigue', 'Dizziness', 'Nausea', 'Pain', 'Fatigue', 'Dizziness'],
        'Treatment': ['Physical Therapy', 'Medication', 'Surgery', 'Medication', 'Physical Therapy', 'Surgery',
                      'Medication', 'Physical Therapy', 'Surgery', 'Medication',
                      'Physical Therapy', 'Surgery', 'Medication', 'Physical Therapy', 'Surgery']
    })

    # Display dataset
    st.write("Example Dataset:")
    st.write(data)

    # Feature engineering
    X = data.drop(columns=['Treatment'])
    X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy variables
    y = data['Treatment']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train decision tree classifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # User input for prediction
    st.write('Enter patient characteristics:')
    age = st.slider('Age', 0, 120, 30)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    smoker = st.selectbox('Smoker', ['Yes', 'No'])
    blood_pressure = st.selectbox('Blood Pressure', ['Normal', 'High'])
    cholesterol = st.selectbox('Cholesterol', ['Normal', 'High'])
    symptoms = st.multiselect('Symptoms', ['Pain', 'Fatigue', 'Dizziness', 'Nausea'])

    # Compute button
    compute_button = st.button('Get Personalized Treatment')

    if compute_button:
        # Create an input DataFrame for prediction with the same structure as the training data
        input_data = {
            'Age': age,
            'Gender_Male': 1 if gender == 'Male' else 0,
            'Smoker_Yes': 1 if smoker == 'Yes' else 0,
            'Blood Pressure_High': 1 if blood_pressure == 'High' else 0,
            'Cholesterol_High': 1 if cholesterol == 'High' else 0
        }
        for symptom in ['Pain', 'Fatigue', 'Dizziness', 'Nausea']:
            input_data[symptom] = 1 if symptom in symptoms else 0

        input_df = pd.DataFrame([input_data])

        # Align the columns of input_df with the training data
        missing_cols = set(X_train.columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0

        input_df = input_df[X_train.columns]

        # Make prediction
        prediction = model.predict(input_df)

        # Display recommendation
        st.write(f'Recommended Treatment: {prediction[0]}')

if __name__ == '_main_':
    run()