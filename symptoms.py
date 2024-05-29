import streamlit as st
import pandas as pd

# Define symptoms and their weights
symptoms = {
    "Fever": 8,
    "Cough": 9,
    "Headache": 7,
    "Fatigue": 8,
    "Nausea": 6,
    "Muscle Pain": 7,
    "Sore Throat": 6,
    "Runny Nose": 5,
    "Shortness of Breath": 10,
    "Chills": 7,
    "High Fever": 9,
    "Persistent Cough": 10,
    "Severe Headache": 9,
    "Extreme Fatigue": 9,
    "Nausea and Vomiting": 8,
    "Muscle Weakness": 8,
    "Difficulty Breathing": 10,
    "Chest Pain": 10,
    "Confusion or Disorientation": 9,
    "Unconsciousness": 10,
    "Seizures": 10,
    "Chest Tightness": 9
}

# Define diseases and their associated symptoms
diseases = {
    "Common Cold": ["Runny Nose", "Sore Throat"],
    "Flu": ["Fever", "Cough", "Muscle Pain", "Chills"],
    "Migraine": ["Severe Headache", "Nausea"],
    "COVID-19": ["Fever", "Cough", "Shortness of Breath", "Fatigue", "Muscle Pain", "Chills"],
    "Pneumonia": ["High Fever", "Persistent Cough", "Difficulty Breathing", "Extreme Fatigue", "Chest Pain"],
    "Meningitis": ["High Fever", "Severe Headache", "Nausea and Vomiting", "Confusion or Disorientation"],
    "Heart Attack": ["Chest Pain", "Nausea and Vomiting", "Extreme Fatigue", "Muscle Weakness", "Shortness of Breath", "Confusion or Disorientation"],
    "Stroke": ["Severe Headache", "Difficulty Breathing", "Confusion or Disorientation", "Unconsciousness"],
    "Bronchitis": ["Cough", "Fatigue", "Shortness of Breath", "Chest Tightness"],
    "Asthma": ["Cough", "Shortness of Breath", "Chest Tightness", "Muscle Weakness"],
    "Allergic Rhinitis": ["Runny Nose", "Sore Throat", "Fatigue"],
    "Epilepsy": ["Seizures", "Confusion or Disorientation"],
}

# Function to calculate the diagnosis based on symptoms
def calculate_diagnosis(selected_symptoms):
    scores = {disease: sum(symptoms[symptom] for symptom in diseases[disease] if symptom in selected_symptoms) for disease in diseases}
    total_score = sum(scores.values())
    percentages = {disease: score / total_score * 100 for disease, score in scores.items()}
    # Sort the diagnoses by percentage and return only the top 3
    sorted_diagnosis = sorted(percentages.items(), key=lambda x: x[1], reverse=True)[:3]
    return {disease: percentage for disease, percentage in sorted_diagnosis}

# Main function to run the Streamlit app
def main():
    st.title("Symptom Diagnosis App")

    # Display checkboxes for symptoms
    st.sidebar.title("Select Symptoms")
    selected_symptoms = [st.sidebar.checkbox(symptom) for symptom in symptoms.keys()]

    # Calculate diagnosis when 'Diagnose' button is clicked
    if st.sidebar.button("Diagnose"):
        diagnosis = calculate_diagnosis([symptom for symptom, selected in zip(symptoms.keys(), selected_symptoms) if selected])
        
        # Display top 3 diagnoses
        st.subheader("Top 3 Diagnoses:")
        for disease, percentage in diagnosis.items():
            st.write(f"- {disease}: {percentage:.2f}%")

        # Visualize the diagnosis percentages
        df = pd.DataFrame(list(diagnosis.items()), columns=["Disease", "Percentage"])
        st.bar_chart(df.set_index("Disease"))

# Run the app
if __name__ == "__main__":
    main()
