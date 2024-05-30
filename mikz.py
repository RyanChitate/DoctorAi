import streamlit as st
import tensorflow as tf #type:ignore
from tensorflow.keras.applications import DenseNet121 #type:ignore
from tensorflow.keras.preprocessing import image #type:ignore
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions #type:ignore
from tensorflow.keras.models import load_model #type:ignore
import numpy as np #type:ignore
import joblib #type:ignore
import nltk #type:ignore
from nltk.tokenize import word_tokenize #type:ignore
from nltk.corpus import stopwords #type:ignore
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained DenseNet121 model + higher level layers
def load_densenet_model():
    base_model = DenseNet121(weights='imagenet')
    return base_model

disease_model = load_densenet_model()

# Function to preprocess image for DenseNet121
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Function for disease diagnosis
def diagnose_disease(image_file):
    img = preprocess_image(image_file)
    prediction = disease_model.predict(img)
    decoded_predictions = decode_predictions(prediction, top=1)[0]
    return decoded_predictions[0][1]  # Return the predicted class name

# Dummy personalized medicine model
def load_dummy_personalized_medicine_model():
    class DummyPersonalizedMedicineModel:
        def predict(self, data):
            return ["Standard treatment"]  # Dummy recommendation
    return DummyPersonalizedMedicineModel()

personalized_medicine_model = load_dummy_personalized_medicine_model()

# Function for personalized medicine recommendation
def recommend_treatment(patient_data):
    prediction = personalized_medicine_model.predict([patient_data])
    return prediction[0]

# Simple symptom checker using keyword matching (dummy example)
def symptom_checker(symptoms):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(symptoms)
    words = [w.lower() for w in words if w.isalpha() and w.lower() not in stop_words and w not in string.punctuation]

    symptom_db = {
        "fever": "You might have the flu.",
        "headache": "It could be a migraine.",
        "cough": "You might have a common cold.",
        "sore throat": "You might have a throat infection."
    }

    diagnosis = "No diagnosis found."
    for word in words:
        if word in symptom_db:
            diagnosis = symptom_db[word]
            break
           
    return diagnosis

# Streamlit app
st.markdown("<h1 style='text-align: center; color: blue;'>🩺 DoctorAI</h1>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("*Your AI-powered medical assistant*")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose your Doctor", ["🩺 Disease Diagnosis", "💊 Personalized Medicine", "🩹 Symptom Checker"])

if app_mode == "🩺 Disease Diagnosis":
    st.header("🩺 Disease Diagnosis from Medical Images")
    uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("Diagnose"):
            result = diagnose_disease(uploaded_file)
            st.write(f"Diagnosis result: {result}")

elif app_mode == "💊 Personalized Medicine":
    st.header("💊 Personalized Medicine Recommendations")
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    weight = st.number_input("Weight (kg)", min_value=0, max_value=200, value=70)
    height = st.number_input("Height (cm)", min_value=0, max_value=250, value=170)
    # Add more patient characteristics as needed
    if st.button("Recommend Treatment"):
        patient_data = [age, weight, height]  # Extend this list based on your model's requirements
        treatment = recommend_treatment(patient_data)
        st.write(f"Recommended treatment: {treatment}")

elif app_mode == "🩹 Symptom Checker":
    st.header("🩹 Symptom Checker Chatbot")
    user_symptoms = st.text_area("Describe your symptoms")
    if st.button("Check Symptoms"):
        diagnosis = symptom_checker(user_symptoms)
        st.write(diagnosis)
