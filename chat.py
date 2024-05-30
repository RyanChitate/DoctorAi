import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define symptoms
symptoms = {
    "Fever": ["fever", "high temperature"],
    "Cough": ["cough"],
    "Headache": ["headache"],
    "Fatigue": ["fatigue", "tiredness"],
    "Nausea": ["nausea"],
    "Muscle Pain": ["muscle pain", "body ache"],
    "Sore Throat": ["sore throat"],
    "Runny Nose": ["runny nose", "nasal congestion"],
    "Shortness of Breath": ["shortness of breath", "difficulty breathing"],
    "Chills": ["chills", "shivering"],
    "High Fever": ["high fever", "very high temperature"],
    "Persistent Cough": ["persistent cough", "continual cough"],
    "Severe Headache": ["severe headache"],
    "Extreme Fatigue": ["extreme fatigue", "extreme tiredness"],
    "Nausea and Vomiting": ["nausea and vomiting"],
    "Muscle Weakness": ["muscle weakness"],
    "Difficulty Breathing": ["difficulty breathing", "breathing difficulty"],
    "Chest Pain": ["chest pain"],
    "Confusion or Disorientation": ["confusion", "disorientation"],
    "Unconsciousness": ["unconsciousness", "fainting"],
    "Seizures": ["seizures", "convulsions"],
    "Chest Tightness": ["chest tightness", "tight chest"]
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

# Bot responses
GREETINGS_IN = ["hello", "hi", "hey", "hola", "welcome", "good morning", "good afternoon", "good evening"]
GREETINGS_OUT = ["Hello!", "Hi!", "Hey!", "Nice to see you!", "Welcome!", "Greetings!"]

THANKS_IN = ["thank you", "thanks", "appreciate it", "thanks a lot"]
THANKS_OUT = ["You're welcome!", "No problem!", "Happy to help!", "My pleasure!"]

FAREWELLS_IN = ["bye", "goodbye", "see you later", "see ya", "take care"]
FAREWELLS_OUT = ["Goodbye!", "See you later!", "Take care!", "Bye-bye!"]

# Function to lemmatize text
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Lemmatize text
    text = lemmatize_text(text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and single-character tokens
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Function to calculate the diagnosis based on symptoms
def calculate_diagnosis(input_text):
    input_text = preprocess_text(input_text)
    selected_symptoms = []
    for symptom, keywords in symptoms.items():
        for keyword in keywords:
            if keyword in input_text:
                selected_symptoms.append(symptom)
                break
    if not selected_symptoms:
        return [("No diagnosis", 0)]  # If no symptoms match, return a message
    diagnoses = {disease: min(92, len(set(symptoms).intersection(selected_symptoms)) / len(diseases[disease]) * 100) for disease, symptoms in diseases.items()}
    sorted_diagnosis = sorted(diagnoses.items(), key=lambda x: x[1], reverse=True)[:3]
    return sorted_diagnosis

# Function to generate bot response
def get_bot_response(user_input, conversation_history):
    user_input = preprocess_text(user_input)
    if any(greeting in user_input for greeting in GREETINGS_IN):
        return random.choice(GREETINGS_OUT), conversation_history
    elif any(thanks in user_input for thanks in THANKS_IN):
        return random.choice(THANKS_OUT), conversation_history
    elif any(farewell in user_input for farewell in FAREWELLS_IN):
        return random.choice(FAREWELLS_OUT), conversation_history
    else:
        diagnosis = calculate_diagnosis(user_input)
        response = ""
        for idx, (disease, percentage) in enumerate(diagnosis, 1):
            if idx == 1 and percentage == 0:
                percentage = 0
                response += f"No diagnosis (0%)\n"
                response += "Please contact a healthcare professional immediately for further assistance.\n\n"
            else:
                response += f"{idx}. {disease} ({percentage:.2f}%)\n"
                response += get_advice_for_disease(disease) + "\n\n"
        conversation_history.insert(0, {"user": user_input, "bot": response.strip()})
        return response.strip(), conversation_history

# Function to get advice for a diagnosed disease
def get_advice_for_disease(disease):
    advice_map = {
        "Common Cold": "Common colds are usually mild and self-limiting. You can manage symptoms by staying hydrated, getting plenty of rest, and using over-the-counter cold medications. If symptoms persist or worsen, consult a healthcare professional.",
        "Flu": "Influenza (flu) can be severe. It's essential to rest, stay hydrated, and take over-the-counter medications to alleviate symptoms. Consider seeing a doctor if symptoms worsen, especially if you have difficulty breathing, persistent chest pain, or confusion.",
        "Migraine": "Migraines can be debilitating. To manage migraines, try to identify and avoid triggers, practice relaxation techniques, and take prescribed medications as directed. If migraines are severe or frequent, consult a healthcare professional for personalized treatment.",
        "COVID-19": "COVID-19 can range from mild to severe. If you suspect you have COVID-19, self-isolate immediately and monitor your symptoms. Seek medical attention if you experience difficulty breathing, persistent chest pain, confusion, or bluish lips or face.",
        "Pneumonia": "Pneumonia can be serious, especially in older adults and individuals with weakened immune systems. Follow your doctor's treatment plan, which may include antibiotics, rest, and hydration. Seek medical help if you have difficulty breathing or chest pain.",
        "Meningitis": "Meningitis is a medical emergency. If you suspect meningitis, seek immediate medical attention. Symptoms may include high fever, severe headache, neck stiffness, nausea, and sensitivity to light. Treatment typically involves hospitalization and antibiotics.",
        "Heart Attack": "A heart attack requires immediate medical attention. Call emergency services if you experience chest pain or discomfort, shortness of breath, nausea, lightheadedness, or discomfort in other areas of the upper body. Follow your doctor's advice for managing heart health.",
        "Stroke": "A stroke is a medical emergency. If you suspect a stroke, remember the acronym FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services. Act quickly, as prompt treatment is crucial for minimizing damage.",
        "Bronchitis": "Bronchitis can cause coughing, wheezing, and chest discomfort. Get plenty of rest, stay hydrated, and use over-the-counter medications to relieve symptoms. If symptoms persist or worsen, consult a healthcare professional.",
        "Asthma": "Asthma requires ongoing management. Follow your asthma action plan, avoid triggers, and take prescribed medications regularly. Seek medical help if you experience severe asthma symptoms or have difficulty controlling your condition.",
        "Allergic Rhinitis": "Allergic rhinitis (hay fever) can cause sneezing, runny nose, and itchy eyes. Try to avoid allergens, use antihistamines or nasal corticosteroids, and consider allergy testing for long-term management.",
        "Epilepsy": "Epilepsy treatment aims to prevent seizures and improve quality of life. Take prescribed medications regularly, get enough sleep, and manage stress. Follow up with your healthcare provider for adjustments to your treatment plan as needed."
    }

    return advice_map.get(disease, "Follow up with a healthcare professional for personalized advice.")

# CSS styles
st.markdown(
    """
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        margin-bottom: 10px;
    }
    .user-bubble {
        background-color: #5F9EA0;
        color: white;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 10px;
        max-width: 70%;
    }
    .bot-bubble {
        background-color: #FFA07A;
        color: white;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 10px;
        max-width: 70%;
        align-self: flex-end;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main function to run the Streamlit app
def main():
    st.title("Symptom Diagnosis Chatbot")

    # Initialize session state for conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # User input for conversation
    user_input = st.text_input("You can talk to me here:")

    # If user inputs something
    if user_input:
        # Get bot's response and update conversation history
        bot_response, st.session_state.conversation_history = get_bot_response(user_input, st.session_state.conversation_history)

    # Display conversation history in reverse order
    for msg in reversed(st.session_state.conversation_history):
        if msg["user"]:
            st.markdown('<div class="chat-container"><div class="user-bubble">' + msg["user"] + '</div></div>', unsafe_allow_html=True)
        if msg["bot"]:
            st.markdown('<div class="chat-container"><div class="bot-bubble">' + msg["bot"] + '</div></div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
