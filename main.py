import streamlit as st
from tensorflow.keras.applications import ResNet50 #type:ignore
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions #type:ignore
import numpy as np
import cv2

# Sample user database (initially pre-populated)
if 'secrets' not in st.session_state:
    st.session_state.secrets = {
        "user1": "password1",
        "user2": "password2"
    }

# Function to check login credentials
def check_login(username, password):
    return st.session_state.secrets.get(username) == password

# Function to sign up new users
def sign_up(username, password):
    if username in st.session_state.secrets:
        return False  # User already exists
    st.session_state.secrets[username] = password
    return True

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ''

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #000033, #000066, #000099, #0000cc, #0000ff);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .stTextInput > div > div > input {
        color: black;
    }
    .stButton button {
        color: white;
        background-color: #0044cc;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: white;
        color: #0044cc;
    }
    .stTitle h1 {
        color: white;
        text-align: center;
    }
    .centered-title {
        text-align: center;
        font-size: 2.5em;
        margin-top: 0;
        margin-bottom: 0.5em;
    }
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #0044cc 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def login():

    # Center the image with padding
    st.markdown("<div style='text-align: center; margin-left: 100px; margin-right: auto;'>", unsafe_allow_html=True)
    st.image("imgs/logo.png", width=150)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h1><center>Login Page<center></h1>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful")
        else:
            st.error("Invalid username or password")

    if st.button("Go to Sign Up"):
        st.session_state.show_signup = True
        st.experimental_rerun()




def signup():
    st.markdown("<h1>🩺 Sign Up Page</h1>", unsafe_allow_html=True)

    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if password != confirm_password:
            st.error("Passwords do not match")
        else:
            if sign_up(username, password):
                st.success("Sign up successful. Please login.")
                st.session_state.show_signup = False
                st.experimental_rerun()
            else:
                st.error("Username already exists")

    if st.button("Go to Login"):
        st.session_state.show_signup = False
        st.experimental_rerun()

def main_page():
    st.markdown("<h1 style='text-align: center; color: blue;'>DoctorAI</h1>", unsafe_allow_html=True)

    # Move "Hi, user!" to the top right corner
    st.markdown("<div style='position: absolute; top: 10px; right: 10px; color: white;'>Hi, " + st.session_state.username + "!</div>", unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose your Doctor", ["Disease Diagnosis", "Personalized Medicine", "Symptom Checker", "Support Page"])

    if app_mode == "Disease Diagnosis":
        st.header("Disease Diagnosis from Medical Images")
        uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            if st.button("Diagnose"):
                # Read the image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                # Display the uploaded image
                st.image(img, caption='Uploaded Chest X-ray', use_column_width=True)
                # Preprocess the image
                img = cv2.resize(img, (224, 224))
                img = preprocess_input(img)
                img = np.expand_dims(img, axis=0)
                # Load pre-trained ResNet50 model
                model = ResNet50(weights='imagenet')
                # Make prediction
                prediction = model.predict(img)
                decoded_predictions = decode_predictions(prediction, top=1)[0]
                result = decoded_predictions[0][1]
                st.write(f"Diagnosis result: {result}")

    elif app_mode == "Personalized Medicine":
        st.header("Personalized Medicine Recommendations")
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
        weight = st.number_input("Weight (kg)", min_value=0, max_value=200, value=70)
        height = st.number_input("Height (cm)", min_value=0, max_value=250, value=170)
        # Add more patient characteristics as needed
        if st.button("Recommend Treatment"):
            patient_data = [age, weight, height]  # Extend this list based on your model's requirements
            treatment = recommend_treatment(patient_data)
            st.write(f"Recommended treatment: {treatment}")

    elif app_mode == "Symptom Checker":
        st.header("Symptom Checker Chatbot")
        user_symptoms = st.text_area("Describe your symptoms")
        if st.button("Check Symptoms"):
            diagnosis = symptom_checker(user_symptoms)
            st.write(diagnosis)

    elif app_mode == "Support Page":
        st.markdown('<div class="centered-title">Support Page ☎️</div>', unsafe_allow_html=True)
        st.write("If you have any questions or need support, please fill out the form below and we will get back to you as soon as possible.")

        with st.form(key='support_form'):
            name = st.text_input("Name")
            email = st.text_input("Email")
            category = st.selectbox("Category", ["Technical Issue", "Account Issue", "General Inquiry", "Other"])
            message = st.text_area("Message")

            submit_button = st.form_submit_button(label='Submit')

            if submit_button:
                st.write("Thank you for your submission!")
                st.write("**Name:**", name)
                st.write("**Email:**", email)
                st.write("**Category:**", category)
                st.write("**Message:**", message)

        st.subheader("Submitted Queries")
        if submit_button:
            st.write("**Name:**", name)
            st.write("**Email:**", email)
            st.write("**Category:**", category)
            st.write("**Message:**", message)

    st.write(f"Hello, {st.session_state.username}!")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.experimental_rerun()

def recommend_treatment(patient_data):
    # Placeholder implementation
    return "Standard Treatment Plan"

def symptom_checker(user_symptoms):
    # Placeholder implementation
    return "Possible Diagnosis: Common Cold"

if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False

if st.session_state.logged_in:
    main_page()
else:
    if st.session_state.show_signup:
        signup()
    else:
        login()
