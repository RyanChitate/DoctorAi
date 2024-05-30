import streamlit as st
import model  # Ensure the model.py is in the same directory or properly referenced

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
        try:
            model.run()
        except Exception as e:
            st.error(f"An error occurred while running the model: {e}")
            st.stop()

    elif app_mode == "Personalized Medicine":
        import treatment
        treatment.run()

    elif app_mode == "Symptom Checker":
        import chat
        chat.main()

    elif app_mode == "Support Page":
        import support
        support.run()

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.experimental_rerun()

if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False

if st.session_state.logged_in:
    main_page()
else:
    if st.session_state.show_signup:
        signup()
    else:
        login()
