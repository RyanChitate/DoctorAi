import streamlit as st

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
    </style>
    """, unsafe_allow_html=True)

def login():
    st.markdown("<h1>🩺 Login Page</h1>", unsafe_allow_html=True)

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
        st.rerun()

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
                st.rerun()
            else:
                st.error("Username already exists")

    if st.button("Go to Login"):
        st.session_state.show_signup = False
        st.rerun()

def main_page():
    st.markdown("<h1>🩺 Welcome to the Main Page</h1>", unsafe_allow_html=True)
    st.write(f"Hello, {st.session_state.username}!")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.rerun()

if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False

if st.session_state.logged_in:
    main_page()
else:
    if st.session_state.show_signup:
        signup()
    else:
        login()
