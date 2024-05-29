import streamlit as st

# Sample user database
USER_DB = {
    "user1": "password1",
    "user2": "password2"
}

# Function to check login credentials
def check_login(username, password):
    return USER_DB.get(username) == password

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ''

# Define secrets directly within the code
secrets = {
    "user1": "password1",
    "user2": "password2"
}

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
    if username in secrets:
        password = secrets[username]
    else:
        password = ""
    password = st.text_input("Password", value=password, type="password")
    
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful")
        else:
            st.error("Invalid username or password")

def main_page():
    st.markdown("<h1>🩺 Welcome to the Main Page</h1>", unsafe_allow_html=True)
    st.write(f"Hello, {st.session_state.username}!")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.experimental_rerun()

if st.session_state.logged_in:
    main_page()
else:
    login()
