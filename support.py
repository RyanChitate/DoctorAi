import streamlit as st

st.markdown(
    """
    <style>
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
    """, 
    unsafe_allow_html=True
)


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

