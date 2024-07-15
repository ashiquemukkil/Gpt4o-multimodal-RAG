# brew install ffmpeg
import streamlit as st
from streamlit_modal import Modal
from streamlit_modal import use_modal

# Create a modal instance
modal = Modal(key="login_modal")

# Function to display the login popup
def show_login():
    modal.open()
    with modal.container():
        st.write("## Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "password":
                st.session_state["authenticated"] = True
                modal.close()
            else:
                st.error("Invalid username or password")

# Main app function
def main():
    # Initialize session state for authentication
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    # If not authenticated, show the login popup
    if not st.session_state["authenticated"]:
        show_login()
    else:
        st.write("Welcome to the main app!")
        if st.button("Logout"):
            st.session_state["authenticated"] = False

if __name__ == "__main__":
    main()

