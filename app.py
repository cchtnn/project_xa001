"""
Diné College Assistant - Main Application
Refactored version with modular components and query type classification
"""

import streamlit as st
import time
import os
import logging
logging.getLogger("watchdog").setLevel(logging.ERROR)

# Import custom modules
from config import PAGE_CONFIG
from ui_components import (
    setup_page_config,
    load_custom_css,
    display_top_logo,
    display_header,
    setup_language_selector,
    create_language_dropdown,
    create_logout_button,
    # create_logout_handler,
    # check_logout_click,
    create_input_section,
    display_answer,
    display_qa_history,
    setup_containers,
    close_containers
)
from vectorstore_manager import initialize_vectorstore, get_collection
from session_manager import (
    initialize_session,
    set_user_query,
    set_language,
    add_to_history,
    get_language,
    get_user_query,
    clear_history,
    update_current_time
)
from query_handler import create_query_handler
import logic
from auth_db import init_auth_db, validate_user, add_user, list_users, get_user
from student_transcript_csv_handler import process_transcript_query
# Initialize auth DB on app start
init_auth_db()

def login_form():
    """Display login form"""
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = validate_user(username, password)
        if user:
            st.session_state["authenticated"] = True
            st.session_state["username"] = user["username"]
            st.session_state["role"] = user["role"]
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

def logout_button():
    """Display logout button - deprecated, now handled in UI components"""
    pass

def admin_user_management():
    """Admin user management interface"""
    st.header("👤 User Management (Admin Only)")
    with st.form("register_user"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        new_role = st.selectbox("Role", ["user", "admin"])
        if st.form_submit_button("Register User"):
            if get_user(new_username):
                st.error("Username already exists.")
            else:
                add_user(new_username, new_password, new_role, st.session_state["username"])
                st.success(f"User '{new_username}' registered as {new_role}.")
    
    st.subheader("All Users")
    for u in list_users():
        st.write(f"Username: {u[0]}, Role: {u[1]}, Created by: {u[2]}, Created at: {u[3]}")

def main_app():
    """Main application with chatbot functionality"""
    # Setup page configuration
    setup_page_config()
    
    # Initialize session state
    initialize_session()
    
    # Initialize vectorstore
    try:
        index, metadata, tab_data = initialize_vectorstore()
        collection = get_collection()
        data_loading_error = None
    except Exception as e:
        data_loading_error = str(e)
        print(f"❌ Error loading data: {e}")
        # Create empty fallbacks
        index, metadata, tab_data = [], [], {}
        collection = None
    
    # Load custom CSS
    load_custom_css()
    
    # Display top logo
    display_top_logo()
    
    # Display header
    display_header()
    
    # Extract first name from username (assuming format like "chetan.mishra")
    username = st.session_state['username']
    first_name = username.split('.')[0].capitalize() if '.' in username else username.capitalize()
    
    # User info section - positioned beside logout button
    st.markdown(f"""
    <div style="position: fixed; top: 18px; right: 280px; z-index: 10001; color: white; font-weight: 600; font-size: 16px;">
        Hello {first_name} !!
    </div>
    """, unsafe_allow_html=True)

    # Admin user management (only for admin users)
    if st.session_state["role"] == "admin":
        with st.expander("👤 User Management (Admin Only)"):
            admin_user_management()
        st.divider()
    
    # Setup language selector with logout functionality
    setup_language_selector()
    
    # Create language dropdown
    lang = create_language_dropdown()
    set_language(lang)
    
    # Check for logout button click (positioned over the logout icon)
    logout_clicked = create_logout_button()
    if logout_clicked:
        for key in ["authenticated", "username", "role"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    # Create input section
    query, submit = create_input_section()
    
    # Setup main containers
    setup_containers()
    
    # --- Left Sidebar for New Chat and PDF Upload ---
    with st.sidebar:
        st.markdown("<h2 style='color: #FFD700; text-align: left; font-family: Arial, sans-serif; padding-bottom: 5px;'>🛠️ Tools</h2>", unsafe_allow_html=True)
        
        # New Chat Section
        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        if st.button("🆕 New Chat", key="sidebar_new_chat", help="Start a new conversation"):
            clear_history()
            st.session_state.user_query = ""
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Divider
        st.markdown("<hr style='border: 1px solid #444; margin: 10px 0;'>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: #FFFFFF; text-align: left; font-family: Arial, sans-serif; padding-bottom: 0px;  font-size: 20px;'>📄 Upload PDF</h3>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        
        is_private = st.checkbox("Private", key="sidebar_private_upload", help="Upload privately for your use only")
        uploaded_file = st.file_uploader(" ", type=["pdf"], key="sidebar_pdf_uploader")

        st.markdown("</div>", unsafe_allow_html=True)
        # In the sidebar section, replace the uploaded_file handling block with:
        if uploaded_file is not None:
            # Check if this file was already processed
            file_key = f"processed_file_{uploaded_file.name}_{is_private}"
            
            # Define paths
            user_folder = f"data/user_uploads/{st.session_state['username']}"
            private_csv_path = os.path.join(user_folder, "csv_files", "student_transcript.csv")
            public_csv_path = "data/public_uploads/csv_files/student_transcript.csv"
            
            # Check if the final merged CSV already exists
            target_csv_path = private_csv_path if is_private else public_csv_path
            if os.path.exists(target_csv_path) and file_key in st.session_state:
                # Skip reprocessing if CSV exists and file was previously processed
                st.session_state["private_csv_path"] = private_csv_path if is_private else None
                st.session_state["active_transcript_csv_path"] = target_csv_path
                st.success(f"Using existing processed data from {('private' if is_private else 'public')} space!")
            else:
                temp_pdf_path = f"temp_{uploaded_file.name}"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if is_private:
                    os.makedirs(user_folder, exist_ok=True)
                    save_path = os.path.join(user_folder, uploaded_file.name)
                    save_scope = "private"
                    st.session_state["private_csv_path"] = private_csv_path
                else:
                    public_folder = "data/public_uploads"
                    os.makedirs(public_folder, exist_ok=True)
                    save_path = os.path.join(public_folder, uploaded_file.name)
                    save_scope = "public"
                    st.session_state["private_csv_path"] = None
                os.replace(temp_pdf_path, save_path)
                logic.parse_and_index_pdf(save_path, user=st.session_state['username'], private=is_private)
                st.session_state[file_key] = True
                st.session_state["active_transcript_csv_path"] = target_csv_path
                st.success(f"PDF uploaded and processed successfully! Using {save_scope} space.")

        # Replace the CSV path selection logic at the bottom of main_app() with:
        # Determine which CSV to use for transcript queries
        if st.session_state.get("private_csv_path") and os.path.exists(st.session_state["private_csv_path"]):
            transcript_csv_path = st.session_state["private_csv_path"]
            print(f"Using private transcript CSV path: {transcript_csv_path}")
        else:
            transcript_csv_path = "data/public_uploads csv_files/student_transcript.csv"
            if os.path.exists(transcript_csv_path):
                print(f"Using public transcript CSV path: {transcript_csv_path}")
            else:
                print(f"Public CSV not found, no transcript data available")
                transcript_csv_path = None

        # Store for use in query handler
        st.session_state["active_transcript_csv_path"] = transcript_csv_path
            # st.success(f"PDF uploaded and saved to {save_scope} space!")
        
    # Handle form submission
    if submit and query:
        set_user_query(query)
        
        # Show loading spinner while generating the answer
        with st.spinner("🔄 Please wait while I find the best answer for you..."):
            if data_loading_error:
                answer = type('obj', (object,), {
                    'content': 'Sorry, I cannot answer questions right now due to a data loading error.'
                })
                query_type = 'ERROR'
                confidence_score = 0.0
            else:
                time.sleep(1)
                
                # Create query handler and process the query
                query_handler = create_query_handler(collection, tab_data)
                answer, query_type, confidence_score = query_handler.process_query(
                    get_user_query(), 
                    get_language()
                )
        
        # Display the answer with query type info
        display_answer(answer)
        
        # Display query classification info (optional - can be removed in production)
        if query_type != 'ERROR':
            st.info(f"🔍 Query Type: **{query_type}** (Confidence: {confidence_score:.3f})")
        
        # Add to history
        answer_content = answer.content if hasattr(answer, 'content') else "No answer available."
        add_to_history(get_user_query(), answer_content)
    
    # Display Q&A history
    display_qa_history()
    
    # Close containers
    close_containers()

    # Determine which CSV to use for transcript queries (only set the path, don't process)
    if st.session_state.get("private_csv_path") and os.path.exists(st.session_state["private_csv_path"]):
        transcript_csv_path = st.session_state["private_csv_path"]
        print(f"Using private transcript CSV path: {transcript_csv_path}")
    else:
        transcript_csv_path = "data/public_uploads/csv_files/student_transcript.csv"
        print(f"Using public transcript CSV path: {transcript_csv_path}")

    # Store for use in query handler
    st.session_state["active_transcript_csv_path"] = transcript_csv_path

def main():
    """Main entry point"""
    if not st.session_state.get("authenticated"):
        login_form()
    else:
        main_app()

if __name__ == "__main__":
    main()