"""
Diné College Assistant - Main Application
Refactored version with modular components and query type classification
"""

import streamlit as st
import time

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
    
    # Add a "New Chat" button
    if st.button("🆕 New Chat"):
        clear_history()
        st.session_state.user_query = ""
        st.rerun()
    
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

def main():
    """Main entry point"""
    if not st.session_state.get("authenticated"):
        login_form()
    else:
        main_app()

if __name__ == "__main__":
    main()