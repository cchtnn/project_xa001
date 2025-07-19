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


def main():
    """Main application function"""
    
    # DO NOT call clear_history() here!
    
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
    
    # Setup language selector
    setup_language_selector()
    
    # Create language dropdown
    lang = create_language_dropdown()
    set_language(lang)
    
    # Create input section
    query, submit = create_input_section()
    
    # Setup main containers
    setup_containers()

    # Add a "New Chat" button
    if st.button("🆕 New Chat (Clear History)"):
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


if __name__ == "__main__":
    main()