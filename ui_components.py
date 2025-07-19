"""
UI Components for Diné College Assistant
Contains all UI-related functions and components
Fixed version with logout button properly positioned beside globe icon
"""

import streamlit as st
from datetime import datetime
from pathlib import Path
from logic import load_css, get_image_base64, load_svg_base64
from config import LOGO_URLS, LANGUAGE_OPTIONS, PATHS


def setup_page_config():
    """Setup Streamlit page configuration"""
    from config import PAGE_CONFIG
    st.set_page_config(**PAGE_CONFIG)


def load_custom_css():
    """Load and apply custom CSS"""
    css = load_css(PATHS["css_template"])
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def display_top_logo():
    """Display the top-left logo"""
    st.markdown(
        f"""
        <div class='top-left-logo'>
            <img src='{LOGO_URLS["dine_college"]}' alt='Logo'>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_header():
    """Display the main header with Jericho logo"""
    jericho_logo_base64 = get_image_base64(PATHS["jericho_logo"])
    current_time = datetime.now().strftime("%A, %d %B %Y %H:%M:%S")
    
    st.markdown(f"""
        <div class='jericho-header'>
            <div class='jericho-logo'>
                <img src='data:image/jpeg;base64,{jericho_logo_base64}' alt='Jericho Logo'>
            </div>
            <p class='tagline'>Ask me any question, and I'll find the best answer for you!</p>
            <p class='timestamp'>{current_time}</p>
        </div>
    """, unsafe_allow_html=True)


def setup_language_selector():
    """Setup the language selector with custom styling and logout button"""
    svg_file = Path(PATHS["globe_svg"])
    globe_base64 = load_svg_base64(svg_file)
    
    # Custom CSS for language selector and logout button positioning
    st.markdown("""
        <style>
            /* fixed position container for language selector */
            .language-selector {
                position: fixed;
                top: 10px;
                right: 30px;
                z-index: 9999;
                display: flex;
                align-items: center;
            }

            /* Globe icon styling */
            .globe-icon {
                width: 20px;
                height: 20px;
                margin-right: 5px;
                position: fixed !important;
                top: 20px !important;
                right: 155px !important;
                z-index: 10000 !important;
            }

            /* Dropdown container */
            .dropdown-container {
                width: 120px;
            }

            /* Target and position the selectbox over our placeholder */
            div[data-testid="stSelectbox"] {
                position: fixed !important;
                top: 10px !important;
                right: 30px !important;
                width: 120px !important;
                margin-top: 0 !important;
                z-index: 9999 !important;
            }

            /* Hide the label */
            div[data-testid="stSelectbox"] > label {
                display: none !important;
            }

            /* Position logout button beside globe icon */
            .stButton[data-testid="logout_btn"] {
                position: fixed !important;
                top: 12px !important;
                right: 185px !important;
                z-index: 10001 !important;
                width: auto !important;
            }

            .stButton[data-testid="logout_btn"] > button {
                background-color: #ff4444 !important;
                color: white !important;
                border: none !important;
                border-radius: 4px !important;
                padding: 4px 12px !important;
                font-size: 12px !important;
                font-weight: 500 !important;
                cursor: pointer !important;
                transition: all 0.3s ease !important;
                height: 28px !important;
                line-height: 1 !important;
            }

            .stButton[data-testid="logout_btn"] > button:hover {
                background-color: #cc3333 !important;
                transform: scale(1.05) !important;
            }

            .stButton[data-testid="logout_btn"] > button:focus {
                outline: none !important;
                box-shadow: 0 0 0 2px rgba(255, 68, 68, 0.3) !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Create placeholder for language selector without logout icon
    placeholder = st.empty()
    placeholder.markdown(f"""
        <div class="language-selector">
            <img src="data:image/svg+xml;base64,{globe_base64}" class="globe-icon">
            <div class="dropdown-container" id="dropdown-placeholder"></div>
        </div>
    """, unsafe_allow_html=True)
    
    return placeholder


def create_language_dropdown():
    """Create the language dropdown selector"""
    col1, col2 = st.columns([9, 1])
    
    with col2:
        lang = st.selectbox(
            "Language",
            LANGUAGE_OPTIONS,
            index=0,
            key="language_select",
            label_visibility="collapsed"
        )
    
    return lang


def create_logout_button():
    """Create a logout button positioned beside the globe icon"""
    # Create the logout button with a specific key for CSS targeting
    logout_clicked = st.button("Logout", key="logout_btn", help="Click to logout")
    return logout_clicked


def create_input_section():
    """Create the query input section"""
    col1, col2 = st.columns([8, 1])
    
    with col1:
        query = st.text_input(
            "", 
            key="input_query", 
            label_visibility="collapsed", 
            placeholder="Please type your question here...",
            value=""
        )
    
    with col2:
        submit = st.button("Enter", key="submit")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return query, submit


def display_answer(answer):
    """Display the generated answer"""
    if hasattr(answer, 'content') and answer.content:
        st.markdown(f"""
        <div class="latest-answer-container">
            <div class="answer-heading">Answer:</div>
            <div class="answer-content">{answer.content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="latest-answer-container">
            <div class="answer-heading">⚠️ No Answer Available</div>
            <div class="answer-content">Sorry, I couldn't find an answer to your question.</div>
        </div>
        """, unsafe_allow_html=True)


def display_qa_history():
    """Display previous questions and answers"""
    if 'qa_history' in st.session_state and st.session_state.qa_history:
        st.markdown('<div class="previous-qa-heading">📚 Previous Questions and Answers:</div>', unsafe_allow_html=True)
        
        for qa in st.session_state.qa_history:
            st.markdown(f"""
            <div class="previous-qa-item">
                <strong>Q:</strong> {qa['question']}<br>
                <div class="answer-separator"></div>
                <strong>A:</strong> {qa['answer']}
            </div>
            """, unsafe_allow_html=True)


def setup_containers():
    """Setup main containers for the app"""
    st.markdown('<div class="answers-container">', unsafe_allow_html=True)


def close_containers():
    """Close main containers"""
    st.markdown('</div>', unsafe_allow_html=True)