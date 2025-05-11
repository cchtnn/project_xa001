import streamlit as st
import json
from datetime import datetime
import pytz
import time
import base64
from pathlib import Path
import requests
from streamlit_lottie import st_lottie
from utils import (
    get_data_from_website,
    generate_embeddings,
    create_faiss_index,
    load_faiss_index,
    load_metadata,
    append_ferpa_data,
    append_civil_rights_data,
    append_file_complaint_data,
    append_fafsa_data,
    get_data_from_pdf,
    preprocess_tab_data,
    chunk_text,
    clean_text,
    truncate_docs
)
from logic import search_query, generate_answer
from utils import get_model

# Load model
model = get_model()

st.session_state.current_time = datetime.now().strftime("%A, %d %B %Y %H:%M:%S")

# First, let's hide the default Streamlit header completely
st.set_page_config(
    page_title="Diné College Assistant",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide all Streamlit default elements that might interfere with our custom layout
st.markdown(
    """
    <style>
        /* Hide the Streamlit header bar completely */
        header {
            visibility: hidden !important;
            height: 0px !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Reset the main content area to start from the top and reduce padding */
        .main .block-container {
            padding-top: 0 !important;
            margin-top: 0 !important;
            max-width: 1000px !important;
        }

        /* Hide hamburger menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Hide deploy button */
        .st-emotion-cache-1avcm0n {
            display: none !important;
        }
        button[title="Deploy"] {
            display: none !important;
        }

        /* Now properly position our logo without interference */
        .top-left-logo {
            position: fixed;
            top: 10px;
            left: 10px;
            height: auto;
            width: 80px;
            z-index: 9999; /* Increased z-index to ensure visibility */
        }
        .top-left-logo img {
            display: block;
            width: 100%;
            height: auto;
        }

        /* Style for the main header container - REPOSITIONED TO TOP */
        .jericho-header {
            text-align: center;
            position: relative;
            top: -90px;
            left: 50%;
            transform: translateX(-50%);
            padding: 10px;
            z-index: 100;
            width: 500px;
            margin-bottom: -150px;
        }

        .jericho-header h1 {
            font-size: 48px;
            margin: 0;
            padding: 0;
            line-height: 1.2;
        }

        .jericho-header p.tagline {
            font-size: 16px;
            color: gray;
            margin: 0;
            padding: 0;
        }

        .jericho-header p.timestamp {
            font-size: 12px;
            color: #888;
            margin: 0;
            padding: 0;
        }

        /* Input container styles - moved up by adding custom positioning */
        .input-container {
            position: fixed;
            top: 130px; /* Position directly below the header */
            left: 50%;
            transform: translateX(-50%);
            max-width: 800px;
            width: 90%;
            z-index: 99;
            margin-bottom: 0px; /* Reduced bottom margin */
        }

        /* Input styles */
        div[data-baseweb="input"] > div {
            border: 1px solid #ccc !important;
            border-radius: 6px !important;
            background-color: white !important;
            box-shadow: none !important;
        }

        /* Button style */
        .stButton > button {
            background-color: #7792E3 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
        }

        /* Make sure inputs and buttons line up properly */
        .stTextInput, .stButton {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        /* Adjust spacing below the input area for answers */
        .answers-container {
            margin-top: 0; /* Remove default top margin */
            padding-top: 10px; /* Adjust padding to control space below input */
            position: relative;
            z-index: 10;
        }

        /* New styles for the latest answer container */
        .latest-answer-container {
            background-color: #f0f0f0; /* Light grey background */
            border-radius: 15px; /* Rounded corners */
            padding: 20px; /* Padding inside the container */
            margin-bottom: 30px; /* Space between latest answer and previous answers */
            box-shadow: 0 2px 5px rgba(0,0,0,0.05); /* Subtle shadow */
            position: relative;
        }

        /* Larger font for the answer heading */
        .answer-heading {
            font-size: 24px !important; /* Larger font size */
            font-weight: 600 !important; /* Make it bold */
            margin-bottom: 15px !important; /* Space after heading */
            color: #333 !important; /* Darker text color */
        }

        /* Regular answer text */
        .answer-content {
            font-size: 16px !important;
            line-height: 1.5 !important;
        }

        .answer-separator {
            margin-top: 10px; /* Add a small gap between question and answer */
            margin-bottom: 15px;
            border-top: 1px solid #eee;
        }

        /* Center empty space */
        .container-centerize {
            display: flex;
            justify-content: center;
            width: 100%;
        }
        
        /* Previous Q&A section styling */
        .previous-qa-heading {
            font-size: 22px;
            font-weight: 600;
            margin-top: 30px;
            margin-bottom: 20px;
            color: #333;
        }
        
        .previous-qa-item {
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display the logo in the top-left corner
st.markdown(
    f"""
    <div class='top-left-logo'>
        <img src='https://www.dinecollege.edu/wp-content/uploads/2024/12/dc_logoFooter.png' alt='Logo'>
    </div>
    """,
    unsafe_allow_html=True,
)

# Create the header with all text
st.markdown(f"""
    <div class='jericho-header'>
        <h1>Jericho</h1>
        <p class='tagline'>Ask me any question, and I'll find the best answer for you!</p>
        <p class='timestamp'>{st.session_state.current_time}</p>
    </div>
""", unsafe_allow_html=True)

# --- Load SVG as base64 ---
def load_svg_base64(svg_path):
    with open(svg_path, "rb") as f:
        svg_data = f.read()
    return base64.b64encode(svg_data).decode("utf-8")

# Use forward slashes and make sure file path is correct
svg_file = Path("templates/globe_.svg")
globe_base64 = load_svg_base64(svg_file)

# --- Custom CSS for language selector positioning ---
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
    </style>
""", unsafe_allow_html=True)

# Create an empty element at the top of the app that we'll replace with our custom HTML
placeholder = st.empty()

# Insert our custom HTML with the globe icon directly into the DOM
placeholder.markdown(f"""
    <div class="language-selector">
        <img src="data:image/svg+xml;base64,{globe_base64}" class="globe-icon">
        <div class="dropdown-container" id="dropdown-placeholder"></div>
    </div>
""", unsafe_allow_html=True)

# Create a container with columns to position the dropdown at the right location
col1, col2 = st.columns([9, 1])

with col2:
    # Render the actual dropdown
    lang = st.selectbox(
        "Language",
        ["English", "Spanish", "Navajo"],
        index=0,
        key="language_select",
        label_visibility="collapsed"
    )

# Store selected language
st.session_state.language = lang

# Input section - using standard Streamlit components with styling
st.markdown('<div class="input-container">', unsafe_allow_html=True)
col1, col2 = st.columns([8, 1])

with col1:
    query = st.text_input("", key="input_query", label_visibility="collapsed", placeholder="Please type your question here...")

with col2:
    submit = st.button("Enter", key="submit")
st.markdown('</div>', unsafe_allow_html=True)

# Container for answers with proper spacing
st.markdown('<div class="answers-container">', unsafe_allow_html=True)

# Process the query when the button is clicked
if submit and query:
    # Your existing processing code here
    st.session_state.user_query = query
    # Your processing code here
    if "index" not in st.session_state:
        # get_data_from_website("https://www.dinecollege.edu/academics/academic-policies/")
        # append_ferpa_data("https://studentprivacy.ed.gov/ferpa")
        # append_civil_rights_data("https://www.ed.gov/laws-and-policy/civil-rights-laws")
        # append_file_complaint_data("https://www.ed.gov/laws-and-policy/civil-rights-laws/file-complaint")
        # append_fafsa_data("https://www.ed.gov/higher-education/paying-college/better-fafsa")

        # Adding pdf data
        # pdf_directory="data\\hr_policies"
        # get_data_from_pdf(pdf_directory,output_filename="data/tab_data.json")

        with open('data/tab_data.json', 'r', encoding='utf-8') as f:
            tab_data = json.load(f)

        documents = [f"{key}: {value}" for key, value in tab_data.items()]
        embeddings = generate_embeddings(documents)
        index = create_faiss_index(embeddings)
        metadata = list(tab_data.keys())
        with open('data/faiss_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        st.session_state.index = index
        st.session_state.metadata = metadata
        st.session_state.tab_data = tab_data
    else:
        index = st.session_state.index
        metadata = st.session_state.metadata
        tab_data = st.session_state.tab_data

    # Show loading spinner while generating the answer
    with st.spinner("🔄 Please wait while I find the best answer for you..."):
        time.sleep(1)
        retrieved_titles, retrieved_docs, distances = search_query(st.session_state.user_query, index, model, metadata, tab_data)
        answer = generate_answer(st.session_state.user_query, retrieved_titles, tab_data, st.session_state.language)

    # Display the latest answer in a styled container
    if hasattr(answer, 'content') and answer.content:
        st.markdown(f"""
        <div class="latest-answer-container">
            <div class="answer-heading">Answer: Filing a Civil Rights Complaint</div>
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
    
    # Save to history
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    st.session_state.qa_history.append({"question": st.session_state.user_query, "answer": answer.content if hasattr(answer, 'content') else "No answer available."})
    
    # Display previous questions and answers
    st.markdown('<div class="previous-qa-heading">📚 Previous Questions and Answers:</div>', unsafe_allow_html=True)
    
    for qa in st.session_state.qa_history:
        st.markdown(f"""
        <div class="previous-qa-item">
            <strong>Q:</strong> {qa['question']}<br>
            <div class="answer-separator"></div>
            <strong>A:</strong> {qa['answer']}
        </div>
        """, unsafe_allow_html=True)

# Close the answers container
st.markdown('</div>', unsafe_allow_html=True)