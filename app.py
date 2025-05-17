import streamlit as st
import json, os
from datetime import datetime
import hashlib
import pytz
import time
import base64
from pathlib import Path
import requests
from streamlit_lottie import st_lottie
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

from logic import count_tokens
import logic
# from utils import get_model

st.set_page_config(
    page_title="Diné College Assistant",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="jericho_documents")

# # Load model
# model = get_model()

# File to store hash metadata
METADATA_FILE = "data/metadata.json"

# Function to calculate hash of a file
def calculate_file_hash(file_path):
    """Calculate MD5 hash of file to detect changes"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Function to load or initialize metadata
def get_metadata():
    """Load metadata from file or create default"""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    else:
        return {"tab_data_hash": "", "last_updated": ""}

# Function to save metadata
def save_metadata(metadata):
    """Save metadata to file"""
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)

# Function to load CSS from file
def load_css(css_file):
    with open(css_file, 'r') as f:
        return f.read()

# Function to load HTML template from file
def load_html_template(template_file):
    with open(template_file, 'r') as f:
        return f.read()
    
# Function to get base64 encoded image
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Initialize or update vectorstore based on file hash
@st.cache_resource(show_spinner=False)
def initialize_vectorstore():
    # Get the current hash of tab_data.json
    tab_data_path = 'data/tab_data.json'
    current_hash = calculate_file_hash(tab_data_path)
    
    # Load metadata
    metadata_info = get_metadata()
    stored_hash = metadata_info.get("tab_data_hash", "")
    
    # Load tab_data regardless (we'll need it for reference)
    with open(tab_data_path, 'r', encoding='utf-8') as f:
        tab_data = json.load(f)
    
    # Check if data has changed or collection is empty
    if current_hash != stored_hash or collection.count() == 0:
        print(f"💾 Data changed or collection empty. Processing data...")
        print(f"Previous hash: {stored_hash}")
        print(f"Current hash: {current_hash}")
        
        # If collection already has data, first delete it
        if collection.count() > 0:
            print("🗑️ Clearing existing collection data")
            # Get all IDs in collection
            results = collection.get()
            if results and 'ids' in results and results['ids']:
                collection.delete(ids=results['ids'])
        
        # Text splitter for chunking documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Process each document: chunk, embed, and add to ChromaDB
        document_chunks = []
        chunk_ids = []
        chunk_metadata = []
        chunk_count = 0

        for title, content in tab_data.items():
            # Create document with title and content
            document = f"{title}: {content}"
            
            # Split document into chunks
            chunks = text_splitter.split_text(document)
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                chunk_id = f"chunk_{chunk_count}"
                chunk_count += 1
                
                # Store chunk with its metadata
                document_chunks.append(chunk)
                chunk_ids.append(chunk_id)
                chunk_metadata.append({"title": title, "chunk_index": i, "source": "tab_data"})

        embeddings = generate_embeddings(document_chunks)
        
        collection.add(
            embeddings=embeddings.tolist(),
            documents=document_chunks,
            metadatas=chunk_metadata,
            ids=chunk_ids
        )

        # Update metadata with new hash
        metadata_info["tab_data_hash"] = current_hash
        metadata_info["last_updated"] = datetime.now().isoformat()
        save_metadata(metadata_info)
        
        print(f"✅ Added {len(document_chunks)} chunks to ChromaDB collection")
        return chunk_ids, chunk_metadata, tab_data
    else:
        print(f"📚 Using existing collection data (hash match: {current_hash})")
        # Return placeholder values for compatibility
        return list(range(collection.count())), [], tab_data

# Initialize data once at app startup, but will update if hash changes
try:
    index, metadata, tab_data = initialize_vectorstore()
    data_loading_error = None
except Exception as e:
    data_loading_error = str(e)
    print(f"❌ Error loading data: {e}")
    # Create empty fallbacks
    index, metadata, tab_data = [], [], {}

st.session_state.current_time = datetime.now().strftime("%A, %d %B %Y %H:%M:%S")

# Load CSS from template file
css = load_css("templates/styles.css")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Display the logo in the top-left corner
st.markdown(
    f"""
    <div class='top-left-logo'>
        <img src='https://www.dinecollege.edu/wp-content/uploads/2024/12/dc_logoFooter.png' alt='Logo'>
    </div>
    """,
    unsafe_allow_html=True,
)

# Get Jericho logo as base64
jericho_logo_path = "templates/jericho_image.jpg"
jericho_logo_base64 = get_image_base64(jericho_logo_path)

# Create the header with logo image instead of text
st.markdown(f"""
    <div class='jericho-header'>
        <div class='jericho-logo'>
            <img src='data:image/jpeg;base64,{jericho_logo_base64}' alt='Jericho Logo'>
        </div>
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

col1, col2 = st.columns([8, 1])

with col1:
    query = st.text_input("", key="input_query", label_visibility="collapsed", placeholder="Please type your question here...",value="")

with col2:
    submit = st.button("Enter", key="submit")
st.markdown('</div>', unsafe_allow_html=True)

# Container for answers with proper spacing
st.markdown('<div class="answers-container">', unsafe_allow_html=True)

if submit and query:
    # Store the query
    st.session_state.user_query = query
    
    # Show loading spinner while generating the answer
    with st.spinner("🔄 Please wait while I find the best answer for you..."):
        if data_loading_error:
            answer = type('obj', (object,), {'content': 'Sorry, I cannot answer questions right now due to a data loading error.'})
        else:
            time.sleep(1)
            # Use logic module functions with the collection parameter
            retrieved_titles, retrieved_chunks, distances = logic.search_query(st.session_state.user_query, collection)
            # Print token counts for each chunk
            print("📏 Token counts for each retrieved chunk:")
            for i, chunk in enumerate(retrieved_chunks):
                print(f"  Chunk {i+1}: {logic.count_tokens(chunk)} tokens")
            answer = logic.generate_answer(st.session_state.user_query, retrieved_chunks, tab_data, st.session_state.language)
            # Count tokens in the response
            response_tokens = logic.count_tokens(answer.content)
            print(f"📊 Response contains {response_tokens} tokens")
    
    # Display the latest answer in a styled container
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