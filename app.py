import streamlit as st
import json
from datetime import datetime
import pytz
import time
import requests
# from streamlit_extras.st_autorefresh import st_autorefresh
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

# --------------------------
# Theme Toggle and Styling
# --------------------------
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
    st.rerun()

# Place this AFTER user_query input
st.sidebar.button("Toggle Theme 🌞🌙", on_click=toggle_theme)

# Dark Theme
if st.session_state.theme == 'dark':
    st.markdown("""
        <style>
        .stApp {
            background-color: #273346;
        }
        .stTextInput > div > div > input {
            background-color: #1e293b;
            color: white;
        }
        .stButton>button {
            background-color: #7792E3;
            color: white;
        }
        .stMarkdown, .css-1cpxqw2, .css-ffhzg2, h1 {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Light Theme
else:
    st.markdown("""
        <style>
        .stApp {
            background-color: #FFFFFF;
        }
        .stTextInput > div > div > input {
            background-color: white;
            color: black;
        }
        .stButton>button {
            background-color: #7792E3;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# --------------------------
# Sidebar Navigation
# --------------------------
page = st.sidebar.radio("Go to", ["Home", "About"])

# --------------------------
# About Page
# --------------------------
if page == "About":
    st.title("About This App")
    st.write("""
    Welcome to the **RAG Chatbot**! This intelligent assistant is designed to help you with questions 
    related to academic policies, FERPA, civil rights laws, and more.

    **Tech Stack:**
    - Streamlit for frontend
    - FAISS for semantic search
    - OpenAI model for generating answers

    Built with ❤️ to make education policy easy to understand.
    """)
    st.stop()

# --------------------------
# Right-Aligned Date/Time
# --------------------------
# Timezone
tz = pytz.timezone('Asia/Kolkata')
now = datetime.now(tz)
current_time = now.strftime("%A, %d %b %Y %H:%M:%S")
st.markdown(f"<div style='text-align: right; font-size: 18px;'>{current_time}</div>", unsafe_allow_html=True)


# --------------------------
# Lottie Animation
# --------------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets10.lottiefiles.com/packages/lf20_totrpclr.json"
lottie_json = load_lottieurl(lottie_url)
st_lottie(lottie_json, height=200, key="lottie-animation")

# --------------------------
# RAG Chatbot Interface
# --------------------------
st.title("🤖 Jericho Chatbot")
st.write("Ask me any question, and I'll find the best answer for you!")

if "user_query" not in st.session_state:
    st.session_state.user_query = ""

st.session_state.user_query = st.text_input("Your Question:", value=st.session_state.user_query)
user_query = st.session_state.user_query

if user_query:
    if 'index' not in st.session_state:
        get_data_from_website("https://www.dinecollege.edu/academics/academic-policies/")
        append_ferpa_data("https://studentprivacy.ed.gov/ferpa")
        append_civil_rights_data("https://www.ed.gov/laws-and-policy/civil-rights-laws")
        append_file_complaint_data("https://www.ed.gov/laws-and-policy/civil-rights-laws/file-complaint")
        append_fafsa_data("https://www.ed.gov/higher-education/paying-college/better-fafsa")

        # Adding pdf data
        pdf_directory="data\\hr_policies"
        get_data_from_pdf(pdf_directory,output_filename="data/tab_data.json")

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
        time.sleep(1)  # Optional: simulate delay for smooth UI
        retrieved_titles, retrieved_docs, distances = search_query(st.session_state.user_query, index, model, metadata, tab_data
)
        answer = generate_answer(st.session_state.user_query, retrieved_titles, tab_data)

    # Create a placeholder to display the final answer dynamically
    answer_placeholder = st.empty()
    if hasattr(answer, 'content') and answer.content:
        answer_placeholder.markdown(f"**Answer:** {answer.content}")
    else:
        answer_placeholder.markdown("⚠️ No answer returned.")
    print("Answer :-", answer.content)
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    st.session_state.qa_history.append({"question": user_query, "answer": answer.content})
    st.subheader("📚 Previous Questions and Answers:")
    for qa in st.session_state.qa_history:
        st.write(f"**Q:** {qa['question']}")
        st.write(f"**A:** {qa['answer']}")