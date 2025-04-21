import streamlit as st
import json
from utils import get_data_from_website, generate_embeddings, create_faiss_index, load_faiss_index, load_metadata
from logic import search_query, generate_answer
from utils import get_model

model = get_model()

# Frontend layout
st.title("RAG Chatbot")
st.write("Ask me any question, and I'll find the best answer for you!")

# Input field for user query
user_query = st.text_input("Your Question:")

if user_query:
    # Loading data and index
    if 'index' not in st.session_state:
        get_data_from_website("https://www.dinecollege.edu/academics/academic-policies/")
        with open('data/tab_data.json', 'r', encoding='utf-8') as f:
            tab_data = json.load(f)
        documents = [f"{key}: {value}" for key, value in tab_data.items()]
        embeddings = generate_embeddings(documents)
        index = create_faiss_index(embeddings)

        # Saving metadata
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

    # Search for the most relevant documents
    retrieved_titles, retrieved_docs, distances = search_query(user_query, index, model, metadata, tab_data)

    # Generate the answer
    answer = generate_answer(user_query, retrieved_titles, tab_data)

    # Display the results
    st.write(f"Answer: {answer.content}")

    # Save question and answer on the frontend
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []

    st.session_state.qa_history.append({"question": user_query, "answer": answer.content})

    st.subheader("Previous Questions and Answers:")
    for qa in st.session_state.qa_history:
        st.write(f"Q: {qa['question']}")
        st.write(f"A: {qa['answer']}")
