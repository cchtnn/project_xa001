from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

def search_query(user_query, index, model, metadata, tab_data, top_k=2,distance_threshold=0.5):
    query_vector = model.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, top_k)
    
    # retrieved_titles = [metadata[idx] for idx in indices[0]]
    # retrieved_docs = [tab_data[title] for title in retrieved_titles]

    valid_indices = [
        i for i, dist in enumerate(distances[0])
        if dist < distance_threshold and indices[0][i] < len(metadata)
    ]

    retrieved_titles = [metadata[indices[0][i]] for i in valid_indices]
    retrieved_docs = [tab_data[title] for title in retrieved_titles if title in tab_data]

    return retrieved_titles, retrieved_docs, distances



def generate_answer(user_query, retrieved_titles, tab_data):
    """
    Generates an answer to the user's query using the LLaMA model (via ChatGroq).
    """
    retrieved_docs = "\n\n".join([f"{title}: {tab_data[title]}" for title in retrieved_titles if title in tab_data])

    prompt = f"""
        You are an expert assistant helping users understand resources related to FAFSA. 
        Answer the user's question **only** based on the information provided below. 
        Do **not** use any external knowledge or make assumptions beyond the provided documents.

        When answering:
        - Use a friendly, guiding tone.
        - Structure the answer in a **clear, easy-to-follow manner**.
        - Use **storytelling** where appropriate to guide the user step-by-step.
        - Highlight important tools or resources using bullet points or bold text.
        - Group information by relevant audience (e.g., students, educators, officials) if applicable.

        If the answer is **not present** in the information, reply with: 
        "I'm sorry, but that question is outside the scope of the provided information."

        Information:
        {retrieved_docs}

        Question: {user_query}

        Answer:
        """

    llm = ChatGroq(
        model="Llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        max_tokens=8000, # 4192
        timeout=60,
        max_retries=2,
    )
    response = llm.invoke(prompt)
    return response
