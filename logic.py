from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

def search_query(user_query, index, model, metadata, tab_data, top_k=2, distance_threshold=0.5):
    def get_matches(threshold):
        query_vector = model.encode([user_query], convert_to_numpy=True)
        distances, indices = index.search(query_vector, top_k)

        if threshold is None:
            valid_indices = [
                i for i in range(len(indices[0]))
                if indices[0][i] < len(metadata)
            ]
        else:
            valid_indices = [
                i for i, dist in enumerate(distances[0])
                if dist < threshold and indices[0][i] < len(metadata)
            ]

        retrieved_titles = [metadata[indices[0][i]] for i in valid_indices]
        retrieved_docs = [tab_data[title] for title in retrieved_titles if title in tab_data]

        return retrieved_titles, retrieved_docs, distances

    # First attempt: use threshold
    retrieved_titles, retrieved_docs, distances = get_matches(distance_threshold)

    # Fallback attempt: no threshold if nothing found
    if not retrieved_titles or not retrieved_docs:
        retrieved_titles, retrieved_docs, distances = get_matches(None)

    return retrieved_titles, retrieved_docs, distances

def generate_answer(user_query, retrieved_titles, tab_data, communication_language):
    """
    Generates an answer to the user's query using the LLaMA model (via ChatGroq).
    """
    retrieved_docs = "\n\n".join([f"{title}: {tab_data[title]}" for title in retrieved_titles if title in tab_data])

    fallback_messages = {
    "English": "I'm sorry, but that question is outside the scope of the provided information.",
    "Spanish": "Lo siento, pero esa pregunta está fuera del alcance de la información proporcionada.",
    "French": "Je suis désolé, mais cette question dépasse le cadre des informations fournies."
    }
    fallback_response = fallback_messages.get(communication_language, fallback_messages["English"])

    prompt = f"""
                You are an expert assistant helping users. 
                Answer the user's question **only using the information provided below**. 
                If the information to answer the question is **not clearly found** in the provided text, respond with:

                "{fallback_response}"

                Do not rely on your own knowledge. Only refer to the content in the documents.

                ---

                ### Provided Information:
                {retrieved_docs}

                ---

                ### Question:
                {user_query}

                ---

                ### Instructions for Answering:
                - The language of communication must be in user chosen {communication_language} language, you must respond in {communication_language} language.
                - If the answer **is found** in the provided information, respond clearly and helpfully.
                - Use a warm and helpful tone.
                - Use bullet points, bold text, or headings if it improves clarity.
                - If **no relevant information is found**, simply respond:  
                *"{fallback_response}"*

                ---

                ### Answer:
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
