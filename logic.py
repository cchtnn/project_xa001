from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
import tiktoken
from dotenv import load_dotenv
from utils import generate_embeddings
import hashlib
import json
import base64

load_dotenv()

# Remove the circular import from app.py
# Instead, we'll pass the collection as a parameter to the functions

def search_query(user_query, collection, top_k=3):
    """Search ChromaDB for relevant documents based on user query"""
    # Generate embedding for the query
    query_embedding = generate_embeddings(user_query)
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    retrieved_chunks = results['documents'][0]  # Top k chunks
    chunk_metadata = results['metadatas'][0]    # Metadata for each chunk
    distances = results['distances'][0]         # Distance scores
    
    # Get the unique titles from the retrieved chunks
    retrieved_titles = list(set([metadata['title'] for metadata in chunk_metadata]))
    
    return retrieved_titles, retrieved_chunks, distances

def generate_answer(user_query, retrieved_chunks, tab_data, communication_language):
    """
    Generates an answer to the user's query using the LLaMA model (via ChatGroq).
    """
    chunk_context = "\n\n".join(retrieved_chunks)

    fallback_messages = {
    "English": "I'm sorry, but that question is outside the scope of the provided information.",
    "Spanish": "Lo siento, pero esa pregunta está fuera del alcance de la información proporcionada.",
    "French": "Je suis désolé, mais cette question dépasse le cadre des informations fournies."
    }
    fallback_response = fallback_messages.get(communication_language, fallback_messages["English"])

    prompt = f"""
                You are an expert assistant helping users. 
                Answer the user's question primarily using the information provided below.
                
                ---

                ### Provided Information:
                {chunk_context}

                ---

                ### Question:
                {user_query}

                ---

                ### Instructions for Answering:
                - The language of communication must be in user chosen {communication_language} language, you must respond in {communication_language} language.
                - First, check if the answer is found in the provided information:
                  * If the answer IS found in the provided information, respond clearly using that information.
                  * If the answer is PARTIALLY found, use what's available from the documents and clearly indicate which parts of your response come from the provided information.
                  * If the answer is NOT found in the provided information, you may provide a helpful response based on your general knowledge, but preface it with: "This information is not found in the provided documents. Based on general knowledge: "
                - Use a warm and helpful tone.
                - Use bullet points, bold text, or headings if it improves clarity.
                - Always be transparent about the source of your information (documents vs. general knowledge).
                - If you're completely uncertain about information outside the provided documents, acknowledge the limitations with: "I don't have specific information about this in the provided documents or in my general knowledge. {fallback_response}"

                ---

                ### Answer:
                """

    llm = ChatGroq(
        model="Llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        max_tokens=4192,
        timeout=60,
        max_retries=2,
    )

    # Count tokens in the prompt
    token_count = count_tokens(prompt)
    print(f"📊 Sending {token_count} tokens to the LLM")

    # Check if we're close to the limit
    if token_count > 6000:
        print(f"⚠️ WARNING: Token count ({token_count}) is approaching or exceeding Groq's limit of 6000 TPM")
        
    response = llm.invoke(prompt)
    return response

def count_tokens(text, model="cl100k_base"):
    """Count the number of tokens in a text string using tiktoken"""
    try:
        encoder = tiktoken.get_encoding(model)
        tokens = encoder.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Rough estimation if tiktoken fails
        return len(text.split()) * 1.3
    

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
    
# --- Load SVG as base64 ---
def load_svg_base64(svg_path):
    with open(svg_path, "rb") as f:
        svg_data = f.read()
    return base64.b64encode(svg_data).decode("utf-8")