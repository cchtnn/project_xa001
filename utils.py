import os
import json
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model="Llama3-8b-8192",
#     temperature=0,
#     max_tokens=4192,
#     timeout=30,
#     max_retries=2,
# )

def get_data_from_website(url):
    """
    Fetches data from a website and saves it as a JSON file.
    """
    response = requests.get(url)
    if response.status_code == 500:
        print("Server error")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    tab_titles = soup.find_all("div", class_="elementor-tab-title")
    tab_data = {}

    for title_div in tab_titles:
        tab_id = title_div.get("data-tab")
        tab_title = title_div.get_text(strip=True)
        matching_content = soup.find("div", class_="elementor-tab-content", attrs={"data-tab": tab_id})
        tab_content = matching_content.get_text(separator="\n", strip=True) if matching_content else ""
        tab_data[tab_title] = tab_content

    os.makedirs("data", exist_ok=True)
    with open("data/tab_data.json", "w", encoding="utf-8") as f:
        json.dump(tab_data, f, ensure_ascii=False, indent=2)
    print("Data saved to data/tab_data.json")


def generate_embeddings(documents):
    """
    Generates embeddings for a list of documents using a SentenceTransformer model.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


def create_faiss_index(embeddings):
    """
    Creates and saves a FAISS index for the given embeddings.
    """
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance metric
    index.add(embeddings)
    faiss.write_index(index, 'data/faiss_index.idx')

    return index


def load_faiss_index():
    """
    Loads the saved FAISS index.
    """
    index = faiss.read_index('data/faiss_index.idx')
    return index


def load_metadata():
    """
    Loads the metadata (e.g., tab titles) used for reverse lookup in FAISS index.
    """
    with open('data/faiss_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata

def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
