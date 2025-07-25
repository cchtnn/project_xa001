"""
Vector Store Management for Diné College Assistant
Handles ChromaDB operations and vectorstore initialization
"""

import streamlit as st
import json
import chromadb
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import generate_embeddings
from logic import calculate_file_hash, get_metadata, save_metadata
from config import CHROMA_COLLECTION_NAME, TEXT_SPLITTER_CONFIG, PATHS
import logging
logging.getLogger("watchdog").setLevel(logging.ERROR)


class VectorStoreManager:
    """Manages ChromaDB vector store operations"""
    
    def __init__(self):
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
            chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
            length_function=len,
            separators=TEXT_SPLITTER_CONFIG["separators"]
        )
    
    def get_collection(self):
        """Get the ChromaDB collection"""
        return self.collection
    
    def load_tab_data(self):
        """Load tab data from JSON file"""
        with open(PATHS["tab_data"], 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def clear_collection(self):
        """Clear existing collection data"""
        if self.collection.count() > 0:
            print("🗑️ Clearing existing collection data")
            results = self.collection.get()
            if results and 'ids' in results and results['ids']:
                self.collection.delete(ids=results['ids'])
    
    def process_documents(self, tab_data):
        """Process documents into chunks and embeddings"""
        document_chunks = []
        chunk_ids = []
        chunk_metadata = []
        chunk_count = 0

        for title, content in tab_data.items():
            # Create document with title and content
            document = f"{title}: {content}"
            
            # Split document into chunks
            chunks = self.text_splitter.split_text(document)
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                chunk_id = f"chunk_{chunk_count}"
                chunk_count += 1
                
                # Store chunk with its metadata
                document_chunks.append(chunk)
                chunk_ids.append(chunk_id)
                chunk_metadata.append({
                    "title": title, 
                    "chunk_index": i, 
                    "source": "tab_data"
                })

        return document_chunks, chunk_ids, chunk_metadata
    
    def add_to_collection(self, document_chunks, chunk_ids, chunk_metadata):
        """Add documents to ChromaDB collection"""
        embeddings = generate_embeddings(document_chunks)
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=document_chunks,
            metadatas=chunk_metadata,
            ids=chunk_ids
        )
        
        print(f"✅ Added {len(document_chunks)} chunks to ChromaDB collection")
    
    def update_metadata(self, current_hash):
        """Update metadata with new hash and timestamp"""
        metadata_info = get_metadata()
        metadata_info["tab_data_hash"] = current_hash
        metadata_info["last_updated"] = datetime.now().isoformat()
        save_metadata(metadata_info)
    
    def initialize_vectorstore(self):
        """Initialize or update vectorstore based on file hash"""
        # Get the current hash of tab_data.json
        current_hash = calculate_file_hash(PATHS["tab_data"])
        
        # Load metadata
        metadata_info = get_metadata()
        stored_hash = metadata_info.get("tab_data_hash", "")
        
        # Load tab_data regardless (we'll need it for reference)
        tab_data = self.load_tab_data()
        
        # Check if data has changed or collection is empty
        if current_hash != stored_hash or self.collection.count() == 0:
            print(f"💾 Data changed or collection empty. Processing data...")
            print(f"Previous hash: {stored_hash}")
            print(f"Current hash: {current_hash}")
            
            # Clear existing collection data if it exists
            self.clear_collection()
            
            # Process documents
            document_chunks, chunk_ids, chunk_metadata = self.process_documents(tab_data)
            
            # Add to collection
            self.add_to_collection(document_chunks, chunk_ids, chunk_metadata)
            
            # Update metadata
            self.update_metadata(current_hash)
            
            return chunk_ids, chunk_metadata, tab_data
        else:
            print(f"📚 Using existing collection data (hash match: {current_hash})")
            # Return placeholder values for compatibility
            return list(range(self.collection.count())), [], tab_data


# Initialize vectorstore with caching
# @st.cache_resource(show_spinner=False)
def get_vectorstore_manager():
    """Get cached vectorstore manager instance"""
    return VectorStoreManager()


def initialize_vectorstore():
    """Initialize vectorstore - wrapper function for compatibility"""
    try:
        manager = get_vectorstore_manager()
        return manager.initialize_vectorstore()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # Create empty fallbacks
        return [], [], {}


def get_collection():
    """Get ChromaDB collection"""
    manager = get_vectorstore_manager()
    return manager.get_collection()