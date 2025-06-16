"""
Student Transcript Handler for Diné College Assistant
Handles student transcript queries using FAISS-based search
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


class StudentTranscriptHandler:
    """Handles student transcript queries with FAISS-based search"""
    
    def __init__(self, transcript_data_path="data/transcripts_student.json"):
        self.transcript_data_path = transcript_data_path
        self.model = None
        self.index = None
        self.all_texts = []
        self.all_metadata = []
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the transcript handler with data and embeddings"""
        try:
            print("📚 Loading student transcript data...")
            
            # Check if transcript data file exists
            if not os.path.exists(self.transcript_data_path):
                print(f"❌ Transcript data file not found: {self.transcript_data_path}")
                return False
            
            # Load student data
            with open(self.transcript_data_path, "r", encoding='utf-8') as f:
                full_data = json.load(f)
            
            if not full_data:
                print("❌ No student data found in transcript file")
                return False
            
            # Create text chunks
            self.all_texts, self.all_metadata = self._create_text_chunks(full_data)
            print(f"✅ Created {len(self.all_texts)} text chunks")
            
            # Create embeddings
            embeddings, self.model = self._create_embeddings(self.all_texts)
            print(f"✅ Created embeddings with shape: {embeddings.shape}")
            
            # Create FAISS index
            self.index = self._create_faiss_index(np.array(embeddings))
            print("✅ FAISS index created successfully!")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Error initializing student transcript handler: {e}")
            return False
    
    def _create_text_chunks(self, all_student_data: Dict[str, str], chunk_size=500, chunk_overlap=100) -> Tuple[List[str], List[str]]:
        """Create text chunks directly from student data"""
        all_texts = []
        all_metadata = []
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=[". ", "\n", "- ", ", ", " ", ""]
        )
        
        for student_key, content in all_student_data.items():
            # Split the content into chunks
            chunks = splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                # Add student context to each chunk
                contextualized_chunk = f"Student ID: {student_key}\n{chunk}"
                all_texts.append(contextualized_chunk)
                all_metadata.append(f"{student_key}|chunk_{i}")
        
        return all_texts, all_metadata
    
    def _create_embeddings(self, texts: List[str], model_name="all-MiniLM-L6-v2"):
        """Create embeddings for text chunks"""
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings, model
    
    def _create_faiss_index(self, embeddings: np.ndarray):
        """Create FAISS index for similarity search"""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index
    
    def _search(self, query: str, top_k=10):
        """Search for relevant chunks using FAISS"""
        if not self.is_initialized:
            return []
        
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vec), top_k)
        
        results = []
        
        print("\n🔍 Top Matches from FAISS Index:")
        for i, idx in enumerate(indices[0]):
            if idx < len(self.all_texts):
                metadata = self.all_metadata[idx]
                student_key = metadata.split('|')[0]
                text_preview = self.all_texts[idx][:200] + "..." if len(self.all_texts[idx]) > 200 else self.all_texts[idx]
                
                print(f"  {i+1}. Student: {student_key}")
                print(f"     Text: {text_preview}")
                print(f"     Distance: {distances[0][i]:.4f}")
                print()
                
                results.append({
                    'text': self.all_texts[idx],
                    'metadata': metadata,
                    'distance': distances[0][i],
                    'student_key': student_key
                })
        
        return results
    
    def _generate_answer(self, user_query: str, search_results: List[Dict], language='English'):
        """Generate comprehensive answer using LLM"""
        
        if not search_results:
            fallback_messages = {
                "English": "I couldn't find any relevant student transcript information for your query.",
                "Spanish": "No pude encontrar información relevante del expediente académico para tu consulta.",
                "Navajo": "Óltaʼgi bééhániih éí doo tʼáá álʼįį da."
            }
            return fallback_messages.get(language, fallback_messages["English"])
        
        # Prepare context from search results
        context_parts = []
        for result in search_results:
            context_parts.append(result['text'])
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are an expert academic advisor assistant. Based on the student transcript data provided below, answer the user's question accurately and comprehensively.

IMPORTANT INSTRUCTIONS:
1. Use ONLY the information provided in the academic data below
2. The language of communication must be in {language} language, you must respond in {language} language.
3. Provide specific details from the transcript data when available
4. If asking about multiple students, organize the information clearly
5. Include relevant academic metrics like GPA, credit hours, grades when mentioned in the data
6. Be precise and factual based on the provided transcript information

ACADEMIC DATA:
{context}

USER QUESTION: {user_query}

ANSWER:"""

        try:
            llm = ChatGroq(
                model="llama3-8b-8192",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.1,
                max_tokens=4000,
                timeout=60,
                max_retries=2,
            )
            
            response = llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            error_messages = {
                "English": "I encountered an error while processing your transcript query. Please try again.",
                "Spanish": "Encontré un error al procesar tu consulta del expediente académico. Por favor, inténtalo de nuevo.",
                "Navajo": "Bééhániih ályaa éí átʼé. Náábah ílį́."
            }
            return error_messages.get(language, error_messages["English"])
    
    def process_query(self, user_query: str, language='English', top_k=15):
        """Main function to search and generate response for transcript queries"""
        
        if not self.is_initialized:
            print("❌ Student transcript handler not initialized")
            init_success = self.initialize()
            if not init_success:
                error_messages = {
                    "English": "Student transcript system is not available. Please ensure the transcript data file exists.",
                    "Spanish": "El sistema de expedientes académicos no está disponible. Asegúrate de que el archivo de datos del expediente existe.",
                    "Navajo": "Óltaʼgi bééhániih éí doo áhólł̥ǫ́ǫ da."
                }
                return error_messages.get(language, error_messages["English"])
        
        print(f"🔎 Processing student transcript query: '{user_query}'")
        print("=" * 80)
        
        # Search for relevant information
        search_results = self._search(user_query, top_k=top_k)
        
        if not search_results:
            print("❌ No matching transcript results found.")
            no_results_messages = {
                "English": "No relevant student transcript information was found for your query.",
                "Spanish": "No se encontró información relevante del expediente académico para tu consulta.",
                "Navajo": "Óltaʼgi bééhániih éí doo tʼáá álʼįį da."
            }
            return no_results_messages.get(language, no_results_messages["English"])
        
        print(f"📊 Found {len(search_results)} relevant transcript chunks")
        print("=" * 80)
        
        # Generate comprehensive answer
        answer = self._generate_answer(user_query, search_results, language)
        
        print("🧠 FINAL TRANSCRIPT ANSWER:")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        
        return answer


# Global transcript handler instance with caching
@st.cache_resource(show_spinner=False)
def get_transcript_handler():
    """Get cached transcript handler instance"""
    return StudentTranscriptHandler()


def process_transcript_query(user_query: str, language='English'):
    """
    Convenience function to process transcript queries
    
    Args:
        user_query (str): The user's question about transcripts
        language (str): Language for response
        
    Returns:
        str: Generated answer
    """
    handler = get_transcript_handler()
    return handler.process_query(user_query, language)