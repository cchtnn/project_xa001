"""
Query Classification Module for Diné College Assistant
Classifies user queries as either STUDENT TRANSCRIPT TYPE or POLICY TYPE
"""

from sentence_transformers import SentenceTransformer, util
import streamlit as st
import logging
logging.getLogger("watchdog").setLevel(logging.ERROR)


class QueryClassifier:
    """Classifies user queries into different types"""
    
    def __init__(self, similarity_threshold=0.6):
        self.similarity_threshold = similarity_threshold
        self.model = None
        self.transcript_embeddings = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the SentenceTransformer model and precompute embeddings"""
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self._precompute_transcript_embeddings()
        except Exception as e:
            print(f"❌ Error initializing query classifier: {e}")
            self.model = None
    
    def _precompute_transcript_embeddings(self):
        """Precompute embeddings for transcript example queries"""
        transcript_examples = [
            "What is the student's GPA?",
            "Tell me the academic information of the student",
            "Which term and sub term is the course taken?",
            "Give me the types of courses the student is pursuing",
            "Show me the career total",
            "What is the student's credit hour total?",
            "How has the student performed in each semester?",
            "What courses did the student take?",
            "Show me the student's grades",
            "What is the student's academic progress?",
            "Display the student's transcript",
            "What are the student's course completions?",
            "Show me the student's enrollment history",
            "What is the student's academic standing?",
            "Tell me about the student's degree progress",
            "Sort students in descending order of GPA",
            "How many students have GPA >= 4.2",
            "How many Students have A grade in Fall 2024-2025 and their details",
            "List of students from Murray State College",
            "Give me the GPA details of  Trista Barrett.",
            "Tell me the courses which Joshua Don Gaitan has enrolled?",
            "Tell me the course name which Leslie Nichole Bright has enrolled?",
            "How many Students have A grade in 2024-2025 Fall and their details",
            "Name of student name where organization is NEWMAN UNIVERSITY",
            "Calculate average GPA of students and sort that in descending order.",
            "Tell me the courses which Trista Denay Barrett has enrolled?",
            "give me all Student Name whose advisor is Laura Lyndsey",
            "Tell me the course number and Term information in which student 'Trista Denay Barrett' has got 'A' grade?"
        ]
        
        if self.model:
            self.transcript_embeddings = self.model.encode(
                transcript_examples, 
                convert_to_tensor=True
            )
    
    def classify_query(self, user_query):
        """
        Classify user query as either STUDENT_TRANSCRIPT or POLICY
        
        Args:
            user_query (str): The user's question
            
        Returns:
            tuple: (query_type, confidence_score)
                query_type: 'STUDENT_TRANSCRIPT' or 'POLICY'
                confidence_score: float between 0 and 1
        """
        if not self.model or self.transcript_embeddings is None:
            # Fallback to POLICY type if model is not available
            print("⚠️ Query classifier not available, defaulting to POLICY type")
            return 'POLICY', 0.0
        
        try:
            # Embed the user question
            user_embedding = self.model.encode(user_query, convert_to_tensor=True)
            
            # Compute cosine similarities with transcript references
            cos_scores = util.cos_sim(user_embedding, self.transcript_embeddings)
            
            # Get the highest similarity score
            max_score = cos_scores.max().item()
            
            # Classify based on threshold
            if max_score >= self.similarity_threshold:
                query_type = 'STUDENT_TRANSCRIPT'
            else:
                query_type = 'POLICY'
                # query_type = 'STUDENT_TRANSCRIPT'
            
            print(f"🔍 Query Classification:")
            print(f"   Query: {user_query}")
            print(f"   Type: {query_type}")
            print(f"   Confidence: {max_score:.3f}")
            print(f"   Threshold: {self.similarity_threshold}")
            
            return query_type, max_score
            
        except Exception as e:
            print(f"❌ Error during query classification: {e}")
            # Fallback to POLICY type on error
            return 'POLICY', 0.0
    
    def update_threshold(self, new_threshold):
        """Update the similarity threshold"""
        self.similarity_threshold = new_threshold
        print(f"📊 Updated similarity threshold to: {new_threshold}")


# Global classifier instance with caching
@st.cache_resource(show_spinner=False)
def get_query_classifier():
    """Get cached query classifier instance"""
    return QueryClassifier()


def classify_user_query(user_query, threshold=0.6):
    """
    Convenience function to classify a user query
    
    Args:
        user_query (str): The user's question
        threshold (float): Similarity threshold for classification
        
    Returns:
        tuple: (query_type, confidence_score)
    """
    classifier = get_query_classifier()
    if classifier.similarity_threshold != threshold:
        classifier.update_threshold(threshold)
    
    return classifier.classify_query(user_query)


# Constants for query types
class QueryType:
    STUDENT_TRANSCRIPT = 'STUDENT_TRANSCRIPT'
    POLICY = 'POLICY'