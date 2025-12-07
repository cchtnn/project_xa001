"""
Query Handler for Diné College Assistant - Updated Version
Handles different types of queries based on classification
Now uses CSV-based student transcript handler instead of FAISS
"""

import time
import streamlit as st
from query_classifier import classify_user_query, QueryType
# Updated import - now using CSV handler instead of FAISS handler
from student_transcript_csv_handler import process_transcript_query
from docx_parser import PayrollCSVAgent
import logic
import json
import os
import logging
logging.getLogger("watchdog").setLevel(logging.ERROR)


class QueryHandler:
    """Handles different types of user queries"""
    
    def __init__(self, collection, tab_data):
        self.collection = collection
        self.tab_data = tab_data
    
    def process_query(self, user_query, language='English', payroll_csv_path=None):
        """
        Process user query based on its type
        
        Args:
            user_query (str): The user's question
            language (str): Selected language for response
            payroll_csv_path (str): Path to payroll CSV file (if available)
            
        Returns:
            tuple: (answer, query_type, confidence_score)
        """
        # Classify the query
        query_type, confidence_score = classify_user_query(user_query)
        
        # Process based on query type
        if query_type == QueryType.STUDENT_TRANSCRIPT:
            answer = self._handle_student_transcript_query(user_query, language)
        elif query_type == QueryType.PAYROLL_CALENDAR:
            answer = self._handle_payroll_query(user_query, language, payroll_csv_path)
        else:  # POLICY type
            answer = self._handle_policy_query(user_query, language)
        
        return answer, query_type, confidence_score
    
    def _handle_student_transcript_query(self, user_query, language):
        """
        Handle student transcript type queries using CSV agent
        
        Args:
            user_query (str): The user's question
            language (str): Selected language for response
            
        Returns:
            object: Answer object with content attribute
        """
        print("📊 Processing STUDENT TRANSCRIPT query with CSV Agent...")
        
        try:
            csv_path = st.session_state.get("active_transcript_csv_path", None)
            # Use the CSV-based student transcript handler to process the query
            answer_content = process_transcript_query(user_query, language, csv_path=csv_path)
            
            # Create answer object compatible with existing UI
            answer = type('obj', (object,), {
                'content': answer_content
            })
            
            return answer
            
        except Exception as e:
            print(f"❌ Error processing student transcript query with CSV agent: {e}")
            
            # Return error message in appropriate language
            error_messages = {
                "English": "I encountered an error while processing your student transcript query. Please try again or contact support.",
                "Spanish": "Encontré un error al procesar tu consulta del expediente académico. Por favor, inténtalo de nuevo o contacta al soporte.",
                "French": "J'ai rencontré une erreur lors du traitement de votre requête de relevé de notes. Veuillez réessayer ou contacter le support.",
                "Navajo": "Bééhániih ályaa éí átʼé. Náábah ílį́ éí doodaii' ánáhwiiłtááh."
            }
            
            error_content = error_messages.get(language, error_messages["English"])
            
            answer = type('obj', (object,), {
                'content': error_content
            })
            
            return answer
    
    def _handle_payroll_query(self, user_query, language, payroll_csv_path=None):
        """
        Handle payroll calendar type queries using PayrollCSVAgent
        
        Args:
            user_query (str): The user's question
            language (str): Selected language for response
            payroll_csv_path (str): Path to payroll CSV file
            
        Returns:
            object: Answer object with content attribute
        """
        print("📅 Processing PAYROLL CALENDAR query...")
        
        try:
            if not payroll_csv_path or not os.path.exists(payroll_csv_path):
                error_content = "No payroll calendar data found. Please upload a payroll calendar document (.docx) first."
                return type('obj', (object,), {'content': error_content})
            
            # Initialize and use PayrollCSVAgent
            payroll_agent = PayrollCSVAgent(csv_path=payroll_csv_path)
            
            if payroll_agent.initialize():
                answer_content = payroll_agent.query(user_query)
            else:
                answer_content = "Failed to initialize payroll calendar system. Please try again."
            
            # Create answer object compatible with existing UI
            answer = type('obj', (object,), {
                'content': answer_content
            })
            
            return answer
            
        except Exception as e:
            print(f"❌ Error processing payroll query: {e}")
            
            # Return error message in appropriate language
            error_messages = {
                "English": "I encountered an error while processing your payroll calendar query. Please try again or contact support.",
                "Spanish": "Encontré un error al procesar tu consulta del calendario de nómina. Por favor, inténtalo de nuevo o contacta al soporte.",
                "French": "J'ai rencontré une erreur lors du traitement de votre requête de calendrier de paie. Veuillez réessayer ou contacter le support.",
                "Navajo": "Béédahodeesnih bee ákonízin. T'áá íiyisí naaltsoos."
            }
            
            error_content = error_messages.get(language, error_messages["English"])
            
            answer = type('obj', (object,), {
                'content': error_content
            })
            
            return answer

    def _handle_policy_query(self, user_query, language):
        """
        Handle policy type queries (existing logic)
        
        Args:
            user_query (str): The user's question
            language (str): Selected language for response
            
        Returns:
            object: Answer object with content attribute
        """
        print("📋 Processing POLICY query...")
        
        try:
            # Use existing logic for policy queries
            retrieved_titles, retrieved_chunks, distances = logic.search_query(
                user_query, 
                self.collection
            )
            
            # Print token counts for each chunk
            print("📏 Token counts for each retrieved chunk:")
            for i, chunk in enumerate(retrieved_chunks):
                print(f"  Chunk {i+1}: {logic.count_tokens(chunk)} tokens")
            
            answer = logic.generate_answer(
                user_query, 
                retrieved_chunks, 
                self.tab_data, 
                language
            )
            
            # Count tokens in the response
            if hasattr(answer, 'content'):
                response_tokens = logic.count_tokens(answer.content)
                print(f"📊 Response contains {response_tokens} tokens")
            
            return answer
            
        except Exception as e:
            print(f"❌ Error processing policy query: {e}")
            
            # Return error message in appropriate language
            error_messages = {
                "English": "I encountered an error while processing your policy query. Please try again or rephrase your question.",
                "Spanish": "Encontré un error al procesar tu consulta de política. Por favor, inténtalo de nuevo o reformula tu pregunta.",
                "French": "J'ai rencontré une erreur lors du traitement de votre requête de politique. Veuillez réessayer ou reformuler votre question.",
                "Navajo": "Bééhódeilnih ályaa éí átʼé. Náábah ílį́ éí doodaii' saad naaltsoos."
            }
            
            error_content = error_messages.get(language, error_messages["English"])
            
            answer = type('obj', (object,), {
                'content': error_content
            })
            
            return answer


def create_query_handler(collection, tab_data):
    """
    Factory function to create a QueryHandler instance
    
    Args:
        collection: ChromaDB collection
        tab_data: Policy data dictionary
        
    Returns:
        QueryHandler: Configured query handler instance
    """
    return QueryHandler(collection, tab_data)