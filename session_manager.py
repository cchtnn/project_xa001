import streamlit as st
from datetime import datetime
import session_db
import tempfile
import os
import logging
logging.getLogger("watchdog").setLevel(logging.ERROR)

class SessionManager:
    """Manages Streamlit session state with DB persistence"""
    
    @staticmethod
    def initialize_session():
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = None
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = []
        if 'current_time' not in st.session_state:
            st.session_state.current_time = datetime.now().strftime("%A, %d %B %Y %H:%M:%S")
        if 'language' not in st.session_state:
            st.session_state.language = session_db.load_meta("language", "English")

    @staticmethod
    def set_user_query(query):
        st.session_state.user_query = query

    @staticmethod
    def set_language(language):
        st.session_state.language = language
        session_db.save_meta("language", language)

    @staticmethod
    def start_new_session(username, session_name="New Chat"):
        session_id = session_db.create_new_session(username, session_name)
        st.session_state.current_session_id = session_id
        st.session_state.qa_history = []
        return session_id

    @staticmethod
    def load_session_history(session_id):
        st.session_state.current_session_id = session_id
        st.session_state.qa_history = session_db.load_qa_history(session_id)

    @staticmethod
    def add_to_history(question, answer):
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = []
        st.session_state.qa_history.append({"question": question, "answer": answer})
        if st.session_state.current_session_id:
            session_db.save_qa_history(st.session_state.current_session_id, st.session_state.qa_history)

    @staticmethod
    def get_history():
        return st.session_state.get('qa_history', [])

    @staticmethod
    def get_language():
        return st.session_state.get('language', 'English')

    @staticmethod
    def get_user_query():
        return st.session_state.get('user_query', '')

    @staticmethod
    def clear_history():
        if st.session_state.current_session_id:
            session_db.save_qa_history(st.session_state.current_session_id, [])
        st.session_state.qa_history = []

    @staticmethod
    def update_current_time():
        st.session_state.current_time = datetime.now().strftime("%A, %d %B %Y %H:%M:%S")

    @staticmethod
    def get_current_session_id():
        return st.session_state.get('current_session_id')

# Convenience functions for easier access
def initialize_session():
    SessionManager.initialize_session()

def set_user_query(query):
    SessionManager.set_user_query(query)

def set_language(language):
    SessionManager.set_language(language)

def start_new_session(username, session_name="New Chat"):
    return SessionManager.start_new_session(username, session_name)

def load_session_history(session_id):
    SessionManager.load_session_history(session_id)

def add_to_history(question, answer):
    SessionManager.add_to_history(question, answer)

def get_history():
    return SessionManager.get_history()

def get_language():
    return SessionManager.get_language()

def get_user_query():
    return SessionManager.get_user_query()

def clear_history():
    SessionManager.clear_history()

def update_current_time():
    SessionManager.update_current_time()

def get_current_session_id():
    return SessionManager.get_current_session_id()