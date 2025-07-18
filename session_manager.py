"""
Session State Management for Diné College Assistant
Handles all session state operations and history management
"""

import streamlit as st
from datetime import datetime
import session_db


class SessionManager:
    """Manages Streamlit session state with DB persistence"""

    @staticmethod
    def initialize_session():
        session_db.init_db()
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = session_db.load_qa_history()
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
    def add_to_history(question, answer):
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = []
        st.session_state.qa_history.append({"question": question, "answer": answer})
        session_db.save_qa_history(st.session_state.qa_history)

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
        st.session_state.qa_history = []
        session_db.save_qa_history([])

    @staticmethod
    def update_current_time():
        st.session_state.current_time = datetime.now().strftime("%A, %d %B %Y %H:%M:%S")


# Convenience functions for easier access
def initialize_session():
    """Initialize session state"""
    SessionManager.initialize_session()


def set_user_query(query):
    """Set user query in session"""
    SessionManager.set_user_query(query)


def set_language(language):
    """Set language in session"""
    SessionManager.set_language(language)


def add_to_history(question, answer):
    """Add Q&A to history"""
    SessionManager.add_to_history(question, answer)


def get_history():
    """Get Q&A history"""
    return SessionManager.get_history()


def get_language():
    """Get selected language"""
    return SessionManager.get_language()


def get_user_query():
    """Get current user query"""
    return SessionManager.get_user_query()