"""
Session State Management for Diné College Assistant
Handles all session state operations and history management
"""

import streamlit as st
from datetime import datetime
import session_db  # Only for language persistence if needed
import tempfile
import os


class SessionManager:
    """Manages Streamlit session state with DB persistence"""
    
    @staticmethod
    def _get_temp_history_file():
        """Get temporary file path for chat history that gets cleared on server restart"""
        # Use temp directory that gets cleared on server restart
        temp_dir = tempfile.gettempdir()
        return os.path.join(temp_dir, f"streamlit_chat_history_{hash('dine_college_assistant')}.txt")

    @staticmethod
    def _load_temp_history():
        """Load chat history from temporary file"""
        import json
        temp_file = SessionManager._get_temp_history_file()
        try:
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    @staticmethod
    def _save_temp_history(history):
        """Save chat history to temporary file"""
        import json
        temp_file = SessionManager._get_temp_history_file()
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(history, f)
        except Exception:
            pass  # Fail silently if can't write to temp file

    @staticmethod
    def initialize_session():
        if 'qa_history' not in st.session_state:
            # Load chat history from temporary file (persists on refresh, clears on server restart)
            st.session_state.qa_history = SessionManager._load_temp_history()
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
        # Save to temporary file (persists on refresh, clears on server restart)
        SessionManager._save_temp_history(st.session_state.qa_history)

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
        # Clear from temporary file as well
        SessionManager._save_temp_history([])

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


def clear_history():
    """Clear Q&A history"""
    SessionManager.clear_history()


def update_current_time():
    """Update the current time in session"""
    SessionManager.update_current_time()