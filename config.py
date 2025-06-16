"""
Configuration file for Diné College Assistant
Contains all configuration constants and settings
"""

# Page configuration
PAGE_CONFIG = {
    "page_title": "Diné College Assistant",
    "page_icon": "🏛️",
    "layout": "centered",
    "initial_sidebar_state": "collapsed"
}

# ChromaDB configuration
CHROMA_COLLECTION_NAME = "jericho_documents"

# File paths
PATHS = {
    "tab_data": "data/tab_data.json",
    "css_template": "templates/styles.css",
    "jericho_logo": "templates/jericho_image.jpg",
    "globe_svg": "templates/globe_.svg"
}

# Text splitter configuration
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "separators": ["\n\n", "\n", ". ", " ", ""]
}

# Language options
LANGUAGE_OPTIONS = ["English", "Spanish", "Navajo"]

# Logo URLs
LOGO_URLS = {
    "dine_college": "https://www.dinecollege.edu/wp-content/uploads/2024/12/dc_logoFooter.png"
}