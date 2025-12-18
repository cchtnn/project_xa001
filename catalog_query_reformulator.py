"""
Catalog Query Reformulator for Diné College Assistant
Reformulates course catalog queries to improve retrieval performance
Similar to PayrollQueryReformulator but specialized for catalog data
"""

import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from typing import Dict, Any
import warnings
from dotenv import load_dotenv
import logging
import re

logging.getLogger("watchdog").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
load_dotenv()


class CatalogQueryReformulator:
    """Reformulates course catalog queries for better retrieval performance"""
    
    def __init__(self, groq_api_key: str = None, model_name: str = "llama-3.1-8b-instant"):
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.model_name = model_name
        self.llm = None
        self._setup_llm()
        
    def _setup_llm(self):
        """Setup the ChatGroq LLM for query reformulation"""
        try:
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name=self.model_name,
                temperature=0,
                max_tokens=2048,
                streaming=False,
                request_timeout=30
            )
            print("✅ Catalog Query Reformulator LLM setup completed")
        except Exception as e:
            print(f"⚠️ Error initializing Catalog Query Reformulator LLM: {str(e)}")
            self.llm = None
    
    def _detect_query_type(self, user_query: str) -> Dict[str, Any]:
        """
        Detect the type of catalog query
        
        Types:
        1. COURSE_DETAIL: Asking for details about a specific course
        2. DEPARTMENT_LISTING: Asking for all courses in a department/category
        3. COURSE_SEARCH: Searching for courses by topic/keyword
        4. COURSE_ATTRIBUTE: Asking about specific attributes (credits, prerequisites, etc.)
        """
        query_lower = user_query.lower()
        
        query_info = {
            "type": None,
            "course_code": None,
            "department": None,
            "keywords": [],
            "attributes": []
        }
        
        # Pattern 1: Course code detection (e.g., AGR 323, ENV 105)
        course_code_pattern = r'\b([A-Z]{2,4})\s*(\d{3})\b'
        course_match = re.search(course_code_pattern, user_query, re.IGNORECASE)
        if course_match:
            query_info["course_code"] = f"{course_match.group(1).upper()} {course_match.group(2)}"
            query_info["type"] = "COURSE_DETAIL"
            print(f"🔍 Detected course code: {query_info['course_code']}")
        
        # Pattern 2: Department listing (e.g., "all courses under AGRICULTURE")
        department_patterns = [
            r'\ball\s+(?:course|courses)\s+(?:in|under|from|for)\s+([A-Z\s]+(?:\([A-Z]+\))?)',
            r'\bcourses?\s+(?:in|under)\s+([A-Z\s]+)',
            r'\blist\s+(?:all\s+)?courses?\s+(?:in|under)\s+([A-Z\s]+)',
            r'\bcourse\s+codes?\s+(?:in|under|from)\s+([A-Z\s]+)',
        ]
        
        for pattern in department_patterns:
            dept_match = re.search(pattern, user_query, re.IGNORECASE)
            if dept_match:
                dept_name = dept_match.group(1).strip()
                query_info["department"] = dept_name
                if not query_info["type"]:
                    query_info["type"] = "DEPARTMENT_LISTING"
                print(f"🏫 Detected department: {dept_name}")
                break
        
        # Pattern 3: Course attributes (credits, prerequisites, etc.)
        attribute_keywords = {
            "credits": ["credit", "credits", "credit hour"],
            "prerequisites": ["prerequisite", "prereq", "required before"],
            "description": ["description", "about", "details", "information"],
            "level": ["level", "upper division", "lower division"],
        }
        
        for attr_type, keywords in attribute_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    query_info["attributes"].append(attr_type)
        
        # Pattern 4: Topic-based search
        topic_keywords = ["related to", "about", "covering", "on the topic"]
        for keyword in topic_keywords:
            if keyword in query_lower:
                if not query_info["type"]:
                    query_info["type"] = "COURSE_SEARCH"
        
        # Default to COURSE_DETAIL if course code found, otherwise COURSE_SEARCH
        if not query_info["type"]:
            if query_info["course_code"]:
                query_info["type"] = "COURSE_DETAIL"
            else:
                query_info["type"] = "COURSE_SEARCH"
        
        print(f"📋 Query type detected: {query_info['type']}")
        return query_info
    
    def create_reformulation_prompt(self, query_info: Dict[str, Any]) -> str:
        """
        Create a detailed prompt for catalog query reformulation based on query type
        """
        
        prompt = f"""
You are an expert query reformulator for course catalog data. Your task is to convert natural language questions into precise, structured search queries that a document retrieval system can understand.

QUERY TYPE: {query_info['type']}

CATALOG DATA STRUCTURE:
- Course Code: Department abbreviation + number (e.g., AGR 323, ENV 105)
- Course Title: Full course name (e.g., "Mushroom and Molds", "Climate Change for Tribal Peoples")
- Department/Category: Subject area (e.g., AGRICULTURE (AGR), ENVIRONMENTAL SCIENCE AND TECHNOLOGY)
- Description: Detailed course information
- Attributes: Credits, prerequisites, level, etc.

REFORMULATION RULES BY QUERY TYPE:

1. COURSE_DETAIL (specific course information):
   - Extract exact course code
   - Include course title if mentioned
   - Specify what information is needed
   - Keep query focused on that specific course
   
   Examples:
   "Give me details about AGR 323" → "Course AGR 323 Mushroom and Molds: full description, prerequisites, credits"
   "What is ENV 105 about?" → "Course ENV 105 Climate Change for Tribal Peoples: description and overview"

2. DEPARTMENT_LISTING (all courses in a category):
   - Identify department name clearly
   - Request list of all course codes
   - Include department abbreviation if available
   
   Examples:
   "All courses under AGRICULTURE (AGR)" → "List all course codes and titles in AGRICULTURE (AGR) department"
   "Courses in Environmental Science" → "List all course codes in ENVIRONMENTAL SCIENCE AND TECHNOLOGY category"

3. COURSE_SEARCH (topic-based search):
   - Extract key topics/keywords
   - Make search broad enough to find relevant courses
   - Include related terms
   
   Examples:
   "Courses about climate change" → "Courses covering climate change, environmental impact, sustainability"
   "Classes on tribal peoples" → "Courses related to tribal peoples, indigenous communities, native cultures"

4. COURSE_ATTRIBUTE (specific attribute query):
   - Focus on the specific attribute requested
   - Include course code if mentioned
   
   Examples:
   "How many credits is AGR 323?" → "Credits for course AGR 323"
   "Prerequisites for ENV 105?" → "Prerequisites and requirements for course ENV 105"

CRITICAL INSTRUCTIONS:
1. Keep reformulated queries CLEAR and CONCISE (max 20 words)
2. Always preserve exact course codes (e.g., AGR 323, ENV 105)
3. Use full department names when available
4. For department listings, request "all course codes" explicitly
5. For course details, specify what information is needed
6. Remove filler words like "give me", "tell me", "I want"
7. Make queries optimized for document search

DETECTED INFORMATION:
- Course Code: {query_info.get('course_code', 'None')}
- Department: {query_info.get('department', 'None')}
- Attributes: {', '.join(query_info.get('attributes', [])) or 'None'}

Now reformulate the following catalog query:
"""
        return prompt
    
    def _reformulate_query(self, user_query: str, query_info: Dict[str, Any] = None) -> str:
        """
        Reformulate catalog query to be more specific for retrieval
        """
        if query_info is None:
            query_info = self._detect_query_type(user_query)
        
        if self.llm is None:
            print("⚠️ LLM not available, returning original query")
            return user_query
        
        try:
            system_prompt = self.create_reformulation_prompt(query_info)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User Query: {user_query}\n\nProvide only the reformulated query, no JSON, no explanations.")
            ]
            
            print(f"🔄 Sending query to reformulator...")
            response = self.llm.invoke(messages)
            print(f"📝 LLM reformulator raw response: {response.content[:500]}...")
            
            reformulated_query = response.content.strip()
            
            # Clean up any prefixes
            prefixes_to_remove = [
                "Reformulated:", "Reformulated Query:", "Query:", 
                "Answer:", "Response:", "Instructions:", "Search for:"
            ]
            for prefix in prefixes_to_remove:
                if reformulated_query.startswith(prefix):
                    reformulated_query = reformulated_query.replace(prefix, "").strip()
            
            # Remove quotes if present
            reformulated_query = reformulated_query.strip('"').strip("'")
            
            print(f"🔄 Catalog query reformulated:")
            print(f"   📥 Original: {user_query}")
            print(f"   📤 Reformulated: {reformulated_query}")
            
            return reformulated_query
            
        except Exception as e:
            print(f"❌ Error reformulating catalog query: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return user_query
    
    def process_catalog_query(self, user_query: str) -> Dict[str, Any]:
        """
        Complete pipeline: analyze query and reformulate

        Args:
            user_query (str): Original user query

        Returns:
            Dict containing reformulated query and metadata
        """
        print(f"\n{'='*60}")
        print(f"📚 CATALOG QUERY REFORMULATION PIPELINE")
        print(f"{'='*60}")
        
        result = {
            "original_query": user_query,
            "reformulated_query": user_query,
            "success": True,
            "query_type": None,
            "course_code": None,
            "department": None,
        }

        try:
            # Analyze query type
            print(f"🔍 Analyzing query type...")
            query_info = self._detect_query_type(user_query)
            
            result["query_type"] = query_info["type"]
            result["course_code"] = query_info.get("course_code")
            result["department"] = query_info.get("department")
            
            print(f"📋 Query Analysis:")
            print(f"   - Type: {query_info['type']}")
            print(f"   - Course Code: {query_info.get('course_code', 'N/A')}")
            print(f"   - Department: {query_info.get('department', 'N/A')}")

            # Skip reformulation if LLM is not available
            if self.llm is None:
                print("⚠️ LLM not available, skipping query reformulation")
                result["message"] = "Query reformulation skipped - LLM not available"
                return result

            # Reformulate the query
            print(f"🔄 Starting query reformulation...")
            reformulated = self._reformulate_query(user_query, query_info)
            
            # If reformulated is empty or None, fallback to original query
            if not reformulated or not reformulated.strip():
                print("⚠️ Reformulator returned empty result, using original query")
                reformulated = user_query
            
            result["reformulated_query"] = reformulated
            print(f"✅ Query reformulation completed successfully")
            print(f"{'='*60}\n")
            
            return result

        except Exception as e:
            print(f"❌ Error processing catalog query: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # Return original query as fallback
            result["success"] = True  # Set to True so system continues with original query
            result["error"] = str(e)
            result["reformulated_query"] = user_query
            result["message"] = "Query reformulation failed, using original query"
            print(f"{'='*60}\n")
            return result


# Singleton instance for caching
_catalog_reformulator_instance = None

def get_catalog_reformulator() -> CatalogQueryReformulator:
    """Get cached catalog reformulator instance"""
    global _catalog_reformulator_instance
    if _catalog_reformulator_instance is None:
        _catalog_reformulator_instance = CatalogQueryReformulator()
    return _catalog_reformulator_instance


def reformulate_catalog_query(user_query: str) -> str:
    """
    Convenience function to reformulate catalog queries
    
    Args:
        user_query (str): Original user query
        
    Returns:
        str: Reformulated query ready for retrieval
    """
    reformulator = get_catalog_reformulator()
    result = reformulator.process_catalog_query(user_query)
    
    if result["success"]:
        return result["reformulated_query"]
    else:
        print(f"⚠️ Query reformulation failed: {result.get('error', 'Unknown error')}")
        return user_query  # Return original query as fallback