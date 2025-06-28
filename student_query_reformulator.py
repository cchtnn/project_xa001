"""
Student Query Reformulator for Diné College Assistant
Reformulates student transcript queries to improve CSV agent performance
Integrates with existing StudentTranscriptCSVHandler
"""

import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from typing import Dict, Any
import json
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()


class StudentQueryReformulator:
    """Reformulates student transcript queries for better CSV agent performance"""
    
    def __init__(self, groq_api_key: str = None, model_name: str = "llama3-8b-8192"):
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.model_name = model_name
        self.llm = None
        self.csv_structure = None
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
            print("✅ Query Reformulator LLM setup completed")
        except Exception as e:
            raise Exception(f"Error initializing Query Reformulator LLM: {str(e)}")
    
    def analyze_csv_structure(self, csv_path: str) -> Dict[str, Any]:
        """
        Analyze CSV structure to understand columns, data types, and sample values
        """
        try:
            df = pd.read_csv(csv_path)
            
            structure = {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
                "sample_values": {},
                "row_count": len(df),
                "unique_values_count": {}
            }
            
            # Get sample values for each column (first 5 non-null unique values)
            for col in df.columns:
                non_null_values = df[col].dropna().unique()[:5]
                structure["sample_values"][col] = [str(val) for val in non_null_values]
                structure["unique_values_count"][col] = df[col].nunique()
            
            self.csv_structure = structure
            print(f"📊 CSV structure analyzed: {len(structure['columns'])} columns, {structure['row_count']} rows")
            return structure
            
        except Exception as e:
            print(f"❌ Error analyzing CSV structure: {str(e)}")
            return None
    
    def create_reformulation_prompt(self, csv_structure: Dict[str, Any]) -> str:
        """
        Create a detailed prompt for query reformulation based on CSV structure
        """
        columns_info = []
        for col in csv_structure["columns"]:
            dtype = csv_structure["dtypes"][col]
            samples = csv_structure["sample_values"][col]
            unique_count = csv_structure["unique_values_count"][col]
            columns_info.append(f"- '{col}' ({dtype}, {unique_count} unique values): Examples: {samples}")
        
        prompt = f"""
You are an expert query reformulator for student transcript CSV data analysis. Your task is to convert natural language questions into precise, structured queries that a CSV agent can understand and execute accurately.

STUDENT TRANSCRIPT CSV STRUCTURE:
Total rows: {csv_structure["row_count"]}
Available columns:
{chr(10).join(columns_info)}

REFORMULATION RULES:
1. Always use EXACT column names as they appear in the CSV (case-sensitive, including spaces)
2. Be specific about column references - use phrases like "from the 'Column Name' column"
3. Convert vague terms to specific column references based on available columns
4. Maintain the original intent while being more explicit
5. For filtering, be specific about column names and use exact values when possible
6. For aggregations, clearly specify the column to aggregate and the operation
7. Use proper pandas/SQL-like syntax concepts that the CSV agent can understand
8. When looking for students, always reference the student name column specifically
9. When looking for courses, reference course-related columns specifically
10. When looking for grades, reference grade-related columns specifically

COMMON QUERY PATTERNS:
- "student who studies at X" → "student name from the 'Student Name' column where the 'College Name' or 'Organization Name' column equals 'X'"
- "courses for student X" → "course information from relevant course columns where 'Student Name' column equals 'X'"
- "students with grade X" → "student names from 'Student Name' column where grade column equals 'X'"
- "GPA information" → "GPA values from 'GPA' column for specified conditions"

EXAMPLES:
Original: "Tell me the student who is studying in college xyz"
Reformulated: "Show me the student name from the 'Student Name' column where the 'College Name' column equals 'xyz'"

Original: "Name of student where organization is NEWMAN UNIVERSITY"
Reformulated: "Give me the unique student names from the 'Student Name' column where the 'Organization Name' or 'College Name' column equals 'NEWMAN UNIVERSITY'"

Original: "What courses did John take?"
Reformulated: "Show me all course information from course-related columns where the 'Student Name' column equals 'John'"

Now reformulate the following user query to be more specific and actionable for the CSV agent:
"""
        return prompt
    
    def reformulate_query(self, user_query: str, csv_structure: Dict[str, Any] = None) -> str:
        """
        Reformulate user query to be more specific for CSV agent
        """
        if csv_structure is None:
            csv_structure = self.csv_structure
            
        if csv_structure is None:
            print("⚠️ No CSV structure available, returning original query")
            return user_query
        
        try:
            system_prompt = self.create_reformulation_prompt(csv_structure)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User Query: {user_query}\n\nProvide only the reformulated query, no explanations or prefixes.")
            ]
            
            response = self.llm.invoke(messages)
            reformulated_query = response.content.strip()
            
            # Clean up any prefixes that might be added
            prefixes_to_remove = ["Reformulated:", "Reformulated Query:", "Query:", "Answer:", "Response:"]
            for prefix in prefixes_to_remove:
                if reformulated_query.startswith(prefix):
                    reformulated_query = reformulated_query.replace(prefix, "").strip()
            
            print(f"🔄 Query reformulated:")
            print(f"   Original: {user_query}")
            print(f"   Reformulated: {reformulated_query}")
            
            return reformulated_query
            
        except Exception as e:
            print(f"❌ Error reformulating query: {str(e)}")
            print(f"   Returning original query: {user_query}")
            return user_query
    
    def validate_query_feasibility(self, user_query: str, csv_structure: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate if the query can be answered with available columns and suggest alternatives
        """
        if csv_structure is None:
            csv_structure = self.csv_structure
            
        if csv_structure is None:
            return {"can_answer": True, "confidence": "unknown", "message": "No structure analysis available"}
        
        try:
            validation_prompt = f"""
Analyze if the following student transcript query can be answered using the available CSV columns.

AVAILABLE COLUMNS: {', '.join(csv_structure['columns'])}
SAMPLE DATA: {json.dumps(csv_structure['sample_values'], indent=2)}

USER QUERY: {user_query}

Respond with a JSON object containing:
{{
    "can_answer": true/false,
    "confidence": "high"/"medium"/"low",
    "required_columns": ["list", "of", "columns", "needed"],
    "missing_info": "what information is missing if any",
    "suggestions": ["alternative queries that can be answered"]
}}
"""
            
            messages = [
                SystemMessage(content=validation_prompt),
                HumanMessage(content="Analyze the query feasibility and respond with JSON only:")
            ]
            
            response = self.llm.invoke(messages)
            
            # Try to parse JSON response
            try:
                validation_result = json.loads(response.content)
                return validation_result
            except json.JSONDecodeError:
                # Fallback to simple validation
                return {
                    "can_answer": True, 
                    "confidence": "medium", 
                    "message": "Could not parse validation response",
                    "validation_text": response.content
                }
                
        except Exception as e:
            print(f"❌ Error validating query: {str(e)}")
            return {"can_answer": True, "confidence": "unknown", "error": str(e)}
    
    def process_student_query(self, user_query: str, csv_path: str = None) -> Dict[str, Any]:
        """
        Complete pipeline: analyze CSV (if needed), validate, and reformulate query
        
        Args:
            user_query (str): Original user query
            csv_path (str, optional): Path to CSV file for structure analysis
            
        Returns:
            Dict containing reformulated query and metadata
        """
        result = {
            "original_query": user_query,
            "reformulated_query": user_query,
            "success": True,
            "validation": None,
            "csv_structure_available": False
        }
        
        try:
            # Analyze CSV structure if path provided and not already analyzed
            if csv_path and self.csv_structure is None:
                structure = self.analyze_csv_structure(csv_path)
                if structure:
                    result["csv_structure_available"] = True
            elif self.csv_structure:
                result["csv_structure_available"] = True
            
            # Validate query feasibility if structure is available
            if self.csv_structure:
                validation = self.validate_query_feasibility(user_query)
                result["validation"] = validation
                
                # If confidence is very low, provide feedback
                if validation.get("confidence") == "low" and not validation.get("can_answer", True):
                    result["success"] = False
                    result["message"] = "Query may not be answerable with available data"
                    result["suggestions"] = validation.get("suggestions", [])
                    return result
            
            # Reformulate the query
            reformulated = self.reformulate_query(user_query)
            result["reformulated_query"] = reformulated
            
            print(f"✅ Query processing completed successfully")
            return result
            
        except Exception as e:
            print(f"❌ Error processing student query: {str(e)}")
            result["success"] = False
            result["error"] = str(e)
            return result


# Singleton instance for caching
_query_reformulator_instance = None

def get_query_reformulator() -> StudentQueryReformulator:
    """Get cached query reformulator instance"""
    global _query_reformulator_instance
    if _query_reformulator_instance is None:
        _query_reformulator_instance = StudentQueryReformulator()
    return _query_reformulator_instance


def reformulate_student_query(user_query: str, csv_path: str = None) -> str:
    """
    Convenience function to reformulate student transcript queries
    
    Args:
        user_query (str): Original user query
        csv_path (str, optional): Path to CSV file for structure analysis
        
    Returns:
        str: Reformulated query ready for CSV agent
    """
    reformulator = get_query_reformulator()
    result = reformulator.process_student_query(user_query, csv_path)
    
    if result["success"]:
        return result["reformulated_query"]
    else:
        print(f"⚠️ Query reformulation failed: {result.get('error', 'Unknown error')}")
        return user_query  # Return original query as fallback