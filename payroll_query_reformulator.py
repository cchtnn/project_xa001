"""
Payroll Query Reformulator for Diné College Assistant
Reformulates payroll calendar queries to improve CSV agent performance
Similar to StudentQueryReformulator but specialized for payroll data
"""

import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from typing import Dict, Any
import json
import warnings
from dotenv import load_dotenv
import logging
import re
from datetime import datetime

logging.getLogger("watchdog").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
load_dotenv()


class PayrollQueryReformulator:
    """Reformulates payroll calendar queries for better CSV agent performance"""
    
    def __init__(self, groq_api_key: str = None, model_name: str = "llama-3.1-8b-instant"):
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.model_name = model_name
        self.llm = None
        self.csv_structure = None
        self.current_year = datetime.now().year
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
            print("✅ Payroll Query Reformulator LLM setup completed")
        except Exception as e:
            print(f"⚠️ Error initializing Payroll Query Reformulator LLM: {str(e)}")
            self.llm = None
    
    def analyze_csv_structure(self, csv_path: str) -> Dict[str, Any]:
        """
        Analyze payroll CSV structure to understand columns, data types, and sample values
        """
        try:
            df = pd.read_csv(csv_path)
            
            structure = {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
                "sample_values": {},
                "row_count": len(df),
                "unique_values_count": {},
                "date_columns": []
            }
            
            # Get sample values for each column
            for col in df.columns:
                non_null_values = df[col].dropna().unique()[:3]
                structure["sample_values"][col] = [str(val) for val in non_null_values]
                structure["unique_values_count"][col] = df[col].nunique()
                
                # Detect date columns
                if any(keyword in col.lower() for keyword in ['date', 'start', 'end', 'check']):
                    structure["date_columns"].append(col)
            
            self.csv_structure = structure
            print(f"📊 Payroll CSV structure analyzed: {len(structure['columns'])} columns, {structure['row_count']} rows")
            print(f"📅 Date columns detected: {structure['date_columns']}")
            return structure
            
        except Exception as e:
            print(f"❌ Error analyzing payroll CSV structure: {str(e)}")
            return None
    
    def _detect_temporal_references(self, user_query: str) -> Dict[str, Any]:
        """
        Detect temporal references in the query (current year, this year, 2026, etc.)
        """
        query_lower = user_query.lower()
        
        temporal_info = {
            "has_temporal": False,
            "year": None,
            "is_current_year": False,
            "temporal_phrases": []
        }
        
        # Patterns for temporal references
        current_year_patterns = [
            r'\bcurrent\s+year\b',
            r'\bthis\s+year\b',
            r'\bin\s+\d{4}\b',
            r'\byear\s+\d{4}\b'
        ]
        
        # Check for current year references
        for pattern in current_year_patterns:
            match = re.search(pattern, query_lower)
            if match:
                temporal_info["has_temporal"] = True
                temporal_info["temporal_phrases"].append(match.group(0))
                
                # Check if it's explicitly current year
                if 'current' in match.group(0) or 'this' in match.group(0):
                    temporal_info["is_current_year"] = True
        
        # Extract explicit year mentions
        year_match = re.search(r'\b(20\d{2})\b', user_query)
        if year_match:
            temporal_info["year"] = int(year_match.group(1))
            temporal_info["has_temporal"] = True
        
        # If current year mentioned but no explicit year, use current year
        if temporal_info["is_current_year"] and not temporal_info["year"]:
            temporal_info["year"] = self.current_year
        
        if temporal_info["has_temporal"]:
            print(f"🕐 Detected temporal reference: Year={temporal_info['year']}, Current={temporal_info['is_current_year']}")
        
        return temporal_info
    
    def _detect_count_query(self, user_query: str) -> bool:
        """
        Detect if query is asking for a count
        """
        query_lower = user_query.lower()
        count_patterns = [
            r'\bhow many\b',
            r'\bcount\b',
            r'\bnumber of\b',
            r'\btotal\b',
        ]
        
        for pattern in count_patterns:
            if re.search(pattern, query_lower):
                return True
        return False
    
    def create_reformulation_prompt(self, csv_structure: Dict[str, Any]) -> str:
        """
        Create a detailed prompt for payroll query reformulation based on CSV structure
        """
        columns_info = []
        for col in csv_structure["columns"]:
            dtype = csv_structure["dtypes"][col]
            samples = csv_structure["sample_values"][col]
            unique_count = csv_structure["unique_values_count"][col]
            columns_info.append(f"- '{col}' ({dtype}, {unique_count} unique values): Examples: {samples}")
        
        prompt = f"""
You are an expert query reformulator for payroll calendar CSV data analysis. Your task is to convert natural language questions into precise, structured queries that a CSV agent with pandas can understand and execute accurately.

PAYROLL CALENDAR CSV STRUCTURE:
Total rows: {csv_structure["row_count"]}
Date columns: {csv_structure["date_columns"]}
Available columns:
{chr(10).join(columns_info)}

CRITICAL REFORMULATION RULES:
1. Keep queries SHORT and SIMPLE - one concise instruction, max 15 words
2. Always use EXACT column names as they appear in the CSV (case-sensitive)
3. Do NOT include pandas imports or datetime conversions - agent handles this
4. For date filtering, just specify the date value: "where start_date is '1/3/2026'"
5. For calculations, use simple expressions: "calculate (check_date - start_date) in days"
6. For counting queries: "count rows where [condition]"
7. For temporal references, use the year number directly
8. Focus on WHAT to find, not HOW to do it

PAYROLL-SPECIFIC PATTERNS:
- "payroll periods in YEAR" → "count rows where start_date year equals YEAR"
- "check date for period X to Y" → "find check_date where start_date is X and end_date is Y"
- "when is next pay" → "find earliest check_date after today"
- "withholding deadline" → "find optional_withholdings_changes_by for given period"
- "payroll number X" → "find row where payroll_no equals X"

    EXAMPLES:

    Original: "How many payroll periods do we have in Current year 2026?"
    Reformulated: "Filter rows where year of start_date equals 2026 and count them"

    Original: "Tell me the check date where optional withholdings changes is 2/27/2026?"
    Reformulated: "Filter rows where optional_withholdings_changes_by equals '2/27/2026' and return check_date"

    Original: "When is the check date for payroll period between 01/03/2026 to 01/16/2026?"
    Reformulated: "Filter rows where start_date is '1/3/2026' and end_date is '1/16/2026', return check_date"

    Original: "What is payroll number 5?"
    Reformulated: "Filter rows where payroll_no equals 5 and return all columns"

    Original: "What is the days difference between start date and check date for payroll number 10?"
    Reformulated: "Filter rows where payroll_no equals 10, then calculate (check_date - start_date) in days"

    Original: "Show me all check dates"
    Reformulated: "Return all unique values from check_date column sorted by date"

IMPORTANT NOTES:
- Date format in CSV is M/D/YYYY (not MM/DD/YYYY) - single digit months/days have no leading zero
- Always specify format='%m/%d/%Y' when using pd.to_datetime()
- When filtering by date, consider both formats: '1/3/2026' and '01/03/2026' might be used
- For year extraction, always convert to datetime first before accessing .dt.year
- Current year is {self.current_year}

CRITICAL INSTRUCTIONS FOR REFORMULATION:
1. Keep reformulated queries SHORT and SIMPLE (one sentence, max 15 words)
2. Do NOT include step-by-step instructions - make it ONE concise instruction
3. Do NOT mention "import pandas" - the agent already has pandas available
4. Do NOT mention datetime conversion - the agent handles this automatically
5. Focus ONLY on what data to filter and what to return
6. For calculations, use simple math expressions like (check_date - start_date).days

BAD REFORMULATION (too verbose):
"Import pandas as pd. Convert 'start_date' and 'check_date' columns to datetime using format M/D/YYYY. Filter rows where 'payroll_no' equals 10. Calculate the absolute difference in days between 'start_date' and 'check_date' for matching rows. Return this difference as a new column named 'days_difference'."

GOOD REFORMULATION (concise):
"For payroll_no 10, calculate days between start_date and check_date"

Now reformulate the following payroll calendar query:
"""
        return prompt
    
    def _reformulate_query(self, user_query: str, csv_structure: Dict[str, Any] = None) -> str:
        """
        Reformulate payroll query to be more specific for CSV agent
        """
        if csv_structure is None:
            csv_structure = self.csv_structure
            
        if csv_structure is None:
            print("⚠️ No CSV structure available, returning original query")
            return user_query
        
        if self.llm is None:
            print("⚠️ LLM not available, returning original query")
            return user_query
        
        try:
            # Detect temporal references and count queries
            temporal_info = self._detect_temporal_references(user_query)
            is_count_query = self._detect_count_query(user_query)
            
            # Build context for reformulation
            context_notes = []
            
            if temporal_info["has_temporal"]:
                if temporal_info["year"]:
                    context_notes.append(f"TEMPORAL CONTEXT: Query refers to year {temporal_info['year']}")
                if temporal_info["is_current_year"]:
                    context_notes.append(f"Note: 'current year' means {self.current_year}")
            
            if is_count_query:
                context_notes.append("QUERY TYPE: This is a counting query - result should be a number")
            
            context = "\n".join(context_notes) if context_notes else ""
            
            system_prompt = self.create_reformulation_prompt(csv_structure)
            if context:
                system_prompt += f"\n\nADDITIONAL CONTEXT:\n{context}\n"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User Query: {user_query}\n\nProvide only the reformulated query as clear instructions, no JSON, no explanations.")
            ]
            
            print(f"🔄 Sending query to reformulator...")
            response = self.llm.invoke(messages)
            print(f"📝 LLM reformulator raw response: {response.content[:500]}...")
            
            reformulated_query = response.content.strip()
            
            # Clean up any prefixes
            prefixes_to_remove = [
                "Reformulated:", "Reformulated Query:", "Query:", 
                "Answer:", "Response:", "Instructions:"
            ]
            for prefix in prefixes_to_remove:
                if reformulated_query.startswith(prefix):
                    reformulated_query = reformulated_query.replace(prefix, "").strip()
            
            print(f"🔄 Payroll query reformulated:")
            print(f"   📥 Original: {user_query}")
            print(f"   📤 Reformulated: {reformulated_query}")
            
            return reformulated_query
            
        except Exception as e:
            print(f"❌ Error reformulating payroll query: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return user_query
    
    def process_payroll_query(self, user_query: str, csv_path: str = None) -> Dict[str, Any]:
        """
        Complete pipeline: analyze CSV (if needed) and reformulate query

        Args:
            user_query (str): Original user query
            csv_path (str, optional): Path to CSV file for structure analysis

        Returns:
            Dict containing reformulated query and metadata
        """
        print(f"\n{'='*60}")
        print(f"🔧 PAYROLL QUERY REFORMULATION PIPELINE")
        print(f"{'='*60}")
        
        result = {
            "original_query": user_query,
            "reformulated_query": user_query,
            "success": True,
            "csv_structure_available": False,
            "temporal_info": None,
            "is_count_query": False
        }

        try:
            # Analyze CSV structure if path provided and not already analyzed
            if csv_path and self.csv_structure is None:
                print(f"📊 Analyzing CSV structure: {csv_path}")
                structure = self.analyze_csv_structure(csv_path)
                if structure:
                    result["csv_structure_available"] = True
                    print(f"✅ CSV structure loaded: {len(structure['columns'])} columns")
            elif self.csv_structure:
                result["csv_structure_available"] = True
                print(f"✅ Using cached CSV structure")

            # Detect query characteristics
            temporal_info = self._detect_temporal_references(user_query)
            is_count_query = self._detect_count_query(user_query)
            
            result["temporal_info"] = temporal_info
            result["is_count_query"] = is_count_query
            
            print(f"🔍 Query Analysis:")
            print(f"   - Has temporal reference: {temporal_info['has_temporal']}")
            print(f"   - Year: {temporal_info.get('year', 'N/A')}")
            print(f"   - Is count query: {is_count_query}")

            # Skip reformulation if LLM is not available
            if self.llm is None:
                print("⚠️ LLM not available, skipping query reformulation")
                result["message"] = "Query reformulation skipped - LLM not available"
                return result

            # Reformulate the query
            print(f"🔄 Starting query reformulation...")
            reformulated = self._reformulate_query(user_query)
            
            # If reformulated is empty or None, fallback to original query
            if not reformulated or not reformulated.strip():
                print("⚠️ Reformulator returned empty result, using original query")
                reformulated = user_query
            
            result["reformulated_query"] = reformulated
            print(f"✅ Query reformulation completed successfully")
            print(f"{'='*60}\n")
            
            return result

        except Exception as e:
            print(f"❌ Error processing payroll query: {str(e)}")
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
_payroll_reformulator_instance = None

def get_payroll_reformulator() -> PayrollQueryReformulator:
    """Get cached payroll reformulator instance"""
    global _payroll_reformulator_instance
    if _payroll_reformulator_instance is None:
        _payroll_reformulator_instance = PayrollQueryReformulator()
    return _payroll_reformulator_instance


def reformulate_payroll_query(user_query: str, csv_path: str = None) -> str:
    """
    Convenience function to reformulate payroll calendar queries
    
    Args:
        user_query (str): Original user query
        csv_path (str, optional): Path to CSV file for structure analysis
        
    Returns:
        str: Reformulated query ready for CSV agent
    """
    reformulator = get_payroll_reformulator()
    result = reformulator.process_payroll_query(user_query, csv_path)
    
    if result["success"]:
        return result["reformulated_query"]
    else:
        print(f"⚠️ Query reformulation failed: {result.get('error', 'Unknown error')}")
        return user_query  # Return original query as fallback