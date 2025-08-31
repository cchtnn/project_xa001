import os
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
import warnings
import streamlit as st
from dotenv import load_dotenv
from student_query_reformulator import get_query_reformulator
import re
import logging
import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
logging.getLogger("watchdog").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")
load_dotenv()


class StudentTranscriptCSVHandler:
    """Handles student transcript queries using CSV Agent with query reformulation"""
    
    def __init__(self, csv_path=None, model_name="llama3-8b-8192"):
        self.csv_path = csv_path or st.session_state.get("active_transcript_csv_path", "data/csv_folder/student_transcript.csv")
        self.model_name = model_name
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.llm = None
        self.summarizer_llm = None
        self.agent = None
        self.df = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.query_reformulator = None
        self.is_initialized = False
        self.csv_structure = None
        self.frequent_queries = self._load_frequent_queries()
        
    def initialize(self):
        """Initialize the CSV handler with LLM, agent, and query reformulator"""
        try:
            print("📚 Initializing Student Transcript CSV Handler...")
            print(f"CSV Path: {self.csv_path}")
            
            # Check if CSV file exists
            if not os.path.exists(self.csv_path):
                print(f"❌ CSV file not found: {self.csv_path}")
                return False
            
            # Setup LLM
            self._setup_llm()
            
            # Setup summarizer
            self._setup_summarizer()
            
            # Setup query reformulator
            self._setup_query_reformulator()
            
            # Load CSV and create agent
            self._load_csv_and_create_agent()
            
            self.is_initialized = True
            print("✅ Student Transcript CSV Handler initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing student transcript CSV handler: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_frequent_queries(self) -> Dict[str, str]:
        """Load frequently asked questions and their pre-reformulated queries"""
        return {
            # GPA related queries
            "Sort students in descending order of GPA": "calculate mean GPA for each student from 'Student Name' and 'GPA' columns, sort in descending order by GPA value",
            "highest gpa student": "find student with maximum GPA value from 'Student Name' and 'GPA' columns",
            "lowest gpa student": "find student with minimum GPA value from 'Student Name' and 'GPA' columns", 
            "average gpa": "calculate overall mean of all GPA values from 'GPA' column",
            "students with gpa above": "filter students from 'Student Name' column where 'GPA' column value is greater than specified threshold",
            
            # Student information queries
            "student at college": "show student names from 'Student Name' column where 'College Name' or 'Organization Name' column matches specified college",
            "all students": "show all unique student names from 'Student Name' column",
            "student count": "count unique students from 'Student Name' column",
            
            # Course related queries  
            "courses for student": "show all course information from course-related columns where 'Student Name' column equals specified student name",
            "course count": "count total courses or course entries in the dataset",
            
            # Grade related queries
            "students with grade": "filter students from 'Student Name' column where grade-related column equals specified grade",
            "grade distribution": "show distribution of grades from grade-related columns",
            
            # Advisor related queries
            "advisor for student": "show advisor information from advisor-related columns where 'Student Name' column equals specified student name",
            "students by advisor": "show students from 'Student Name' column grouped by advisor from advisor-related columns"
        }

    def _find_matching_frequent_query(self, user_query: str) -> str:
        """Find if user query matches any frequent query pattern"""
        user_query_lower = user_query.lower().strip()
        
        # Direct keyword matching - check both directions
        for pattern, reformulated in self.frequent_queries.items():
            pattern_lower = pattern.lower()
            # Check if pattern matches user query or user query matches pattern
            if (pattern_lower in user_query_lower) or (user_query_lower in pattern_lower):
                print(f"🎯 Found matching frequent query pattern: '{pattern}'")
                return reformulated
        
        # Additional exact phrase matching for common variations
        query_variations = {
            "sort students in descending order of gpa": "Sort students in descending order of GPA",
            "sort students by gpa descending": "Sort students in descending order of GPA", 
            "order students by gpa desc": "Sort students in descending order of GPA",
            "rank students by gpa": "Sort students in descending order of GPA",
            "list students by gpa highest first": "Sort students in descending order of GPA"
        }
        
        for variation, pattern in query_variations.items():
            if variation in user_query_lower:
                if pattern in self.frequent_queries:
                    print(f"🎯 Found matching query variation: '{variation}' -> '{pattern}'")
                    return self.frequent_queries[pattern]

    def _setup_llm(self):
        """Setup the ChatGroq LLM for query reformulation"""
        try:
            if not self.groq_api_key:
                print("⚠️ GROQ_API_KEY not found, query reformulation will be disabled")
                self.llm = None
                return
                
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name=self.model_name,
                temperature=0,
                max_tokens=4096,
                streaming=False,
                request_timeout=60
            )
            print("✅ Query Reformulator LLM setup completed")
        except Exception as e:
            print(f"⚠️ Query Reformulator LLM setup failed: {str(e)}")
            self.llm = None
    
    def _setup_summarizer(self):
        """Setup a separate LLM for summarization"""
        try:
            self.summarizer_llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name=self.model_name,
                temperature=0.1,
                max_tokens=4096,
                streaming=False,
                request_timeout=60
            )
            print("✅ Summarizer LLM setup completed")
        except Exception as e:
            raise Exception(f"Error initializing Summarizer LLM: {str(e)}")
    
    def _setup_query_reformulator(self):
        """Setup the query reformulator and analyze CSV structure"""
        try:
            # Get the query reformulator instance
            self.query_reformulator = get_query_reformulator()
            
            # Analyze the CSV structure for the reformulator
            csv_structure = self.query_reformulator.analyze_csv_structure(self.csv_path)
            self.csv_structure = csv_structure  # <-- ADD THIS LINE
            
            if csv_structure:
                print("✅ Query Reformulator setup completed with CSV structure analysis")
            else:
                print("⚠️ Query Reformulator setup completed but CSV structure analysis failed")
                
        except Exception as e:
            print(f"⚠️ Warning: Query Reformulator setup failed: {str(e)}")
            print("   Continuing without query reformulation...")
            self.query_reformulator = None
    
    def _load_csv_and_create_agent(self):
        """Load CSV and create the agent"""
        try:
            # Load and analyze the CSV
            self.df = pd.read_csv(self.csv_path)
            
            # Convert GPA to numeric if it exists
            if 'GPA' in self.df.columns:
                # First, replace empty strings and whitespace with NaN
                self.df['GPA'] = self.df['GPA'].astype(str).str.strip()
                self.df['GPA'] = self.df['GPA'].replace('', pd.NA)
                self.df['GPA'] = self.df['GPA'].replace(' ', pd.NA)
                self.df['GPA'] = self.df['GPA'].replace('nan', pd.NA)
                
                # Convert to numeric, coercing errors to NaN
                self.df['GPA'] = pd.to_numeric(self.df['GPA'], errors='coerce')
                self.df['GPA'] = pd.to_numeric(self.df['GPA'], errors='coerce')
                
                # Save the cleaned CSV back for the agent to use
                self.df.to_csv(self.csv_path, index=False)
                print(f"✅ GPA column cleaned and converted to numeric")
                print(f"   Non-null GPA values: {self.df['GPA'].notna().sum()}")
                print(f"   Null GPA values: {self.df['GPA'].isna().sum()}")
            
            # Get column information for debugging
            column_info = self._get_column_info()
            print(f"📊 CSV loaded successfully:")
            print(f"   Shape: {self.df.shape}")
            print(f"   Columns: {list(self.df.columns)}")
            print(f"   Column Info: {column_info}")
            
            # Create the CSV agent with improved configuration
            self.agent = create_csv_agent(
                llm=self.llm,
                path=self.csv_path,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                allow_dangerous_code=True,
                handle_parsing_errors=True,
                max_iterations=3,
                max_execution_time=60,
                return_intermediate_steps=False,
                include_df_in_prompt=False,
                prefix="""
                You are working with a pandas DataFrame in Python. The DataFrame is loaded from a CSV file.
                You should use the tools below to answer the question posed about the DataFrame.

                CRITICAL INSTRUCTIONS FOR TOOL USAGE:
                1. You have access to ONLY ONE tool: python_repl_ast
                2. ALWAYS use this EXACT format for actions:
                    Action: python_repl_ast
                    Action Input: your_python_code_here

                3. NEVER use descriptive text as the Action name
                4. NEVER say "Use the python_repl_ast to..." - just use "python_repl_ast"
                5. After getting results from an action, IMMEDIATELY provide the full pandas dataframe as Final Answer
                6. Do NOT execute additional actions after finding the answer
                7. For unique values, use .unique() or .drop_duplicates()
                8. Give unique rows only - do not repeat rows in your answers.
                
                RESPONSE FORMAT:
                9. When you find data, Give unique rows only - do not repeat rows in your answers. Write result of the agent after fixed text "Final Answer"
                10. For advisor queries: if multiple rows have same advisor, show unique advisor name only
                11. show unique column rows or columns only
                12. When None is coming as answer then in that case mention "No data found" instead of None.
                
                DATA HANDLING RULES:
                13. pandas is already imported as 'pd' - you don't need to import it again
                14. For GPA calculations, use .mean() method and handle NaN values properly
                15. For groupby operations, use .dropna() if needed to exclude null values
                16. Always check data types before performing operations
                17. Use .sort_values(ascending=False) for descending order sorting

                The DataFrame columns and their types are automatically detected by pandas.
                The GPA column has been pre-processed to be numeric (float type).
                """
            )
            
            print("✅ CSV Agent created successfully")
            
        except Exception as e:
            raise Exception(f"Error loading CSV and creating agent: {str(e)}")
    
    def _get_column_info(self):
        """Analyze the DataFrame and return column type information"""
        if self.df is None:
            return "No data loaded"
        
        column_info = {}
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            null_count = self.df[col].isnull().sum()
            total_count = len(self.df[col])
            
            # Determine the logical data type
            if dtype in ['int64', 'int32', 'int16', 'int8']:
                logical_type = 'integer'
            elif dtype in ['float64', 'float32', 'float16']:
                logical_type = 'float'
            elif dtype == 'object':
                # Check if it's actually numeric stored as string
                try:
                    pd.to_numeric(self.df[col], errors='raise')
                    logical_type = 'numeric_string'
                except:
                    logical_type = 'string'
            elif dtype == 'bool':
                logical_type = 'boolean'
            else:
                logical_type = 'other'
            
            column_info[col] = {
                'pandas_dtype': dtype,
                'logical_type': logical_type,
                'null_count': null_count,
                'null_percentage': round((null_count / total_count) * 100, 2) if total_count > 0 else 0
            }
        
        # Format for display
        info_str = ""
        for col, info in column_info.items():
            info_str += f"- {col}: {info['logical_type']} (pandas: {info['pandas_dtype']}, nulls: {info['null_count']}/{total_count} = {info['null_percentage']}%)\n"
        
        return info_str

    def _reformulate_query(self, user_query: str, csv_structure: Dict[str, Any] = None) -> str:
        """
        Reformulate user query to be more specific for CSV agent
        """
        if csv_structure is None:
            csv_structure = self.csv_structure
            
        if csv_structure is None:
            print("⚠️ No CSV structure available, returning original query")
            return user_query
        
        if self.llm is None:
            print("⚠️ LLM not available for query reformulation, returning original query")
            return user_query
        # First check if query matches any frequent query pattern
        frequent_match = self._find_matching_frequent_query(user_query)
        if frequent_match:
            print(f"🔄 Using pre-reformulated frequent query:")
            print(f"   Original: {user_query}")
            print(f"   Pre-reformulated: {frequent_match}")
            return frequent_match
        
        print("🔄 No frequent query match found, using LLM reformulation...")

        try:
            # FIX: Use the reformulator's prompt creator
            system_prompt = self.query_reformulator.create_reformulation_prompt(csv_structure)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User Query: {user_query}\n\nProvide only the reformulated query, no explanations or prefixes.")
            ]
            response = self.llm.invoke(messages)
            print(f"🔄 LLM reformulator raw response: {response.content}")
            reformulated_query = response.content.strip()
            
            # Try to parse as JSON if it looks like JSON
            if reformulated_query.startswith("{") and reformulated_query.endswith("}"):
                try:
                    data = json.loads(reformulated_query)
                    if "reformulated_query" in data:
                        reformulated_query = data["reformulated_query"]
                except json.JSONDecodeError as json_e:
                    print(f"⚠️ Could not parse reformulator response as JSON: {json_e}")
            
            # Clean up any prefixes that might be added
            prefixes_to_remove = ["Reformulated:", "Reformulated Query:", "Query:", "Answer:", "Response:"]
            for prefix in prefixes_to_remove:
                if reformulated_query.startswith(prefix):
                    reformulated_query = reformulated_query.replace(prefix, "").strip()
            
            # Remove quotes if the entire response is wrapped in quotes
            if (reformulated_query.startswith('"') and reformulated_query.endswith('"')) or \
            (reformulated_query.startswith("'") and reformulated_query.endswith("'")):
                reformulated_query = reformulated_query[1:-1]
            
            # Validate the reformulated query is not empty
            if not reformulated_query or not reformulated_query.strip():
                print("⚠️ Reformulated query is empty, using original query")
                return user_query
                
            print(f"🔄 Query reformulated:")
            print(f"   Original: {user_query}")
            print(f"   Reformulated: {reformulated_query}")
            
            return reformulated_query
            
        except Exception as e:
            print(f"❌ Error reformulating query: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print(f"   Returning original query: {user_query}")
            return user_query
    
    def _extract_raw_data(self, response: str) -> str:
        """Extract the raw data from agent response - IMPROVED GENERIC VERSION"""
        if not response:
            return "No response generated"
        
        print(f"🔍 DEBUG: Extracting raw data from response length: {len(response)}")
        
        # Method 1: Look for Final Answer section first
        final_answer_match = None
        if "Final Answer:" in response:
            final_answer_part = response.split("Final Answer:")[-1].strip()
            print(f"🔍 DEBUG: Final Answer section found, length: {len(final_answer_part)}")
            
            # Clean up the final answer part by removing common agent artifacts
            cleaned_final = final_answer_part
            for artifact in ["```", "python", "```python", "Output:", "Result:"]:
                cleaned_final = cleaned_final.replace(artifact, "")
            
            cleaned_final = cleaned_final.strip()
            
            # If Final Answer contains meaningful data, prioritize it
            if cleaned_final and (
                # Check for tabular data patterns
                re.search(r'\s+\d+\s+\w+.*\n', cleaned_final) or
                # Check for course/academic data
                re.search(r'[A-Z]{2,}\d+|Course Number|Student Name|Grade|Term|GPA', cleaned_final, re.IGNORECASE) or
                # Check for structured data (multiple lines with consistent patterns)
                len([line for line in cleaned_final.split('\n') if line.strip()]) > 2 or
                # Check for pandas Series output (student names with GPA values)
                re.search(r'Name:\s+\w+|dtype:\s+float|^\w+.*\s+\d+\.\d+', cleaned_final, re.MULTILINE)
            ):
                print("🔍 DEBUG: Using Final Answer section - contains meaningful data")
                final_answer_match = cleaned_final
        
        # Method 2: Look for the last substantial data output in the response
        lines = response.split('\n')
        data_blocks = []
        current_block = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip obvious agent execution markers
            if any(marker in line_stripped for marker in [
                '> Entering', '> Finished', 'Let\'s get started!', 'Question:', 
                'Thought:', 'Action:', 'Action Input:', 'chain...', 'python_repl_ast',
                'is not a valid tool', 'Let\'s execute', 'NameError:', 'ValueError:', 'TypeError:'
            ]):
                # If we have a current block, save it
                if current_block:
                    data_blocks.append('\n'.join(current_block))
                    current_block = []
                continue
            
            # Look for data patterns
            is_data_line = (
                # Table headers or data rows
                re.search(r'Course Number|Student Name|Grade|Term|GPA', line_stripped, re.IGNORECASE) or
                # Indexed data (pandas output)
                re.search(r'^\s*\d+\s+[A-Z]', line_stripped) or
                # Course codes
                re.search(r'[A-Z]{2,}\d+', line_stripped) or
                # Pandas Series output (Name: student_name, value)
                re.search(r'^[A-Za-z\s]+\s+\d+\.\d+$', line_stripped) or
                # Series metadata
                re.search(r'Name:\s+\w+|dtype:\s+float', line_stripped) or
                # Data with consistent structure
                (line_stripped and not line_stripped.startswith(('Observation:', 'Final Answer:')))
            )
            
            if is_data_line and line_stripped:
                current_block.append(line)
            elif current_block:
                # End of current block
                data_blocks.append('\n'.join(current_block))
                current_block = []
        
        # Add the last block if it exists
        if current_block:
            data_blocks.append('\n'.join(current_block))
        
        # Method 3: Choose the best data block
        best_data = None
        
        # Prioritize Final Answer if it's substantial
        if final_answer_match and len(final_answer_match.split('\n')) >= 3:
            best_data = final_answer_match
            print("🔍 DEBUG: Using Final Answer as best data")
        
        # Otherwise, find the most substantial data block
        elif data_blocks:
            # Score each block based on data richness
            scored_blocks = []
            for block in data_blocks:
                score = 0
                lines_in_block = [line.strip() for line in block.split('\n') if line.strip()]
                
                # Score based on number of lines
                score += len(lines_in_block)
                
                # Score based on data patterns
                for line in lines_in_block:
                    if re.search(r'[A-Z]{2,}\d+', line):  # Course codes
                        score += 3
                    if re.search(r'Course Number|Student Name|Grade|Term', line, re.IGNORECASE):  # Headers
                        score += 2
                    if re.search(r'\d+\.\d+', line):  # Numbers (GPA, etc.)
                        score += 1
                    if re.search(r'^\s*\d+\s+', line.strip()):  # Indexed data
                        score += 1
                    if re.search(r'^[A-Za-z\s]+\s+\d+\.\d+$', line.strip()):  # Student GPA pairs
                        score += 4
                    if re.search(r'Name:\s+\w+|dtype:\s+float', line):  # Pandas Series
                        score += 2
                
                scored_blocks.append((score, block))
            
            # Get the highest scoring block
            if scored_blocks:
                scored_blocks.sort(key=lambda x: x[0], reverse=True)
                best_data = scored_blocks[0][1]
                print(f"🔍 DEBUG: Using highest scoring data block (score: {scored_blocks[0][0]})")
        
        # Method 4: Fallback to cleaned response
        if not best_data:
            # Clean the entire response
            clean_lines = []
            for line in lines:
                line_stripped = line.strip()
                if line_stripped and not any(marker in line_stripped for marker in [
                    '> Entering', '> Finished', 'Let\'s get started!', 'Question:', 
                    'Thought:', 'Action:', 'Action Input:', 'chain...', 'python_repl_ast',
                    'is not a valid tool', 'Let\'s execute', 'NameError:', 'ValueError:', 'TypeError:'
                ]):
                    clean_lines.append(line)
            
            if clean_lines:
                best_data = '\n'.join(clean_lines).strip()
                print("🔍 DEBUG: Using cleaned response as fallback")
        
        # Final fallback
        if not best_data:
            best_data = response.strip()
            print("🔍 DEBUG: Using original response as final fallback")
        
        print(f"🔍 DEBUG: Final extracted data length: {len(best_data)}")
        print(f"🔍 DEBUG: First 100 chars: {best_data[:100]}...")
        
        return best_data

    def _summarize_response(self, raw_response: str, original_question: str, format_type: str = "auto") -> str:
        """Use separate LLM to summarize and format the response with improved data interpretation"""
        
        print(f"🔄 DEBUG: Summarizing response. Raw response length: {len(raw_response)}")
        print(f"🔄 DEBUG: First 200 chars of raw response: {raw_response}...")
        
        # Check if the raw response indicates an error or timeout
        if "Agent stopped due to iteration limit or time limit" in raw_response:
            return "I encountered a timeout while processing your query. This usually means the data was found but the system took too long to format it. Please try rephrasing your question or contact support."
        
        prompt = f"""
            You are an expert data presentation assistant for academic transcript systems. You must interpret data accurately and present it clearly.

            Original Question: {original_question}
            
            Raw Data Response: {raw_response}

            **CRITICAL DATA INTERPRETATION RULES:**
            - Extract ALL numeric values EXACTLY as they appear in the raw data
            - Do NOT round, modify, or change any numbers unless explicitly showing the original value first
            - The raw data contains the ACTUAL answer - use those exact values

            **DATA PRESENTATION RULES:**
            - Present ALL data found in the raw response
            - Use clear, professional academic language

            **FORMATTING RULES - CHOOSE BEST FORMAT:**

            **Option 1: Simple List Format (Default for single-column data like advisors):**
            * For advisor lists, use this format:
            **Advisor(s) for Student 'Student Name'**
            - Advisor Name 1
            - Advisor Name 2
            - Advisor Name 3

            **Option 2: Table Format (Use for multi-column data like GPA, grades, course details):**
            * When data has multiple columns or comparative information, use HTML table format
            * For tables, use this EXACT HTML structure (NO MARKDOWN TABLES). Do NOT add extra line breaks or blank lines before or after the table:
            <table>
            <tr><th>Column Header 1</th><th>Column Header 2</th></tr>
            <tr><td>Actual Data 1</td><td>Actual Data 2</td></tr>
            <tr><td>Actual Data 3</td><td>Actual Data 4</td></tr>
            </table>

            **ABSOLUTE TABLE RULES - MUST FOLLOW:**
            * NEVER use markdown table format with pipes (|) and dashes (---)
            * ONLY use HTML table format with <table>, <tr>, <th>, <td> tags
            * FORBIDDEN: Any use of |---|, ---, or pipe separators
            * Every <tr> after the header must contain actual student names, GPA values, or real information
            * If you see raw data, immediately put that real data in <td> cells
            * REQUIRED: Start immediately with real data in table rows after the header row

            **FORMAT SELECTION GUIDE:**
            - Use simple list format for: advisor names, course lists, single-column data
            - Use table format for: GPA data, grade reports, multi-column comparisons, detailed course information

            - Be accurate about what the data shows
            - Round GPA values to 2 decimal places for display

            **CRITICAL FORMATTING REQUIREMENTS:**
            - Write a brief intro sentence followed immediately (same line or next line only) by the table with NO blank lines
            - Example format (this exact structure must be followed — no blank lines or line breaks between sentence and table):
            Here are the students sorted by GPA:<table>
            <tr><th>Student Name</th><th>GPA</th></tr>
            <tr><td>Example Student</td><td>3.50</td></tr>
            </table>
            - FORBIDDEN: Any blank line or whitespace between the colon and <table>
            - FORBIDDEN: Markdown bolding using ** for headers
            - FORBIDDEN: Any introductory phrasing like “I will present…” or “Here is…” followed by a blank line
            - Keep content compact, professional, and minimal with no extra spacing
            """
        if format_type == "auto":
            prompt

        else:
            prompt 

        try:
            # Use the summarizer LLM
            summary_response = self.summarizer_llm.invoke(prompt)
            try:
                summary_response = re.sub(r":\s*<table>", ":<table>", summary_response)
            except Exception as e:
                print(f"⚠️ Error cleaning response: {str(e)}")
            
            # Extract the content from the response
            if hasattr(summary_response, 'content'):
                result = summary_response.content
            else:
                result = str(summary_response)
                    
            print(f"✅ DEBUG: Summarization completed. Result length: {len(result)}")
            return result
                    
        except Exception as e:
            print(f"❌ Summarization failed: {str(e)}")
            # Return a formatted version of the raw response
            return self._manual_format_fallback(raw_response, original_question)

    def _manual_format_fallback(self, raw_response: str, original_question: str) -> str:
        """Enhanced manual formatting fallback when summarizer fails"""
        try:
            # Clean the raw response first
            lines = raw_response.split('\n')
            formatted_lines = []
            
            # Extract meaningful data lines
            for line in lines:
                line = line.strip()
                # Skip empty lines and technical markers
                if line and not any(marker in line for marker in [
                    'Action:', 'Thought:', 'Observation:', '> Entering', '> Finished',
                    'python_repl_ast', 'is not a valid tool', 'NameError:', 'ValueError:', 
                    'TypeError:', 'KeyError:', 'AttributeError:'
                ]):
                    formatted_lines.append(line)
            
            if not formatted_lines:
                return f"📋 **Query Results:**\n\n{raw_response}"
            
            # Try different formatting strategies based on content patterns
            
            # Strategy 1: GPA Analysis
            if self._is_gpa_data(formatted_lines):
                return self._format_gpa_data(formatted_lines)
            
            # Strategy 2: Academic Records (Course data)
            if self._is_academic_records(formatted_lines):
                return self._format_academic_records(formatted_lines)
            
            # Strategy 3: Tabular Data (general table format)
            if self._is_tabular_data(formatted_lines):
                return self._format_tabular_data(formatted_lines)
            
            # Strategy 4: List-based Data
            if self._is_list_data(formatted_lines):
                return self._format_list_data(formatted_lines)
            
            # Strategy 5: Statistical/Numerical Data
            if self._is_statistical_data(formatted_lines):
                return self._format_statistical_data(formatted_lines)
            
            # General fallback
            return self._format_general_data(formatted_lines)
            
        except Exception as e:
            print(f"❌ Manual formatting failed: {str(e)}")
            return f"Query results for: {original_question}\n\n{raw_response}"

    def _is_gpa_data(self, lines):
        """Check if data contains GPA information"""
        return any(re.search(r'GPA|\d+\.\d+$', line, re.IGNORECASE) for line in lines)

    def _format_gpa_data(self, lines):
        """Format GPA-related data"""
        gpa_lines = []
        for line in lines:
            if re.search(r'^[A-Za-z\s]+\s+\d+\.\d+$', line.strip()):
                gpa_lines.append(line.strip())
        
        if gpa_lines:
            result = f"📊 Student GPA Analysis ({len(gpa_lines)} records):<table>"
        result += "<tr><th>Student Name</th><th>Average GPA</th></tr>"

        for line in gpa_lines:
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                student_name = parts[0].strip()
                gpa_value = parts[1].strip()
                # round GPA to 2 decimals for display
                try:
                    gpa_value = f"{float(gpa_value):.2f}"
                except:
                    pass
                result += f"<tr><td>{student_name}</td><td>{gpa_value}</td></tr>"

        result += "</table>"
        return result

        
        return self._format_general_data(lines)

    def _is_academic_records(self, lines):
        """Check if data contains academic course information"""
        return any(re.search(r'Course Number|ART\d+|BM\d+|CD\d+|[A-Z]{2,4}\d+', line, re.IGNORECASE) for line in lines)

    def _format_academic_records(self, lines):
        """Format academic course records"""
        # Count actual data rows
        data_rows = [line for line in lines if re.search(r'^\s*\d+\s+[A-Z]', line)]
        count = len(data_rows)
        
        result = f"📚 **Academic Records Found ({count} records):**\n\n"
        
        # Check if it's tabular course data
        if any('Course Number' in line for line in lines):
            result += "| Course Number | Term |\n"
            result += "|---------------|------|\n"
            
            for line in lines:
                if re.search(r'^\s*\d+\s+([A-Z]+\d+)\s+(.+)', line):
                    match = re.search(r'^\s*\d+\s+([A-Z]+\d+)\s+(.+)', line)
                    if match:
                        course = match.group(1)
                        term = match.group(2)
                        result += f"| {course} | {term} |\n"
        else:
            result += '\n'.join(lines)
        
        # Add contextual notes
        if "Transfer Term" in ' '.join(lines):
            result += "\n\n*Note: Transfer Term indicates courses transferred from other institutions.*"
        
        return result

    def _is_tabular_data(self, lines):
        """Check if data appears to be in tabular format"""
        # Look for common table indicators
        header_indicators = ['Name', 'ID', 'Date', 'Score', 'Grade', 'Total', 'Count']
        pipe_separated = any('|' in line for line in lines)
        comma_separated = any(line.count(',') >= 2 for line in lines)
        has_headers = any(any(header in line for header in header_indicators) for line in lines)
        
        return pipe_separated or (comma_separated and has_headers)

    def _format_tabular_data(self, lines):
        """Format general tabular data as HTML table"""
        result = "📊 Data Table:<table>"
        
        # Try to detect delimiter
        if any('|' in line for line in lines):
            # Pipe-separated
            rows = [line.split('|') for line in lines if '|' in line]
            for i, row in enumerate(rows):
                if not row:
                    continue
                cells = [cell.strip() for cell in row if cell.strip()]
                if not cells:
                    continue
                if i == 0:  # header row
                    result += "<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>"
                else:
                    result += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        
        elif any(line.count(',') >= 1 for line in lines):
            # CSV-like format
            rows = [line.split(',') for line in lines if ',' in line]
            for i, row in enumerate(rows):
                cells = [c.strip() for c in row]
                if i == 0:
                    result += "<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>"
                else:
                    result += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        else:
            # fallback: plain rows
            for line in lines:
                result += f"<tr><td>{line}</td></tr>"
        
        result += "</table>"
        return result


    def _is_list_data(self, lines):
        """Check if data appears to be a list"""
        # Look for numbered lists, bullet points, or consistent patterns
        numbered = any(re.search(r'^\d+\.', line) for line in lines)
        bulleted = any(re.search(r'^[-*•]', line) for line in lines)
        consistent_format = len(set(len(line.split()) for line in lines if line)) <= 2
        
        return numbered or bulleted or (len(lines) > 3 and consistent_format)

    def _format_list_data(self, lines):
        """Format list-based data"""
        result = f"📋 **List Results ({len(lines)} items):**\n\n"
        
        # If not already formatted as a list, format it
        if not any(re.search(r'^[-*•\d+\.]', line) for line in lines):
            for i, line in enumerate(lines, 1):
                result += f"{i}. {line}\n"
        else:
            result += '\n'.join(lines)
        
        return result

    def _is_statistical_data(self, lines):
        """Check if data contains statistical information"""
        stat_keywords = ['average', 'mean', 'median', 'total', 'count', 'sum', 'min', 'max', 'std']
        return any(any(keyword in line.lower() for keyword in stat_keywords) for line in lines)

    def _format_statistical_data(self, lines):
        """Format statistical/numerical data"""
        result = "📈 **Statistical Analysis:**\n\n"
        
        # Group statistical lines
        stats = []
        data = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['average', 'mean', 'total', 'count', 'sum']):
                stats.append(line)
            else:
                data.append(line)
        
        if stats:
            result += "**Summary Statistics:**\n"
            for stat in stats:
                result += f"- {stat}\n"
            result += "\n"
        
        if data:
            result += "**Data:**\n"
            result += '\n'.join(data)
        
        return result

    def _format_general_data(self, lines):
        """General formatting for unstructured data"""
        return f"📋 **Query Results ({len(lines)} lines):**\n\n" + '\n'.join(lines)
        
    def _query_csv_agent(self, question: str, max_retries: int = 2, clean_logs: bool = True, use_summarizer: bool = True, format_type: str = "auto"):
        """Query the CSV agent with error handling and optional summarization"""
        if not self.agent:
            raise ValueError("CSV agent not initialized. Please call initialize() first.")

        # First, reformulate the query for better CSV agent performance
        reformulated_question = self._reformulate_query(question)
        
        # Use the reformulated question for the CSV agent
        final_question = reformulated_question

        for attempt in range(max_retries):
            try:
                print(f"🤔 Attempt {attempt + 1}: {final_question}")
                print("-" * 50)
                
                # Query the agent with the reformulated question
                response = self.agent.run(final_question)
                
                print("=" * 60)
                print(f"✅ Agent completed successfully")
                print(f"🔍 DEBUG: Raw agent response length: {len(response)}")
                
                if use_summarizer:
                    print("🔄 Formatting response with summarizer...")
                    # Extract raw data and summarize (use original question for context)
                    raw_data = self._extract_raw_data(response)
                    print(f"🔍 DEBUG: Extracted raw data length: {len(raw_data)}")
                    formatted_response = self._summarize_response(raw_data, question, format_type)
                    print("✅ Summarization completed")
                    return formatted_response
                else:
                    # Apply cleaning based on clean_logs parameter
                    if clean_logs:
                        response = self._extract_raw_data(response)
                    return response
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return f"I encountered an error while processing your transcript query. Please try rephrasing your question or contact support for assistance."
        
        return "Unable to process the query after multiple attempts."
    
    def _generate_multilingual_response(self, csv_response: str, user_query: str, language: str):
        """Generate response in the requested language"""
        if language == 'English':
            return csv_response
        
        # For non-English languages, we could translate the response
        # For now, we'll provide a basic multilingual wrapper
        language_headers = {
            "Spanish": "**Respuesta del Expediente Académico:**\n\n",
            "French": "**Réponse du Relevé de Notes:**\n\n",
            "Navajo": "**Óltaʼgi Bééhániih:**\n\n"
        }
        
        header = language_headers.get(language, "**Transcript Response:**\n\n")
        
        # Add a note about language if not English
        if language != 'English':
            language_notes = {
                "Spanish": "\n\n*Nota: Los datos se muestran en inglés por ser el idioma original de los registros.*",
                "French": "\n\n*Note: Les données sont affichées en anglais car c'est la langue originale des dossiers.*",
                "Navajo": "\n\n*Béhániih: Bilagáana bizaad ílį́ bééhózin áko bílaʼashdlaʼii bee.*"
            }
            note = language_notes.get(language, "")
            return header + csv_response + note
        
        return header + csv_response
    
    def process_query(self, user_query: str, language='English', use_summarizer: bool = True, format_type: str = "auto"):
        """
        Main function to process transcript queries using CSV agent with query reformulation
        
        Args:
            user_query (str): The user's question about transcripts
            language (str): Language for response
            use_summarizer (bool): Whether to use the summarizer for better formatting
            format_type (str): Format type - 'auto' or 'clean'
            
        Returns:
            str: Generated answer
        """
        print(f"🔎 Processing student transcript CSV query: '{user_query}'")
        print("=" * 80)
        
        if not self.is_initialized:
            print("❌ Student transcript CSV handler not initialized")
            init_success = self.initialize()
            if not init_success:
                error_messages = {
                    "English": "Student transcript system is not available. Please ensure the CSV file exists and is accessible.",
                    "Spanish": "El sistema de expedientes académicos no está disponible. Asegúrate de que el archivo CSV existe y está accesible.",
                    "French": "Le système de relevés de notes n'est pas disponible. Assurez-vous que le fichier CSV existe et est accessible.",
                    "Navajo": "Óltaʼgi bééhaniih éí doo áhólł̥ǫ́ǫ da."
                }
                return error_messages.get(language, error_messages["English"])
        
        try:
            # Query the CSV agent (with automatic query reformulation)
            csv_response = self._query_csv_agent(user_query, clean_logs=True, use_summarizer=use_summarizer, format_type=format_type)
            
            # Generate multilingual response
            final_response = self._generate_multilingual_response(csv_response, user_query, language)
            
            print("🧠 FINAL TRANSCRIPT CSV ANSWER:")
            print("=" * 80)
            print(final_response)
            print("=" * 80)
            
            return final_response
            
        except Exception as e:
            print(f"❌ Error processing transcript CSV query: {e}")
            import traceback
            traceback.print_exc()
            
            error_messages = {
                "English": "I encountered an error while processing your transcript query. Please try again or rephrase your question.",
                "Spanish": "Encontré un error al procesar tu consulta del expediente académico. Por favor, inténtalo de nuevo o reformula tu pregunta.",
                "French": "J'ai rencontré une erreur lors du traitement de votre requête de relevé de notes. Veuillez réessayer ou reformuler votre question.",
                "Navajo": "Bééhániih ályaa éí átʼé. Náábah ílį́ éí doodaii' saad naaltsoos."
            }
            return error_messages.get(language, error_messages["English"])


# Global CSV handler instance with caching
@st.cache_resource(show_spinner=False)
def get_csv_transcript_handler(csv_path=None):
    """Get cached CSV transcript handler instance"""
    if csv_path is None:
        csv_path = "data/csv_folder/student_transcript.csv"
    print(f"StudentTranscriptCSVHandler: Using CSV path: {csv_path}")
    return StudentTranscriptCSVHandler(csv_path=csv_path)


def process_transcript_query(user_query: str, language='English', use_summarizer: bool = True, format_type: str = "auto", csv_path=None):
    """
    Convenience function to process transcript queries using CSV agent with query reformulation
    
    Args:
        user_query (str): The user's question about transcripts
        language (str): Language for response
        use_summarizer (bool): Whether to use the summarizer for better formatting
        format_type (str): Format type - 'auto' or 'clean'
        
    Returns:
        str: Generated answer
    """
    print("DEBUG: process_transcript_query csv_path =", csv_path)
    handler = get_csv_transcript_handler(csv_path)
    answer = handler.process_query(user_query, language, use_summarizer, format_type)
    # Out-of-scope detection
    if not answer or answer.strip().lower() in [
        "no relevant data found", 
        "no answer found", 
        "i don't know", 
        "unable to answer", 
        "no data"
    ] or "no relevant" in answer.lower() or "not found" in answer.lower():
        return "Based on the document you uploaded I did not find the answer. Kindly upload the specific document."
    return answer