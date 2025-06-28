"""
Student Transcript CSV Handler for Diné College Assistant
Handles student transcript queries using CSV Agent with LangChain
Replaces the FAISS-based approach with direct CSV querying
Now integrates with StudentQueryReformulator for better query processing
"""

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

warnings.filterwarnings("ignore")
load_dotenv()


class StudentTranscriptCSVHandler:
    """Handles student transcript queries using CSV Agent with query reformulation"""
    
    def __init__(self, csv_path="data/csv_folder/student_transcript.csv", model_name="llama3-8b-8192"):
        self.csv_path = csv_path
        self.model_name = model_name
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.llm = None
        self.summarizer_llm = None
        self.agent = None
        self.df = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.query_reformulator = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the CSV handler with LLM, agent, and query reformulator"""
        try:
            print("📚 Initializing Student Transcript CSV Handler...")
            
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
    
    def _setup_llm(self):
        """Setup the ChatGroq LLM"""
        try:
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name=self.model_name,
                temperature=0,
                max_tokens=4096,
                streaming=False,
                request_timeout=60
            )
            print("✅ LLM setup completed")
        except Exception as e:
            raise Exception(f"Error initializing LLM: {str(e)}")
    
    def _setup_summarizer(self):
        """Setup a separate LLM for summarization"""
        try:
            self.summarizer_llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name=self.model_name,
                temperature=0.3,  # Slightly higher temperature for more natural summaries
                max_tokens=2048,
                streaming=False,
                request_timeout=30
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
                self.df['GPA'] = pd.to_numeric(self.df['GPA'], errors='coerce')
            
            # Get column information for debugging
            column_info = self._get_column_info()
            print(f"📊 CSV loaded successfully:")
            print(f"   Shape: {self.df.shape}")
            print(f"   Columns: {list(self.df.columns)}")
            print(f"   Column Info: {column_info}")
            
            # Create the CSV agent
            self.agent = create_csv_agent(
                llm=self.llm,
                path=self.csv_path,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                allow_dangerous_code=True,
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=120,
                return_intermediate_steps=False,
                include_df_in_prompt=False
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
    
    def _reformulate_query(self, user_query: str) -> str:
        """
        Reformulate user query using the StudentQueryReformulator
        
        Args:
            user_query (str): Original user query
            
        Returns:
            str: Reformulated query or original query if reformulation fails
        """
        if not self.query_reformulator:
            print("⚠️ Query reformulator not available, using original query")
            return user_query
        
        try:
            # Process the query with full pipeline (validation + reformulation)
            result = self.query_reformulator.process_student_query(user_query, self.csv_path)
            
            if result["success"]:
                reformulated_query = result["reformulated_query"]
                
                # Log validation information if available
                if result.get("validation"):
                    validation = result["validation"]
                    print(f"📋 Query validation:")
                    print(f"   Can answer: {validation.get('can_answer', 'unknown')}")
                    print(f"   Confidence: {validation.get('confidence', 'unknown')}")
                    
                    # If validation suggests the query might not be answerable
                    if validation.get("confidence") == "low" and not validation.get("can_answer", True):
                        print(f"⚠️ Low confidence in query answerability")
                        if validation.get("suggestions"):
                            print(f"   Suggestions: {validation['suggestions']}")
                
                return reformulated_query
            else:
                print(f"⚠️ Query reformulation failed: {result.get('error', 'Unknown error')}")
                if result.get("suggestions"):
                    print(f"   Suggestions: {result['suggestions']}")
                return user_query
                
        except Exception as e:
            print(f"⚠️ Error during query reformulation: {str(e)}")
            return user_query
    
    def _clean_response(self, response: str) -> str:
        """Clean the response to extract only the final answer"""
        if not response:
            return "No response generated"
        
        # Find the actual data result by looking for patterns
        lines = response.split('\n')
        result_lines = []
        data_started = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip all agent execution patterns
            skip_patterns = [
                '> Entering new AgentExecutor chain',
                '> Finished chain',
                'Thought:',
                'Action:',
                'Action Input:',
                'Observation:',
                'TypeError:',
                'NameError:',
                'Here\'s the',
                'The error message',
                'I need to',
                'Now that',
                'Final Answer:',
                'The final answer is',
                'Note:',
                'The result is',
                'which is:',
                'pandas Series',
                'student names as',
                'average gpa as',
                'dtype=',
                'Name: GPA,'
            ]
            
            # Skip lines that match agent patterns
            if any(pattern in line for pattern in skip_patterns):
                continue
                
            # Look for the actual data pattern
            if 'Student Name' in line and not any(skip in line for skip in skip_patterns):
                data_started = True
                continue
            
            # If we've found the data section, collect lines that look like results
            if data_started:
                # Check if line contains student name and GPA value
                if any(char.isdigit() for char in line) and any(char.isalpha() for char in line):
                    result_lines.append(line)
        
        # If we found data lines, return them
        if result_lines:
            return '\n'.join(result_lines)
        
        # Fallback: try to extract from Final Answer section more aggressively
        if "Final Answer:" in response:
            final_part = response.split("Final Answer:")[-1]
            
            # Look for lines that contain both letters and numbers (likely data)
            data_lines = []
            for line in final_part.split('\n'):
                line = line.strip()
                if (line and 
                    any(char.isdigit() for char in line) and 
                    any(char.isalpha() for char in line) and
                    not any(skip in line.lower() for skip in ['note:', 'result is', 'pandas', 'dtype'])):
                    data_lines.append(line)
            
            if data_lines:
                return '\n'.join(data_lines)
        
        # If nothing found, return a cleaned version of the original response
        return self._basic_clean_response(response)
    
    def _basic_clean_response(self, response: str) -> str:
        """Basic cleaning of response if advanced cleaning fails"""
        # Remove agent execution traces
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not any(pattern in line for pattern in [
                '> Entering', '> Finished', 'Thought:', 'Action:', 'Observation:'
            ]):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines) if cleaned_lines else response
    
    def _extract_raw_data(self, response: str) -> str:
        """Extract the raw data from agent response"""
        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip()
        return response

    def _summarize_response(self, raw_response: str, original_question: str, format_type: str = "auto") -> str:
        """Use separate LLM to summarize and format the response with intelligent format selection"""
        
        if format_type == "auto":
            prompt = f"""
            You are an intelligent data presentation expert. I have a question and raw data response that needs to be formatted in the most appropriate way for the end user.

            Original Question: {original_question}
            
            Raw Data Response: {raw_response}

            Your task is to analyze the data and automatically choose the BEST presentation format based on the content. Follow these guidelines:

            DECISION CRITERIA:
            1. **Use TABLE format when:**
               - Data contains structured information (like course numbers, student names, GPAs, dates)
               - Data has clear columns/rows that can be organized
               - Data involves comparisons between multiple items
               - Data contains numerical values that need to be compared
               - Question asks for specific records or listings

            2. **Use STORY format when:**
               - Data represents trends, patterns, or insights
               - Question asks for analysis, summary, or explanation
               - Data needs context or interpretation
               - Result is a single value or simple answer
               - Data involves calculations or aggregations that need explanation

            3. **Use BULLET POINT format when:**
               - Data is a simple list without complex structure
               - Multiple unrelated items need to be presented
               - Quick facts or key points need highlighting

            FORMATTING RULES:
            - Remove ALL technical jargon, pandas terms, dtype references
            - Use clear, professional language
            - Add appropriate emojis (📊 for tables, 📖 for stories, 📝 for lists)
            - Include brief explanations where helpful
            - Make it conversational but professional
            - If using table format, use proper markdown table syntax
            - If using story format, create engaging narrative with clear structure

            ANALYZE the data first, then CHOOSE the best format automatically, and PRESENT the data accordingly. Do not ask which format to use - just pick the best one and execute it.

            Provide only the final formatted response, nothing else.
            """
        
        else:  # clean format
            prompt = f"""
            You are a data formatter. I have a question and raw data response that needs to be cleaned up and made user-friendly.

            Original Question: {original_question}
            
            Raw Data Response: {raw_response}

            Please clean up this response by:
            1. Removing technical terms and agent execution details
            2. Presenting the data clearly and concisely
            3. Using proper formatting (bullet points, headers, etc.)
            4. Making it easy to understand for end users
            5. Keeping only the essential information

            Provide only the cleaned, formatted response, nothing else.
            """

        try:
            # Use the summarizer LLM
            summary_response = self.summarizer_llm.invoke(prompt)
            
            # Extract the content from the response
            if hasattr(summary_response, 'content'):
                return summary_response.content
            else:
                return str(summary_response)
                
        except Exception as e:
            print(f"❌ Summarization failed: {str(e)}")
            return f"Summarization failed. Raw response: {raw_response}"
    
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
                
                if use_summarizer:
                    print("🔄 Formatting response with summarizer...")
                    # Extract raw data and summarize (use original question for context)
                    raw_data = self._extract_raw_data(response)
                    formatted_response = self._summarize_response(raw_data, question, format_type)
                    print("✅ Summarization completed")
                    return formatted_response
                else:
                    # Apply cleaning based on clean_logs parameter
                    if clean_logs:
                        response = self._clean_response(response)
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
                    "Spanish": "El sistema de expedientes académicos no está disponible. Asegúrate de que el archivo CSV existe y es accesible.",
                    "French": "Le système de relevés de notes n'est pas disponible. Assurez-vous que le fichier CSV existe et est accessible.",
                    "Navajo": "Óltaʼgi bééhániih éí doo áhólł̥ǫ́ǫ da."
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
def get_csv_transcript_handler():
    """Get cached CSV transcript handler instance"""
    return StudentTranscriptCSVHandler()


def process_transcript_query(user_query: str, language='English', use_summarizer: bool = True, format_type: str = "auto"):
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
    handler = get_csv_transcript_handler()
    return handler.process_query(user_query, language, use_summarizer, format_type)