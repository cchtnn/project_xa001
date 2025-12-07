from docx2python import docx2python
import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")

def normalize_date_str(s: str) -> str:
    """Normalize date strings and fix common OCR issues."""
    s = s.strip()
    s = s.replace("\\", "/").replace("–", "-")
    s = re.sub(r"[^\d/]", "", s)

    # Already correct?
    if DATE_RE.match(s):
        return s

    # Fix cases like 7172026 → 7/17/2026
    m = re.fullmatch(r"(\d{1,2})(\d{1,2})(\d{4})", s)
    if m:
        return f"{int(m.group(1))}/{int(m.group(2))}/{m.group(3)}"

    # Fix cases like 7/172026 → 7/17/2026
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})(\d{4})", s)
    if m:
        return f"{int(m.group(1))}/{int(m.group(2))}/{m.group(3)}"

    return s


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%m/%d/%Y")


def extract_payroll_calendar(docx_path: str, expected_count: int = 27) -> pd.DataFrame:
    # 1) Read entire docx (including textboxes)
    doc = docx2python(docx_path)
    raw_text = doc.text
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    # Fix NBSP
    lines = [ln.replace("\u00A0", " ") for ln in lines]

    # 2) Extract Pay Period ranges
    pay_periods = []
    seen_pp = set()
    i = 0
    while i < len(lines) - 2:
        a = normalize_date_str(lines[i])
        b = lines[i + 1].strip().upper()
        c = normalize_date_str(lines[i + 2])

        if DATE_RE.match(a) and b == "TO" and DATE_RE.match(c):
            tup = (a, c)
            if tup not in seen_pp:
                pay_periods.append(tup)
                seen_pp.add(tup)
            i += 3
        else:
            i += 1

    pay_periods = pay_periods[:expected_count]

    # 3) Extract Check Dates by finding the "Check Date" block
    check_dates = []
    start_idx = None

    for idx, ln in enumerate(lines):
        lo = ln.lower()
        if lo.startswith("check date"):
            start_idx = idx + 1
            break
        if ln.strip().lower() == "check" and idx + 1 < len(lines) and lines[idx + 1].strip().lower().startswith("date"):
            start_idx = idx + 2
            break

    # Fallback: find first occurrence of "Check"
    if start_idx is None:
        for idx, ln in enumerate(lines):
            if "check" in ln.lower():
                start_idx = idx + 1
                break

    if start_idx is not None:
        j = start_idx
        while j < len(lines):
            cand = normalize_date_str(lines[j])
            if DATE_RE.match(cand):
                check_dates.append(cand)
                j += 1
                continue

            # Handle lines with multiple dates
            multiple = re.findall(r"\d{1,2}/\d{1,2}/\d{4}", lines[j])
            if multiple:
                check_dates.extend(multiple)
                j += 1
                continue

            # Stop when check-date block ends
            if check_dates:
                break

            j += 1

    # 4) Deduplicate & trim to expected count
    seen = set()
    unique_checks = []
    for d in check_dates:
        if d not in seen:
            seen.add(d)
            unique_checks.append(d)

    unique_checks = unique_checks[:len(pay_periods)]

    # 5) Standardize check dates
    final_checks = []
    for s in unique_checks:
        try:
            dt = parse_date(s)
            final_checks.append(f"{dt.month}/{dt.day}/{dt.year}")  # Windows-safe formatting
        except:
            final_checks.append(s)

    # 6) Build final DataFrame
    rows = []
    for idx, (start, end) in enumerate(pay_periods, start=1):
        chk = final_checks[idx - 1] if idx - 1 < len(final_checks) else ""
        rows.append({
            "Payroll No": idx,
            "Pay Period Start": start,
            "Pay Period End": end,
            "Check Date": chk
        })

    df = pd.DataFrame(rows)
    return df


class PayrollCSVAgent:
    """Enhanced Payroll retrieval system using LangChain CSV Agent"""
    
    def __init__(self, csv_path: str, model_name: str = "llama-3.1-8b-instant"):
        self.csv_path = csv_path
        self.model_name = model_name
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.llm = None
        self.summarizer_llm = None
        self.agent = None
        self.df = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the CSV agent with LLM"""
        try:
            print("📅 Initializing Payroll CSV Agent...")
            print(f"CSV Path: {self.csv_path}")
            
            if not os.path.exists(self.csv_path):
                print(f"❌ CSV file not found: {self.csv_path}")
                return False
            
            self._setup_llm()
            self._setup_summarizer()
            self._load_csv_and_create_agent()
            
            self.is_initialized = True
            print("✅ Payroll CSV Agent initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing payroll CSV agent: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_llm(self):
        """Setup the ChatGroq LLM"""
        try:
            if not self.groq_api_key:
                print("⚠️ GROQ_API_KEY not found")
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
            print("✅ LLM setup completed")
        except Exception as e:
            print(f"⚠️ LLM setup failed: {str(e)}")
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
    
    def _load_csv_and_create_agent(self):
        """Load CSV and create the agent"""
        try:
            self.df = pd.read_csv(self.csv_path)
            
            print(f"📊 CSV loaded successfully:")
            print(f"   Shape: {self.df.shape}")
            print(f"   Columns: {list(self.df.columns)}")
            
            # Create the CSV agent with improved configuration
            self.agent = create_csv_agent(
                llm=self.llm,
                path=self.csv_path,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                allow_dangerous_code=True,
                handle_parsing_errors=True,
                max_iterations=3,
                max_execution_time=90,
                return_intermediate_steps=True,
                include_df_in_prompt=True,  # Changed to True for better context
                prefix="""
You are working with a pandas DataFrame containing payroll calendar data.
The DataFrame is already loaded as 'df' and pandas is already imported as 'pd'.

Columns:
- payroll_no: Payroll period number (integer)
- start_date: Pay period start date (string format: M/D/YYYY)
- end_date: Pay period end date (string format: M/D/YYYY)  
- check_date: Check/payment date (string format: M/D/YYYY)
- optional_withholdings_changes_by: Deadline for withholding changes (string format: M/D/YYYY)

CRITICAL INSTRUCTIONS:
1. pandas is ALREADY imported as 'pd' - DO NOT import it again
2. df is ALREADY loaded - DO NOT load it again
3. Use ONLY the tool: python_repl_ast
4. ALWAYS use this exact format:
   Action: python_repl_ast
   Action Input: your_code_here

DATE HANDLING - REQUIRED PATTERN:
For date filtering, ALWAYS use this pattern:
```python
# Convert date columns to datetime
df['col_dt'] = pd.to_datetime(df['column_name'], format='%m/%d/%Y')
target_dt = pd.to_datetime('M/D/YYYY', format='%m/%d/%Y')
result = df[df['col_dt'] == target_dt][['payroll_no', 'start_date', 'end_date', 'check_date']]
print(result)
```

For date ranges:
```python
df['start_date_dt'] = pd.to_datetime(df['start_date'], format='%m/%d/%Y')
df['end_date_dt'] = pd.to_datetime(df['end_date'], format='%m/%d/%Y')
target = pd.to_datetime('M/D/YYYY', format='%m/%d/%Y')
result = df[(df['start_date_dt'] <= target) & (df['end_date_dt'] >= target)]
print(result)
```

RESPONSE RULES:
5. After getting pandas output, IMMEDIATELY provide Final Answer
6. Use print() to display the result DataFrame
7. Include ALL relevant columns in the result
8. DO NOT keep reformatting - provide Final Answer after first successful result

EXAMPLES:

Q: "Check date where optional withholdings changes is 2/27/2026"
Action: python_repl_ast
Action Input: df['opt_dt'] = pd.to_datetime(df['optional_withholdings_changes_by'], format='%m/%d/%Y'); target = pd.to_datetime('2/27/2026', format='%m/%d/%Y'); result = df[df['opt_dt'] == target][['payroll_no', 'check_date']]; print(result)
[Wait for observation]
Final Answer: [provide the data from observation]

Q: "Payroll period between 1/3/2026 to 1/16/2026"
Action: python_repl_ast
Action Input: df['start_dt'] = pd.to_datetime(df['start_date'], format='%m/%d/%Y'); df['end_dt'] = pd.to_datetime(df['end_date'], format='%m/%d/%Y'); target1 = pd.to_datetime('1/3/2026', format='%m/%d/%Y'); target2 = pd.to_datetime('1/16/2026', format='%m/%d/%Y'); result = df[(df['start_dt'] == target1) & (df['end_dt'] == target2)]; print(result)
[Wait for observation]
Final Answer: [provide the data]
"""
            )
            
            print("✅ CSV Agent created successfully")
            
        except Exception as e:
            raise Exception(f"Error loading CSV and creating agent: {str(e)}")
    
    def _extract_raw_data(self, response: str, intermediate_steps: list = None) -> str:
        """Extract the raw data from agent response - IMPROVED"""
        if not response:
            return "No response generated"
        
        print(f"🔍 Extracting raw data from response (length: {len(response)})...")
        
        # PRIORITY 1: Check intermediate steps for actual DataFrame output
        if intermediate_steps:
            print(f"🔍 Checking {len(intermediate_steps)} intermediate steps...")
            for i, (action, observation) in enumerate(reversed(intermediate_steps)):
                obs_str = str(observation).strip()
                
                # Look for DataFrame output patterns
                if any(pattern in obs_str for pattern in [
                    'payroll_no', 'start_date', 'end_date', 'check_date',
                    'Empty DataFrame', 'Series([]'
                ]):
                    # Check if it's an empty result
                    if 'Empty DataFrame' in obs_str or 'Series([])' in obs_str:
                        print(f"⚠️ Empty result found in step {len(intermediate_steps) - i}")
                        continue
                    
                    # Found actual data
                    print(f"✅ Found data in intermediate step {len(intermediate_steps) - i}")
                    # Clean up the observation
                    cleaned = obs_str
                    for remove_str in ['Observation:', 'Action:', 'Thought:']:
                        cleaned = cleaned.replace(remove_str, '')
                    return cleaned.strip()
        
        # PRIORITY 2: Look for Final Answer in response
        if "Final Answer:" in response:
            final_part = response.split("Final Answer:")[-1].strip()
            # Clean artifacts
            for artifact in ["```", "python", "Output:", "Result:"]:
                final_part = final_part.replace(artifact, "")
            final_part = final_part.strip()
            
            # Check if it contains meaningful data
            if final_part and len(final_part) > 10:
                if any(keyword in final_part for keyword in [
                    'payroll_no', 'check_date', 'start_date', '\n'
                ]):
                    print("✅ Using Final Answer section")
                    return final_part
        
        # PRIORITY 3: Look for Observation blocks
        lines = response.split('\n')
        in_observation = False
        observation_content = []
        
        for line in lines:
            if 'Observation:' in line:
                in_observation = True
                content = line.split('Observation:')[-1].strip()
                if content:
                    observation_content.append(content)
                continue
            
            if in_observation:
                if any(marker in line for marker in ['Thought:', 'Action:', '> Finished', '> Entering']):
                    in_observation = False
                    continue
                if line.strip():
                    observation_content.append(line.strip())
        
        if observation_content:
            obs_text = '\n'.join(observation_content)
            if 'payroll_no' in obs_text or 'check_date' in obs_text:
                print("✅ Using observation blocks")
                return obs_text
        
        # PRIORITY 4: Extract any DataFrame-like structure
        data_lines = []
        for line in lines:
            line = line.strip()
            # Skip agent markers
            if any(marker in line for marker in [
                'Action:', 'Thought:', '> Entering', '> Finished', 
                'AgentExecutor', 'python_repl_ast', 'Action Input:'
            ]):
                continue
            # Look for data patterns
            if re.search(r'\d+/\d+/\d+', line) or 'payroll_no' in line.lower():
                data_lines.append(line)
        
        if data_lines:
            result = '\n'.join(data_lines)
            print("✅ Using extracted data lines")
            return result
        
        print("⚠️ No structured data found in response")
        return "No data found"
    
    def _summarize_response(self, raw_response: str, original_question: str) -> str:
        """Use LLM to summarize and format the response - IMPROVED"""
        
        print(f"🔄 Summarizing response (raw length: {len(raw_response)})...")
        
        # Check for empty or error responses
        if not raw_response or raw_response == "No data found":
            return "No matching payroll data found for your query. Please check the date format (M/D/YYYY) and try again."
        
        if "Agent stopped due to iteration limit" in raw_response:
            return "Query timeout. Please try rephrasing your question or simplify the query."
        
        # Check if response indicates empty result
        if 'Empty DataFrame' in raw_response or len(raw_response.strip()) < 20:
            return "No matching payroll records found for the specified criteria."
        
        prompt = f"""
You are a payroll data presentation assistant. Format the data clearly and professionally.

Original Question: {original_question}

Raw Data: {raw_response}

**FORMATTING INSTRUCTIONS:**

1. **Single Record** (1 row): Present as readable text
   Example: "Payroll Period #4: 2/14/2026 to 2/27/2026, Check Date: 3/6/2026"

2. **Multiple Records** (2+ rows): Use HTML table
   Format:
   <table>
   <tr><th>Payroll No</th><th>Start Date</th><th>End Date</th><th>Check Date</th></tr>
   <tr><td>1</td><td>1/3/2026</td><td>1/16/2026</td><td>1/23/2026</td></tr>
   </table>

3. **Column Selection**: Include only columns mentioned in the question or all relevant columns

**CRITICAL RULES:**
- Extract dates exactly as shown (M/D/YYYY format)
- If raw data shows a DataFrame with index and columns, extract the actual values
- For DataFrame output like "   payroll_no  check_date\n4          4    3/6/2026", extract: Payroll #4, Check Date: 3/6/2026
- NO blank lines between text and <table> tag
- If data shows empty or no results, say "No matching records found"

Provide clean, formatted output:
"""
        
        try:
            summary_response = self.summarizer_llm.invoke(prompt)
            
            if hasattr(summary_response, 'content'):
                result = summary_response.content
            else:
                result = str(summary_response)
            
            # Clean formatting
            result = re.sub(r":\s+<table>", ":<table>", result)
            
            print(f"✅ Summarization completed")
            return result
                    
        except Exception as e:
            print(f"❌ Summarization failed: {str(e)}")
            return self._manual_format_fallback(raw_response)
    
    def _manual_format_fallback(self, raw_response: str) -> str:
        """Manual formatting fallback - IMPROVED"""
        try:
            # Parse DataFrame-like text output
            lines = [line.strip() for line in raw_response.split('\n') if line.strip()]
            
            # Look for DataFrame structure
            has_header = False
            header_line = None
            data_rows = []
            
            for i, line in enumerate(lines):
                # Check for column headers
                if 'payroll_no' in line.lower() and 'date' in line.lower():
                    has_header = True
                    header_line = line
                    continue
                
                # Extract data rows (lines with numbers and dates)
                if re.search(r'\d+\s+\d+/\d+/\d+', line):
                    data_rows.append(line)
            
            if data_rows:
                formatted = "📅 Payroll Calendar Results:\n\n"
                for row in data_rows:
                    formatted += row + "\n"
                return formatted
            
            # If DataFrame structure found but couldn't parse, return cleaned version
            if 'payroll_no' in raw_response or 'check_date' in raw_response:
                return f"📅 Payroll Data:\n\n{raw_response}"
            
            return "No payroll data found in the response."
            
        except Exception as e:
            print(f"❌ Manual formatting failed: {str(e)}")
            return raw_response
    
    def query(self, question: str, max_retries: int = 2) -> str:
        """Query the CSV agent with a payroll-related question"""
        if not self.agent:
            raise ValueError("CSV agent not initialized. Call initialize() first.")
        
        for attempt in range(max_retries):
            try:
                print(f"🤔 Attempt {attempt + 1}: {question}")
                print("-" * 50)
                
                result = self.agent(question)
                
                response = result.get("output", "")
                intermediate_steps = result.get("intermediate_steps", [])
                
                print("=" * 60)
                print(f"✅ Agent completed")
                print(f"🔍 Response length: {len(response)}")
                print(f"🔍 Intermediate steps: {len(intermediate_steps)}")
                
                # Extract raw data (pass intermediate steps)
                raw_data = self._extract_raw_data(response, intermediate_steps)
                print(f"🔍 Extracted raw data: {raw_data[:200]}...")
                
                # Format and summarize
                formatted_response = self._summarize_response(raw_data, question)
                print("✅ Query completed successfully")
                return formatted_response
                
            except Exception as e:
                error_msg = str(e)
                print(f"⚠️ Attempt {attempt + 1} error: {error_msg[:200]}")
                
                # Try to extract from parsing errors
                if "Final Answer:" in error_msg:
                    try:
                        result_part = error_msg.split("Final Answer:")[-1].strip()
                        # Clean up
                        for marker in ['For troubleshooting', 'visit', 'https']:
                            if marker in result_part:
                                result_part = result_part.split(marker)[0]
                        
                        lines = [line.strip() for line in result_part.split('\n') if line.strip()]
                        
                        if lines and any('payroll' in line.lower() or re.search(r'\d+/\d+/\d+', line) for line in lines):
                            extracted_result = '\n'.join(lines)
                            print(f"✅ Extracted from parsing error")
                            formatted_response = self._summarize_response(extracted_result, question)
                            return formatted_response
                    except Exception as extract_error:
                        print(f"❌ Failed to extract: {extract_error}")
                
                if attempt == max_retries - 1:
                    return "Unable to process your payroll query. Please try rephrasing the question or check the date format (M/D/YYYY)."
        
        return "Query processing failed after multiple attempts."


# # Main usage
# if __name__ == "__main__":
#     input_docx = "data\\payroll_cal\\2026Payroll Calendar.docx"
#     df = extract_payroll_calendar(input_docx, expected_count=27)
#     df.columns = ['payroll_no', 'start_date', 'end_date', 'check_date']
#     print("column names:", df.columns.tolist())
#     print("Shape of the dataframe :- ", df.shape)
#     # Add optional withholdings column
#     df['optional_withholdings_changes_by'] = df['end_date']
    
    
#     print("Extracted Payroll Calendar:")
#     print(df.head(10).to_string(index=False))
#     print("\n" + "="*80 + "\n")
    
#     # Save to CSV
#     csv_output_path = "data\\payroll_cal\\payroll_2026.csv"
#     df.to_csv(csv_output_path, index=False)
#     print(f"✅ Saved to: {csv_output_path}\n")
    
#     # Step 2: Initialize CSV Agent
#     agent = PayrollCSVAgent(csv_path=csv_output_path)
    
#     if agent.initialize():
#         print("\n" + "="*80)
#         print("Testing Payroll CSV Agent:")
#         print("="*80 + "\n")
        
#         # Test queries
#         test_queries = [
#             "Tell me the check date where optional withholdings changes is 2/27/2026?",
#             "When is the check date for payroll period between 01/03/2026 to 01/16/2026?",
#             "What is the payroll period where check date is 2/6/2026?",
#         ]
        
#         for query in test_queries:
#             print(f"\n{'='*80}")
#             print(f"Query: {query}")
#             print('='*80)
            
#             answer = agent.query(query)
#             print(f"\n📋 Answer:\n{answer}\n")
#     else:
#         print("❌ Failed to initialize agent")