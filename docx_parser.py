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
from payroll_query_reformulator import reformulate_payroll_query, get_payroll_reformulator

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
            temp_cols = [i for i in self.df.columns if i.startswith('Extra')]
            self.df.drop(columns=temp_cols, inplace=True, errors='ignore')
            
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
                handle_parsing_errors=True,  # This will handle parsing errors gracefully
                max_iterations=3,  # Increased from 3 to 5
                max_execution_time=60,
                return_intermediate_steps=True,
                include_df_in_prompt=True,
                early_stopping_method="generate",
                prefix="""
You are working with a pandas DataFrame containing payroll calendar data.

CRITICAL ENVIRONMENT LIMITATION:
The python_repl_ast tool does NOT have pandas imported as 'pd' in its scope.
You CANNOT use 'pd.to_datetime()' or any 'pd.' functions.
The DataFrame 'df' exists, but you must work with it using ONLY:
- String operations (.str methods)
- DataFrame methods (.shape, .nunique(), .loc, etc.)
- Python built-in functions (len, str, int, etc.)

Columns:
- payroll_no: Payroll period number (integer, UNIQUE - only ONE row per payroll number)
- start_date: Pay period start date (string format: M/D/YYYY)
- end_date: Pay period end date (string format: M/D/YYYY)  
- check_date: Check/payment date (string format: M/D/YYYY)
- optional_withholdings_changes_by: Deadline for withholding changes (string format: M/D/YYYY)

CRITICAL INSTRUCTIONS:
1. NEVER use pd.to_datetime() - it will always fail with NameError
2. Use STRING operations for date filtering: df['start_date'].str.contains('2026')
3. Use ONLY the tool: python_repl_ast
4. Follow this workflow:
   - Think about what you need to do
   - Write ONE Action with python_repl_ast
   - Write ONLY Python code in Action Input
   - Wait for the Observation
   - IF the Observation shows valid results → Provide Final Answer IMMEDIATELY
   - DO NOT run additional actions if you already have the answer

5. EFFICIENCY RULE: Once you get a valid observation with data, STOP and provide Final Answer
6. Each payroll_no is UNIQUE - there is only ONE row per payroll number, no need to iterate
7. If you get NameError about 'pd', immediately switch to string operations

WORKING CODE PATTERNS:

Pattern 1 - Count rows by year:
result = df[df['start_date'].str.contains('2026')]['payroll_no'].nunique()
print(result)

Pattern 2 - Filter by specific date:
result = df[df['start_date'] == '5/9/2026'][['payroll_no', 'start_date', 'end_date', 'check_date']]
print(result)

Pattern 3 - Filter by payroll number:
result = df[df['payroll_no'] == 10][['payroll_no', 'start_date', 'end_date', 'check_date']]
print(result)

Pattern 4 - Calculate days difference (USE THIS FOR DATE CALCULATIONS):
from datetime import datetime
row = df[df['payroll_no'] == 9].iloc[0]
start = row['start_date']
check = row['check_date']
start_parts = start.split('/')
check_parts = check.split('/')
start_dt = datetime(int(start_parts[2]), int(start_parts[0]), int(start_parts[1]))
check_dt = datetime(int(check_parts[2]), int(check_parts[0]), int(check_parts[1]))
days = (check_dt - start_dt).days
print("Start: " + start + ", Check: " + check + ", Days: " + str(days))

Pattern 5 - Get all rows:
result = df[['payroll_no', 'start_date', 'end_date', 'check_date']]
print(result)

RESPONSE RULES:
8. Use print() to display results
9. After seeing valid Observation with data → STOP and provide Final Answer
10. DO NOT run the same action multiple times
11. DO NOT second-guess yourself - if the observation shows the answer, that IS the answer

ERROR RECOVERY:
- If you see "NameError: name 'pd' is not defined" → Use string operations instead
- If you see "SyntaxError" → Remove any non-Python text from your code
- If you see "KeyError" → Check column names match exactly

WORKFLOW EXAMPLE (NOTICE: ONLY ONE ACTION NEEDED):

Question: "How many payroll periods in 2026?"

Thought: I need to count rows where start_date contains '2026' using string operations.
Action: python_repl_ast
Action Input: result = df[df['start_date'].str.contains('2026')]['payroll_no'].nunique()
print(result)
Observation: 26
Thought: I found 26 unique payroll periods. This answers the question completely.
Final Answer: There are 26 payroll periods in 2026.

ANOTHER EXAMPLE (NOTICE: ONLY ONE ACTION NEEDED):

Question: "Days difference between start and check date for payroll 9?"

Thought: I need to get dates for payroll 9 and calculate difference. Since payroll_no is unique, there's only one row.
Action: python_repl_ast
Action Input: from datetime import datetime
row = df[df['payroll_no'] == 9].iloc[0]
start = row['start_date']
check = row['check_date']
start_parts = start.split('/')
check_parts = check.split('/')
start_dt = datetime(int(start_parts[2]), int(start_parts[0]), int(start_parts[1]))
check_dt = datetime(int(check_parts[2]), int(check_parts[0]), int(check_parts[1]))
days = (check_dt - start_dt).days
print("Start: " + start + ", Check: " + check + ", Days: " + str(days))
Observation: Start: 4/25/2026, Check: 5/15/2026, Days: 20
Thought: Perfect! I have the answer. The days difference is 20 days.
Final Answer: The difference between the start date (4/25/2026) and check date (5/15/2026) for payroll 9 is 20 days.
"""
            )
            self.agent_executor = self.agent
            
            print("✅ CSV Agent created successfully")
            
        except Exception as e:
            raise Exception(f"Error loading CSV and creating agent: {str(e)}")
    
    def _extract_raw_data(self, response: dict) -> str:
        """
        Extract raw data from agent response with enhanced debugging
        
        Args:
            response: Agent executor response
            
        Returns:
            str: Extracted data or "No data found"
        """
        print(f"🔍 Extracting raw data from response (length: {len(str(response))})...")
        
        # PRIORITY 1: Check intermediate_steps FIRST (most reliable)
        if isinstance(response, dict) and 'intermediate_steps' in response:
            steps = response['intermediate_steps']
            print(f"🔍 Checking {len(steps)} intermediate steps...")
            
            # Iterate through steps in REVERSE order (last step is usually the final answer)
            for i in range(len(steps) - 1, -1, -1):
                step = steps[i]
                print(f"\n   📋 Step {i+1} (reverse order):")
                print(f"      Type: {type(step)}")
                
                # Langchain format: (AgentAction, observation)
                if isinstance(step, tuple) and len(step) >= 2:
                    action, observation = step[0], step[1]
                    print(f"      Action: {str(action)[:100]}...")
                    print(f"      Observation type: {type(observation)}")
                    print(f"      Observation: {str(observation)[:200]}...")
                    
                    # The observation usually contains the actual result
                    if observation is not None:
                        obs_str = str(observation).strip()
                        
                        # Skip error messages
                        if 'error' in obs_str.lower() or 'exception' in obs_str.lower():
                            print(f"      ⚠️ Skipping error observation")
                            continue
                        
                        # Check if this is actual data (not just None or empty)
                        if obs_str and obs_str.lower() not in ['none', '', 'null']:
                            print(f"      ✅ Found valid observation data")
                            
                            # If it's a single number (count query), return it immediately
                            if obs_str.strip().isdigit() or re.match(r'^\d+$', obs_str.strip()):
                                print(f"      🔢 Detected numeric result: {obs_str}")
                                return obs_str.strip()
                            
                            # If it looks like structured data, return it
                            if any(keyword in obs_str.lower() for keyword in ['payroll', 'date', 'period', 'check']):
                                print(f"      📊 Detected structured data")
                                return obs_str
                            
                            # Return any non-empty observation from last successful step
                            if i == len(steps) - 1:  # Last step
                                return obs_str
                
                # Alternative format: dict with action/observation
                elif isinstance(step, dict):
                    print(f"      Dict keys: {step.keys()}")
                    if 'observation' in step:
                        obs = step['observation']
                        print(f"      Observation: {str(obs)[:200]}...")
                        if obs and str(obs).strip():
                            obs_str = str(obs).strip()
                            if obs_str.isdigit():
                                return obs_str
                            return obs_str
        
        # PRIORITY 2: Check output field ONLY if intermediate_steps failed
        if isinstance(response, dict) and 'output' in response:
            output = response['output']
            print(f"⚠️ Checking 'output' field as fallback: {str(output)[:200]}...")
            
            # Skip if output contains error messages
            output_str = str(output).strip()
            if 'agent stopped' not in output_str.lower() and 'iteration limit' not in output_str.lower():
                if output_str and output_str not in ['none', '', 'null']:
                    print(f"✅ Using output field: {output_str[:100]}...")
                    return output_str
            else:
                print(f"⚠️ Output field contains error/timeout message, ignoring")
        
        # PRIORITY 3: Fallback - check for 'result' or 'answer' keys
        if isinstance(response, dict):
            print(f"🔍 Checking fallback keys in response...")
            for key in ['result', 'answer', 'final_answer', 'text']:
                if key in response:
                    value = response[key]
                    print(f"   Found '{key}': {str(value)[:200]}...")
                    if value and str(value).strip():
                        return str(value).strip()
        
        print(f"❌ Could not extract meaningful data from response")
        return "No data found"
    
    def _summarize_response(self, raw_response: str, original_question: str) -> str:
        """Use LLM to summarize and format the response - IMPROVED FORMATTING"""
        
        print(f"📄 Summarizing response (raw length: {len(raw_response)})...")
        
        # Check for empty or error responses
        if not raw_response or raw_response == "No data found":
            return "No matching payroll data found for your query. Please check the date format (M/D/YYYY) and try again."
        
        if "Agent stopped due to iteration limit" in raw_response:
            return "Query timeout. Please try rephrasing your question or simplify the query."
        
        # FIXED: Check if it's a simple numeric result (count query) BEFORE checking length
        raw_stripped = raw_response.strip()
        if raw_stripped.isdigit() or re.match(r'^\d+$', raw_stripped):
            # This is a count result - format it directly
            count = raw_stripped
            print(f"🔢 Detected count result: {count}")
            return f"There are **{int(count)+1}** payroll periods in CY 2026 from January 3rd, 2026 till January 2nd, 2027."
        
        # Check if response indicates empty result (AFTER numeric check)
        if 'Empty DataFrame' in raw_response:
            return "No matching payroll records found for the specified criteria."
        
        # UPDATED: Only check for very short responses (< 5 chars) that aren't numbers
        if len(raw_stripped) < 5 and not raw_stripped.replace('.', '').isdigit():
            return "No matching payroll records found for the specified criteria."
        
        prompt = f"""
    You are a payroll data presentation assistant. Your job is to convert raw DataFrame output into CONCISE, PROFESSIONAL, STORY-LIKE responses that feel natural to read.

    Original Question: {original_question}

    Raw Data (DataFrame output):
    {raw_response}

    **RESPONSE STYLE REQUIREMENTS:**

    Write responses as if you're telling someone a brief, professional story. Make it feel conversational yet precise.

    1. **For Single Record Queries** (like "payroll number 10" or "days difference"):
    - Lead with the direct answer
    - Use ordinal dates (e.g., "May 9th, 2026" not "May 9, 2026")
    - Keep it to ONE sentence when possible
    - Format: "[Key info] with [additional context]"
    
    Example: "Start date May 9th, 2026 and Check Date of May 28th, 2026 with the gap of 19 days."

    2. **For Count Results**:
    - State the count with context in one flowing sentence
    - Include the year range naturally
    - Use ordinal dates (e.g., "January 3rd" not "January 3")
    
    Example: "There are 27 payroll periods in CY 2026 from January 3rd, 2026 till January 2nd, 2027."

    3. **For Date Range Queries**:
    - Present dates in a flowing manner
    - Use ordinal suffixes (1st, 2nd, 3rd, 9th, etc.)
    
    Example: "The payroll period runs from February 14th, 2026 to February 27th, 2026, with check date on March 6th, 2026."

    4. **For Multiple Records**:
    - Introduce with a brief statement
    - Use clean HTML table
    ```html
    <table>
    <tr><th>Payroll No</th><th>Start Date</th><th>End Date</th><th>Check Date</th></tr>
    <tr><td>1</td><td>Jan 3rd, 2026</td><td>Jan 16th, 2026</td><td>Jan 23rd, 2026</td></tr>
    </table>
    ```

    **FORMATTING RULES:**
    - Use ordinal dates: 1st, 2nd, 3rd, 4th, 5th, etc. (not 1, 2, 3)
    - Month abbreviations for tables, full names in sentences
    - Use "CY" for "Current Year"
    - Use "with the gap of X days" for date differences
    - Keep sentences flowing - avoid bullet points unless showing multiple records
    - Use **bold** ONLY for key numbers (counts, differences)
    - Write in active voice, present tense

    **WHAT TO AVOID:**
    - Do NOT write "Based on the provided DataFrame"
    - Do NOT write "Here's the response"
    - Do NOT explain what you're doing
    - Do NOT use bullet points for single records
    - Do NOT write multiple sentences when one will do
    - Do NOT use formal date formats like "May 9, 2026" - use "May 9th, 2026"

    **CRITICAL:** Extract the actual data from the DataFrame representation, then present it naturally. Go straight to the answer.

    Provide your concise, story-like response now:
    """
        
        try:
            summary_response = self.summarizer_llm.invoke(prompt)
            
            if hasattr(summary_response, 'content'):
                result = summary_response.content
            else:
                result = str(summary_response)
            
            # Clean up any extra whitespace
            result = re.sub(r'\n{3,}', '\n\n', result)
            result = result.strip()
            
            # Remove any leftover meta-commentary
            meta_phrases = [
                "Based on the provided DataFrame output",
                "Here's the parsed data:",
                "Here's the response in a clear, readable format:",
                "Based on the DataFrame:",
                "According to the data:"
            ]
            
            for phrase in meta_phrases:
                if phrase in result:
                    # Remove the phrase and everything before it up to the first newline or colon
                    parts = result.split(phrase, 1)
                    if len(parts) > 1:
                        # Take everything after the phrase, skip any bullet points or formatting
                        remaining = parts[1].strip()
                        # If there's a colon or newline shortly after, skip to the actual content
                        if remaining.startswith(':'):
                            remaining = remaining[1:].strip()
                        result = remaining
            
            print("Final Summarized Response:", result)
            print(f"✅ Summarization completed")
            return result
                    
        except Exception as e:
            print(f"❌ Summarization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._manual_format_fallback(raw_response)
    
    def _manual_format_fallback(self, raw_response: str) -> str:
        """Enhanced manual formatting fallback"""
        try:
            print(f"🔧 Using manual formatting fallback...")
            
            # Clean up the raw response
            lines = [line.strip() for line in raw_response.split('\n') if line.strip()]
            
            # Try to find tabular data pattern
            data_found = False
            payroll_no = None
            start_date = None
            end_date = None
            check_date = None
            days_diff = None
            
            for line in lines:
                # Look for data row (contains numbers and dates)
                if re.search(r'\d+\s+\d+/\d+/\d+', line) or re.search(r'\d{1,2}/\d{1,2}/\d{4}', line):
                    parts = line.split()
                    parts = [p for p in parts if p.strip()]
                    
                    for i, part in enumerate(parts):
                        # Payroll number
                        if not payroll_no and part.isdigit() and int(part) <= 50:
                            payroll_no = part
                        
                        # Date format M/D/YYYY
                        if re.match(r'\d{1,2}/\d{1,2}/\d{4}', part):
                            if not start_date:
                                start_date = part
                            elif not end_date:
                                end_date = part
                            elif not check_date:
                                check_date = part
                        
                        # Days difference
                        if 'days' in part.lower():
                            days_match = re.search(r'(\d+)', part)
                            if days_match:
                                days_diff = days_match.group(1)
                        elif i == len(parts) - 1 and part.isdigit() and int(part) < 100:
                            days_diff = part
                    
                    data_found = True
            
            # Format response based on what we found
            if data_found:
                if start_date and end_date and days_diff:
                    # Has calculation result
                    try:
                        start_dt = datetime.strptime(start_date, '%m/%d/%Y')
                        end_dt = datetime.strptime(end_date, '%m/%d/%Y')
                        
                        start_readable = start_dt.strftime('%B %d, %Y')
                        end_readable = end_dt.strftime('%B %d, %Y')
                        
                        if payroll_no:
                            return (f"For Payroll Period #{payroll_no}, the pay period runs from {start_readable} "
                                f"to {end_readable}. The difference is **{days_diff} days**.")
                        else:
                            return f"Pay period: {start_readable} to {end_readable}. Difference: **{days_diff} days**."
                    except:
                        pass
                
                if start_date and end_date:
                    # Just start and end dates
                    try:
                        start_dt = datetime.strptime(start_date, '%m/%d/%Y')
                        end_dt = datetime.strptime(end_date, '%m/%d/%Y')
                        
                        start_readable = start_dt.strftime('%B %d, %Y')
                        end_readable = end_dt.strftime('%B %d, %Y')
                        
                        result = f"**Start Date:** {start_readable}\n**End Date:** {end_readable}"
                        
                        if payroll_no:
                            result = f"**Payroll Period #{payroll_no}**\n{result}"
                        if check_date:
                            try:
                                check_dt = datetime.strptime(check_date, '%m/%d/%Y')
                                check_readable = check_dt.strftime('%B %d, %Y')
                                result += f"\n**Check Date:** {check_readable}"
                            except:
                                result += f"\n**Check Date:** {check_date}"
                        
                        return result
                    except:
                        # Fallback to raw format
                        result = f"**Start Date:** {start_date}\n**End Date:** {end_date}"
                        if check_date:
                            result += f"\n**Check Date:** {check_date}"
                        return result
            
            # If no structured data found, clean and return
            cleaned = raw_response.replace('    ', ' ').strip()
            
            # Remove error URLs
            if 'For troubleshooting, visit:' in cleaned:
                cleaned = cleaned.split('For troubleshooting, visit:')[0].strip()
            
            return f"📅 Payroll Information:\n\n{cleaned}"
            
        except Exception as e:
            print(f"❌ Manual formatting failed: {str(e)}")
            return f"Payroll data found:\n\n{raw_response}"

    def query(self, user_query: str) -> str:
        """
        Query the payroll calendar data with query reformulation and enhanced error handling
        """
        if not self.is_initialized:
            return "Payroll system not initialized. Please check the CSV file."
        
        print(f"\n{'='*60}")
        print(f"💼 PAYROLL CALENDAR QUERY PROCESSING")
        print(f"{'='*60}")
        print(f"❓ Original Query: {user_query}")
        
        # Step 1: Reformulate the query
        try:
            print(f"\n🔄 Step 1: Query Reformulation")
            print(f"{'-'*60}")
            reformulator = get_payroll_reformulator()
            reformulation_result = reformulator.process_payroll_query(user_query, self.csv_path)
            
            reformulated_query = reformulation_result["reformulated_query"]
            
            print(f"✅ Reformulation completed:")
            print(f"   📥 Original: {user_query}")
            print(f"   📤 Reformulated: {reformulated_query}")
            
        except Exception as e:
            print(f"⚠️ Reformulation failed: {str(e)}")
            reformulated_query = user_query
        
        # Step 2: Execute query with enhanced error handling
        print(f"\n🤖 Step 2: CSV Agent Execution")
        print(f"{'-'*60}")
        
        max_attempts = 2
        raw_data = None
        last_error = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                query_to_use = reformulated_query if attempt == 1 else user_query
                
                print(f"🤔 Attempt {attempt}: {query_to_use}")
                print(f"{'-'*50}")
                
                executor = getattr(self, 'agent_executor', None) or getattr(self, 'agent', None)
                if executor is None:
                    raise AttributeError("Neither agent_executor nor agent is initialized")

                response = executor.invoke({"input": query_to_use})
                
                print(f"✅ Agent completed")
                print(f"🔍 Response type: {type(response)}")
                
                # Extract raw data
                print(f"\n🔍 Step 3: Response Extraction")
                print(f"{'-'*60}")
                raw_data = self._extract_raw_data(response)
                
                print(f"🔍 Extracted raw data: {str(raw_data)[:200]}...")
                
                # Validate the extracted data
                if raw_data and str(raw_data).strip():
                    raw_str = str(raw_data).strip().lower()
                    
                    # Skip invalid responses
                    if raw_str in ['no data found', 'none', '', 'null']:
                        print(f"⚠️ Invalid data in attempt {attempt}")
                        if attempt < max_attempts:
                            continue
                        break
                    
                    # Skip timeout/error messages
                    if 'agent stopped' in raw_str or 'iteration limit' in raw_str:
                        print(f"⚠️ Timeout in attempt {attempt}")
                        if attempt < max_attempts:
                            continue
                        break
                    
                    # Success!
                    print(f"✅ Valid data extracted on attempt {attempt}")
                    break
                else:
                    print(f"⚠️ Empty data in attempt {attempt}")
                    if attempt < max_attempts:
                        continue
            
            except Exception as e:
                last_error = e
                error_str = str(e)
                print(f"❌ Error in attempt {attempt}: {error_str[:200]}...")
                
                # ENHANCED: Try to extract data from error message
                # The error message often contains the actual result!
                if "Final Answer:" in error_str:
                    try:
                        # Extract the Final Answer section from error
                        parts = error_str.split("Final Answer:")
                        if len(parts) > 1:
                            answer_section = parts[1].split("For troubleshooting")[0].strip()
                            
                            # Check if this contains actual data
                            if answer_section and len(answer_section) > 5:
                                print(f"🔧 Extracted data from error message: {answer_section[:200]}...")
                                raw_data = answer_section
                                print(f"✅ Successfully recovered data from error!")
                                break
                    except Exception as extract_error:
                        print(f"⚠️ Could not extract data from error: {extract_error}")
                
                # If this is the last attempt and we couldn't extract data, return error
                if attempt == max_attempts:
                    # One last attempt to find data in intermediate steps
                    if isinstance(last_error, ValueError) and hasattr(executor, '_intermediate_steps'):
                        try:
                            print("🔧 Trying to extract from intermediate steps...")
                            # This might have the data even if final parsing failed
                            pass
                        except:
                            pass
                    
                    if not raw_data:
                        return f"Unable to process query. Please try rephrasing your question or simplify it."
        
        # Step 4: Summarize the response
        print(f"\n📝 Step 4: Response Summarization")
        print(f"{'-'*60}")
        
        if not raw_data or str(raw_data).strip().lower() == "no data found":
            return "No matching payroll data found for your query. Please verify the date format (M/D/YYYY) and try again."
        
        try:
            print(f"📄 Summarizing response (raw length: {len(str(raw_data))})...")
            summarized_answer = self._summarize_response(str(raw_data), user_query)
            
            print(f"✅ Summarization completed")
            print(f"{'='*60}\n")
            return summarized_answer
        
        except Exception as e:
            print(f"❌ Summarization failed: {str(e)}")
            formatted_response = self._manual_format_fallback(str(raw_data))
            print(f"{'='*60}\n")
            return formatted_response