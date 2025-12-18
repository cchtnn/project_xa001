from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from vectorstore_manager import initialize_vectorstore, get_collection
from conversation_graph import create_conversation_graph
import uvicorn
import sqlite3
import os
import docx_parser
import shutil
import auth_db
import session_db, session_manager, student_transcript_csv_handler, query_handler, logic, config
from query_classifier import classify_user_query, QueryType
from catalog_query_processor import process_catalog_query
import jwt
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
import secrets
import logging
from typing import List
from bor_planner import answer_bor_query

# Load environment variables
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))
STRONG_PASSWORD = os.getenv("STRONG_PASSWORD", "true").lower() == "true"

# Default payroll calendar path
DEFAULT_PAYROLL_CALENDAR_PATH = 'data//payroll_cal//2026Payroll Calendar.docx'
DEFAULT_PAYROLL_CSV_FOLDER = 'data//payroll_cal//csv_files'

DEFAULT_CATALOG_CHUNKS_FILE = 'data//catalog_data//parsed_data//embedding_chunks_robust.json'
DEFAULT_CATALOG_SEARCH_SYSTEM = 'chtn_test_docs//search_system'

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# OAuth2 for JWT
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

def get_username_from_token(request: Request):
    token = request.cookies.get("access_token")
    payload = decode_access_token(token) if token else None
    if payload and payload.get("type") != "refresh":
        return payload["sub"]
    return None

def generate_session_name(question):
    question = question.strip()
    question = re.sub(r'^(what|how|when|where|why|who|can you|tell me|explain|describe)\s+', '', question.lower())
    question = re.sub(r'\?+$', '', question)
    question = re.sub(r'[^\w\s-]', '', question)
    words = question.split()
    stop_words = {'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about', 'me', 'you', 'your', 'my'}
    meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
    if meaningful_words:
        session_name = ' '.join(meaningful_words[:4])
        session_name = ' '.join(word.capitalize() for word in session_name.split())
        if len(session_name) > 30:
            session_name = session_name[:27] + "..."
        return session_name
    else:
        fallback = ' '.join(words[:3])
        fallback = ' '.join(word.capitalize() for word in fallback.split())
        return fallback[:30] if fallback else "New Chat"

# --- Authentication Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    user = auth_db.validate_user(username, password)
    if not user:
        return templates.TemplateResponse("login.html", {
            "request": request, 
            "error": "Invalid credentials"
        })
    
    active_sessions = session_db.get_active_sessions(user["username"])
    if len(active_sessions) >= 10:
        return templates.TemplateResponse("login.html", {
            "request": request, 
            "error": "Too many active sessions"
        })
    
    access_token = create_access_token({"sub": user["username"]})
    refresh_token = create_refresh_token({"sub": user["username"]})
    response = RedirectResponse(url="/chat", status_code=302)
    response.set_cookie(key="access_token", value=access_token, httponly=True, secure=True, samesite="strict")
    response.set_cookie(key="refresh_token", value=refresh_token, httponly=True, secure=True, samesite="strict")
    return response

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = auth_db.validate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    active_sessions = session_db.get_active_sessions(user["username"])
    if len(active_sessions) >= 3:
        raise HTTPException(status_code=429, detail="Too many active sessions")
    access_token = create_access_token({"sub": user["username"]})
    refresh_token = create_refresh_token({"sub": user["username"]})
    response = RedirectResponse(url="/chat", status_code=302)
    response.set_cookie(key="access_token", value=access_token, httponly=True, secure=True, samesite="strict")
    response.set_cookie(key="refresh_token", value=refresh_token, httponly=True, secure=True, samesite="strict")
    return response

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    username = get_username_from_token(request)
    if not username:
        return RedirectResponse(url="/login", status_code=302)
    first_name = username.split('.')[0].capitalize()
    sessions = session_db.get_user_sessions(username)
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "user_authenticated": True, "username": first_name, "sessions": sessions}
    )

@app.post("/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")
    return response

# --- Chat Session Endpoints ---
@app.post("/new_session")
async def new_session(request: Request):
    username = get_username_from_token(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    session_id = session_db.create_new_session(username, "New Chat")
    return {"session_id": session_id}

@app.get("/user_sessions")
async def get_user_sessions(request: Request):
    username = get_username_from_token(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    sessions = session_db.get_user_sessions(username)
    return {"sessions": sessions}

@app.post("/rename_session")
async def rename_session(request: Request, session_id: int = Form(...), new_name: str = Form(...)):
    username = get_username_from_token(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    session_db.rename_session(session_id, new_name)
    return {"success": True}

@app.post("/delete_session")
async def delete_session(request: Request, session_id: int = Form(...)):
    username = get_username_from_token(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    session_db.delete_session(session_id)
    session_folder = f"data/user_uploads/{username}/session_{session_id}"
    if os.path.exists(session_folder):
        shutil.rmtree(session_folder)
    public_folder = f"data/public_uploads/session_{session_id}"
    if os.path.exists(public_folder):
        shutil.rmtree(public_folder)
    return {"success": True}

# --- File Upload Endpoint ---
@app.post("/upload")
async def upload_files(request: Request, session_id: int = Form(...), files: List[UploadFile] = File(...), private: bool = Form(...)):
    username = get_username_from_token(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if private:
        base_output_path = f"data/user_uploads/{username}/session_{session_id}"
    else:
        base_output_path = f"data/public_uploads/session_{session_id}"

    image_output_path = os.path.join(base_output_path, "extracted_images")
    csv_output_path = os.path.join(base_output_path, "csv_files")
    os.makedirs(csv_output_path, exist_ok=True)
    upload_folder = base_output_path

    processed_files = []
    errors = []
    
    if private:
        upload_folder = f"data/user_uploads/{username}/session_{session_id}"
        os.makedirs(upload_folder, exist_ok=True)
    else:
        upload_folder = f"data/public_uploads/session_{session_id}"
        os.makedirs(upload_folder, exist_ok=True)
    
    for file in files:
        try:
            file_path = os.path.join(upload_folder, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            if file.filename.lower().endswith('.zip'):
                extracted_pdfs = logic.extract_and_process_zip_images_only(file_path, image_output_path, csv_output_path)
                processed_files.extend(extracted_pdfs)
                logging.info(f"User {username} uploaded and processed ZIP file {file.filename}")
            elif file.filename.lower().endswith('.pdf'):
                result = logic.parse_pdf_to_individual_csv(file_path, image_output_path, csv_output_path)
                if result:
                    processed_files.append(file.filename)
                logging.info(f"User {username} uploaded PDF file {file.filename}")
            elif file.filename.lower().endswith('.docx'):
                try:
                    print(f"Extract payroll calendar from DOCX")
                    df = docx_parser.extract_payroll_calendar(file_path, expected_count=27)
                    df.columns = ['payroll_no', 'start_date', 'end_date', 'check_date']
                    df['optional_withholdings_changes_by'] = df['end_date']
                    docx_csv_output = os.path.join(csv_output_path, f"{os.path.splitext(file.filename)[0]}_payroll.csv")
                    df.to_csv(docx_csv_output, index=False)
                    processed_files.append(file.filename)
                    logging.info(f"User {username} uploaded and processed DOCX file {file.filename} - extracted {len(df)} payroll records")
                    
                except Exception as docx_error:
                    errors.append(f"Error processing DOCX {file.filename}: {str(docx_error)}")
                    logging.error(f"DOCX processing error for {file.filename}: {docx_error}")
            else:
                errors.append(f"Unsupported file type: {file.filename}")
                
        except Exception as e:
            errors.append(f"Error processing {file.filename}: {str(e)}")
            logging.error(f"Error processing file {file.filename}: {e}")
    
    if processed_files:
        try:
            print("Creating final merged CSV from all individual CSVs...")
            final_merged_csv = logic.create_final_merged_csv(csv_output_path)
            if final_merged_csv:
                logic.fix_term_career_totals(final_merged_csv, final_merged_csv)
                print(f"Final merged CSV created: {os.path.basename(final_merged_csv)}")
        except Exception as e:
            errors.append(f"Error creating final merged CSV: {str(e)}")
            logging.error(f"Error creating final merged CSV: {e}")
    
    session_db.update_upload_paths(session_id, processed_files)
    
    message = f"Processed {len(processed_files)} file(s) successfully."
    if errors:
        message += f" {len(errors)} error(s) occurred."
    
    return {
        "success": True, 
        "message": message,
        "processed_files": processed_files,
        "errors": errors
    }

# --- Query Endpoint (FIXED) ---
@app.post("/query")
async def query_endpoint(
    request: Request,
    user_query: str = Form(..., alias="query"),
    session_id: int = Form(...),
    private: bool = Form(False)
):
    username = get_username_from_token(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")

    session_info = session_db.get_session_info(session_id)
    csv_folder = session_info['upload_paths']
    embedding_index = session_info['embedding_index_path']

    query_type, confidence_score = classify_user_query(user_query)
    print(f"Query classified as: {query_type} (confidence: {confidence_score:.3f})")

    # ==================================================================================
    # CATALOG QUERY HANDLING
    # ==================================================================================
    if query_type == QueryType.CATALOG:
        logging.info(f"User {username} queried CATALOG: {user_query}")
        
        print(f"\n{'='*80}")
        print(f"📚 CATALOG QUERY DETECTED")
        print(f"{'='*80}")
        print(f"User: {username}")
        print(f"Query: {user_query}")
        print(f"Session ID: {session_id}")
        print(f"Private Mode: {private}")
        print(f"Confidence: {confidence_score:.3f}")
        
        # Determine catalog data paths
        catalog_chunks_file = None
        catalog_search_system = None
        user_uploaded_catalog = False
        
        # Check for user-uploaded catalog data (private mode)
        if private:
            user_catalog_folder = f"data/user_uploads/{username}/session_{session_id}/catalog"
            user_chunks_file = os.path.join(user_catalog_folder, "embedding_chunks.json")
            user_search_system = os.path.join(user_catalog_folder, "search_system")
            
            print(f"\n🔍 Checking for private catalog data...")
            print(f"   Chunks file: {user_chunks_file}")
            print(f"   Search system: {user_search_system}")
            
            if os.path.exists(user_search_system):
                catalog_search_system = user_search_system
                user_uploaded_catalog = True
                print(f"✅ Using user's private search system")
            elif os.path.exists(user_chunks_file):
                catalog_chunks_file = user_chunks_file
                catalog_search_system = user_search_system  # Will be created
                user_uploaded_catalog = True
                print(f"✅ Using user's private chunks file")
            else:
                print(f"⚠️ No private catalog data found")
        
        # Check for public/shared catalog data (session-based)
        if not user_uploaded_catalog:
            public_catalog_folder = f"data/public_uploads/session_{session_id}/catalog"
            public_chunks_file = os.path.join(public_catalog_folder, "embedding_chunks.json")
            public_search_system = os.path.join(public_catalog_folder, "search_system")
            
            print(f"\n🔍 Checking for session catalog data...")
            print(f"   Chunks file: {public_chunks_file}")
            print(f"   Search system: {public_search_system}")
            
            if os.path.exists(public_search_system):
                catalog_search_system = public_search_system
                print(f"✅ Using session search system")
            elif os.path.exists(public_chunks_file):
                catalog_chunks_file = public_chunks_file
                catalog_search_system = public_search_system  # Will be created
                print(f"✅ Using session chunks file")
            else:
                print(f"⚠️ No session catalog data found")
        
        # Fall back to default catalog if no user/session data found
        if not catalog_search_system and not catalog_chunks_file:
            print(f"\n🔍 Falling back to default catalog...")
            print(f"   Default chunks: {DEFAULT_CATALOG_CHUNKS_FILE}")
            print(f"   Default system: {DEFAULT_CATALOG_SEARCH_SYSTEM}")
            
            if os.path.exists(DEFAULT_CATALOG_SEARCH_SYSTEM):
                catalog_search_system = DEFAULT_CATALOG_SEARCH_SYSTEM
                print(f"✅ Using default search system")
            elif os.path.exists(DEFAULT_CATALOG_CHUNKS_FILE):
                catalog_chunks_file = DEFAULT_CATALOG_CHUNKS_FILE
                catalog_search_system = DEFAULT_CATALOG_SEARCH_SYSTEM
                print(f"✅ Using default chunks file")
            else:
                print(f"❌ No catalog data available")
                answer = (
                    "📚 **Course Catalog Not Available**\n\n"
                    "I couldn't find any course catalog data to answer your question.\n\n"
                    "**To use the catalog feature:**\n"
                    "1. Upload your course catalog document (PDF with embedded chunks JSON), or\n"
                    "2. Contact your administrator to set up the default catalog system.\n\n"
                    "For now, please try asking about policies, student transcripts, "
                    "payroll calendar, or Board of Regents meetings."
                )
                
                # Save to history
                history_before = session_db.get_session_message_count(session_id)
                session_db.add_single_qa_to_history(session_id, user_query, answer)
                
                if history_before == 0:
                    session_name = generate_session_name(user_query)
                    session_db.rename_session(session_id, session_name)
                    logging.info(f"Updated session {session_id} name to: {session_name}")
                
                return {
                    "answer": answer,
                    "session_id": session_id,
                    "query_type": query_type,
                    "confidence_score": confidence_score,
                    "session_name_updated": history_before == 0,
                    "error": "No catalog data available"
                }
        
        # Process the catalog query
        print(f"\n{'='*80}")
        print(f"🚀 PROCESSING CATALOG QUERY")
        print(f"{'='*80}")
        print(f"Chunks File: {catalog_chunks_file or 'N/A (using pre-built system)'}")
        print(f"Search System: {catalog_search_system}")
        print(f"User Uploaded: {user_uploaded_catalog}")
        print(f"{'='*80}\n")
        
        try:
            answer = process_catalog_query(
                user_query=user_query,
                catalog_path=None,  # Not used in current implementation
                chunks_file=catalog_chunks_file,
                search_system_dir=catalog_search_system
            )
            
            print(f"\n✅ Catalog query processed successfully")
            print(f"   Answer length: {len(answer)} characters")
            
        except Exception as e:
            logging.error(f"Error processing catalog query: {e}")
            import traceback
            traceback.print_exc()
            
            answer = (
                "❌ **Error Processing Catalog Query**\n\n"
                "I encountered an error while processing your course catalog query. "
                "This could be due to:\n\n"
                "- Corrupted or incompatible catalog data\n"
                "- Missing required files\n"
                "- System configuration issues\n\n"
                "Please try again or contact support if the problem persists.\n\n"
                f"**Error Details:** {str(e)}"
            )
            print(f"❌ Error: {str(e)}")
        
        # Save to conversation history
        history_before = session_db.get_session_message_count(session_id)
        session_db.add_single_qa_to_history(session_id, user_query, answer)
        
        # Update session name if this is the first message
        if history_before == 0:
            session_name = generate_session_name(user_query)
            session_db.rename_session(session_id, session_name)
            logging.info(f"Updated session {session_id} name to: {session_name}")
        
        print(f"\n{'='*80}")
        print(f"📊 CATALOG QUERY COMPLETE")
        print(f"{'='*80}")
        print(f"Session Name Updated: {history_before == 0}")
        print(f"Response Length: {len(answer)} chars")
        print(f"{'='*80}\n")
        
        return {
            "answer": answer,
            "session_id": session_id,
            "query_type": query_type,
            "confidence_score": confidence_score,
            "session_name_updated": history_before == 0,
            "catalog_source": "user_private" if user_uploaded_catalog and private else 
                            "session_shared" if user_uploaded_catalog else "default"
        }
    
    if query_type == QueryType.BOR_MEETING:
        today = datetime.utcnow().date()
        answer = answer_bor_query(user_query, today=today)

        history_before = session_db.get_session_message_count(session_id)
        session_db.add_single_qa_to_history(session_id, user_query, answer)

        if history_before == 0:
            session_name = generate_session_name(user_query)
            session_db.rename_session(session_id, session_name)
            logging.info(f"Updated session {session_id} name to: {session_name}")

        return {
            "answer": answer,
            "session_id": session_id,
            "query_type": query_type,
            "confidence_score": confidence_score,
            "session_name_updated": history_before == 0,
        }

    print("Private flag is", private)
    if private:
        user_csv_folder = f"data/user_uploads/{username}/session_{session_id}/csv_files"
        has_csv = False

        if os.path.exists(user_csv_folder):
            csv_files = [f for f in os.listdir(user_csv_folder) if f.lower().endswith('.csv')]
            if csv_files:
                has_csv = True

        if query_type == QueryType.STUDENT_TRANSCRIPT and has_csv:
            csv_files = sorted([f for f in os.listdir(user_csv_folder) if f.lower().endswith('.csv')])
            csv_path = os.path.join(user_csv_folder, csv_files[-1])
            print("Using private CSV for transcript query:", csv_path)
            answer = student_transcript_csv_handler.process_transcript_query(
                user_query, csv_path=csv_path
            )
            
            history_before = session_db.get_session_message_count(session_id)
            session_db.add_single_qa_to_history(session_id, user_query, answer)

            if history_before == 0:
                session_name = generate_session_name(user_query)
                session_db.rename_session(session_id, session_name)
                logging.info(f"Updated session {session_id} name to: {session_name}")

            return {
                "answer": answer, 
                "session_id": session_id,
                "query_type": query_type,
                "confidence_score": confidence_score,
                "session_name_updated": history_before == 0
            }
        
        if query_type == QueryType.STUDENT_TRANSCRIPT and not has_csv:
            answer = "Based on the document you uploaded I did not find the answer. Kindly upload the specific document."
        elif query_type == QueryType.POLICY:
            answer = "I can only answer questions based on your private uploaded documents when private mode is enabled. Please uncheck the private option to access general policy information, or upload relevant documents to get answers from your private data."
        else:
            answer = "I can only provide answers based on your private uploaded documents when private mode is enabled. Please upload relevant documents or uncheck the private option."
        
        history_before = session_db.get_session_message_count(session_id)
        session_db.add_single_qa_to_history(session_id, user_query, answer)

        if history_before == 0:
            session_name = generate_session_name(user_query)
            session_db.rename_session(session_id, session_name)
            logging.info(f"Updated session {session_id} name to: {session_name}")

        return {
            "answer": answer,
            "session_id": session_id,
            "query_type": query_type,
            "confidence_score": confidence_score,
            "session_name_updated": history_before == 0
        }

    public_upload_folder = "data/public_uploads"
    public_csv_folder = os.path.join(public_upload_folder, "csv_files")
    has_public_csv = False
    
    if os.path.exists(public_csv_folder):
        csv_files = [f for f in os.listdir(public_csv_folder) if f.lower().endswith('.csv')]
        if csv_files:
            has_public_csv = True

    print("User has public CSV:", has_public_csv)
    
    if query_type == QueryType.PAYROLL_CALENDAR:
        logging.info(f"User {username} queried PAYROLL_CALENDAR: {user_query}")
        
        if private:
            payroll_csv_folder = f"data/user_uploads/{username}/session_{session_id}/csv_files"
        else:
            payroll_csv_folder = f"data/public_uploads/session_{session_id}/csv_files"
        
        print(f"Looking for payroll CSV in: {payroll_csv_folder}")
        
        payroll_csv_path = None
        user_uploaded_file = False
        
        if os.path.exists(payroll_csv_folder):
            csv_files = [f for f in os.listdir(payroll_csv_folder) if f.lower().endswith('.csv') and 'payroll' in f.lower()]
            print(f"Found CSV files with 'payroll': {csv_files}")
            if csv_files:
                merged_files = [f for f in csv_files if 'merged' not in f.lower()]
                if merged_files:
                    payroll_csv_path = os.path.join(payroll_csv_folder, merged_files[0])
                    print(f"Using user-uploaded payroll CSV: {payroll_csv_path}")
                    user_uploaded_file = True
                else:
                    payroll_csv_path = os.path.join(payroll_csv_folder, csv_files[-1])
                    print(f"Using user-uploaded payroll CSV: {payroll_csv_path}")
                    user_uploaded_file = True
        else:
            print(f"Payroll CSV folder does not exist: {payroll_csv_folder}")
        
        if not payroll_csv_path:
            print(f"No user-uploaded payroll data found. Checking default payroll calendar...")
            
            os.makedirs(DEFAULT_PAYROLL_CSV_FOLDER, exist_ok=True)
            
            default_csv_files = [f for f in os.listdir(DEFAULT_PAYROLL_CSV_FOLDER) if f.lower().endswith('.csv') and 'payroll' in f.lower()]
            
            if default_csv_files:
                payroll_csv_path = os.path.join(DEFAULT_PAYROLL_CSV_FOLDER, default_csv_files[-1])
                print(f"Using existing default payroll CSV: {payroll_csv_path}")
            elif os.path.exists(DEFAULT_PAYROLL_CALENDAR_PATH):
                print(f"Parsing default payroll calendar from: {DEFAULT_PAYROLL_CALENDAR_PATH}")
                try:
                    df = docx_parser.extract_payroll_calendar(DEFAULT_PAYROLL_CALENDAR_PATH, expected_count=27)
                    df.columns = ['payroll_no', 'start_date', 'end_date', 'check_date']
                    df['optional_withholdings_changes_by'] = df['end_date']
                    
                    default_csv_path = os.path.join(DEFAULT_PAYROLL_CSV_FOLDER, "2026Payroll_Calendar_payroll.csv")
                    df.to_csv(default_csv_path, index=False)
                    payroll_csv_path = default_csv_path
                    
                    print(f"Default payroll CSV created: {payroll_csv_path}")
                    logging.info(f"Created default payroll CSV from {DEFAULT_PAYROLL_CALENDAR_PATH} with {len(df)} records")
                except Exception as e:
                    logging.error(f"Error processing default payroll calendar: {e}")
                    print(f"Error processing default payroll calendar: {e}")
            else:
                print(f"Default payroll calendar not found at: {DEFAULT_PAYROLL_CALENDAR_PATH}")
        
        answer = None
        
        if payroll_csv_path and os.path.exists(payroll_csv_path):
            from docx_parser import PayrollCSVAgent
            
            try:
                print(f"\n{'='*60}")
                print(f"INITIALIZING PAYROLL QUERY PROCESSING")
                print(f"{'='*60}")
                print(f"CSV Path: {payroll_csv_path}")
                print(f"Query: {user_query}")
                print(f"Private: {private}")
                
                payroll_agent = PayrollCSVAgent(csv_path=payroll_csv_path)
                if payroll_agent.initialize():
                    print(f"Payroll agent initialized successfully")
                    answer = payroll_agent.query(user_query)
                    print(f"Query processed successfully")
                    print(f"Answer length: {len(answer)} characters")
                    print(f"{'='*60}\n")
                else:
                    answer = "Failed to initialize payroll calendar system. Please try again."
                    print(f"Failed to initialize payroll agent")
            except Exception as e:
                logging.error(f"Error processing payroll query: {e}")
                import traceback
                traceback.print_exc()
                answer = "Error processing payroll query. Please ensure you have uploaded the payroll calendar document or try again later."
        else:
            answer = "No payroll calendar data found. Please upload a payroll calendar document (.docx) or contact support if the default calendar should be available."
        
        history_before = session_db.get_session_message_count(session_id)
        session_db.add_single_qa_to_history(session_id, user_query, answer)

        if history_before == 0:
            session_name = generate_session_name(user_query)
            session_db.rename_session(session_id, session_name)
            logging.info(f"Updated session {session_id} name to: {session_name}")

        return {
            "answer": answer,
            "session_id": session_id,
            "query_type": query_type,
            "confidence_score": confidence_score,
            "session_name_updated": history_before == 0
        }
    
    if query_type == QueryType.STUDENT_TRANSCRIPT and has_public_csv and not private:
        csv_files = sorted([f for f in os.listdir(public_csv_folder) if f.lower().endswith('.csv')])
        csv_path = os.path.join(public_csv_folder, csv_files[-1])
        answer = student_transcript_csv_handler.process_transcript_query(
            user_query, csv_path=csv_path
        )
        
        history_before = session_db.get_session_message_count(session_id)
        session_db.add_single_qa_to_history(session_id, user_query, answer)

        if history_before == 0:
            session_name = generate_session_name(user_query)
            session_db.rename_session(session_id, session_name)
            logging.info(f"Updated session {session_id} name to: {session_name}")

        return {
            "answer": answer, 
            "session_id": session_id,
            "query_type": query_type,
            "confidence_score": confidence_score,
            "session_name_updated": history_before == 0
        }

    logging.info(f"User {username} queried: {user_query} (session_id: {session_id}) - Using vectorstore for {query_type} query")
    try:
        index, metadata, tab_data = initialize_vectorstore()
        collection = get_collection()
    except Exception as e:
        logging.error(f"Error loading data for user {username}: {e}")
        collection, tab_data = None, {}

    chat_history = session_db.get_contextual_history(session_id, limit=5)
    user_context = {
        "username": username,
        "session_id": session_id,
        "language": "English",
        "active_transcript_csv_path": None
    }
    conv_graph = create_conversation_graph(collection, tab_data)
    result = conv_graph.process_conversation(user_query, chat_history, user_context)
    response_content = result["response"]

    history_before = session_db.get_session_message_count(session_id)
    session_db.add_single_qa_to_history(session_id, user_query, response_content)

    if history_before == 0:
        session_name = generate_session_name(user_query)
        session_db.rename_session(session_id, session_name)
        logging.info(f"Updated session {session_id} name to: {session_name}")

    return {
        "answer": response_content,
        "session_id": session_id,
        "query_type": query_type,
        "confidence_score": confidence_score,
        "contextual": len(chat_history) > 0,
        "session_name_updated": history_before == 0
    }

@app.get("/history")
async def get_history(request: Request, session_id: int):
    username = get_username_from_token(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    user_sessions = session_db.get_user_sessions(username)
    session_ids = [s["session_id"] for s in user_sessions]
    
    if session_id not in session_ids:
        raise HTTPException(status_code=403, detail="Access denied")
    
    history = session_db.load_qa_history(session_id)
    return {"history": history}

# --- Debug Endpoint ---
@app.get("/debug/sessions")
async def debug_sessions():
    import sqlite3
    conn = sqlite3.connect("data/session_state.db")
    c = conn.cursor()
    c.execute("SELECT session_id, username, session_name, upload_paths, embedding_index_path FROM chat_sessions")
    rows = c.fetchall()
    conn.close()
    return {"sessions": rows}

# --- Register Admin Routes ---
from admin_routes import register_admin_routes
register_admin_routes(app)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)