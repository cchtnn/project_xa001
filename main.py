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
    # Delete from DB
    session_db.delete_session(session_id)
    # Delete session folder (private uploads)
    session_folder = f"data/user_uploads/{username}/session_{session_id}"
    if os.path.exists(session_folder):
        shutil.rmtree(session_folder)
    # Optionally, delete public uploads if you support public sessions
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
    upload_folder = base_output_path  # Always upload to the base_output_path

    processed_files = []
    errors = []
    
    # Determine output folders based on private flag
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
            
            # Check file extension
            if file.filename.lower().endswith('.zip'):
                # Process ZIP file (only extract images and create individual CSVs)
                extracted_pdfs = logic.extract_and_process_zip_images_only(file_path, image_output_path, csv_output_path)
                processed_files.extend(extracted_pdfs)
                logging.info(f"User {username} uploaded and processed ZIP file {file.filename}")
            elif file.filename.lower().endswith('.pdf'):
                # Process single PDF (only extract images and create individual CSVs)
                result = logic.parse_pdf_to_individual_csv(file_path, image_output_path, csv_output_path)
                if result:
                    processed_files.append(file.filename)
                logging.info(f"User {username} uploaded PDF file {file.filename}")
            elif file.filename.lower().endswith('.docx'):
                # Process DOCX file (payroll calendar extraction)
                try:
                    print(f"Extract payroll calendar from DOCX")
                    df = docx_parser.extract_payroll_calendar(file_path, expected_count=27)
                    df.columns = ['payroll_no', 'start_date', 'end_date', 'check_date']
                    
                    # Add optional withholdings column
                    df['optional_withholdings_changes_by'] = df['end_date']
                    
                    # Save extracted data to CSV in the session folder
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
    
    # After processing all files, create ONE final merged CSV
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
    
    # After upload, update session_db with file paths
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
async def query(
    request: Request,
    query: str = Form(...),
    session_id: int = Form(...),
    private: bool = Form(False)
):
    username = get_username_from_token(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")

    session_info = session_db.get_session_info(session_id)
    csv_folder = session_info['upload_paths']  # or parse JSON
    embedding_index = session_info['embedding_index_path']

    # CLASSIFICATION FIRST - Always classify the query regardless of uploaded data
    query_type, confidence_score = classify_user_query(query)
    print(f"Query classified as: {query_type} (confidence: {confidence_score:.3f})")

    # --- BOR_MEETING queries (Board of Regents) ---
    if query_type == QueryType.BOR_MEETING:
        # Use current UTC date; adjust if you want local time
        today = datetime.utcnow().date()
        answer = answer_bor_query(query, today=today)

        # Save to history
        history_before = session_db.get_session_message_count(session_id)
        session_db.add_single_qa_to_history(session_id, query, answer)

        # If this was the first question, update the session name
        if history_before == 0:
            session_name = generate_session_name(query)
            session_db.rename_session(session_id, session_name)
            logging.info(f"Updated session {session_id} name to: {session_name}")

        return {
            "answer": answer,
            "session_id": session_id,
            "query_type": query_type,
            "confidence_score": confidence_score,
            "session_name_updated": history_before == 0,
        }


    # Scenario a: Private checked - use ONLY user's private data (no fallback to common pool)
    print("Private flag is", private)
    if private:
        # FIX: Use session-specific folder for CSVs
        user_csv_folder = f"data/user_uploads/{username}/session_{session_id}/csv_files"
        has_csv = False

        if os.path.exists(user_csv_folder):
            csv_files = [f for f in os.listdir(user_csv_folder) if f.lower().endswith('.csv')]
            if csv_files:
                has_csv = True

        # NEW LOGIC: Check if it's actually a transcript query
        if query_type == QueryType.STUDENT_TRANSCRIPT and has_csv:
            # Use private CSV for transcript queries
            csv_files = sorted([f for f in os.listdir(user_csv_folder) if f.lower().endswith('.csv')])
            csv_path = os.path.join(user_csv_folder, csv_files[-1])
            print("Using private CSV for transcript query:", csv_path)
            answer = student_transcript_csv_handler.process_transcript_query(
                query, csv_path=csv_path
            )
            
            # Check if this is the first question in the session BEFORE adding to history
            history_before = session_db.get_session_message_count(session_id)
            session_db.add_single_qa_to_history(session_id, query, answer)

            # If this was the first question, update the session name
            if history_before == 0:
                session_name = generate_session_name(query)
                session_db.rename_session(session_id, session_name)
                logging.info(f"Updated session {session_id} name to: {session_name}")

            return {
                "answer": answer, 
                "session_id": session_id,
                "query_type": query_type,
                "confidence_score": confidence_score,
                "session_name_updated": history_before == 0
            }
        
        # UPDATED LOGIC: If private is checked, ALWAYS stay within private scope
        # Don't fall through to common pool - return out of scope message
        if query_type == QueryType.STUDENT_TRANSCRIPT and not has_csv:
            answer = "Based on the document you uploaded I did not find the answer. Kindly upload the specific document."
        elif query_type == QueryType.POLICY:
            answer = "I can only answer questions based on your private uploaded documents when private mode is enabled. Please uncheck the private option to access general policy information, or upload relevant documents to get answers from your private data."
        else:
            answer = "I can only provide answers based on your private uploaded documents when private mode is enabled. Please upload relevant documents or uncheck the private option."
        
        history_before = session_db.get_session_message_count(session_id)
        session_db.add_single_qa_to_history(session_id, query, answer)

        if history_before == 0:
            session_name = generate_session_name(query)
            session_db.rename_session(session_id, session_name)
            logging.info(f"Updated session {session_id} name to: {session_name}")

        return {
            "answer": answer,
            "session_id": session_id,
            "query_type": query_type,
            "confidence_score": confidence_score,
            "session_name_updated": history_before == 0
        }

    # Check if user has uploaded files to public folder (only for transcript queries)
    public_upload_folder = "data/public_uploads"
    public_csv_folder = os.path.join(public_upload_folder, "csv_files")
    has_public_csv = False
    
    if os.path.exists(public_csv_folder):
        csv_files = [f for f in os.listdir(public_csv_folder) if f.lower().endswith('.csv')]
        if csv_files:
            has_public_csv = True

    print("User has public CSV:", has_public_csv)
    
    # NEW LOGIC: Only use public CSV if it's actually a transcript query
    # Handle PAYROLL_CALENDAR queries
    if query_type == QueryType.PAYROLL_CALENDAR:
        logging.info(f"User {username} queried PAYROLL_CALENDAR: {query}")
        
        # Determine payroll CSV path based on private flag
        if private:
            payroll_csv_folder = f"data/user_uploads/{username}/session_{session_id}/csv_files"
        else:
            payroll_csv_folder = f"data/public_uploads/session_{session_id}/csv_files"
        
        print(f"Looking for payroll CSV in: {payroll_csv_folder}")
        
        # Look for payroll CSV (contains "payroll" in filename)
        payroll_csv_path = None
        if os.path.exists(payroll_csv_folder):
            csv_files = [f for f in os.listdir(payroll_csv_folder) if f.lower().endswith('.csv') and 'payroll' in f.lower()]
            print(f"Found CSV files with 'payroll': {csv_files}")
            if csv_files:
                # Prefer merged files if available
                merged_files = [f for f in csv_files if 'merged' not in f.lower()]
                if merged_files:
                    payroll_csv_path = os.path.join(payroll_csv_folder, merged_files[0])
                    print(f"Using merged payroll CSV: {payroll_csv_path}")
                else:
                    payroll_csv_path = os.path.join(payroll_csv_folder, csv_files[-1])
                    print(f"Using payroll CSV: {payroll_csv_path}")
        else:
            print(f"Payroll CSV folder does not exist: {payroll_csv_folder}")
        
        if payroll_csv_path and os.path.exists(payroll_csv_path):
            # Import and use PayrollCSVAgent with reformulation
            from docx_parser import PayrollCSVAgent
            
            try:
                print(f"\n{'='*60}")
                print(f"🚀 INITIALIZING PAYROLL QUERY PROCESSING")
                print(f"{'='*60}")
                print(f"📄 CSV Path: {payroll_csv_path}")
                print(f"❓ Query: {query}")
                print(f"🔒 Private: {private}")
                
                payroll_agent = PayrollCSVAgent(csv_path=payroll_csv_path)
                if payroll_agent.initialize():
                    print(f"✅ Payroll agent initialized successfully")
                    
                    # The query() method now includes reformulation internally
                    answer = payroll_agent.query(query)
                    
                    print(f"✅ Query processed successfully")
                    print(f"📊 Answer length: {len(answer)} characters")
                    print(f"{'='*60}\n")
                else:
                    answer = "Failed to initialize payroll calendar system. Please try again."
                    print(f"❌ Failed to initialize payroll agent")
            except Exception as e:
                logging.error(f"Error processing payroll query: {e}")
                import traceback
                traceback.print_exc()
                answer = "Error processing payroll query. Please ensure you have uploaded the payroll calendar document."
        else:
            answer = "No payroll calendar data found. Please upload a payroll calendar document (.docx) first."
        
        # Save to history
        history_before = session_db.get_session_message_count(session_id)
        session_db.add_single_qa_to_history(session_id, query, answer)

        if history_before == 0:
            session_name = generate_session_name(query)
            session_db.rename_session(session_id, session_name)
            logging.info(f"Updated session {session_id} name to: {session_name}")

        return {
            "answer": answer,
            "session_id": session_id,
            "query_type": query_type,
            "confidence_score": confidence_score,
            "session_name_updated": history_before == 0
        }
    
    # NEW LOGIC: Only use public CSV if it's actually a transcript query
    if query_type == QueryType.STUDENT_TRANSCRIPT and has_public_csv and not private:
        csv_files = sorted([f for f in os.listdir(public_csv_folder) if f.lower().endswith('.csv')])
        csv_path = os.path.join(public_csv_folder, csv_files[-1])
        answer = student_transcript_csv_handler.process_transcript_query(
            query, csv_path=csv_path
        )
        
        # Check if this is the first question in the session BEFORE adding to history
        history_before = session_db.get_session_message_count(session_id)
        session_db.add_single_qa_to_history(session_id, query, answer)

        # If this was the first question, update the session name
        if history_before == 0:
            session_name = generate_session_name(query)
            session_db.rename_session(session_id, session_name)
            logging.info(f"Updated session {session_id} name to: {session_name}")

        return {
            "answer": answer, 
            "session_id": session_id,
            "query_type": query_type,
            "confidence_score": confidence_score,
            "session_name_updated": history_before == 0
        }

    # Scenario c: Either no uploads found OR it's a POLICY query - use common data (existing vectorstore logic)
    logging.info(f"User {username} queried: {query} (session_id: {session_id}) - Using vectorstore for {query_type} query")
    try:
        index, metadata, tab_data = initialize_vectorstore()
        collection = get_collection()
    except Exception as e:
        logging.error(f"Error loading data for user {username}: {e}")
        collection, tab_data = None, {}

    # Existing logic for chat_history, user_context, conversation graph
    chat_history = session_db.get_contextual_history(session_id, limit=5)
    user_context = {
        "username": username,
        "session_id": session_id,
        "language": "English",
        "active_transcript_csv_path": None
    }
    conv_graph = create_conversation_graph(collection, tab_data)
    result = conv_graph.process_conversation(query, chat_history, user_context)
    response_content = result["response"]

    # Check if this is the first question in the session BEFORE adding to history
    history_before = session_db.get_session_message_count(session_id)
    session_db.add_single_qa_to_history(session_id, query, response_content)

    # If this was the first question, update the session name
    if history_before == 0:
        session_name = generate_session_name(query)
        session_db.rename_session(session_id, session_name)
        logging.info(f"Updated session {session_id} name to: {session_name}")

    return {
        "answer": response_content,
        "session_id": session_id,
        "query_type": query_type,  # Now returns the actual classified type
        "confidence_score": confidence_score,  # Now returns the actual confidence
        "contextual": len(chat_history) > 0,
        "session_name_updated": history_before == 0
    }

@app.get("/history")
async def get_history(request: Request, session_id: int):
    username = get_username_from_token(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Verify user owns this session
    user_sessions = session_db.get_user_sessions(username)
    session_ids = [s["session_id"] for s in user_sessions]
    
    if session_id not in session_ids:
        raise HTTPException(status_code=403, detail="Access denied")
    
    history = session_db.load_qa_history(session_id)
    return {"history": history}

# --- Admin Endpoints ---
@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    username = get_username_from_token(request)
    if not username or not auth_db.validate_admin(username):
        return RedirectResponse(url="/admin/login", status_code=302)
    users_raw = auth_db.list_users()
    users = [{"username": u[0], "role": u[1]} for u in users_raw]
    active_sessions = session_db.get_all_active_sessions()
    csrf_token = secrets.token_urlsafe(32)
    session_db.save_meta(f"csrf_{username}", csrf_token)
    
    # Check for success/error parameters
    add_success = request.query_params.get("add_success") == "true"
    
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "user_authenticated": True,
        "users": users,
        "active_sessions": active_sessions,
        "csrf_token": csrf_token,
        "add_success": add_success
    })

@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_get(request: Request):
    return templates.TemplateResponse("admin_login.html", {"request": request})

@app.post("/admin/login", response_class=HTMLResponse)
async def admin_login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    user = auth_db.validate_user(username, password)
    if not user or user["role"] != "admin":
        return templates.TemplateResponse(
            "admin_login.html",
            {"request": request, "error": "Invalid credentials"}
        )
    access_token = create_access_token({"sub": username}, expires_delta=timedelta(minutes=15))
    response = RedirectResponse(url="/admin", status_code=302)
    response.set_cookie(key="access_token", value=access_token, httponly=True, secure=True, samesite="strict")
    logging.info(f"Admin {username} logged in")
    return response

@app.get("/admin/add_user")
async def admin_add_user_get(request: Request):
    """Redirect GET requests to admin page"""
    username = get_username_from_token(request)
    if not username or not auth_db.validate_admin(username):
        return RedirectResponse(url="/admin/login", status_code=302)
    return RedirectResponse(url="/admin", status_code=302)

@app.get("/admin/delete_user")
async def admin_delete_user_get(request: Request):
    """Redirect GET requests to admin page"""
    username = get_username_from_token(request)
    if not username or not auth_db.validate_admin(username):
        return RedirectResponse(url="/admin/login", status_code=302)
    return RedirectResponse(url="/admin", status_code=302)

@app.get("/admin/edit_user")
async def admin_edit_user_get(request: Request):
    """Redirect GET requests to admin page"""
    username = get_username_from_token(request)
    if not username or not auth_db.validate_admin(username):
        return RedirectResponse(url="/admin/login", status_code=302)
    return RedirectResponse(url="/admin", status_code=302)

@app.post("/admin/refresh_sessions")
async def refresh_sessions(request: Request, csrf_token: str = Form(...)):
    admin_username = get_username_from_token(request)
    if not admin_username or not auth_db.validate_admin(admin_username):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if session_db.load_meta(f"csrf_{admin_username}") != csrf_token:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    
    try:
        # Clean up inactive sessions first
        session_db.cleanup_inactive_sessions()
        
        # Get fresh session data
        active_sessions = session_db.get_all_active_sessions()
        
        logging.info(f"Admin {admin_username} refreshed active sessions - found {len(active_sessions)} sessions")
        
        return {"success": True, "session_count": len(active_sessions)}
        
    except Exception as e:
        logging.error(f"Error refreshing sessions for admin {admin_username}: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh sessions")

@app.get("/admin/kill_session")
async def admin_kill_session_get(request: Request):
    """Redirect GET requests to admin page"""
    username = get_username_from_token(request)
    if not username or not auth_db.validate_admin(username):
        return RedirectResponse(url="/admin/login", status_code=302)
    return RedirectResponse(url="/admin", status_code=302)

@app.post("/admin/add_user")
async def add_user(request: Request, username: str = Form(...), password: str = Form(...), role: str = Form(...), csrf_token: str = Form(...)):
    admin_username = get_username_from_token(request)
    if not admin_username or not auth_db.validate_admin(admin_username):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if session_db.load_meta(f"csrf_{admin_username}") != csrf_token:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    try:
        auth_db.add_user(username, password, role, created_by=admin_username)
        logging.info(f"Admin {admin_username} added user {username} with role {role}")
        return RedirectResponse(url="/admin?add_success=true", status_code=302)
    
    except sqlite3.IntegrityError:
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "user_authenticated": True,
            "users": [{"username": u[0], "role": u[1]} for u in auth_db.list_users()],
            "active_sessions": session_db.get_all_active_sessions(),
            "csrf_token": session_db.load_meta(f"csrf_{admin_username}"),
            "error": f"Username '{username}' already exists. Please choose a different username."
        })
    except ValueError as e:
        if STRONG_PASSWORD:
            return templates.TemplateResponse("admin.html", {
                "request": request,
                "user_authenticated": True,
                "users": [{"username": u[0], "role": u[1]} for u in auth_db.list_users()],
                "active_sessions": session_db.get_all_active_sessions(),
                "csrf_token": session_db.load_meta(f"csrf_{admin_username}"),
                "error": str(e)
            })
        else:
            try:
                logging.warning(f"Admin {admin_username} added user {username} with weak password")
                auth_db.add_user(username, password, role, created_by=admin_username, bypass_password_validation=True)
                return templates.TemplateResponse("admin.html", {
                    "request": request,
                    "user_authenticated": True,
                    "users": [{"username": u[0], "role": u[1]} for u in auth_db.list_users()],
                    "active_sessions": session_db.get_all_active_sessions(),
                    "csrf_token": session_db.load_meta(f"csrf_{admin_username}"),
                    "add_success": True
                })
            except sqlite3.IntegrityError:
                return templates.TemplateResponse("admin.html", {
                    "request": request,
                    "user_authenticated": True,
                    "users": [{"username": u[0], "role": u[1]} for u in auth_db.list_users()],
                    "active_sessions": session_db.get_all_sessions(),
                    "csrf_token": session_db.load_meta(f"csrf_{admin_username}"),
                    "error": f"Username '{username}' already exists. Please choose a different username."
                })

@app.post("/admin/edit_user")
async def edit_user(request: Request, original_username: str = Form(...), username: str = Form(...), role: str = Form(...), csrf_token: str = Form(...)):
    admin_username = get_username_from_token(request)
    if not admin_username or not auth_db.validate_admin(admin_username):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if session_db.load_meta(f"csrf_{admin_username}") != csrf_token:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    try:
        auth_db.update_user(original_username, username, None, role)
        logging.info(f"Admin {admin_username} edited user {original_username} to {username} with role {role}")
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "user_authenticated": True,
            "users": [{"username": u[0], "role": u[1]} for u in auth_db.list_users()],
            "csrf_token": session_db.load_meta(f"csrf_{admin_username}"),
            "edit_success": True
        })
    except Exception as e:
        logging.error(f"Error editing user {original_username}: {e}")
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "user_authenticated": True,
            "users": [{"username": u[0], "role": u[1]} for u in auth_db.list_users()],
            "csrf_token": session_db.load_meta(f"csrf_{admin_username}"),
            "edit_error": True
        })

@app.post("/admin/delete_user")
async def delete_user(request: Request, username: str = Form(...), csrf_token: str = Form(...)):
    admin_username = get_username_from_token(request)
    if not admin_username or not auth_db.validate_admin(admin_username):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if session_db.load_meta(f"csrf_{admin_username}") != csrf_token:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    try:
        auth_db.delete_user(username)
        logging.info(f"Admin {admin_username} deleted user {username}")
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "user_authenticated": True,
            "users": [{"username": u[0], "role": u[1]} for u in auth_db.list_users()],
            "csrf_token": session_db.load_meta(f"csrf_{admin_username}"),
            "delete_success": True
        })
    except Exception as e:
        logging.error(f"Error deleting user {username}: {e}")
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "user_authenticated": True,
            "users": [{"username": u[0], "role": u[1]} for u in auth_db.list_users()],
            "csrf_token": session_db.load_meta(f"csrf_{admin_username}"),
            "delete_error": True
        })


@app.post("/admin/kill_session")
async def kill_session(request: Request, session_id: int = Form(...), csrf_token: str = Form(...)):
    admin_username = get_username_from_token(request)
    if not admin_username or not auth_db.validate_admin(admin_username):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if session_db.load_meta(f"csrf_{admin_username}") != csrf_token:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    
    try:
        session_info = session_db.kill_user_session(session_id)
        if session_info:
            logging.info(f"Admin {admin_username} killed session {session_id} for user {session_info[0]}")
            users_raw = auth_db.list_users()
            users = [{"username": u[0], "role": u[1]} for u in users_raw]
            active_sessions = session_db.get_all_active_sessions()
            add_success = request.query_params.get("add_success") == "true"
            return templates.TemplateResponse("admin.html", {
                "request": request,
                "user_authenticated": True,
                "users": users,
                "active_sessions": active_sessions,
                "csrf_token": csrf_token,
                "add_success": add_success
            })
        else:
            raise Exception("Session not found")
    except Exception as e:
        logging.error(f"Error killing session {session_id}: {e}")
        users_raw = auth_db.list_users()
        users = [{"username": u[0], "role": u[1]} for u in users_raw]
        active_sessions = session_db.get_all_active_sessions()
        
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "user_authenticated": True,
            "users": users,
            "active_sessions": active_sessions,
            "csrf_token": session_db.load_meta(f"csrf_{admin_username}"),
            "session_kill_error": True
        })
    
# Add this to main.py for quick debugging (remove after use)
@app.get("/debug/sessions")
async def debug_sessions():
    import sqlite3
    conn = sqlite3.connect("data/session_state.db")
    c = conn.cursor()
    c.execute("SELECT session_id, username, session_name, upload_paths, embedding_index_path FROM chat_sessions")
    rows = c.fetchall()
    conn.close()
    return {"sessions": rows}
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)