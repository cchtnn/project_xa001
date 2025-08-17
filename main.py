from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from vectorstore_manager import initialize_vectorstore, get_collection
from conversation_graph import create_conversation_graph
import uvicorn
import os
import shutil
import auth_db
import session_db, session_manager, student_transcript_csv_handler, query_handler, logic, config
import jwt
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
import secrets
import logging
from typing import List

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
    session_db.delete_session(session_id)
    return {"success": True}

# --- File Upload Endpoint ---
@app.post("/upload")
async def upload_files(request: Request, files: List[UploadFile] = File(...), private: bool = Form(...)):
    username = get_username_from_token(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    upload_folder = f"/secure_uploads/{username}" if private else "/secure_uploads/public"
    os.makedirs(upload_folder, exist_ok=True)
    
    processed_files = []
    errors = []
    
    # Determine output folders based on private flag
    if private:
        base_output_path = f"data/user_uploads/{username}"
        image_output_path = os.path.join(base_output_path, "extracted_images")
        csv_output_path = os.path.join(base_output_path, "csv_files")
    else:
        base_output_path = "data/public_uploads"
        image_output_path = os.path.join(base_output_path, "extracted_images")
        csv_output_path = os.path.join(base_output_path, "csv_files")
    
    # Create base directories
    os.makedirs(base_output_path, exist_ok=True)
    
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
    
    message = f"Processed {len(processed_files)} file(s) successfully."
    if errors:
        message += f" {len(errors)} error(s) occurred."
    
    return {
        "success": True, 
        "message": message,
        "processed_files": processed_files,
        "errors": errors
    }

# --- Query Endpoint ---
@app.post("/query")
async def query(request: Request, query: str = Form(...), session_id: int = Form(...)):
    username = get_username_from_token(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    query = query.strip()
    if not query or len(query) > 500:
        raise HTTPException(status_code=400, detail="Invalid query length")
    if re.search(r'[<>{};]', query):
        raise HTTPException(status_code=400, detail="Invalid characters in query")
    
    logging.info(f"User {username} queried: {query} (session_id: {session_id})")
    
    try:
        # Initialize vectorstore and data
        index, metadata, tab_data = initialize_vectorstore()
        collection = get_collection()
        data_loading_error = None
    except Exception as e:
        data_loading_error = str(e)
        logging.error(f"Error loading data for user {username}: {e}")
        index, metadata, tab_data = [], [], {}
        collection = None
    
    if len(query.split()) > 100:
        raise HTTPException(status_code=400, detail="Query too complex")
    
    try:
        # Get conversation history for context
        chat_history = session_db.get_contextual_history(session_id, limit=5)
        
        # Create user context
        user_context = {
            "username": username,
            "session_id": session_id,
            "language": "English",  # You can make this dynamic
            "active_transcript_csv_path": None  # Add logic to set this if needed
        }
        
        # Create and use conversation graph
        conv_graph = create_conversation_graph(collection, tab_data)
        result = conv_graph.process_conversation(query, chat_history, user_context)
        
        # Extract response and metadata
        response_content = result["response"]
        query_type = result["query_type"]
        confidence_score = result["confidence_score"]
        contextual_query = result.get("contextual_query", query)
        entities = result.get("entities", {})
        
        # Save conversation context for future use
        session_db.save_conversation_context(session_id, entities, contextual_query)
        
        # Handle session naming for new sessions
        if len(chat_history) == 0:
            session_name = generate_session_name(query)
            session_db.rename_session(session_id, session_name)
        
        # Save Q&A to history
        success = session_db.add_single_qa_to_history(session_id, query, response_content)
        if not success:
            # Fallback to old method if new method fails
            history = session_db.load_qa_history(session_id)
            history.append({"question": query, "answer": response_content})
            session_db.save_qa_history(session_id, history)
        
        print(f"Processed query with context - Type: {query_type}, Confidence: {confidence_score:.2f}, contextual: {len(chat_history)}")
        
        return {
            "answer": response_content, 
            "session_id": session_id,
            "query_type": query_type,
            "confidence_score": confidence_score,
            "contextual": len(chat_history) > 0  # Indicate if context was used
        }
        
    except Exception as e:
        print(f"Error processing contextual query for user {username}: {e}")
        
        # Fallback to original query handler
        try:
            handler = query_handler.create_query_handler(collection, tab_data)
            answer_obj, query_type, confidence_score = handler.process_query(query, language="English")
            
            # Save using original method
            history = session_db.load_qa_history(session_id)
            if len(history) == 0:
                session_name = generate_session_name(query)
                session_db.rename_session(session_id, session_name)
            history.append({"question": query, "answer": answer_obj.content})
            session_db.save_qa_history(session_id, history)
            
            return {"answer": answer_obj.content, "session_id": session_id}
            
        except Exception as fallback_error:
            logging.error(f"Fallback query processing failed for user {username}: {fallback_error}")
            return {
                "answer": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "session_id": session_id
            }

# --- History Endpoint ---
@app.get("/history")
async def history(request: Request, session_id: int):
    username = get_username_from_token(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        history = session_db.load_qa_history(session_id)
        return {"history": history}
    except Exception as e:
        logging.error(f"Error loading history for user {username}: {e}")
        return {"history": []}

# --- Admin Endpoints ---
@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, access_token: str = Cookie(None)):
    payload = decode_access_token(access_token)
    if not payload or not auth_db.validate_admin(payload["sub"]):
        return RedirectResponse(url="/admin/login", status_code=302)
    users_raw = auth_db.list_users()
    users = [{"username": u[0], "role": u[1]} for u in users_raw]
    active_sessions = session_db.get_all_active_sessions()
    csrf_token = secrets.token_urlsafe(32)
    session_db.save_meta(f"csrf_{payload['sub']}", csrf_token)
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "user_authenticated": True,
        "users": users,
        "active_sessions": active_sessions,
        "csrf_token": csrf_token
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
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "user_authenticated": True,
            "users": [{"username": u[0], "role": u[1]} for u in auth_db.list_users()],
            "csrf_token": session_db.load_meta(f"csrf_{admin_username}"),
            "add_success": True
        })
    except ValueError as e:
        if STRONG_PASSWORD:
            return templates.TemplateResponse("admin.html", {
                "request": request,
                "user_authenticated": True,
                "users": [{"username": u[0], "role": u[1]} for u in auth_db.list_users()],
                "csrf_token": session_db.load_meta(f"csrf_{admin_username}"),
                "error": str(e)
            })
        else:
            logging.warning(f"Admin {admin_username} added user {username} with weak password")
            auth_db.add_user(username, password, role, created_by=admin_username, bypass_password_validation=True)
            return templates.TemplateResponse("admin.html", {
                "request": request,
                "user_authenticated": True,
                "users": [{"username": u[0], "role": u[1]} for u in auth_db.list_users()],
                "csrf_token": session_db.load_meta(f"csrf_{admin_username}"),
                "add_success": True
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
            
            return templates.TemplateResponse("admin.html", {
                "request": request,
                "user_authenticated": True,
                "users": users,
                "active_sessions": active_sessions,
                "csrf_token": session_db.load_meta(f"csrf_{admin_username}"),
                "session_kill_success": True
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)