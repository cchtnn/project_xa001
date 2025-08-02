from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from vectorstore_manager import initialize_vectorstore, get_collection
import uvicorn
import os
import shutil
import auth_db, session_db, session_manager, student_transcript_csv_handler, query_handler, logic, config
import jwt
from datetime import datetime, timedelta
import re

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# OAuth2 for JWT
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = "chetan@123"  # Change to a secure random key!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
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
    return payload["sub"] if payload and "sub" in payload else None

def generate_session_name(question):
    """Generate a meaningful session name based on the first question"""
    # Clean the question
    question = question.strip()
    
    # Remove common question words and clean up
    question = re.sub(r'^(what|how|when|where|why|who|can you|tell me|explain|describe)\s+', '', question.lower())
    question = re.sub(r'\?+$', '', question)  # Remove question marks
    question = re.sub(r'[^\w\s-]', '', question)  # Remove special characters except hyphens
    
    # Extract key terms and create a meaningful name
    words = question.split()
    
    # Filter out common words
    stop_words = {'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about', 'me', 'you', 'your', 'my'}
    meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Take first 3-4 meaningful words and capitalize them
    if meaningful_words:
        session_name = ' '.join(meaningful_words[:4])
        session_name = ' '.join(word.capitalize() for word in session_name.split())
        
        # Limit length to 30 characters
        if len(session_name) > 30:
            session_name = session_name[:27] + "..."
        
        return session_name
    else:
        # Fallback to first few words of the original question
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
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = auth_db.validate_user(form_data.username, form_data.password)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid credentials"}
        )
    token = create_access_token({"sub": user["username"]})
    response = RedirectResponse(url="/chat", status_code=302)
    response.set_cookie(key="access_token", value=token, httponly=True)
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
    return response

# --- Chat Session Endpoints ---
@app.post("/new_session")
async def new_session(request: Request):
    username = get_username_from_token(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    session_id = session_db.create_new_session(username, "New Chat")  # Start with default name
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
async def upload_pdf(request: Request, file: UploadFile = File(...), private: bool = Form(...)):
    username = get_username_from_token(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    upload_folder = f"data/user_uploads/{username}" if private else "data/public_uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logic.parse_and_index_pdf(file_path, user=username, private=private)
    return {"success": True, "message": "File uploaded and processed."}

# --- Query Endpoint ---
@app.post("/query")
async def query(request: Request, query: str = Form(...), session_id: int = Form(...)):
    username = get_username_from_token(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        index, metadata, tab_data = initialize_vectorstore()
        collection = get_collection()
        data_loading_error = None
    except Exception as e:
        data_loading_error = str(e)
        print(f"❌ Error loading data: {e}")
        index, metadata, tab_data = [], [], {}
        collection = None

    handler = query_handler.create_query_handler(collection, tab_data)
    answer_obj, query_type, confidence_score = handler.process_query(query, language="English")

    # --- Save Q&A to history ---
    try:
        history = session_db.load_qa_history(session_id)
        
        # If this is the first question in the session, auto-rename it
        if len(history) == 0:
            session_name = generate_session_name(query)
            session_db.rename_session(session_id, session_name)
        
        history.append({"question": query, "answer": answer_obj.content})
        session_db.save_qa_history(session_id, history)
    except Exception as e:
        print(f"Error saving to history: {e}")
        # Continue without saving history if there's an error

    return {"answer": answer_obj.content, "session_id": session_id}

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
        print(f"Error loading history: {e}")
        return {"history": []}

# --- Admin Endpoints ---
@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, access_token: str = Cookie(None)):
    if access_token != "FAKE_ADMIN_TOKEN":
        return RedirectResponse(url="/admin/login", status_code=302)
    users_raw = auth_db.list_users()
    users = [{"username": u[0], "role": u[1]} for u in users_raw]
    return templates.TemplateResponse("admin.html", {"request": request, "user_authenticated": True, "users": users})

@app.post("/admin/add_user")
async def add_user(request: Request, username: str = Form(...), password: str = Form(...), role: str = Form(...)):
    auth_db.add_user(username, password, role, created_by="admin")
    return RedirectResponse(url="/admin", status_code=302)

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
    response = RedirectResponse(url="/admin", status_code=302)
    response.set_cookie(key="access_token", value="FAKE_ADMIN_TOKEN", httponly=True)
    return response

@app.post("/admin/edit_user")
async def edit_user(request: Request, original_username: str = Form(...), username: str = Form(...), role: str = Form(...)):
    auth_db.update_user(original_username, username, None, role)
    return RedirectResponse(url="/admin", status_code=302)

@app.post("/admin/delete_user")
async def delete_user(request: Request, username: str = Form(...)):
    auth_db.delete_user(username)
    return RedirectResponse(url="/admin", status_code=302)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)