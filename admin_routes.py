from fastapi import Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import secrets
import logging
import sqlite3
import auth_db
import session_db

templates = Jinja2Templates(directory="templates")

def get_username_from_token(request: Request):
    from main import decode_access_token
    token = request.cookies.get("access_token")
    payload = decode_access_token(token) if token else None
    if payload and payload.get("type") != "refresh":
        return payload["sub"]
    return None

def register_admin_routes(app):
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
        from main import create_access_token
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
        username = get_username_from_token(request)
        if not username or not auth_db.validate_admin(username):
            return RedirectResponse(url="/admin/login", status_code=302)
        return RedirectResponse(url="/admin", status_code=302)

    @app.get("/admin/delete_user")
    async def admin_delete_user_get(request: Request):
        username = get_username_from_token(request)
        if not username or not auth_db.validate_admin(username):
            return RedirectResponse(url="/admin/login", status_code=302)
        return RedirectResponse(url="/admin", status_code=302)

    @app.get("/admin/edit_user")
    async def admin_edit_user_get(request: Request):
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
            session_db.cleanup_inactive_sessions()
            active_sessions = session_db.get_all_active_sessions()
            logging.info(f"Admin {admin_username} refreshed active sessions - found {len(active_sessions)} sessions")
            return {"success": True, "session_count": len(active_sessions)}
            
        except Exception as e:
            logging.error(f"Error refreshing sessions for admin {admin_username}: {e}")
            raise HTTPException(status_code=500, detail="Failed to refresh sessions")

    @app.get("/admin/kill_session")
    async def admin_kill_session_get(request: Request):
        username = get_username_from_token(request)
        if not username or not auth_db.validate_admin(username):
            return RedirectResponse(url="/admin/login", status_code=302)
        return RedirectResponse(url="/admin", status_code=302)

    @app.post("/admin/add_user")
    async def add_user(request: Request, username: str = Form(...), password: str = Form(...), role: str = Form(...), csrf_token: str = Form(...)):
        from main import STRONG_PASSWORD
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