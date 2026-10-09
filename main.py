import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from bs4 import BeautifulSoup
import httpx

from database import get_db, init_db
from gemini_service import GeminiServiceError, generate_gemini_reply
from models import Client, Website

logger = logging.getLogger(__name__)
app = FastAPI(title="AI Web Agent SaaS Backend", version="1.0")

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- EMBEDDING HEADERS ---
@app.middleware("http")
async def add_embedding_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = "frame-ancestors *"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


PROJECT_DIR = Path(__file__).parent

# SECURE ENVIRONMENT VARIABLES (No hardcoded secrets)
SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret-change-in-prod")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="admin_session",
    max_age=8 * 60 * 60,
    same_site="lax",
)


class ClientOnboardRequest(BaseModel):
    name: str
    email: str
    website_url: str


class ChatRequest(BaseModel):
    api_key: str
    message: str
    history: Optional[List[dict]] = []
    language: str = "English"


class AdminLoginRequest(BaseModel):
    username: str
    password: str


async def require_admin(request: Request) -> bool:
    if request.session.get("admin_authenticated") is not True:
        raise HTTPException(status_code=401, detail="Admin sign-in required.")
    return True


async def scrape_website(client_id: int, url: str):
    logger.info(f"Starting background scrape for client {client_id}: {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string if soup.title else "No title"
        headings = [
            h.get_text(strip=True)
            for h in soup.find_all(["h1", "h2", "h3"])
            if h.get_text(strip=True)
        ]
        paragraphs = [
            p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)
        ][:10]
        links = [
            {"text": a.get_text(strip=True), "href": a.get("href")}
            for a in soup.find_all("a", href=True)
            if a.get_text(strip=True)
        ][:15]

        site_map = {
            "url": url,
            "title": title,
            "headings": headings,
            "summary_paragraphs": paragraphs,
            "navigation_links": links,
        }

        map_path = PROJECT_DIR / f"client_{client_id}_map.json"
        map_path.write_text(json.dumps(site_map, indent=2), encoding="utf-8")
        logger.info(f"Successfully saved map for client {client_id}")

    except Exception as e:
        logger.error(f"Failed to scrape website for client {client_id}: {e}")


@app.on_event("startup")
async def startup():
    await init_db()
    logger.info("Database initialized.")


# --- NEW HEALTH CHECK ENDPOINT (Fixes the crashing/405 error) ---
@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    """A lightweight endpoint that accepts HEAD requests to prevent crashes."""
    return PlainTextResponse("ok", status_code=200)


@app.get("/login", response_class=HTMLResponse)
async def read_login(request: Request):
    if request.session.get("admin_authenticated") is True:
        return RedirectResponse(url="/", status_code=303)
    login_path = PROJECT_DIR / "login.html"
    if not login_path.is_file():
        raise HTTPException(status_code=404, detail="Login page not found.")
    return HTMLResponse(content=login_path.read_text(encoding="utf-8"))


@app.post("/api/auth/login")
async def admin_login(payload: AdminLoginRequest, request: Request):
    if payload.username != ADMIN_USERNAME or payload.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    request.session.clear()
    request.session["admin_authenticated"] = True
    return {"authenticated": True}


@app.post("/api/auth/logout")
async def admin_logout(request: Request):
    request.session.clear()
    return {"authenticated": False}


@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    if request.session.get("admin_authenticated") is not True:
        return RedirectResponse(url="/login", status_code=303)
    dashboard_path = PROJECT_DIR / "dashboard.html"
    if not dashboard_path.is_file():
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))


@app.get("/widget.html", response_class=HTMLResponse)
async def read_widget():
    widget_path = PROJECT_DIR / "widget.html"
    if not widget_path.is_file():
        raise HTTPException(status_code=404, detail="Widget not found.")
    return HTMLResponse(content=widget_path.read_text(encoding="utf-8"))


@app.get("/widget-loader.js")
async def get_widget_loader():
    js_code = """
    (function() {
        const urlParams = new URLSearchParams(window.location.search);
        const API_KEY = urlParams.get('api_key');
        if (!API_KEY) { console.error('AI Web Agent: No API key provided.'); return; }
        const scriptSrc = document.currentScript ? document.currentScript.src : '';
        const backendUrl = scriptSrc.split('/widget-loader.js')[0];
        const iframe = document.createElement('iframe');
        iframe.src = backendUrl + '/widget.html?api_key=' + API_KEY;
        iframe.style.cssText = 'position:fixed;bottom:20px;right:20px;width:350px;height:500px;border:none;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,0.2);z-index:999999;background:transparent;';
        iframe.setAttribute('allow', 'clipboard-read; clipboard-write');
        document.body.appendChild(iframe);
        console.log('AI Web Agent loaded for client:', API_KEY.substring(0, 10) + '...');
    })();
    """
    return Response(content=js_code, media_type="application/javascript")


@app.post("/api/onboard")
async def onboard_client(
    request: ClientOnboardRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_admin),
):
    try:
        new_api_key = f"sk-saas-{uuid.uuid4().hex}"
        new_client = Client(name=request.name, email=request.email, api_key=new_api_key)
        db.add(new_client)
        await db.commit()
        await db.refresh(new_client)

        new_website = Website(
            client_id=new_client.id, url=request.website_url, bot_status="scraping"
        )
        db.add(new_website)
        await db.commit()

        background_tasks.add_task(scrape_website, new_client.id, request.website_url)

        return {
            "id": new_client.id,
            "name": new_client.name,
            "email": new_client.email,
            "api_key": new_api_key,
            "website_url": new_website.url,
            "bot_status": "scraping",
        }
    except Exception as exc:
        await db.rollback()
        logger.exception("Client onboarding failed.")
        raise HTTPException(
            status_code=500, detail="Unable to onboard client."
        ) from exc


@app.get("/api/clients")
async def get_clients(
    db: AsyncSession = Depends(get_db), _: bool = Depends(require_admin)
):
    result = await db.execute(select(Client).order_by(Client.id.desc()))
    clients = result.scalars().all()
    response = []
    for client in clients:
        website_result = await db.execute(
            select(Website).where(Website.client_id == client.id)
        )
        website = website_result.scalar_one_or_none()
        response.append(
            {
                "id": client.id,
                "name": client.name,
                "email": client.email,
                "api_key": client.api_key,
                "bot_status": website.bot_status if website else "pending",
                "website_url": website.url if website else "",
                "message_count": client.message_count,
                "last_active": client.last_active,
            }
        )
    return response


@app.post("/api/chat")
async def chat_with_agent(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Client).where(Client.api_key == request.api_key))
    client = result.scalar_one_or_none()

    if client is None:
        return {"reply": "Invalid API key."}

    site_map = "No map available yet."
    map_path = PROJECT_DIR / f"client_{client.id}_map.json"
    if map_path.is_file():
        try:
            site_map = json.loads(map_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Could not read the stored website map for client %s.", client.id
            )

    prompt = (
        f"You are a helpful AI assistant for {client.name}.\n"
        f"Website map: {site_map}\n"
        f"User asked: {request.message}\n"
        "Answer briefly and helpfully based ONLY on the website map provided."
    )

    try:
        reply = await generate_gemini_reply(
            prompt, history=request.history, language=request.language
        )

        client.message_count = (client.message_count or 0) + 1
        client.last_active = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        await db.commit()

        return {"reply": reply}
    except GeminiServiceError as exc:
        return {"reply": str(exc)}
    except Exception:
        logger.exception("Unexpected error while generating a chat response.")
        return {"reply": "The AI assistant is temporarily unavailable."}
