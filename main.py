import hmac
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.sessions import SessionMiddleware

from database import AsyncSessionLocal, get_db, init_db
from gemini_service import GeminiServiceError, generate_gemini_reply
from models import Client, SiteMap, Website

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Web Agent SaaS Backend", version="1.1")
PROJECT_DIR = Path(__file__).parent

SESSION_SECRET = os.getenv("SESSION_SECRET", "")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET or "not-configured",
    session_cookie="admin_session",
    max_age=8 * 60 * 60,
    same_site="lax",
)


@app.middleware("http")
async def framing_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/widget"):
        response.headers["Content-Security-Policy"] = "frame-ancestors *"
    else:
        response.headers["Content-Security-Policy"] = "frame-ancestors 'self'"
    return response


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


class SiteMapUpdate(BaseModel):
    content: str


async def require_admin(request: Request) -> bool:
    if request.session.get("admin_authenticated") is not True:
        raise HTTPException(status_code=401, detail="Admin sign-in required.")
    return True


# ---------------------------------------------------------------- scraper
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}
MAX_PAGES = 5
MAX_MAP_CHARS = 14000
SKIP_EXT = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".pdf",
    ".zip",
    ".css",
    ".js",
    ".ico",
    ".mp4",
)


def extract_page(html: str, url: str):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    meta = soup.find("meta", attrs={"name": "description"})
    description = (meta.get("content") or "").strip() if meta else ""
    headings = [
        h.get_text(" ", strip=True)
        for h in soup.find_all(["h1", "h2", "h3"])
        if h.get_text(strip=True)
    ][:25]
    text = " ".join(soup.get_text(" ", strip=True).split())[:3000]

    contacts = set()
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("tel:", "mailto:")):
            contacts.add(href.split(":", 1)[1].split("?")[0])
            continue
        links.append((a.get_text(" ", strip=True), urljoin(url, href).split("#")[0]))

    page = {
        "url": url,
        "title": title,
        "description": description,
        "headings": headings,
        "contacts": sorted(contacts),
        "text": text,
    }
    return page, links


async def build_site_map(url: str) -> str:
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=20.0, headers=HEADERS
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        home_url = str(resp.url)
        home, links = extract_page(resp.text, home_url)
        pages = [home]

        host = urlparse(home_url).netloc
        seen = {home_url.rstrip("/")}
        navigation = []

        for label, href in links:
            parsed = urlparse(href)
            if parsed.scheme not in ("http", "https") or parsed.netloc != host:
                continue
            clean = href.split("?")[0].rstrip("/")
            if clean.lower().endswith(SKIP_EXT):
                continue
            if label and len(navigation) < 25:
                navigation.append({"text": label, "href": clean})
            if clean in seen or len(pages) >= MAX_PAGES:
                continue
            seen.add(clean)
            try:
                page_resp = await client.get(clean)
                page_resp.raise_for_status()
                page, _ = extract_page(page_resp.text, clean)
                pages.append(page)
            except Exception as exc:
                logger.warning("Skipping page %s: %s", clean, exc)

    total_text = sum(len(p["text"]) for p in pages)
    if total_text < 100:
        raise ValueError(
            "The site returned almost no readable text (it may be rendered by JavaScript). "
            "Add the information manually with PUT /api/clients/{id}/site-map."
        )

    site_map = {"website": home_url, "pages": pages, "navigation": navigation}
    return json.dumps(site_map, ensure_ascii=False)[:MAX_MAP_CHARS]


async def save_site_map(client_id: int, content: Optional[str]) -> str:
    """Save a map (if given) and update the website status. Returns the status."""
    async with AsyncSessionLocal() as db:
        existing = (
            await db.execute(select(SiteMap).where(SiteMap.client_id == client_id))
        ).scalar_one_or_none()
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

        if content is not None:
            if existing:
                existing.content = content
                existing.updated_at = now
            else:
                db.add(SiteMap(client_id=client_id, content=content, updated_at=now))
            status = "ready"
        else:
            status = "ready" if existing and existing.content else "failed"

        websites = (
            (await db.execute(select(Website).where(Website.client_id == client_id)))
            .scalars()
            .all()
        )
        for website in websites:
            website.bot_status = status
        await db.commit()
        return status


async def scrape_website(client_id: int, url: str):
    logger.info("Scraping client %s: %s", client_id, url)
    try:
        content = await build_site_map(url)
        await save_site_map(client_id, content)
        logger.info("Saved site map for client %s (%s chars)", client_id, len(content))
    except Exception as exc:
        logger.error("Scrape failed for client %s: %s", client_id, exc)
        await save_site_map(client_id, None)


# ---------------------------------------------------------------- app routes
@app.on_event("startup")
async def startup():
    if not SESSION_SECRET or not ADMIN_USERNAME or not ADMIN_PASSWORD:
        raise RuntimeError("Set SESSION_SECRET, ADMIN_USERNAME and ADMIN_PASSWORD.")
    await init_db()
    logger.info("Database initialized.")


@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    return PlainTextResponse("ok")


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
    user_ok = hmac.compare_digest(payload.username, ADMIN_USERNAME)
    pass_ok = hmac.compare_digest(payload.password, ADMIN_PASSWORD)
    if not (user_ok and pass_ok):
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
        iframe.style.cssText = 'position:fixed;bottom:0;right:0;width:400px;height:600px;border:none;background:transparent;z-index:999999;';
        document.body.appendChild(iframe);
    })();
    """
    return Response(content=js_code, media_type="application/javascript")


# ---------------------------------------------------------------- admin API
@app.post("/api/onboard")
async def onboard_client(
    request: ClientOnboardRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_admin),
):
    try:
        api_key = f"sk-saas-{uuid.uuid4().hex}"
        client = Client(name=request.name, email=request.email, api_key=api_key)
        db.add(client)
        await db.commit()
        await db.refresh(client)

        website = Website(
            client_id=client.id, url=request.website_url, bot_status="scraping"
        )
        db.add(website)
        await db.commit()

        background_tasks.add_task(scrape_website, client.id, request.website_url)

        return {
            "id": client.id,
            "name": client.name,
            "email": client.email,
            "api_key": api_key,
            "website_url": request.website_url,
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
    clients = (
        (await db.execute(select(Client).order_by(Client.id.desc()))).scalars().all()
    )
    response = []
    for client in clients:
        website = (
            (await db.execute(select(Website).where(Website.client_id == client.id)))
            .scalars()
            .first()
        )
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


@app.get("/api/clients/{client_id}/site-map")
async def view_site_map(
    client_id: int, db: AsyncSession = Depends(get_db), _: bool = Depends(require_admin)
):
    row = (
        await db.execute(select(SiteMap).where(SiteMap.client_id == client_id))
    ).scalar_one_or_none()
    if not row:
        raise HTTPException(
            status_code=404, detail="No site map saved for this client yet."
        )
    return {
        "client_id": client_id,
        "updated_at": row.updated_at,
        "content": row.content,
    }


@app.put("/api/clients/{client_id}/site-map")
async def set_site_map(
    client_id: int,
    payload: SiteMapUpdate,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_admin),
):
    if not (
        await db.execute(select(Client).where(Client.id == client_id))
    ).scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Client not found.")
    status = await save_site_map(client_id, payload.content[:MAX_MAP_CHARS])
    return {"client_id": client_id, "bot_status": status}


@app.post("/api/clients/{client_id}/rescan")
async def rescan_client(
    client_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_admin),
):
    website = (
        (await db.execute(select(Website).where(Website.client_id == client_id)))
        .scalars()
        .first()
    )
    if not website:
        raise HTTPException(status_code=404, detail="Client not found.")
    website.bot_status = "scraping"
    await db.commit()
    background_tasks.add_task(scrape_website, client_id, website.url)
    return {"client_id": client_id, "bot_status": "scraping"}


# ---------------------------------------------------------------- chat
@app.post("/api/chat")
async def chat_with_agent(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    client = (
        await db.execute(select(Client).where(Client.api_key == request.api_key))
    ).scalar_one_or_none()
    if client is None:
        return {"reply": "Invalid API key."}

    row = (
        await db.execute(select(SiteMap).where(SiteMap.client_id == client.id))
    ).scalar_one_or_none()
    site_info = (
        row.content
        if row and row.content
        else "No website information has been added yet."
    )

    history = request.history or []
    # The widget already includes the current question as the last history item.
    if history and history[-1].get("role") == "user":
        history = history[:-1]

    prompt = (
        f"You are the website assistant for {client.name}.\n"
        "Answer the visitor's question using ONLY the website information below. "
        "If the answer is not there, say you are not sure and suggest contacting the business. "
        "Keep answers short and friendly.\n\n"
        f"WEBSITE INFORMATION:\n{site_info}\n\n"
        f"VISITOR QUESTION: {request.message}"
    )

    try:
        reply = await generate_gemini_reply(
            prompt, history=history, language=request.language
        )
    except GeminiServiceError as exc:
        return {"reply": str(exc)}
    except Exception:
        logger.exception("Unexpected error while generating a chat response.")
        return {"reply": "The AI assistant is temporarily unavailable."}

    try:
        client.message_count = (client.message_count or 0) + 1
        client.last_active = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        await db.commit()
    except Exception:
        await db.rollback()
        logger.warning("Could not update usage stats.")

    return {"reply": reply}
