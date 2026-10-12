import logging
import os
import re
import time
from typing import Any, List, Optional

import httpx

logger = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
CACHE_SECONDS = 900
FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]

_cached_models: list = []
_cache_expires = 0.0


class GeminiServiceError(Exception):
    """Raised when Gemini cannot provide a response."""


def _score(model: dict) -> Optional[tuple]:
    name = str(model.get("name", "")).rsplit("/", 1)[-1]
    if "generateContent" not in (model.get("supportedGenerationMethods") or []):
        return None
    match = re.fullmatch(r"gemini-(\d+(?:\.\d+)*)-flash(-lite)?", name)
    if not match:
        return None
    version = tuple(int(p) for p in match.group(1).split("."))
    return version, (0 if match.group(2) else 1)


async def _discover_models(client: httpx.AsyncClient, api_key: str) -> list:
    global _cached_models, _cache_expires
    if _cached_models and time.monotonic() < _cache_expires:
        return list(_cached_models)

    try:
        resp = await client.get(
            f"{API_ROOT}/models",
            headers={"x-goog-api-key": api_key},
            params={"pageSize": 200},
        )
        resp.raise_for_status()
        found = []
        for model in resp.json().get("models", []):
            score = _score(model)
            if score:
                found.append((score, str(model["name"]).rsplit("/", 1)[-1]))
        found.sort(key=lambda item: item[0], reverse=True)
        names = [name for _, name in found]
    except Exception as exc:
        logger.warning("Gemini model discovery failed: %s", exc)
        names = []

    preferred = os.getenv("GEMINI_MODEL", "").strip()
    if preferred:
        names = [preferred] + [n for n in names if n != preferred]
    if not names:
        names = list(FALLBACK_MODELS)

    _cached_models = names
    _cache_expires = time.monotonic() + CACHE_SECONDS
    return list(names)


async def generate_gemini_reply(
    prompt: str,
    history: Optional[List[dict]] = None,
    language: str = "English",
) -> str:
    global _cached_models

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiServiceError(
            "AI chat is not configured. Add GEMINI_API_KEY on the server."
        )

    contents: list = []
    for msg in (history or [])[-6:]:
        text = (msg.get("text") or "").strip()
        if text:
            role = "model" if msg.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": text}]})
    contents.append(
        {"role": "user", "parts": [{"text": f"{prompt}\n\nRespond in {language}."}]}
    )

    last_error = "The AI is busy right now. Please try again in a moment."
    timeout = httpx.Timeout(20.0, connect=8.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        models = await _discover_models(client, api_key)

        for model in models[:2]:
            try:
                resp = await client.post(
                    f"{API_ROOT}/models/{model}:generateContent",
                    headers={"x-goog-api-key": api_key},
                    json={"contents": contents},
                )
            except httpx.TimeoutException:
                last_error = "The AI took too long to answer. Please try again."
                continue
            except httpx.HTTPError as exc:
                logger.error("Gemini network error: %s", exc)
                last_error = "Could not reach the AI service. Please try again."
                continue

            if resp.status_code == 404:
                _cached_models = []  # model was retired; rediscover next time
                logger.warning("Gemini model %s not found", model)
                continue
            if resp.status_code in (429, 500, 503):
                last_error = "The AI is busy right now. Please try again in a moment."
                continue
            if resp.status_code in (400, 401, 403):
                logger.error("Gemini rejected the request: %s", resp.text[:300])
                raise GeminiServiceError(
                    "The AI key is invalid or not allowed. Check GEMINI_API_KEY."
                )
            if not resp.is_success:
                logger.error("Gemini error %s: %s", resp.status_code, resp.text[:300])
                continue

            try:
                parts = resp.json()["candidates"][0]["content"]["parts"]
                answer = "\n".join(
                    p["text"] for p in parts if isinstance(p.get("text"), str)
                ).strip()
            except (KeyError, IndexError, TypeError, ValueError):
                answer = ""
            if answer:
                return answer
            last_error = "The AI returned an empty answer. Please rephrase."

    raise GeminiServiceError(last_error)
