import logging
import os
import time
from typing import List, Optional
import httpx

logger = logging.getLogger(__name__)

GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

# Google retires Gemini models often (1.5 and 2.5 are already gone), so this
# service tries several models and remembers the first one that works.
# Set GEMINI_MODEL in your environment to force a specific model first.
# Flash-Lite is listed first because it is the fastest option for simple chat.
FALLBACK_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
]

_working_model: Optional[str] = None


class GeminiServiceError(Exception):
    """Raised when Gemini cannot provide a response."""


class _ModelNotFound(Exception):
    pass


def _candidate_models() -> List[str]:
    models = []
    if _working_model:
        models.append(_working_model)
    env_model = os.getenv("GEMINI_MODEL", "").strip()
    if env_model:
        models.append(env_model)
    models.extend(FALLBACK_MODELS)
    # de-duplicate, keep order
    return list(dict.fromkeys(models))


def _error_message(response: httpx.Response) -> str:
    try:
        return response.json().get("error", {}).get("message", "Unknown error")
    except ValueError:
        return response.text[:200] or "Unknown error"


async def _discover_models(client: httpx.AsyncClient, api_key: str) -> List[str]:
    """Ask Google which models this API key can actually use for generateContent."""
    try:
        resp = await client.get(
            f"{GEMINI_API_ROOT}/models",
            headers={"x-goog-api-key": api_key},
            params={"pageSize": 200},
        )
        if not resp.is_success:
            return []
        found = []
        for m in resp.json().get("models", []):
            name = m.get("name", "").replace("models/", "")
            methods = m.get("supportedGenerationMethods", [])
            if (
                "generateContent" in methods
                and "flash" in name
                and not any(
                    x in name for x in ("image", "live", "audio", "tts", "embedding")
                )
            ):
                found.append(name)
        # newest-looking names first (e.g. 3.5 before 3.1)
        return sorted(found, reverse=True)
    except Exception as e:
        logger.error(f"Model discovery failed: {e}")
        return []


async def _call_model(
    client: httpx.AsyncClient, api_key: str, model: str, payload: dict
) -> str:
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{GEMINI_API_ROOT}/models/{model}:generateContent",
            headers={"x-goog-api-key": api_key},
            json=payload,
        )
    except httpx.ReadTimeout:
        raise GeminiServiceError("The AI is taking too long. Please try again.")
    except httpx.ConnectTimeout:
        raise GeminiServiceError("Cannot connect to AI service.")
    except Exception as e:
        logger.error(f"Network error: {e}")
        raise GeminiServiceError("Network error occurred.")

    if not response.is_success:
        error_msg = _error_message(response)
        logger.error(
            f"Gemini API Error ({response.status_code}) on {model}: {error_msg}"
        )

        if response.status_code == 404:
            raise _ModelNotFound(model)
        # If a model rejects the thinking setting, retry once without it.
        if (
            response.status_code == 400
            and "thinking" in error_msg.lower()
            and "thinkingConfig" in payload.get("generationConfig", {})
        ):
            lean = {
                **payload,
                "generationConfig": {
                    k: v
                    for k, v in payload["generationConfig"].items()
                    if k != "thinkingConfig"
                },
            }
            return await _call_model(client, api_key, model, lean)
        if response.status_code == 429:
            raise GeminiServiceError("AI rate limit reached. Try again later.")
        if response.status_code in (401, 403) or "API key" in error_msg:
            raise GeminiServiceError("Invalid AI API key.")
        raise GeminiServiceError(f"AI error: {error_msg}")

    try:
        parts = response.json()["candidates"][0]["content"]["parts"]
        answer = "\n".join(
            part["text"] for part in parts if isinstance(part.get("text"), str)
        ).strip()
    except (KeyError, IndexError, TypeError, ValueError):
        answer = ""

    if not answer:
        raise GeminiServiceError("AI returned no text. Try rephrasing.")
    logger.info(f"Gemini {model} replied in {time.perf_counter() - started:.1f}s")
    return answer


async def generate_gemini_reply(
    prompt: str, history: Optional[List[dict]] = None, language: str = "English"
) -> str:
    global _working_model

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiServiceError("AI chat is not configured.")

    timeout = httpx.Timeout(15.0, connect=5.0)

    # Only send last 5 messages to save time
    contents = []
    if history:
        for msg in history[-5:]:
            role = "model" if msg.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg.get("text", "")}]})

    final_prompt = (
        f"{prompt}\n\nIMPORTANT: Respond entirely in {language}. Be brief and helpful."
    )
    contents.append({"role": "user", "parts": [{"text": final_prompt}]})
    payload = {
        "contents": contents,
        "generationConfig": {
            # "minimal" = least thinking, fastest replies (supported by all models above)
            "thinkingConfig": {"thinkingLevel": "minimal"},
            # Short answers finish sooner
            "maxOutputTokens": 400,
        },
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        tried = set()

        for model in _candidate_models():
            tried.add(model)
            try:
                answer = await _call_model(client, api_key, model, payload)
                _working_model = model
                return answer
            except _ModelNotFound:
                continue

        # Every hardcoded model is gone: ask Google what is available now.
        for model in await _discover_models(client, api_key):
            if model in tried:
                continue
            try:
                answer = await _call_model(client, api_key, model, payload)
                _working_model = model
                logger.warning(f"Using auto-discovered model: {model}")
                return answer
            except _ModelNotFound:
                continue

    raise GeminiServiceError(
        "No available AI model found. Set GEMINI_MODEL to a current Gemini model."
    )
