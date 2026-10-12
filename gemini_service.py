import logging
import os
import time
from typing import List, Optional
import httpx

logger = logging.getLogger(__name__)

GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

# Google retires Gemini models often, so this service tries several models and
# remembers the first one that works. Set GEMINI_MODEL to force one first.
FALLBACK_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
]

# ---------------------------------------------------------------------------
# FILL THIS IN: what the assistant knows about your business.
# (Or set it as the BUSINESS_INFO environment variable on Render.)
# ---------------------------------------------------------------------------
BUSINESS_INFO = (
    os.getenv("BUSINESS_INFO", "").strip()
    or """
Business: Mawolangalan Bites, a bakery in Kenya.
Website: https://mawolangalan-bites.vercel.app
What we make: artisan cookies, custom cakes, and premium baked goods. All
products are made with love and the finest ingredients in Kenya.
Website sections: Home, Products, Custom Cakes, Order Now.

HOW TO ORDER:
To add items to your cart, click the "Add to Cart" button on any product page.
You can view your cart by clicking the cart icon in the top right corner of
the website.

CUSTOM CAKES AND SPECIAL REQUESTS:
Contact us for custom cake orders and special requests on +254 784 437 428.

PRICES AND MENU DETAILS:
Product names and prices are listed on the Products page of the website.
If you are asked for a price or item you are not sure about, tell the customer
to check the Products page or call +254 784 437 428.

CONTACT:
Phone: +254 784 437 428
Location: Kenya
"""
)

SYSTEM_INSTRUCTION = (
    "You are the friendly assistant for the Mawolangalan Bites website. "
    "Answer using ONLY the business information below. Keep answers short. "
    "Write in plain text only: never use markdown, asterisks, or bold. "
    "If something is not covered, do not invent it: say you are not sure and "
    "suggest the customer use the Order Now button or the contact details.\n\n"
    f"BUSINESS INFORMATION:\n{BUSINESS_INFO}"
)

_working_model: Optional[str] = None


class GeminiServiceError(Exception):
    """Raised when Gemini cannot provide a response."""


class _TryNextModel(Exception):
    """This model is missing, overloaded, or too slow: try another one."""

    def __init__(self, model: str, reason: str):
        super().__init__(f"{model}: {reason}")
        self.reason = reason


def _candidate_models() -> List[str]:
    models = []
    if _working_model:
        models.append(_working_model)
    env_model = os.getenv("GEMINI_MODEL", "").strip()
    if env_model:
        models.append(env_model)
    models.extend(FALLBACK_MODELS)
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
    except (
        httpx.ReadTimeout,
        httpx.ConnectTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
    ):
        logger.warning(
            f"Gemini {model} timed out after {time.perf_counter() - started:.1f}s"
        )
        raise _TryNextModel(model, "timeout")
    except Exception as e:
        logger.error(f"Network error: {e}")
        raise GeminiServiceError("Network error occurred.")

    if not response.is_success:
        error_msg = _error_message(response)
        logger.error(
            f"Gemini API Error ({response.status_code}) on {model}: {error_msg}"
        )

        # Missing, overloaded or erroring model: move on to the next one.
        if response.status_code in (404, 500, 503):
            raise _TryNextModel(model, str(response.status_code))
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

    # The chat widget shows plain text, so strip any markdown bold markers.
    answer = answer.replace("**", "").replace("__", "").strip()

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

    # Per-attempt limit. A slow model is abandoned quickly and the next one is tried.
    timeout = httpx.Timeout(10.0, connect=5.0)

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
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": contents,
        "generationConfig": {
            "thinkingConfig": {"thinkingLevel": "minimal"},
            "maxOutputTokens": 400,
        },
    }

    timed_out = False
    async with httpx.AsyncClient(timeout=timeout) as client:
        tried = set()

        for model in _candidate_models():
            tried.add(model)
            try:
                answer = await _call_model(client, api_key, model, payload)
                _working_model = model
                return answer
            except _TryNextModel as e:
                timed_out = timed_out or e.reason == "timeout"
                continue

        # Every listed model failed with "not found": ask Google what exists now.
        if not timed_out:
            for model in await _discover_models(client, api_key):
                if model in tried:
                    continue
                try:
                    answer = await _call_model(client, api_key, model, payload)
                    _working_model = model
                    logger.warning(f"Using auto-discovered model: {model}")
                    return answer
                except _TryNextModel:
                    continue

    if timed_out:
        raise GeminiServiceError("The AI is taking too long. Please try again.")
    raise GeminiServiceError(
        "No available AI model found. Set GEMINI_MODEL to a current Gemini model."
    )
