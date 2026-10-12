import logging
import os
from typing import List, Optional
import httpx

logger = logging.getLogger(__name__)

GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

# gemini-1.5-flash was retired by Google. Override with the GEMINI_MODEL env var if needed.
DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiServiceError(Exception):
    """Raised when Gemini cannot provide a response."""


async def generate_gemini_reply(
    prompt: str, history: Optional[List[dict]] = None, language: str = "English"
) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiServiceError("AI chat is not configured.")

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    # STRICT TIMEOUT: 15 seconds max
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
        # Skip "thinking" so replies are fast enough for the 15s timeout
        "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}},
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
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
        try:
            error_msg = response.json().get("error", {}).get("message", "Unknown error")
        except ValueError:
            error_msg = response.text[:200] or "Unknown error"
        logger.error(f"Gemini API Error ({response.status_code}): {error_msg}")

        if response.status_code == 404:
            raise GeminiServiceError(
                f"AI model '{model}' is unavailable. Update GEMINI_MODEL."
            )
        if response.status_code == 429:
            raise GeminiServiceError("AI rate limit reached. Try again later.")
        if response.status_code in (401, 403):
            raise GeminiServiceError("Invalid AI API key.")
        raise GeminiServiceError(f"AI error: {error_msg}")

    try:
        response_data = response.json()
        parts = response_data["candidates"][0]["content"]["parts"]
        answer = "\n".join(
            part["text"] for part in parts if isinstance(part.get("text"), str)
        ).strip()
    except (KeyError, IndexError, TypeError, ValueError):
        answer = ""

    if answer:
        return answer

    raise GeminiServiceError("AI returned no text. Try rephrasing.")
