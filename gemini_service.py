import logging
import os
from typing import List, Optional
import httpx

logger = logging.getLogger(__name__)

GEMINI_API_ROOT = "   GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1"


class GeminiServiceError(Exception):
    """Raised when Gemini cannot provide a response."""


async def generate_gemini_reply(
    prompt: str, history: Optional[List[dict]] = None, language: str = "English"
) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiServiceError("AI chat is not configured.")

    # STRICT TIMEOUT: 15 seconds max to prevent 2-4 minute waits
    timeout = httpx.Timeout(15.0, connect=5.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        # USE THE CORRECT, FAST MODEL
        model = "gemini-1.5-flash"

        # Only send last 5 messages to save processing time
        contents = []
        if history:
            for msg in history[-5:]:
                role = "model" if msg.get("role") == "assistant" else "user"
                contents.append(
                    {"role": role, "parts": [{"text": msg.get("text", "")}]}
                )

        final_prompt = f"{prompt}\n\nIMPORTANT: Respond entirely in {language}. Be brief and helpful."
        contents.append({"role": "user", "parts": [{"text": final_prompt}]})

        payload = {"contents": contents}

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
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", "Unknown error")
            logger.error(f"Gemini API Error: {error_msg}")

            if response.status_code == 429:
                raise GeminiServiceError("AI rate limit reached. Try again later.")
            if response.status_code in [400, 401, 403]:
                raise GeminiServiceError("Invalid AI API key.")
            raise GeminiServiceError(f"AI error: {error_msg}")

        try:
            response_data = response.json()
            parts = response_data["candidates"][0]["content"]["parts"]
            answer = "\n".join(
                part["text"] for part in parts if isinstance(part.get("text"), str)
            ).strip()
        except (KeyError, IndexError, TypeError):
            answer = ""

        if answer:
            return answer

        raise GeminiServiceError("AI returned no text. Try rephrasing.")
