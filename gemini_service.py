import asyncio
import logging
import os
import re
import time
from typing import Any, List, Optional

import httpx

logger = logging.getLogger(__name__)

GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
MODEL_CACHE_SECONDS = 900

_cached_models: list[str] = []
_model_cache_expires_at = 0.0


class GeminiServiceError(Exception):
    """Raised when Gemini cannot provide a response."""


def _flash_model_score(model: dict[str, Any]) -> tuple[tuple[int, ...], int] | None:
    model_name = str(model.get("name", "")).rsplit("/", 1)[-1]
    supported_methods = model.get("supportedGenerationMethods") or []
    if "generateContent" not in supported_methods:
        return None

    match = re.fullmatch(r"gemini-(\d+(?:\.\d+)*)-flash(-lite)?", model_name)
    if not match:
        return None

    version = tuple(int(part) for part in match.group(1).split("."))
    standard_flash_preference = 0 if match.group(2) else 1
    return version, standard_flash_preference


async def _discover_flash_models(
    client: httpx.AsyncClient,
    api_key: str,
    *,
    force_refresh: bool = False,
) -> list[str]:
    global _cached_models, _model_cache_expires_at

    now = time.monotonic()
    if _cached_models and now < _model_cache_expires_at and not force_refresh:
        return _cached_models.copy()

    models: list[dict[str, Any]] = []
    page_token: str | None = None
    for _ in range(5):
        params: dict[str, str | int] = {"pageSize": 100}
        if page_token:
            params["pageToken"] = page_token

        response = await client.get(
            f"{GEMINI_API_ROOT}/models",
            headers={"x-goog-api-key": api_key},
            params=params,
        )
        if not response.is_success:
            logger.warning(
                "Gemini model discovery failed with HTTP %s",
                response.status_code,
            )
            raise GeminiServiceError(
                "The AI service could not load its available models."
            )

        catalog = response.json()
        models.extend(catalog.get("models", []))
        page_token = catalog.get("nextPageToken")
        if not page_token:
            break

    candidates = [
        (score, str(model["name"]).rsplit("/", 1)[-1])
        for model in models
        if (score := _flash_model_score(model)) is not None
    ]
    if not candidates:
        raise GeminiServiceError(
            "No supported Gemini Flash model is available for this API key."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    requested_model = os.getenv("GEMINI_MODEL", "").strip()
    available_names = {name for _, name in candidates}
    if requested_model and requested_model in available_names:
        ordered_models = [requested_model] + [
            name for _, name in candidates if name != requested_model
        ]
    else:
        if requested_model:
            logger.info(
                "Configured Gemini model is unavailable; selecting from live catalog."
            )
        ordered_models = [name for _, name in candidates]

    _cached_models = ordered_models
    _model_cache_expires_at = time.monotonic() + MODEL_CACHE_SECONDS
    logger.info("Discovered supported Gemini Flash models from the live catalog.")
    return ordered_models.copy()


def _model_was_removed(status_code: int, error_data: dict[str, Any]) -> bool:
    if status_code == 404:
        return True
    message = str(error_data.get("error", {}).get("message", "")).lower()
    return status_code == 400 and ("not found" in message or "not supported" in message)


async def generate_gemini_reply(
    prompt: str, history: Optional[List[dict]] = None, language: str = "English"
) -> str:
    global _cached_models, _model_cache_expires_at

    # SECURE: Reads the key from Replit Secrets / Environment Variables ONLY
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiServiceError(
            "AI chat is not configured yet. Add GEMINI_API_KEY to Replit Secrets."
        )

    timeout = httpx.Timeout(35.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        models = await _discover_flash_models(client, api_key)
        max_models_to_try = min(len(models), 3)
        model_index = 0
        retries_for_model = 0
        refreshed_catalog = False

        while model_index < max_models_to_try:
            model = models[model_index]

            contents = []
            if history:
                for msg in history:
                    role = "model" if msg.get("role") == "assistant" else "user"
                    contents.append(
                        {"role": role, "parts": [{"text": msg.get("text", "")}]}
                    )

            final_prompt = f"{prompt}\n\nIMPORTANT INSTRUCTION: You must respond entirely in {language}. Do not use any other language."
            contents.append({"role": "user", "parts": [{"text": final_prompt}]})

            payload = {"contents": contents}

            response = await client.post(
                f"{GEMINI_API_ROOT}/models/{model}:generateContent",
                headers={"x-goog-api-key": api_key},
                json=payload,
            )

            try:
                response_data = response.json()
            except ValueError:
                response_data = {}

            if not response.is_success:
                if not refreshed_catalog and _model_was_removed(
                    response.status_code,
                    response_data,
                ):
                    _cached_models = []
                    _model_cache_expires_at = 0.0
                    refreshed_models = await _discover_flash_models(
                        client,
                        api_key,
                        force_refresh=True,
                    )
                    refreshed_catalog = True
                    if refreshed_models != models:
                        models = refreshed_models
                        max_models_to_try = min(len(models), 3)
                        model_index = 0
                        retries_for_model = 0
                        continue

                if response.status_code == 503 and retries_for_model == 0:
                    retries_for_model += 1
                    await asyncio.sleep(0.8)
                    continue

                if (
                    response.status_code in {429, 503}
                    and model_index + 1 < max_models_to_try
                ):
                    logger.info(
                        "Gemini model is temporarily unavailable; trying another supported Flash model."
                    )
                    model_index += 1
                    retries_for_model = 0
                    continue

                logger.warning(
                    "Gemini content request failed with HTTP %s",
                    response.status_code,
                )
                if response.status_code == 503:
                    raise GeminiServiceError(
                        "Gemini is temporarily busy. Please try again shortly."
                    )
                if response.status_code == 429:
                    raise GeminiServiceError(
                        "The AI request limit has been reached. Please try again later."
                    )
                raise GeminiServiceError(
                    "The AI service could not complete this request. Please try again."
                )

            try:
                parts = response_data["candidates"][0]["content"]["parts"]
                answer = "\n".join(
                    part["text"] for part in parts if isinstance(part.get("text"), str)
                ).strip()
            except (KeyError, IndexError, TypeError):
                answer = ""

            if answer:
                return answer
            raise GeminiServiceError(
                "The AI service returned no text. Please try rephrasing your message."
            )

    raise GeminiServiceError("Gemini is temporarily busy. Please try again shortly.")
