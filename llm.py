"""
llm.py — Google Gemini wrapper: chat + embeddings, with model detection and error handling.
(Builds on the client from Module 03.)
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import numpy as np
from google import genai
from google.genai import types

REQUEST_TIMEOUT_S = 60
FALLBACK_CHAT_MODELS = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-flash-lite-latest"]
FALLBACK_EMBED_MODELS = ["gemini-embedding-001", "text-embedding-004"]
EMBED_BATCH = 100  # max texts per embedding request


def get_api_key() -> str | None:
    """Streamlit secrets -> .env -> environment variable. Never hard-code keys."""
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("GEMINI_API_KEY")


# --------------------------------------------------------------------------- errors
class LLMError(Exception):
    def __init__(self, user_message: str, retryable: bool = False, detail: str = ""):
        super().__init__(user_message)
        self.user_message, self.retryable, self.detail = user_message, retryable, detail


def friendly_error(e: Exception) -> LLMError:
    if isinstance(e, LLMError) or hasattr(e, "user_message"):
        return e
    code, text, name = getattr(e, "code", None), str(e), type(e).__name__.lower()
    if code == 429 or "RESOURCE_EXHAUSTED" in text:
        err = LLMError("⏳ Free-tier rate limit reached. Wait about a minute and try again.", True)
    elif code in (401, 403) or "API_KEY_INVALID" in text or "API key not valid" in text:
        err = LLMError("🔑 The API key is invalid or not allowed. Check GEMINI_API_KEY.")
    elif code == 404:
        err = LLMError("🤖 This model isn't available for your key. Pick another model in the sidebar.")
    elif code == 400:
        err = LLMError(f"⚠️ The request was rejected (400): {text[:200]}")
    elif code is not None and code >= 500:
        err = LLMError("🛠️ Gemini servers are busy. Please try again in a moment.", True)
    elif "timeout" in name or "timed out" in text.lower() or "timeout" in text.lower():
        err = LLMError(f"⌛ Gemini didn't answer within {REQUEST_TIMEOUT_S}s. Try again or use a Flash-Lite model.")
    elif "connect" in name or "connection" in text.lower():
        err = LLMError("🌐 Network problem while contacting Gemini. Please try again.", True)
    else:
        err = LLMError(f"Unexpected error: {text[:200]}")
    err.detail = f"{type(e).__name__}: {text[:500]}"
    return err


def with_retries(fn, retries: int = 2):
    """Run fn(); retry temporary errors (429 / 5xx / network) with back-off."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            err = friendly_error(e)
            if not err.retryable or attempt == retries:
                raise err from e
            time.sleep(4 * (attempt + 1))


# --------------------------------------------------------------------------- usage
@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    def record(self, um) -> None:
        if um:
            self.input_tokens = um.prompt_token_count or 0
            self.output_tokens = (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)


def _version(name: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


# --------------------------------------------------------------------------- client
class Gemini:
    def __init__(self, api_key: str | None):
        if not api_key:
            raise LLMError("🔑 No API key found. Add GEMINI_API_KEY to Streamlit secrets or a .env file.")
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_S * 1000, retry_options=types.HttpRetryOptions(attempts=1)))
        self._one_by_one: set[str] = set()  # embedding models that merge lists into one vector

    # ---- model discovery (Google renames/retires models often)
    def _models_with(self, action: str) -> list[str]:
        out = []
        for m in self.client.models.list():
            if action in (getattr(m, "supported_actions", None) or []):
                out.append((m.name or "").removeprefix("models/"))
        return out

    def chat_models(self) -> list[str]:
        skip = ("image", "tts", "audio", "live", "embedding", "robotics", "computer-use", "veo", "imagen")
        names = [n for n in self._models_with("generateContent")
                 if n.startswith("gemini") and not any(s in n for s in skip)]

        def rank(n):
            tier = 0 if ("flash" in n and "lite" not in n) else 1 if "flash-lite" in n else 2
            return (tier, "preview" in n or "exp" in n, -_version(n), n)
        return sorted(set(names), key=rank) or FALLBACK_CHAT_MODELS

    def embedding_models(self) -> list[str]:
        names = self._models_with("embedContent")
        return sorted(set(names), key=lambda n: ("gemini" not in n, "exp" in n or "preview" in n,
                                                 -_version(n), n)) or FALLBACK_EMBED_MODELS

    # ---- embeddings
    def _embed_call(self, model, contents, task_type):
        try:
            return with_retries(lambda: self.client.models.embed_content(
                model=model, contents=contents, config=types.EmbedContentConfig(task_type=task_type)))
        except LLMError as e:
            if "400" not in e.user_message:
                raise
            return with_retries(lambda: self.client.models.embed_content(model=model, contents=contents))

    def embed(self, texts: list[str], model: str, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        """One embedding per text. task_type: RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for questions.
        Tries batches of 100 first; newer models merge a list into ONE vector, so then we go one by one."""
        vectors = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            if model not in self._one_by_one:
                resp = self._embed_call(model, batch, task_type)
                if len(resp.embeddings) == len(batch):
                    vectors.extend(e.values for e in resp.embeddings)
                    continue
                self._one_by_one.add(model)
            for t in batch:
                vectors.append(self._embed_call(model, t, task_type).embeddings[0].values)
                if len(batch) > 1:
                    time.sleep(0.7)  # stay under the free-tier requests-per-minute limit
        return np.asarray(vectors, dtype=np.float32)

    # ---- generation
    def generate(self, prompt: str, model: str, system: str | None = None,
                 temperature: float = 0.3, max_tokens: int = 2048) -> tuple[str, Usage]:
        cfg = {"temperature": temperature, "max_output_tokens": max_tokens}
        if system:
            cfg["system_instruction"] = system
        usage, t0 = Usage(), time.time()
        resp = with_retries(lambda: self.client.models.generate_content(
            model=model, contents=prompt, config=types.GenerateContentConfig(**cfg)))
        usage.record(resp.usage_metadata)
        usage.latency_s = round(time.time() - t0, 2)
        return resp.text or "", usage
