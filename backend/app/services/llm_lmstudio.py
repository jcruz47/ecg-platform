import os
import re
from typing import Dict, Any, Optional, List
from openai import OpenAI

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://192.168.56.1:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "meta-llama-3.1-8b-instruct")
USE_LLM = os.getenv("USE_LLM", "1") == "1"

SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    (
        "Eres un asistente clínico experto en ECG.\n"
        "Responde SIEMPRE en español, de forma clara y concisa.\n"
        "No inventes hallazgos ni emitas diagnósticos definitivos.\n"
        "No prescribas tratamiento.\n"
        "Si faltan datos o la calidad es insuficiente, dilo claramente.\n"
        "Solo hablas español."
    )
)

_client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    global _client
    if not USE_LLM:
        return None
    if _client is None:
        _client = OpenAI(
            base_url=LM_STUDIO_BASE_URL,
            api_key="lm-studio"
        )
    return _client


def get_llm_backend_info() -> Dict[str, Any]:
    return {
        "backend": "lm_studio" if USE_LLM else "disabled",
        "base_url": LM_STUDIO_BASE_URL,
        "model": LM_STUDIO_MODEL,
        "use_llm": USE_LLM,
    }


_ES_STOP = {
    "de","la","que","el","en","y","a","los","se","del","las","por","un","para","con","no","una",
    "su","al","lo","como","más","pero","sus","le","ya","o","este","sí","porque","esta","entre",
    "cuando","muy","sin","sobre","también","me","hasta","hay","donde","quien","desde","todo","nos"
}
_EN_STOP = {
    "the","of","and","to","in","a","is","that","for","it","on","as","with","are","this","by",
    "be","or","from","at","an","which","was","have","has","will","can","if","not","more","but",
    "we","you","they","their","our","your","he","she","its","there","here","what","when","how"
}


def _lang_guess(text: str) -> str:
    words = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ]+", (text or "").lower())
    if not words:
        return "es"
    es_hits = sum((w in _ES_STOP) or any(c in w for c in "áéíóúñ") for w in words)
    en_hits = sum(w in _EN_STOP for w in words)
    return "es" if es_hits >= en_hits else "en"


def ensure_spanish(text: str) -> str:
    if not text or _lang_guess(text) == "es":
        return text

    client = _get_client()
    if client is None:
        return "⚠️ Texto detectado en otro idioma; se devuelve tal cual.\n\n" + text

    prompt = (
        "Convierte el siguiente texto exactamente al ESPAÑOL NEUTRO. "
        "No agregues ni quites nada. Respeta cifras y formato.\n\n"
        f"---\n{text}\n---"
    )

    try:
        result = client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=[
                {"role": "system", "content": "Respondes únicamente en ESPAÑOL. Devuelve solo el texto convertido."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        out = (result.choices[0].message.content or "").strip()
        return out if _lang_guess(out) == "es" else text
    except Exception:
        return text


def chat_complete(messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 400) -> str:
    client = _get_client()
    if client is None:
        return (
            "Modo sin LLM: responderé de forma básica y siempre en español "
            "sobre las métricas visibles."
        )

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)

    try:
        resp = client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return ensure_spanish(raw)
    except Exception as e:
        return f"[ERROR] {e}"