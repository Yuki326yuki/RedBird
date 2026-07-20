from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .database import query_policies


CompletionFn = Callable[[List[Dict[str, str]], Dict[str, Any]], str]


DEFAULT_CHATBOT_CONFIG = {
    "top_k": 6,
    "model": {
        "backend": "api",
        "name": "gpt-4.1-mini",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0,
        "max_tokens": 1500,
        "json_mode": False,
    },
}


def answer_question(
    db_path: Path,
    question: str,
    prompt_path: Path,
    config: Optional[Dict[str, Any]] = None,
    completion_fn: Optional[CompletionFn] = None,
) -> Dict[str, Any]:
    chatbot_config = {**DEFAULT_CHATBOT_CONFIG, **(config or {})}
    top_k = int(chatbot_config.get("top_k", 6))
    results = query_policies(
        db_path=db_path,
        query=question,
        validation_status="valid",
        require_reviewed=True,
        limit=top_k,
    )
    if not results:
        return {
            "answer": "The database does not contain enough reviewed evidence to answer this question.",
            "sources": [],
            "status": "no_evidence",
        }

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "retrieved_policies": [chat_context_record(row) for row in results],
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]
    model_config = chatbot_config.get("model") or {}
    try:
        complete = completion_fn or call_openai_compatible
        answer = complete(messages, model_config)
        return {
            "answer": answer.strip(),
            "sources": source_records(results),
            "status": "answered",
        }
    except Exception as exc:
        return {
            "answer": f"Chatbot API error: {type(exc).__name__}: {exc}",
            "sources": source_records(results),
            "status": f"error:{type(exc).__name__}",
        }


def call_openai_compatible(messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
    if config.get("backend", "api") != "api":
        raise ValueError("Chatbot currently supports only the api backend.")
    api_key_env = config.get("api_key_env") or "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {api_key_env}")
    api_base = str(config.get("api_base") or "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": config.get("name") or "gpt-4.1-mini",
        "messages": messages,
        "temperature": float(config.get("temperature", 0)),
        "max_tokens": int(config.get("max_tokens", 1500)),
    }
    request = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"API request failed: HTTP {exc.code}: {detail}") from exc
    return str(data["choices"][0]["message"]["content"])


def chat_context_record(record: Dict[str, Any]) -> Dict[str, Any]:
    content = record.get("policy_content_en") or record.get("policy_content")
    summary = record.get("policy_summary_en") or record.get("policy_summary")
    title = record.get("policy_title_en") or record.get("policy_title")
    return {
        "id": record.get("id"),
        "country": record.get("country"),
        "policy_title": title,
        "publication_date": record.get("publication_date"),
        "policy_content": truncate_text(content, 1600),
        "policy_summary": truncate_text(summary, 600),
        "sdg_goal": record.get("sdg_goal"),
        "sdg_targets": record.get("sdg_targets"),
        "review_status": record.get("review_status"),
        "review_confidence": record.get("review_confidence"),
        "source_filename": record.get("source_filename"),
    }


def source_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "id": record.get("id"),
            "country": record.get("country"),
            "policy_title": record.get("policy_title_en") or record.get("policy_title"),
            "source_filename": record.get("source_filename"),
            "sdg_targets": record.get("sdg_targets"),
        }
        for record in records
    ]


def truncate_text(value: Any, limit: int) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
