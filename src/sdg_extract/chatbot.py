from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .database import database_summary, distinct_values, legacy_policy_passage, query_passages, query_policies
from .models import parse_json_object
from .multilingual import (
    AGGREGATE_TERMS,
    COMPARE_TERMS,
    COUNTRY_ALIASES,
    DATABASE_TERMS,
    QUERY_STOP_PHRASES,
    SUPPORTED_LANGUAGE_CODES,
    expand_multilingual_query,
)


CompletionFn = Callable[[List[Dict[str, str]], Dict[str, Any]], str]
QUERY_MODES = {"general", "database_search", "database_aggregate", "compare", "mixed"}


DEFAULT_CHATBOT_CONFIG = {
    "top_k": 8,
    "aggregate_limit": 200,
    "require_reviewed": False,
    "allow_general_knowledge": True,
    "conversation_turn_limit": 8,
    "llm_query_planning": False,
    "llm_reranking": False,
    "model": {
        "backend": "api",
        "name": "gpt-5.6-luna",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "api_protocol": "auto",
        "temperature": 0,
        "reasoning_effort": "medium",
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
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    chatbot_config = deep_merge(DEFAULT_CHATBOT_CONFIG, config or {})
    model_config = chatbot_config.get("model") or {}
    conversation = sanitize_history(history, int(chatbot_config.get("conversation_turn_limit", 8)))
    available_countries = distinct_values(db_path, "country")
    plan, planner_diagnostics = plan_question(question, conversation, available_countries)
    if chatbot_config.get("llm_query_planning") and completion_fn is None:
        try:
            plan = llm_plan_question(question, conversation, model_config, plan, available_countries)
            planner_diagnostics["planner"] = "model"
        except Exception as exc:
            planner_diagnostics["planner_error"] = f"{type(exc).__name__}: {exc}"

    results, retrieval = retrieve_for_plan(db_path, plan, chatbot_config)
    retrieval.update(planner_diagnostics)
    retrieval["plan"] = plan
    if results and chatbot_config.get("llm_reranking") and completion_fn is None and plan["mode"] not in {"database_aggregate", "compare"}:
        try:
            results = rerank_passages(question, results, model_config, int(chatbot_config.get("top_k", 8)))
            retrieval["reranker"] = "gpt-5.6-luna"
        except Exception as exc:
            retrieval["reranker_error"] = f"{type(exc).__name__}: {exc}"
            results = results[: int(chatbot_config.get("top_k", 8))]
    elif plan["mode"] not in {"database_aggregate", "compare"}:
        results = results[: int(chatbot_config.get("top_k", 8))]
    allow_general = bool(chatbot_config.get("allow_general_knowledge", True))
    if not results and not allow_general:
        return {
            "answer": "数据库中没有检索到足以回答该问题的政策证据。你可以关闭‘仅使用已审查政策’，或查看检索详情调整问题。",
            "sources": [],
            "status": "no_evidence",
            "answer_mode": "no_database_evidence",
            "retrieval": retrieval,
        }

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    messages = [
        {"role": "system", "content": prompt},
        *conversation,
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "query_plan": plan,
                    "database_status": {
                        "matched_passages": len(results),
                        "eligible_policies": retrieval.get("eligible_policies"),
                        "database_evidence_available": bool(results),
                    },
                    "retrieved_passages": [chat_context_record(row) for row in results],
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]
    try:
        complete = completion_fn or call_openai_compatible
        answer = complete(messages, model_config).strip()
        if not answer:
            raise RuntimeError(
                f"The API returned an empty answer for {model_config.get('name') or 'configured model'}."
            )
        answer_mode = "database" if results else "general"
        return {
            "answer": answer,
            "sources": source_records(results),
            "status": "answered",
            "answer_mode": answer_mode,
            "retrieval": retrieval,
        }
    except Exception as exc:
        return {
            "answer": f"Chatbot API error: {type(exc).__name__}: {exc}",
            "sources": source_records(results),
            "status": f"error:{type(exc).__name__}",
            "retrieval": retrieval,
        }


def plan_question(
    question: str,
    history: Sequence[Dict[str, str]],
    available_countries: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    standalone = resolve_follow_up(question, history)
    lower = standalone.lower()
    countries = [canonical for alias, canonical in COUNTRY_ALIASES.items() if contains_alias(lower, alias)]
    for country in available_countries or []:
        if contains_alias(lower, str(country).lower()):
            countries.append(str(country))
    countries = list(dict.fromkeys(countries))
    targets = list(dict.fromkeys(re.findall(r"(?<!\d)(13\.[123ab])(?!\w)", lower)))
    years = list(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", standalone)))
    has_database_signal = bool(countries or targets or any(term in lower for term in DATABASE_TERMS))
    if any(term in lower for term in COMPARE_TERMS):
        mode = "compare"
    elif any(term in lower for term in AGGREGATE_TERMS) and has_database_signal:
        mode = "database_aggregate"
    elif has_database_signal:
        mode = "database_search"
    else:
        mode = "general"
    semantic = semantic_query(standalone, countries)
    plan = {
        "mode": mode,
        "standalone_question": standalone,
        "semantic_query_zh": semantic if contains_cjk(semantic) else "",
        "semantic_query_en": semantic if not contains_cjk(semantic) else expand_query_aliases(semantic),
        "countries": countries,
        "sdg_goal": "SDG 13" if re.search(r"(?i)sdg\s*13|目标\s*13|气候行动", standalone) else None,
        "sdg_targets": targets,
        "years": years,
        "request_all": mode in {"database_aggregate", "compare"},
    }
    return plan, {"planner": "deterministic", "standalone_question": standalone}


def llm_plan_question(
    question: str,
    history: Sequence[Dict[str, str]],
    model_config: Dict[str, Any],
    fallback: Dict[str, Any],
    available_countries: Sequence[str],
) -> Dict[str, Any]:
    planner_config = dict(model_config)
    planner_config["reasoning_effort"] = "low"
    planner_config["max_tokens"] = min(int(model_config.get("max_tokens", 1500)), 700)
    planner_config["json_mode"] = True
    messages = [
        {
            "role": "system",
            "content": (
                "Convert the user question into a database query plan. Return JSON only. "
                "Modes: general, database_search, database_aggregate, compare, mixed. "
                "Extract only countries present in available_countries, SDG13 targets, years, and both original-language and English semantic queries. "
                f"Supported corpus language codes: {SUPPORTED_LANGUAGE_CODES}."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"question": question, "history": list(history), "available_countries": list(available_countries), "deterministic_fallback": fallback},
                ensure_ascii=False,
            ),
        },
    ]
    response = call_openai_compatible_json(messages, planner_config)
    plan = dict(fallback)
    if str(response.get("mode") or "") in QUERY_MODES:
        plan["mode"] = str(response["mode"])
    for key in ["standalone_question", "semantic_query_zh", "semantic_query_en", "sdg_goal"]:
        if response.get(key) is not None:
            plan[key] = response.get(key)
    for key in ["countries", "sdg_targets", "years"]:
        if isinstance(response.get(key), list):
            plan[key] = [str(value) for value in response[key] if value not in {None, ""}]
    plan["countries"] = [COUNTRY_ALIASES.get(value.lower(), value) for value in plan.get("countries", [])]
    plan["request_all"] = bool(response.get("request_all", plan.get("request_all")))
    return plan


def retrieve_for_plan(
    db_path: Path,
    plan: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    summary = database_summary(db_path)
    diagnostics: Dict[str, Any] = {
        **summary,
        "mode": plan["mode"],
        "query_variants": [],
        "fallback_stage": None,
    }
    if plan["mode"] == "general":
        return [], diagnostics
    countries = plan.get("countries") or None
    targets = plan.get("sdg_targets") or None
    year = (plan.get("years") or [None])[0]
    require_reviewed = bool(config.get("require_reviewed", False))
    if plan["mode"] in {"database_aggregate", "compare"}:
        rows = query_policies(
            db_path,
            countries=countries,
            sdg_targets=targets,
            publication_year=year,
            validation_status="valid",
            require_reviewed=require_reviewed,
            limit=int(config.get("aggregate_limit", 200)),
        )
        diagnostics["aggregate_rows"] = len(rows)
        diagnostics["fallback_stage"] = "structured_sql"
        return [legacy_policy_passage(row) for row in rows], diagnostics

    variants = query_variants(plan)
    diagnostics["query_variants"] = variants
    merged: Dict[Any, Dict[str, Any]] = {}
    stages: List[Dict[str, Any]] = []
    for variant in variants:
        rows, stage = query_passages(
            db_path,
            query=variant,
            countries=countries,
            sdg_targets=targets,
            publication_year=year,
            validation_status="valid",
            require_reviewed=require_reviewed,
            limit=max(int(config.get("top_k", 8)), 12),
        )
        stages.append(stage)
        for row in rows:
            key = row.get("passage_id")
            current = merged.get(key)
            if current is None or float(row.get("retrieval_score") or 0) > float(current.get("retrieval_score") or 0):
                merged[key] = row
    ranked = sorted(
        merged.values(),
        key=lambda row: (float(row.get("retrieval_score") or 0), str(row.get("review_status") or "").startswith("reviewed")),
        reverse=True,
    )
    diagnostics["stages"] = stages
    diagnostics["final_hits"] = len(ranked)
    if not ranked:
        diagnostics["fallback_stage"] = "no_database_evidence"
    candidate_limit = int(config.get("rerank_candidates", 20)) if config.get("llm_reranking") else int(config.get("top_k", 8))
    return ranked[:candidate_limit], diagnostics



def rerank_passages(
    question: str,
    passages: Sequence[Dict[str, Any]],
    model_config: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    config = dict(model_config)
    config["reasoning_effort"] = "low"
    config["max_tokens"] = min(int(model_config.get("max_tokens", 1500)), 700)
    config["json_mode"] = True
    candidates = [
        {
            "passage_id": row.get("passage_id"),
            "country": row.get("country"),
            "policy_title": row.get("policy_title_en") or row.get("policy_title"),
            "sdg_target": row.get("sdg_target") or row.get("sdg_targets"),
            "text": truncate_text(row.get("text_en") or row.get("text_original"), 1000),
        }
        for row in passages
    ]
    response = call_openai_compatible_json(
        [
            {
                "role": "system",
                "content": (
                    "Rank database passages by their ability to answer the user question. "
                    "Return JSON only as {\"ranked_passage_ids\":[...]}. Use only supplied IDs."
                ),
            },
            {"role": "user", "content": json.dumps({"question": question, "candidates": candidates}, ensure_ascii=False)},
        ],
        config,
    )
    ranked_ids = response.get("ranked_passage_ids")
    if not isinstance(ranked_ids, list):
        return list(passages)[:limit]
    by_id = {str(row.get("passage_id")): row for row in passages}
    ordered: List[Dict[str, Any]] = []
    seen = set()
    for passage_id in ranked_ids:
        key = str(passage_id)
        if key in by_id and key not in seen:
            ordered.append(by_id[key])
            seen.add(key)
    ordered.extend(row for row in passages if str(row.get("passage_id")) not in seen)
    return ordered[:limit]

def query_variants(plan: Dict[str, Any]) -> List[str]:
    variants = [
        str(plan.get("semantic_query_zh") or "").strip(),
        str(plan.get("semantic_query_en") or "").strip(),
    ]
    if plan.get("sdg_goal"):
        variants.append(str(plan["sdg_goal"]))
    variants.extend(expand_query_aliases(value) for value in list(variants) if value)
    return list(dict.fromkeys(value for value in variants if value)) or [""]


def semantic_query(question: str, countries: Sequence[str]) -> str:
    value = question
    for alias in sorted(COUNTRY_ALIASES, key=len, reverse=True):
        value = re.sub(re.escape(alias), " ", value, flags=re.IGNORECASE)
    for country in countries:
        value = re.sub(re.escape(country), " ", value, flags=re.IGNORECASE)
    for phrase in sorted(QUERY_STOP_PHRASES, key=len, reverse=True):
        value = re.sub(re.escape(phrase), " ", value, flags=re.IGNORECASE)
    return " ".join(value.split()).strip("，。？！,? ")


def expand_query_aliases(query: str) -> str:
    return expand_multilingual_query(query)


def resolve_follow_up(question: str, history: Sequence[Dict[str, str]]) -> str:
    if not re.search(r"这些|它们|其中|上述|those|these|them|above", question, flags=re.IGNORECASE):
        return question
    prior_users = [message["content"] for message in history if message.get("role") == "user"]
    if not prior_users:
        return question
    return f"{prior_users[-1]}；追问：{question}"


def chat_context_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "passage_id": record.get("passage_id"),
        "language": record.get("language"),
        "country": record.get("country"),
        "policy_title": record.get("policy_title_en") or record.get("policy_title"),
        "publication_date": record.get("publication_date"),
        "matched_passage": truncate_text(record.get("text_en") or record.get("text_original"), 3200),
        "passage_summary": truncate_text(record.get("summary_en") or record.get("summary_original"), 1000),
        "sdg_goal": record.get("sdg_goal"),
        "sdg_target": record.get("sdg_target") or record.get("sdg_targets"),
        "page_start": record.get("page_start"),
        "page_end": record.get("page_end"),
        "review_status": record.get("review_status") or "unreviewed",
        "review_confidence": record.get("review_confidence"),
        "source_filename": record.get("source_filename"),
    }


def source_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "country": record.get("country"),
            "language": record.get("language"),
            "policy_title": record.get("policy_title_en") or record.get("policy_title"),
            "sdg_target": record.get("sdg_target") or record.get("sdg_targets"),
            "page": page_label(record.get("page_start"), record.get("page_end")),
            "source_filename": record.get("source_filename"),
            "review_status": record.get("review_status") or "unreviewed",
            "retrieval_score": record.get("retrieval_score"),
        }
        for record in records
    ]


def call_openai_compatible(messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
    data = call_openai_api(messages, config, json_mode=False)
    return extract_api_text(data)


def call_openai_compatible_json(messages: List[Dict[str, str]], config: Dict[str, Any]) -> Dict[str, Any]:
    data = call_openai_api(messages, config, json_mode=True)
    return parse_json_object(extract_api_text(data))


def call_openai_api(messages: List[Dict[str, str]], config: Dict[str, Any], json_mode: bool) -> Dict[str, Any]:
    if config.get("backend", "api") != "api":
        raise ValueError("Chatbot currently supports only the api backend.")
    api_key_env = str(config.get("api_key_env") or "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {api_key_env}")
    protocol = str(config.get("api_protocol") or "auto").lower()
    if protocol in {"auto", "responses"}:
        try:
            return request_responses_api(messages, config, api_key, json_mode)
        except ApiCapabilityError:
            if protocol == "responses":
                raise
    return request_chat_completions(messages, config, api_key, json_mode)


def request_responses_api(
    messages: List[Dict[str, str]], config: Dict[str, Any], api_key: str, json_mode: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": config.get("name") or "gpt-5.6-luna",
        "input": messages,
        "max_output_tokens": int(config.get("max_tokens", 1500)),
    }
    if config.get("reasoning_effort"):
        payload["reasoning"] = {"effort": str(config["reasoning_effort"])}
    if json_mode:
        payload["text"] = {"format": {"type": "json_object"}}
    return api_request(config, api_key, "/responses", payload, capability_fallback=True)


def request_chat_completions(
    messages: List[Dict[str, str]], config: Dict[str, Any], api_key: str, json_mode: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": config.get("name") or "gpt-5.6-luna",
        "messages": messages,
        "temperature": float(config.get("temperature", 0)),
        "max_tokens": int(config.get("max_tokens", 1500)),
    }
    if config.get("reasoning_effort"):
        payload["reasoning_effort"] = str(config["reasoning_effort"])
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return api_request(config, api_key, "/chat/completions", payload, capability_fallback=False)


def api_request(
    config: Dict[str, Any], api_key: str, endpoint: str, payload: Dict[str, Any], capability_fallback: bool,
) -> Dict[str, Any]:
    api_base = str(config.get("api_base") or "https://api.openai.com/v1").rstrip("/")
    request = urllib.request.Request(
        f"{api_base}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=int(config.get("timeout", 180))) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        if capability_fallback and exc.code in {400, 404, 405, 422}:
            raise ApiCapabilityError(f"Responses API unavailable: HTTP {exc.code}: {detail}") from exc
        raise RuntimeError(f"API request failed: HTTP {exc.code}: {detail}") from exc


def extract_api_text(data: Dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data["output_text"]).strip()
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, list):
            return "".join(str(part.get("text") or "") for part in content if isinstance(part, dict)).strip()
        return str(content or "").strip()
    texts: List[str] = []
    for output in data.get("output", []) if isinstance(data.get("output"), list) else []:
        for part in output.get("content", []) if isinstance(output, dict) else []:
            if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                texts.append(str(part.get("text") or ""))
    return "".join(texts).strip()


def sanitize_history(history: Optional[List[Dict[str, str]]], turn_limit: int) -> List[Dict[str, str]]:
    sanitized = [
        {"role": str(message.get("role")), "content": str(message.get("content") or "").strip()}
        for message in history or []
        if message.get("role") in {"user", "assistant"} and str(message.get("content") or "").strip()
    ]
    max_messages = max(0, turn_limit) * 2
    return sanitized[-max_messages:] if max_messages else []


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result



def truncate_text(value: Any, limit: int) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."

def page_label(start: Any, end: Any) -> Optional[str]:
    if start in {None, ""}:
        return None
    return str(start) if end in {None, "", start} else f"{start}-{end}"



def contains_alias(text: str, alias: str) -> bool:
    value = alias.lower()
    if value.isascii() and all(char.isalnum() or char in {" ", ".", "-"} for char in value):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", text))
    return value in text

def contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value))


class ApiCapabilityError(RuntimeError):
    pass
