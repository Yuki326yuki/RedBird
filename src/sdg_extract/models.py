from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ModelBackend(ABC):
    @abstractmethod
    def complete_json(self, messages: List[Dict[str, str]], schema_hint: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError


class OpenAICompatibleBackend(ModelBackend):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.name = config.get("name") or "unknown-api-model"
        self.api_base = str(config.get("api_base") or "").rstrip("/")
        self.api_key_env = config.get("api_key_env") or "OPENAI_API_KEY"
        self.temperature = float(config.get("temperature", 0))
        self.max_tokens = int(config.get("max_tokens", 4000))
        self.json_mode = bool(config.get("json_mode", True))

    def complete_json(self, messages: List[Dict[str, str]], schema_hint: str) -> Dict[str, Any]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
        payload = {
            "model": self.name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
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
        content = data["choices"][0]["message"]["content"]
        return parse_json_object(content)

    def model_name(self) -> str:
        return self.name


class LocalTransformersBackend(ModelBackend):
    def __init__(self, config: Dict[str, Any]) -> None:
        model_path = config.get("local_model_path")
        if not model_path:
            raise RuntimeError("model.local_model_path is required for local_transformers backend.")
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "local_transformers backend requires optional dependencies: transformers and torch."
            ) from exc

        self.name = str(config.get("name") or model_path)
        self.max_tokens = int(config.get("max_tokens", 4000))
        self.temperature = float(config.get("temperature", 0))
        self.use_chat_template = bool(config.get("use_chat_template", False))
        self.enable_thinking = bool(config.get("enable_thinking", True))
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dtype_value = config.get("torch_dtype")
        torch_dtype = (
            "auto"
            if dtype_value == "auto"
            else torch.float16
            if torch.cuda.is_available()
            else None
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = tokenizer
        self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def complete_json(self, messages: List[Dict[str, str]], schema_hint: str) -> Dict[str, Any]:
        prompt = self.build_prompt(messages)
        outputs = self.generator(
            prompt,
            max_new_tokens=self.max_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 0.01),
            return_full_text=False,
        )
        content = outputs[0]["generated_text"]
        try:
            return parse_json_object(content)
        except (json.JSONDecodeError, ValueError) as exc:
            path = write_parse_failure(content)
            print(
                f"[WARN] Could not parse local model JSON response; "
                f"treating this chunk as 0 records. Debug output: {path}. Error: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return {"records": []}

    def model_name(self) -> str:
        return self.name

    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        prompt = "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)
        return f"{prompt}\n\nReturn JSON only."


class HeuristicBackend(ModelBackend):
    """Offline fallback for smoke tests and dry runs.

    This backend is intentionally conservative. It only emits records for
    paragraphs that explicitly mention SDG terms.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.name = config.get("name") or "heuristic"

    def complete_json(self, messages: List[Dict[str, str]], schema_hint: str) -> Dict[str, Any]:
        user_text = messages[-1]["content"].rsplit("Source text:\n", 1)[-1]
        records: List[Dict[str, Any]] = []
        for paragraph_id, page, text in re.findall(
            r"\[(p\d+)\s+page=([^\]]+)\]\s+(.+?)(?=\n\n\[p\d+\s+page=|\Z)",
            user_text,
            flags=re.DOTALL,
        ):
            sdg_goal = infer_sdg_goal(text)
            if not sdg_goal:
                continue
            records.append(
                {
                    "sdg_goal": sdg_goal,
                    "sdg_target": infer_sdg_target(text),
                    "sdg_confidence": 0.5,
                    "policy_content_original": text.strip(),
                    "policy_content_summary": None,
                    "evidence_quote": text.strip()[:500],
                    "evidence_page": int(page) if page.isdigit() else None,
                    "evidence_paragraph_id": paragraph_id,
                    "validation_status": "heuristic",
                    "notes": "Generated by heuristic backend for local smoke testing.",
                }
            )
        return {"records": records}

    def model_name(self) -> str:
        return self.name


def build_backend(config: Dict[str, Any]) -> ModelBackend:
    backend = config.get("backend", "api")
    if backend == "api":
        return OpenAICompatibleBackend(config)
    if backend == "local_transformers":
        return LocalTransformersBackend(config)
    if backend == "heuristic":
        return HeuristicBackend(config)
    raise ValueError(f"Unsupported model backend: {backend}")


def parse_json_object(text: str) -> Dict[str, Any]:
    candidates = [text.strip(), *extract_fenced_json(text)]
    first_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return normalize_json_payload(json.loads(candidate))
        except json.JSONDecodeError as exc:
            first_error = first_error or exc
        except ValueError:
            continue
    parsed = first_json_payload(text, first_error)
    return normalize_json_payload(parsed)


def extract_fenced_json(text: str) -> List[str]:
    return re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)


def first_json_payload(text: str, original_error: json.JSONDecodeError | None) -> Any:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    if original_error is not None:
        raise original_error
    raise json.JSONDecodeError("No JSON object or array found", text, 0)


def normalize_json_payload(parsed: Any) -> Dict[str, Any]:
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"records": parsed}
    raise ValueError("Model response must be a JSON object or records array.")


def write_parse_failure(content: str) -> Path:
    debug_dir = Path("data/work/model_parse_failures")
    debug_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()[:12]
    path = debug_dir / f"local_model_response_{digest}.txt"
    path.write_text(content, encoding="utf-8", errors="replace")
    return path


def infer_sdg_goal(text: str) -> str:
    explicit = re.search(r"\bSDG\s*([1-9]|1[0-7])\b", text, flags=re.IGNORECASE)
    if explicit:
        return f"SDG {explicit.group(1)}"
    lower = text.lower()
    keyword_map = {
        "poverty": "SDG 1",
        "hunger": "SDG 2",
        "health": "SDG 3",
        "education": "SDG 4",
        "gender": "SDG 5",
        "water": "SDG 6",
        "energy": "SDG 7",
        "climate": "SDG 13",
        "biodiversity": "SDG 15",
    }
    if "sustainable development goal" not in lower and "sdg" not in lower:
        return ""
    for keyword, goal in keyword_map.items():
        if keyword in lower:
            return goal
    return ""


def infer_sdg_target(text: str) -> Any:
    match = re.search(r"\b([1-9]|1[0-7])\.[0-9]+[a-z]?\b", text, flags=re.IGNORECASE)
    return match.group(0) if match else None
