import argparse
import base64
import csv
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import requests
from openai import OpenAI


BROWSECOMP_SUBSET_PATH = os.path.join(os.path.dirname(__file__), "browsecomp_subset_125.csv")
DEFAULT_GRADER_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_GRADER_TEMPERATURE = 0.5
DEFAULT_MAX_TOOL_ROUNDS = 12
DEFAULT_SEARCH_RESULTS = 5
DEFAULT_FETCH_CHAR_LIMIT = 12000
DEFAULT_SEARCH_ENGINE = "serper"
SEARCH_ENGINE_CHOICES = ("duckduckgo", "serper", "auto")
ACTIVE_SEARCH_ENGINE = os.environ.get("BROWSECOMP_SEARCH_ENGINE", DEFAULT_SEARCH_ENGINE).lower()
MAX_SEARCH_QUERY_CHARS = 512
CONTEXT_BUDGET_SAFETY_MARGIN = 32
MIN_CONTEXT_RETRY_MAX_TOKENS = 16
OUTPUT_FIELDS_TO_DROP = {"annotations", "audio", "function_call", "reasoning", "text"}
RESPONSE_FIELDS_TO_DROP = OUTPUT_FIELDS_TO_DROP - {"text"}

QUERY_TEMPLATE_NO_GET_DOCUMENT = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search and open tools provided. Please perform reasoning and use the tools step by step, in an interleaved manner. Use search to find relevant pages and open(url) to inspect promising pages. You may use the tools multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

GRADER_TEMPLATE = r"""
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below. Evaluate the answer conveyed by the full [response]; do not require a separately parsed answer field.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

[correct_answer]: {correct_answer}

reasoning: Explain why the answer conveyed by [response] is correct or incorrect based on [correct_answer], focusing only on meaningful differences between [correct_answer] and [response]. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if the answer conveyed by [response] matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the response is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the public web for pages relevant to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Web search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open",
            "description": "Fetch and extract readable text from a public web page URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
]


@dataclass
class SampleResult:
    index: int
    task_id: str
    score: bool | None
    question: str
    correct_answer: str
    model_response: str
    grader_response: str
    tool_call_count: int
    search_tool_call_count: int
    open_tool_call_count: int
    other_tool_call_count: int
    raw_trajectory_path: str
    error: str | None = None


class ChatCompletionsSampler:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ):
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def clone(self) -> "ChatCompletionsSampler":
        return ChatCompletionsSampler(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            system_message=self.system_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    @staticmethod
    def pack_message(role: str, content: str) -> dict[str, Any]:
        return {"role": role, "content": content}

    def prepare_messages(self, message_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.system_message and (not message_list or message_list[0].get("role") != "system"):
            return [self.pack_message("system", self.system_message)] + message_list
        return message_list

    def complete(
        self,
        message_list: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        message_list = self.prepare_messages(message_list)

        trial = 0
        request_max_tokens = self.max_tokens
        while True:
            started_at = utc_now()
            try:
                request_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": message_list,
                    "max_tokens": request_max_tokens,
                }
                if self.temperature is not None:
                    request_kwargs["temperature"] = self.temperature
                if tools:
                    request_kwargs["tools"] = tools
                    request_kwargs["tool_choice"] = tool_choice or "auto"

                response = self.client.chat.completions.create(
                    model=self.model,
                    **{k: v for k, v in request_kwargs.items() if k != "model"},
                )
                response_dict = serialize_response(response)
                assistant_message = extract_assistant_message(response_dict)
                content = render_content(assistant_message.get("content"))
                request_trace = {
                    **request_kwargs,
                    "configured_max_tokens": self.max_tokens,
                    "base_url": self.base_url,
                    "system_message": self.system_message,
                    "tool_choice": request_kwargs.get("tool_choice"),
                }
                return content, {
                    "request": request_trace,
                    "response": response_dict,
                    "assistant_message": assistant_message,
                    "started_at": started_at,
                    "completed_at": utc_now(),
                }
            except Exception as error:
                reduced_max_tokens = reduced_max_tokens_for_context_error(error, request_max_tokens)
                if reduced_max_tokens is not None:
                    print(
                        (
                            "[context-budget] Reducing max_tokens "
                            f"from {request_max_tokens} to {reduced_max_tokens} for {self.model}"
                        ),
                        flush=True,
                    )
                    request_max_tokens = reduced_max_tokens
                    trial = 0
                    continue
                if is_context_budget_error(error):
                    raise
                if trial >= 5:
                    raise
                time.sleep(2**trial)
                trial += 1

    def __call__(self, message_list: list[dict[str, Any]]) -> str:
        content, _ = self.complete(message_list)
        return content


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prune_fields(value: Any, fields_to_drop: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            str(k): prune_fields(v, fields_to_drop)
            for k, v in value.items()
            if k not in fields_to_drop
        }
    if isinstance(value, list):
        return [prune_fields(item, fields_to_drop) for item in value]
    return value


def prune_response_fields(value: Any) -> Any:
    return prune_fields(value, RESPONSE_FIELDS_TO_DROP)


def prune_output_fields(value: Any) -> Any:
    return prune_fields(value, OUTPUT_FIELDS_TO_DROP)


def serialize_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return prune_response_fields(response.model_dump(mode="json"))
    if hasattr(response, "to_dict"):
        return prune_response_fields(response.to_dict())
    if hasattr(response, "model_dump_json"):
        return prune_response_fields(json.loads(response.model_dump_json()))
    raise TypeError(f"Unsupported response type: {type(response)!r}")


def render_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "".join(parts)
    return str(content)


def extract_assistant_message(response_dict: dict[str, Any]) -> dict[str, Any]:
    choices = response_dict.get("choices") or []
    if not choices:
        raise ValueError("Chat completion returned no choices")
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        raise ValueError("Chat completion returned an invalid assistant message")
    message.setdefault("role", "assistant")
    return message


def parse_tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") or {}
    raw_args = function.get("arguments", "{}")
    if isinstance(raw_args, dict):
        return raw_args
    try:
        return json.loads(raw_args)
    except json.JSONDecodeError:
        return {"raw": raw_args}


def safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [safe_json_value(item) for item in value]
    return str(value)


def make_exception_payload(error: Exception) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
    }

    for attr in ("status_code", "code", "param", "request_id"):
        value = getattr(error, attr, None)
        if value is not None:
            payload[attr] = safe_json_value(value)

    body = getattr(error, "body", None)
    if body is not None:
        payload["body"] = safe_json_value(body)

    response = getattr(error, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            payload["status_code"] = status_code
        url = getattr(response, "url", None)
        if url:
            payload["url"] = str(url)
        response_text = getattr(response, "text", "")
        if isinstance(response_text, str) and response_text.strip():
            payload["response_text"] = response_text.strip()[:1000]

    return payload


def get_error_text(error: Exception) -> str:
    payload = make_exception_payload(error)
    parts = [str(error)]
    body = payload.get("body")
    if body is not None:
        parts.append(json.dumps(body, ensure_ascii=False))
    response_text = payload.get("response_text")
    if response_text:
        parts.append(str(response_text))
    return "\n".join(parts)


def reduced_max_tokens_for_context_error(
    error: Exception,
    requested_max_tokens: int,
    *,
    safety_margin: int = CONTEXT_BUDGET_SAFETY_MARGIN,
) -> int | None:
    text = get_error_text(error)
    patterns = (
        r"maximum context length is\s+(\d+)\s+tokens.*?request has\s+(\d+)\s+input tokens",
        r"model'?s maximum context length is\s+(\d+)\s+tokens.*?request has\s+(\d+)\s+input tokens",
    )

    max_context: int | None = None
    input_tokens: int | None = None
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            max_context = int(match.group(1))
            input_tokens = int(match.group(2))
            break

    if max_context is None or input_tokens is None:
        match = re.search(r"\d+\s*>\s*(\d+)\s*-\s*(\d+)", text)
        if match:
            max_context = int(match.group(1))
            input_tokens = int(match.group(2))

    if max_context is None or input_tokens is None:
        return None

    available = max_context - input_tokens
    if available < MIN_CONTEXT_RETRY_MAX_TOKENS:
        return None

    reduced = max(MIN_CONTEXT_RETRY_MAX_TOKENS, available - safety_margin)
    if reduced >= requested_max_tokens:
        return None
    return reduced


def is_context_budget_error(error: Exception) -> bool:
    text = get_error_text(error).lower()
    return (
        "maximum context length" in text
        and "input tokens" in text
        and ("max_tokens" in text or "max_completion_tokens" in text)
    )


class PageTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag in {"p", "br", "div", "li", "section", "article", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "br", "div", "li", "section", "article"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data.strip())

    def get_text(self) -> str:
        text = " ".join(self.parts)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


class DuckDuckGoResultParser(HTMLParser):
    def __init__(self, max_results: int):
        super().__init__()
        self.max_results = max_results
        self.results: list[dict[str, str]] = []
        self._capture: str | None = None
        self._capture_tag: str | None = None
        self._parts: list[str] = []
        self._pending_url = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {name: value or "" for name, value in attrs}
        class_names = attrs_dict.get("class", "")

        if tag == "a" and ("result__a" in class_names or "result-link" in class_names):
            self._capture = "title"
            self._capture_tag = tag
            self._parts = []
            self._pending_url = normalize_duckduckgo_url(attrs_dict.get("href", ""))
            return

        if self.results and ("result__snippet" in class_names or "result-snippet" in class_names):
            self._capture = "snippet"
            self._capture_tag = tag
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._capture or tag != self._capture_tag:
            return

        text = clean_search_text(" ".join(self._parts))
        if self._capture == "title" and text and self._pending_url:
            if not any(result["url"] == self._pending_url for result in self.results):
                self.results.append({"title": text, "url": self._pending_url, "snippet": ""})
        elif self._capture == "snippet" and text and self.results:
            self.results[-1]["snippet"] = text

        self._capture = None
        self._capture_tag = None
        self._parts = []
        self._pending_url = ""


def search_web(query: str) -> dict[str, Any]:
    max_results = DEFAULT_SEARCH_RESULTS
    search_engine = ACTIVE_SEARCH_ENGINE
    if search_engine == "auto":
        if os.environ.get("SERPER_API_KEY"):
            try:
                return search_serper(query, max_results)
            except requests.RequestException as error:
                result = search_duckduckgo(query, max_results)
                result["fallback_from"] = make_tool_error_payload(error)
                return result
        return search_duckduckgo(query, max_results)
    if search_engine == "duckduckgo":
        return search_duckduckgo(query, max_results)
    if search_engine == "serper":
        return search_serper(query, max_results)
    return {
        "engine": search_engine,
        "query": query,
        "results": [],
        "error": f"Unknown search engine: {search_engine}",
    }


def clean_search_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_search_query(query: str) -> str:
    return clean_search_text(str(query))[:MAX_SEARCH_QUERY_CHARS]


def normalize_duckduckgo_url(url: str) -> str:
    if url.startswith("//"):
        url = f"https:{url}"
    elif url.startswith("/"):
        url = f"https://duckduckgo.com{url}"

    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        uddg = parse_qs(parsed.query).get("uddg")
        if uddg:
            return unquote(uddg[0])
    return url


def search_duckduckgo(query: str, max_results: int = DEFAULT_SEARCH_RESULTS) -> dict[str, Any]:
    query = normalize_search_query(query)
    if not query:
        return {"engine": "duckduckgo/html", "query": query, "results": [], "error": "empty query"}

    endpoints = (
        "https://html.duckduckgo.com/html/",
        "https://duckduckgo.com/html/",
        "https://lite.duckduckgo.com/lite/",
    )
    errors: list[dict[str, Any]] = []

    for endpoint in endpoints:
        try:
            response = requests.get(
                endpoint,
                params={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
                timeout=60,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            errors.append(make_tool_error_payload(error))
            continue

        parser = DuckDuckGoResultParser(max_results=max_results)
        parser.feed(response.text)
        results = parser.results[:max_results]
        if results:
            return {
                "engine": "duckduckgo/html",
                "query": query,
                "results": results,
            }

        errors.append({
            "type": "NoResultsParsed",
            "message": f"No DuckDuckGo results parsed from {endpoint}",
            "retryable": False,
            "url": response.url,
            "status_code": response.status_code,
        })

    return {
        "engine": "duckduckgo/html",
        "query": query,
        "results": [],
        "error": errors[-1] if errors else "No DuckDuckGo results",
    }


def search_serper(query: str, max_results: int = DEFAULT_SEARCH_RESULTS) -> dict[str, Any]:
    query = normalize_search_query(query)
    serper_api_key = os.environ.get("SERPER_API_KEY", "")
    if not serper_api_key:
        return {"engine": "serper", "query": query, "results": [], "error": "SERPER_API_KEY not set"}
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": serper_api_key, "Content-Type": "application/json"},
        json={"q": query, "num": max_results},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    results: list[dict[str, str]] = []
    for item in data.get("organic", [])[:max_results]:
        results.append({
            "title": item.get("title", item.get("link", "")),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })
    return results


def open_url(url: str, max_chars: int = DEFAULT_FETCH_CHAR_LIMIT) -> dict[str, Any]:
    parsed = urlparse(str(url))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {
            "url": url,
            "error": {
                "type": "InvalidURL",
                "message": "open requires an absolute http(s) URL",
                "retryable": False,
            },
        }

    max_chars = max(1000, min(max_chars, 50000))
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    text = response.text
    if "html" in content_type.lower():
        parser = PageTextExtractor()
        parser.feed(text)
        text = parser.get_text()
    return {
        "url": response.url,
        "status_code": response.status_code,
        "content_type": content_type,
        "text": text[:max_chars],
    }


def make_tool_error_payload(error: Exception) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
        "retryable": False,
    }

    if isinstance(error, requests.HTTPError):
        response = error.response
        if response is not None:
            payload["status_code"] = response.status_code
            payload["url"] = response.url
            payload["retryable"] = response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
            response_text = response.text.strip()
            if response_text:
                payload["response_text"] = response_text[:1000]
    elif isinstance(error, requests.Timeout):
        payload["retryable"] = True
    elif isinstance(error, requests.RequestException):
        request = getattr(error, "request", None)
        if request is not None and getattr(request, "url", None):
            payload["url"] = request.url
        payload["retryable"] = True

    return payload


def make_tool_error_result(error: Exception) -> dict[str, Any]:
    payload = make_tool_error_payload(error)
    return {"error": payload}


def execute_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") or {}
    name = function.get("name", "")
    arguments = parse_tool_arguments(tool_call)

    try:
        if name in {"search", "search_web"}:
            result = search_web(query=str(arguments.get("query", "")))
        elif name in {"open", "open_url"}:
            result = open_url(url=str(arguments.get("url", "")))
        else:
            result = {"error": {"type": "UnknownTool", "message": f"Unknown tool: {name}", "retryable": False}}
    except Exception as error:
        result = make_tool_error_result(error)

    return {
        "tool_name": name,
        "arguments": arguments,
        "result": result,
    }


def make_task_id(question: str, index: int) -> str:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
    return f"browsecomp_{index:04d}_{digest}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_run_paths(output_dir: str, target_model: str) -> dict[str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", target_model)
    run_prefix = f"browsecomp_{safe_model}_{timestamp}"
    return {
        "run_prefix": run_prefix,
        "all_results_path": os.path.join(output_dir, f"{run_prefix}_all_results.json"),
        "raw_dir": os.path.join(output_dir, run_prefix),
    }


def write_raw_trajectory(raw_dir: str, task_id: str, payload: dict[str, Any]) -> str:
    ensure_dir(raw_dir)
    out_path = os.path.join(raw_dir, f"{task_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prune_output_fields(payload), f, indent=2)
    return out_path


def compute_accuracy(results: list[SampleResult]) -> float | None:
    if not results:
        return None
    return sum(1 for result in results if result.score is True) / len(results)


def format_accuracy(accuracy: float | None) -> str:
    return "n/a" if accuracy is None else f"{accuracy:.3f}"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def estimate_eta_seconds(
    *,
    elapsed_seconds: float,
    completed_pending: int,
    total_pending: int,
) -> float | None:
    if completed_pending <= 0:
        return None
    remaining = max(0, total_pending - completed_pending)
    return elapsed_seconds * remaining / completed_pending


def format_progress_log(
    *,
    results: list[SampleResult],
    total_examples: int,
    resumed_examples: int,
    total_pending: int,
    started_monotonic: float,
) -> str:
    elapsed_seconds = time.monotonic() - started_monotonic
    completed_pending = min(max(0, len(results) - resumed_examples), total_pending)
    eta_seconds = estimate_eta_seconds(
        elapsed_seconds=elapsed_seconds,
        completed_pending=completed_pending,
        total_pending=total_pending,
    )
    return (
        f"[{len(results)}/{total_examples}] "
        f"accuracy={format_accuracy(compute_accuracy(results))} "
        f"errors={sum(1 for item in results if item.error)} "
        f"tool_calls={sum(item.tool_call_count for item in results)} "
        f"search={sum(item.search_tool_call_count for item in results)} "
        f"open={sum(item.open_tool_call_count for item in results)} "
        f"other_tools={sum(item.other_tool_call_count for item in results)} "
        f"elapsed={format_duration(elapsed_seconds)} "
        f"eta={format_duration(eta_seconds)}"
    )


def build_resume_config(
    *,
    target_model: str,
    grader_model: str,
    target_base_url: str,
    grader_base_url: str | None,
    num_examples_requested: int | None,
    dataset_subset: str,
    temperature: float | None,
    max_tokens: int,
    grader_temperature: float,
    enable_search_tools: bool,
    search_engine: str,
    max_tool_rounds: int,
) -> dict[str, Any]:
    return {
        "eval_name": "browsecomp",
        "model_name": target_model,
        "grader_model": grader_model,
        "target_base_url": target_base_url,
        "grader_base_url": grader_base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "search_tools_enabled": enable_search_tools,
        "search_engine": search_engine,
        "max_tool_rounds": max_tool_rounds,
        "requested_examples": num_examples_requested,
        "grader_temperature": grader_temperature,
        "dataset_subset": dataset_subset,
    }


def make_eval_stats(
    results: list[SampleResult],
    *,
    total_examples: int,
    resumed_examples: int,
    evaluated_examples: int,
) -> dict[str, int]:
    scored_examples = sum(1 for result in results if result.score is not None)
    error_examples = sum(1 for result in results if result.error)
    total_tool_calls = sum(result.tool_call_count for result in results)
    total_search_tool_calls = sum(result.search_tool_call_count for result in results)
    total_open_tool_calls = sum(result.open_tool_call_count for result in results)
    total_other_tool_calls = sum(result.other_tool_call_count for result in results)
    return {
        "total_examples": total_examples,
        "scored_examples": scored_examples,
        "error_examples": error_examples,
        "unscored_examples": total_examples - scored_examples,
        "resumed_examples": resumed_examples,
        "evaluated_examples": evaluated_examples,
        "total_tool_calls": total_tool_calls,
        "total_search_tool_calls": total_search_tool_calls,
        "total_open_tool_calls": total_open_tool_calls,
        "total_other_tool_calls": total_other_tool_calls,
    }


def read_json_file(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def resolve_recorded_path(output_dir: str, recorded_path: Any) -> str:
    if not isinstance(recorded_path, str) or not recorded_path:
        return ""
    if os.path.isabs(recorded_path) or os.path.exists(recorded_path):
        return recorded_path
    candidate = os.path.join(output_dir, os.path.basename(recorded_path))
    return candidate if os.path.exists(candidate) else recorded_path


def manifest_matches_resume_config(
    manifest: dict[str, Any],
    resume_config: dict[str, Any],
) -> bool:
    return all(manifest.get(key) == value for key, value in resume_config.items())


def read_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def tool_call_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") or {}
    return str(function.get("name", ""))


def tool_call_count_bucket(name: str) -> str:
    if name in {"search", "search_web"}:
        return "search"
    if name in {"open", "open_url"}:
        return "open"
    return "other"


def result_from_record(
    record: dict[str, Any],
    *,
    expected_task_ids: dict[int, str],
) -> SampleResult | None:
    score = record.get("score")
    if not isinstance(score, bool) or record.get("error"):
        return None
    try:
        index = int(record["index"])
    except (KeyError, TypeError, ValueError):
        return None
    task_id = str(record.get("task_id", ""))
    expected_task_id = expected_task_ids.get(index)
    if expected_task_id is None or task_id != expected_task_id:
        return None
    return SampleResult(
        index=index,
        task_id=task_id,
        score=score,
        question=str(record.get("question", "")),
        correct_answer=str(record.get("correct_answer", "")),
        model_response=str(record.get("model_response", "")),
        grader_response=str(record.get("grader_response", "")),
        tool_call_count=read_int(record.get("tool_call_count"), 0),
        search_tool_call_count=read_int(record.get("search_tool_call_count"), 0),
        open_tool_call_count=read_int(record.get("open_tool_call_count"), 0),
        other_tool_call_count=read_int(record.get("other_tool_call_count"), 0),
        raw_trajectory_path=str(record.get("raw_trajectory_path", "")),
        error=None,
    )


def load_resume_results(
    *,
    output_dir: str,
    resume_config: dict[str, Any],
    expected_task_ids: dict[int, str],
) -> dict[int, SampleResult]:
    if not os.path.isdir(output_dir):
        return {}

    resumed: dict[int, SampleResult] = {}
    legacy_manifest_paths = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.endswith("_manifest.json")
    ]
    legacy_manifest_paths.sort(key=lambda path: os.path.getmtime(path))

    for manifest_path in legacy_manifest_paths:
        manifest = read_json_file(manifest_path)
        if not manifest or not manifest_matches_resume_config(manifest, resume_config):
            continue

        details_path = resolve_recorded_path(output_dir, manifest.get("details_path"))
        details = read_json_file(details_path)
        if not details:
            continue
        records = details.get("results", [])
        if not isinstance(records, list):
            continue

        for record in records:
            if not isinstance(record, dict):
                continue
            result = result_from_record(record, expected_task_ids=expected_task_ids)
            if result is not None:
                resumed[result.index] = result

    all_results_paths = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, name))
        and (name.endswith("_all_results.json") or name.endswith("_allresults.json"))
    ]
    all_results_paths.sort(key=lambda path: os.path.getmtime(path))

    for all_results_path in all_results_paths:
        all_results = read_json_file(all_results_path)
        if not all_results or not manifest_matches_resume_config(all_results, resume_config):
            continue

        records = all_results.get("results", [])
        if not isinstance(records, list):
            continue

        for record in records:
            if not isinstance(record, dict):
                continue
            result = result_from_record(record, expected_task_ids=expected_task_ids)
            if result is not None:
                resumed[result.index] = result

    return resumed


def derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def load_examples(num_examples: int | None) -> list[dict[str, str]]:
    with open(BROWSECOMP_SUBSET_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if num_examples is not None:
        if num_examples > len(rows):
            raise ValueError(
                f"--examples={num_examples} exceeds local BrowseComp subset size {len(rows)}"
            )
        rng = random.Random(0)
        rows = rng.sample(rows, num_examples)
    return rows


def expected_task_ids_for_rows(rows: list[dict[str, str]]) -> dict[int, str]:
    expected_task_ids: dict[int, str] = {}
    for idx, row in enumerate(rows, start=1):
        try:
            question = decrypt(row["problem"], row["canary"])
        except Exception:
            continue
        expected_task_ids[idx] = make_task_id(question, idx)
    return expected_task_ids


def run_target_sample(
    target_sampler: ChatCompletionsSampler,
    *,
    prompt: str,
    enable_search_tools: bool,
    max_tool_rounds: int,
) -> tuple[str, dict[str, Any]]:
    conversation: list[dict[str, Any]] = [target_sampler.pack_message("user", prompt)]
    transcript: list[dict[str, Any]] = list(target_sampler.prepare_messages(conversation))
    terminated_due_to_max_tool_rounds = False
    used_search_tools = False
    tool_call_count = 0
    search_tool_call_count = 0
    open_tool_call_count = 0
    other_tool_call_count = 0
    final_text = ""
    last_trace: dict[str, Any] | None = None

    for _ in range(max_tool_rounds + 1):
        content, trace = target_sampler.complete(
            conversation,
            tools=SEARCH_TOOLS if enable_search_tools else None,
            # tool_choice=None,
        )
        last_trace = trace
        assistant_message = trace["assistant_message"]
        transcript.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls") or []
        tool_call_count += len(tool_calls)
        for tool_call in tool_calls:
            bucket = tool_call_count_bucket(tool_call_name(tool_call))
            if bucket == "search":
                search_tool_call_count += 1
            elif bucket == "open":
                open_tool_call_count += 1
            else:
                other_tool_call_count += 1
        if enable_search_tools and tool_calls:
            used_search_tools = True
            msg = {
                "role": "assistant",
                "content": assistant_message.get("content"),
                "tool_calls": tool_calls,
                "reasoning_content": assistant_message.get("reasoning_content"),
            }
            conversation.append(msg)
            for tool_call in tool_calls:
                tool_result = execute_tool_call(tool_call)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "name": tool_result.get("tool_name", ""),
                    "content": json.dumps(tool_result.get("result"), ensure_ascii=False),
                }
                conversation.append(tool_message)
                transcript.append(tool_message)
            continue

        final_text = render_content(assistant_message.get("content")).strip() or content.strip()
        msg = {
            "role": "assistant",
            "content": assistant_message.get("content"),
            "reasoning_content": assistant_message.get("reasoning_content"),
        }
        conversation.append(msg)
        break
    else:
        terminated_due_to_max_tool_rounds = True

    if not final_text and conversation and conversation[-1].get("role") == "assistant":
        final_text = render_content(conversation[-1].get("content")).strip()

    return final_text, {
        "started_at": (last_trace or {}).get("started_at", ""),
        "completed_at": (last_trace or {}).get("completed_at", utc_now()),
        "transcript": transcript,
        "terminated_due_to_max_tool_rounds": terminated_due_to_max_tool_rounds,
        "used_search_tools": used_search_tools,
        "tool_call_count": tool_call_count,
        "search_tool_call_count": search_tool_call_count,
        "open_tool_call_count": open_tool_call_count,
        "other_tool_call_count": other_tool_call_count,
    }


def grade_sample(
    grader: ChatCompletionsSampler,
    *,
    question: str,
    correct_answer: str,
    response: str,
) -> tuple[bool, str, dict[str, Any]]:
    grader_prompt = GRADER_TEMPLATE.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )
    grader_response, grader_trace = grader.complete([grader.pack_message("user", grader_prompt)])
    match = re.search(r"correct:\s*(yes|no)", grader_response, re.IGNORECASE)
    return bool(match and match.group(1).lower() == "yes"), grader_response, grader_trace


def evaluate_sample(
    *,
    idx: int,
    row: dict[str, str],
    target_sampler: ChatCompletionsSampler,
    grader_sampler: ChatCompletionsSampler,
    raw_dir: str,
    enable_search_tools: bool,
    max_tool_rounds: int,
) -> SampleResult:
    question = decrypt(row["problem"], row["canary"])
    correct_answer = decrypt(row["answer"], row["canary"])
    task_id = make_task_id(question, idx)
    prompt = QUERY_TEMPLATE_NO_GET_DOCUMENT.format(Question=question)
    model_response, target_trace = run_target_sample(
        target_sampler,
        prompt=prompt,
        enable_search_tools=enable_search_tools,
        max_tool_rounds=max_tool_rounds,
    )
    tool_call_count = int(target_trace.get("tool_call_count", 0) or 0)
    search_tool_call_count = int(target_trace.get("search_tool_call_count", 0) or 0)
    open_tool_call_count = int(target_trace.get("open_tool_call_count", 0) or 0)
    other_tool_call_count = int(target_trace.get("other_tool_call_count", 0) or 0)
    grader_input = model_response
    if not grader_input.strip():
        error_message = "Grading skipped because target response was empty."
        grader_response = error_message
        raw_trajectory_path = write_raw_trajectory(
            raw_dir,
            task_id,
            {
                "eval_name": "browsecomp",
                "task_id": task_id,
                "index": idx,
                "created_at": utc_now(),
                "status": "error",
                "score": None,
                "question": question,
                "correct_answer": correct_answer,
                "tool_call_count": tool_call_count,
                "search_tool_call_count": search_tool_call_count,
                "open_tool_call_count": open_tool_call_count,
                "other_tool_call_count": other_tool_call_count,
                "error": {
                    "type": "EmptyGraderInput",
                    "message": error_message,
                },
                "target": target_trace,
                "grader": {
                    "skipped": True,
                    "reason": "empty_grader_input",
                    "input_text": grader_input,
                },
            },
        )
        return SampleResult(
            index=idx,
            task_id=task_id,
            score=None,
            question=question,
            correct_answer=correct_answer,
            model_response=model_response,
            grader_response=grader_response,
            tool_call_count=tool_call_count,
            search_tool_call_count=search_tool_call_count,
            open_tool_call_count=open_tool_call_count,
            other_tool_call_count=other_tool_call_count,
            raw_trajectory_path=raw_trajectory_path,
            error=error_message,
        )

    is_correct, grader_response, grader_trace = grade_sample(
        grader_sampler,
        question=question,
        correct_answer=correct_answer,
        response=grader_input,
    )
    raw_trajectory_path = write_raw_trajectory(
        raw_dir,
        task_id,
        {
            "eval_name": "browsecomp",
            "task_id": task_id,
            "index": idx,
            "created_at": utc_now(),
            "score": is_correct,
            "question": question,
            "correct_answer": correct_answer,
            "tool_call_count": tool_call_count,
            "search_tool_call_count": search_tool_call_count,
            "open_tool_call_count": open_tool_call_count,
            "other_tool_call_count": other_tool_call_count,
            "target": target_trace,
            "grader": {
                **grader_trace,
                "input_text": grader_input,
            },
        },
    )
    return SampleResult(
        index=idx,
        task_id=task_id,
        score=is_correct,
        question=question,
        correct_answer=correct_answer,
        model_response=model_response,
        grader_response=grader_response,
        tool_call_count=tool_call_count,
        search_tool_call_count=search_tool_call_count,
        open_tool_call_count=open_tool_call_count,
        other_tool_call_count=other_tool_call_count,
        raw_trajectory_path=raw_trajectory_path,
    )


def make_failed_sample_result(
    *,
    idx: int,
    row: dict[str, str],
    raw_dir: str,
    error: Exception,
) -> SampleResult:
    error_payload = make_exception_payload(error)
    try:
        question = decrypt(row["problem"], row["canary"])
        correct_answer = decrypt(row["answer"], row["canary"])
        task_id = make_task_id(question, idx)
    except Exception as decrypt_error:
        question = ""
        correct_answer = ""
        task_id = f"browsecomp_{idx:04d}_failed"
        error_payload["decrypt_error"] = make_exception_payload(decrypt_error)

    model_response = ""
    grader_response = "Grading skipped because sample execution failed."
    raw_trajectory_path = write_raw_trajectory(
        raw_dir,
        task_id,
        {
            "eval_name": "browsecomp",
            "task_id": task_id,
            "index": idx,
            "created_at": utc_now(),
            "status": "error",
            "score": None,
            "question": question,
            "correct_answer": correct_answer,
            "tool_call_count": 0,
            "search_tool_call_count": 0,
            "open_tool_call_count": 0,
            "other_tool_call_count": 0,
            "error": error_payload,
            "target": {
                "error": error_payload,
            },
            "grader": {
                "skipped": True,
            },
        },
    )
    return SampleResult(
        index=idx,
        task_id=task_id,
        score=None,
        question=question,
        correct_answer=correct_answer,
        model_response=model_response,
        grader_response=grader_response,
        tool_call_count=0,
        search_tool_call_count=0,
        open_tool_call_count=0,
        other_tool_call_count=0,
        raw_trajectory_path=raw_trajectory_path,
        error=error_payload["message"],
    )


def run_eval(
    *,
    target_sampler: ChatCompletionsSampler,
    grader_sampler: ChatCompletionsSampler,
    num_examples: int | None,
    output_dir: str,
    raw_dir: str,
    enable_search_tools: bool,
    max_tool_rounds: int,
    n_concurrent: int,
    resume_config: dict[str, Any],
) -> tuple[float | None, list[SampleResult], dict[str, int]]:
    rows = load_examples(num_examples)
    expected_task_ids = expected_task_ids_for_rows(rows)
    resumed_by_index = load_resume_results(
        output_dir=output_dir,
        resume_config=resume_config,
        expected_task_ids=expected_task_ids,
    )
    pending_items = [
        (idx, row)
        for idx, row in enumerate(rows, start=1)
        if idx not in resumed_by_index
    ]
    results: list[SampleResult] = [
        resumed_by_index[idx] for idx in sorted(resumed_by_index)
    ]

    if resumed_by_index:
        print(
            (
                f"Resumed {len(resumed_by_index)} scored samples from {output_dir}; "
                f"evaluating {len(pending_items)} errored or missing samples."
            ),
            flush=True,
        )

    progress_started_monotonic = time.monotonic()
    resumed_count = len(resumed_by_index)
    pending_count = len(pending_items)

    if n_concurrent <= 1:
        for idx, row in pending_items:
            try:
                result = evaluate_sample(
                    idx=idx,
                    row=row,
                    target_sampler=target_sampler.clone(),
                    grader_sampler=grader_sampler.clone(),
                    raw_dir=raw_dir,
                    enable_search_tools=enable_search_tools,
                    max_tool_rounds=max_tool_rounds,
                )
            except Exception as error:
                result = make_failed_sample_result(idx=idx, row=row, raw_dir=raw_dir, error=error)
                print(f"[{idx}/{len(rows)}] sample failed: {type(error).__name__}: {error}", flush=True)
            results.append(result)
            print(
                format_progress_log(
                    results=results,
                    total_examples=len(rows),
                    resumed_examples=resumed_count,
                    total_pending=pending_count,
                    started_monotonic=progress_started_monotonic,
                ),
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
            future_to_idx = {
                executor.submit(
                    evaluate_sample,
                    idx=idx,
                    row=row,
                    target_sampler=target_sampler.clone(),
                    grader_sampler=grader_sampler.clone(),
                    raw_dir=raw_dir,
                    enable_search_tools=enable_search_tools,
                    max_tool_rounds=max_tool_rounds,
                ): idx
                for idx, row in pending_items
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = make_failed_sample_result(
                        idx=idx,
                        row=rows[idx - 1],
                        raw_dir=raw_dir,
                        error=error,
                    )
                    print(f"[{idx}/{len(rows)}] sample failed: {type(error).__name__}: {error}", flush=True)
                results.append(result)
                print(
                    format_progress_log(
                        results=results,
                        total_examples=len(rows),
                        resumed_examples=resumed_count,
                        total_pending=pending_count,
                        started_monotonic=progress_started_monotonic,
                    ),
                    flush=True,
                )

    results.sort(key=lambda item: item.index)

    accuracy = compute_accuracy(results)
    stats = make_eval_stats(
        results,
        total_examples=len(rows),
        resumed_examples=len(resumed_by_index),
        evaluated_examples=len(pending_items),
    )
    return accuracy, results, stats


def write_outputs(
    *,
    run_paths: dict[str, str],
    target_model: str,
    grader_model: str,
    target_base_url: str,
    grader_base_url: str | None,
    num_examples_requested: int | None,
    dataset_subset: str,
    temperature: float | None,
    max_tokens: int,
    grader_temperature: float,
    enable_search_tools: bool,
    search_engine: str,
    max_tool_rounds: int,
    n_concurrent: int,
    run_started_at: str,
    run_completed_at: str,
    accuracy: float | None,
    results: list[SampleResult],
    stats: dict[str, int],
) -> str:
    all_results_path = run_paths["all_results_path"]
    raw_dir = run_paths["raw_dir"]

    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(
            prune_output_fields({
                "eval_name": "browsecomp",
                "run_prefix": run_paths["run_prefix"],
                "model_name": target_model,
                "grader_model": grader_model,
                "target_base_url": target_base_url,
                "grader_base_url": grader_base_url,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "grader_temperature": grader_temperature,
                "search_tools_enabled": enable_search_tools,
                "search_engine": search_engine,
                "max_tool_rounds": max_tool_rounds,
                "n_concurrent": n_concurrent,
                "dataset_subset": dataset_subset,
                "requested_examples": num_examples_requested,
                "num_examples": len(results),
                "completed_examples": len(results),
                "scored_examples": stats["scored_examples"],
                "error_examples": stats["error_examples"],
                "failed_examples": stats["error_examples"],
                "unscored_examples": stats["unscored_examples"],
                "resumed_examples": stats["resumed_examples"],
                "evaluated_examples": stats["evaluated_examples"],
                "total_tool_calls": stats["total_tool_calls"],
                "total_search_tool_calls": stats["total_search_tool_calls"],
                "total_open_tool_calls": stats["total_open_tool_calls"],
                "total_other_tool_calls": stats["total_other_tool_calls"],
                "accuracy": accuracy,
                "stats": stats,
                "run_started_at": run_started_at,
                "run_completed_at": run_completed_at,
                "raw_trajectory_dir": raw_dir,
                "results": [
                    {
                        "index": result.index,
                        "task_id": result.task_id,
                        "status": "error" if result.error else "scored",
                        "score": result.score,
                        "question": result.question,
                        "correct_answer": result.correct_answer,
                        "model_response": result.model_response,
                        "grader_response": result.grader_response,
                        "tool_call_count": result.tool_call_count,
                        "search_tool_call_count": result.search_tool_call_count,
                        "open_tool_call_count": result.open_tool_call_count,
                        "other_tool_call_count": result.other_tool_call_count,
                        "raw_trajectory_path": result.raw_trajectory_path,
                        "error": result.error,
                    }
                    for result in results
                ],
            }),
            f,
            indent=2,
        )

    return all_results_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BrowseComp against an OpenAI-compatible chat completions endpoint "
            "while keeping the grader on standard OpenAI by default."
        )
    )
    parser.add_argument("--model", required=True, help="Target model name on the compatible endpoint")
    parser.add_argument("--base-url", required=True, help="Target endpoint base URL, e.g. http://host:8000/v1")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TARGET_OPENAI_API_KEY"),
        help="Target endpoint API key. Defaults to TARGET_OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=None,
        help="Number of examples to sample from the local BrowseComp subset. Omit to run the full subset.",
    )
    parser.add_argument(
        "--grader-model",
        default=DEFAULT_GRADER_MODEL,
        help="Standard OpenAI grader model.",
    )
    parser.add_argument(
        "--grader-api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="Standard OpenAI API key for grading. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--grader-base-url",
        default=None,
        help="Optional alternate base URL for the grader. Leave unset to use standard OpenAI.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(".", "results", "browsecomp"),
        help="Directory for all-results JSON and trajectory dumps.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for the target model. Omit to use the endpoint default.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max completion tokens for the target model.",
    )
    parser.add_argument(
        "--disable-search-tools",
        action="store_true",
        help="Disable the local search tool and query the model in a single shot.",
    )
    parser.add_argument(
        "--search-engine",
        choices=SEARCH_ENGINE_CHOICES,
        default=os.environ.get("BROWSECOMP_SEARCH_ENGINE", DEFAULT_SEARCH_ENGINE).lower(),
        help=(
            "Search provider for the search tool. Use duckduckgo to avoid Serper credits, "
            "serper for the paid Serper API, or auto to try Serper then fall back to DuckDuckGo."
        ),
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=DEFAULT_MAX_TOOL_ROUNDS,
        help="Maximum assistant tool-calling rounds for the target model.",
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=32,
        help="Number of samples to evaluate concurrently.",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("Missing target API key. Pass --api-key or set TARGET_OPENAI_API_KEY.")
    if not args.grader_api_key:
        parser.error("Missing grader API key. Pass --grader-api-key or set OPENAI_API_KEY.")
    if args.search_engine not in SEARCH_ENGINE_CHOICES:
        parser.error(f"--search-engine must be one of: {', '.join(SEARCH_ENGINE_CHOICES)}")
    return args


def main() -> None:
    args = parse_args()
    global ACTIVE_SEARCH_ENGINE
    ACTIVE_SEARCH_ENGINE = args.search_engine
    ensure_dir(args.output_dir)
    run_paths = build_run_paths(args.output_dir, args.model)
    ensure_dir(run_paths["raw_dir"])
    run_started_at = utc_now()

    target_sampler = ChatCompletionsSampler(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        system_message=None,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    grader_sampler = ChatCompletionsSampler(
        model=args.grader_model,
        api_key=args.grader_api_key,
        base_url=args.grader_base_url,
        system_message=None,
        temperature=DEFAULT_GRADER_TEMPERATURE,
        max_tokens=20000,
    )
    resume_config = build_resume_config(
        target_model=args.model,
        grader_model=args.grader_model,
        target_base_url=args.base_url,
        grader_base_url=args.grader_base_url,
        num_examples_requested=args.examples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        grader_temperature=DEFAULT_GRADER_TEMPERATURE,
        enable_search_tools=not args.disable_search_tools,
        search_engine=args.search_engine,
        max_tool_rounds=args.max_tool_rounds,
        dataset_subset=os.path.basename(BROWSECOMP_SUBSET_PATH),
    )

    accuracy, results, stats = run_eval(
        target_sampler=target_sampler,
        grader_sampler=grader_sampler,
        num_examples=args.examples,
        output_dir=args.output_dir,
        raw_dir=run_paths["raw_dir"],
        enable_search_tools=not args.disable_search_tools,
        max_tool_rounds=args.max_tool_rounds,
        n_concurrent=args.n_concurrent,
        resume_config=resume_config,
    )
    run_completed_at = utc_now()
    all_results_path = write_outputs(
        run_paths=run_paths,
        target_model=args.model,
        grader_model=args.grader_model,
        target_base_url=args.base_url,
        grader_base_url=args.grader_base_url,
        num_examples_requested=args.examples,
        dataset_subset=os.path.basename(BROWSECOMP_SUBSET_PATH),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        grader_temperature=DEFAULT_GRADER_TEMPERATURE,
        enable_search_tools=not args.disable_search_tools,
        search_engine=args.search_engine,
        max_tool_rounds=args.max_tool_rounds,
        n_concurrent=args.n_concurrent,
        run_started_at=run_started_at,
        run_completed_at=run_completed_at,
        accuracy=accuracy,
        results=results,
        stats=stats,
    )

    print(f"Final accuracy: {format_accuracy(accuracy)}")
    print(f"Scored examples: {stats['scored_examples']}/{stats['total_examples']}")
    print(f"Error examples: {stats['error_examples']}")
    print(
        (
            f"Total tool calls: {stats['total_tool_calls']} "
            f"(search={stats['total_search_tool_calls']}, "
            f"open={stats['total_open_tool_calls']}, "
            f"other={stats['total_other_tool_calls']})"
        )
    )
    print(f"All results written to {all_results_path}")
    print(f"Raw trajectories written to {run_paths['raw_dir']}")


if __name__ == "__main__":
    main()
