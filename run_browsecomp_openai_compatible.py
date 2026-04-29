import argparse
import base64
import csv
import hashlib
import io
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


BROWSECOMP_DATASET_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
)
DEFAULT_TARGET_SYSTEM_MESSAGE = """
You are a web research assistant solving BrowseComp questions.

Use the provided web search tools to gather evidence when they help answer the question.
Do not rely only on internal memory or produce long unsupported reasoning.
Search the web, open promising pages, gather evidence, then answer when you have enough support.

When you are ready to answer, use exactly this format:
Explanation: <brief evidence-backed explanation>
Exact Answer: <succinct final answer>
Confidence: <0% to 100%>
""".strip()
DEFAULT_GRADER_SYSTEM_MESSAGE = "You are a helpful assistant."
DEFAULT_GRADER_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_MAX_TOOL_ROUNDS = 12
DEFAULT_SEARCH_RESULTS = 5
DEFAULT_FETCH_CHAR_LIMIT = 12000
DEFAULT_SEARCH_ENGINE = "duckduckgo"
SEARCH_ENGINE_CHOICES = ("duckduckgo", "serper", "auto")
ACTIVE_SEARCH_ENGINE = os.environ.get("BROWSECOMP_SEARCH_ENGINE", DEFAULT_SEARCH_ENGINE).lower()
MAX_SEARCH_QUERY_CHARS = 512

QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

GRADER_TEMPLATE = """
Judge whether the following response to the question is correct based on the precise and unambiguous correct answer below.

Question: {question}

Response: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the response. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

Correct answer: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on the correct answer, focusing only on meaningful differences between the correct answer and the extracted_final_answer. Do not comment on background to the problem, do not attempt to solve the problem, and do not argue for any answer different than the correct answer.

correct: Answer 'yes' if extracted_final_answer matches the correct answer above, or is within a small margin of error for numerical problems. Answer 'no' otherwise.

confidence: The extracted confidence score between 0% and 100% from the response. Put 100 if there is no confidence score available.
""".strip()

SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the public web for pages relevant to the question. "
                "Use this first to discover candidate pages before opening them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Web search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return.",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": (
                "Fetch the text content of a web page so you can read evidence from it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute URL to fetch."},
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum number of characters to keep from the fetched page.",
                        "minimum": 1000,
                        "maximum": 50000,
                    },
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
    score: bool
    question: str
    correct_answer: str
    model_response: str
    extracted_final_answer: str
    grader_response: str
    raw_trajectory_path: str


class ChatCompletionsSampler:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        system_message: str | None = None,
        temperature: float = 0.5,
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
        while True:
            started_at = utc_now()
            try:
                request_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": message_list,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
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
                return content, {
                    "request": {
                        "model": self.model,
                        "messages": message_list,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "base_url": self.base_url,
                        "system_message": self.system_message,
                        "tools": tools,
                        "tool_choice": tool_choice or ("auto" if tools else None),
                    },
                    "response": response_dict,
                    "assistant_message": assistant_message,
                    "started_at": started_at,
                    "completed_at": utc_now(),
                }
            except Exception:
                if trial >= 5:
                    raise
                time.sleep(2**trial)
                trial += 1

    def __call__(self, message_list: list[dict[str, Any]]) -> str:
        content, _ = self.complete(message_list)
        return content


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "model_dump_json"):
        return json.loads(response.model_dump_json())
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
        return {"_raw": raw_args}


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


def search_web(query: str, max_results: int = DEFAULT_SEARCH_RESULTS) -> dict[str, Any]:
    max_results = max(1, min(max_results, 10))
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
    return {
        "engine": "serper",
        "query": query,
        "results": results,
    }


def open_url(url: str, max_chars: int = DEFAULT_FETCH_CHAR_LIMIT) -> dict[str, Any]:
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
    text = text[:max_chars]
    return {
        "url": url,
        "status_code": response.status_code,
        "content_type": content_type,
        "text": text,
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
        if name == "search_web":
            result = search_web(
                query=str(arguments.get("query", "")),
                max_results=int(arguments.get("max_results", DEFAULT_SEARCH_RESULTS)),
            )
        elif name == "open_url":
            result = open_url(
                url=str(arguments.get("url", "")),
                max_chars=int(arguments.get("max_chars", DEFAULT_FETCH_CHAR_LIMIT)),
            )
        else:
            result = {"error": {"type": "UnknownTool", "message": f"Unknown tool: {name}", "retryable": False}}
    except Exception as error:
        result = make_tool_error_result(error)

    return {
        "tool_name": name,
        "arguments": arguments,
        "result": result,
    }


def extract_exact_answer(response_text: str) -> str:
    if not response_text:
        return ""

    match = re.search(
        r"(?im)^Exact Answer:\s*(.+?)(?=^\w[\w ]*:\s|\Z)",
        response_text.strip(),
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    return ""


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
        "summary_path": os.path.join(output_dir, f"{run_prefix}.json"),
        "details_path": os.path.join(output_dir, f"{run_prefix}_allresults.json"),
        "manifest_path": os.path.join(output_dir, f"{run_prefix}_manifest.json"),
        "raw_dir": os.path.join(output_dir, f"{run_prefix}_raw"),
    }


def write_raw_trajectory(raw_dir: str, task_id: str, payload: dict[str, Any]) -> str:
    ensure_dir(raw_dir)
    out_path = os.path.join(raw_dir, f"{task_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


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
    response = requests.get(BROWSECOMP_DATASET_URL, timeout=120)
    response.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(response.text)))
    if num_examples is not None:
        rng = random.Random(0)
        rows = rng.sample(rows, num_examples)
    return rows


def run_target_sample(
    target_sampler: ChatCompletionsSampler,
    *,
    prompt: str,
    enable_search_tools: bool,
    max_tool_rounds: int,
) -> tuple[str, dict[str, Any]]:
    conversation: list[dict[str, Any]] = [target_sampler.pack_message("user", prompt)]
    transcript: list[dict[str, Any]] = list(target_sampler.prepare_messages(conversation))
    response_history: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    terminated_due_to_max_tool_rounds = False
    used_search_tools = False
    final_text = ""
    last_trace: dict[str, Any] | None = None

    for _ in range(max_tool_rounds + 1):
        content, trace = target_sampler.complete(
            conversation,
            tools=SEARCH_TOOLS if enable_search_tools else None,
            tool_choice=None,
        )
        last_trace = trace
        assistant_message = trace["assistant_message"]
        response_history.append(trace["response"])
        transcript.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls") or []
        step_record: dict[str, Any] = {
            "assistant_message": assistant_message,
            "response": trace["response"],
            "started_at": trace["started_at"],
            "completed_at": trace["completed_at"],
            "tool_calls": tool_calls,
            "observations": [],
        }
        if enable_search_tools and tool_calls:
            used_search_tools = True
            conversation.append({
                "role": "assistant",
                "content": assistant_message.get("content"),
                "tool_calls": tool_calls,
            })
            for tool_call in tool_calls:
                tool_result = execute_tool_call(tool_call)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
                conversation.append(tool_message)
                transcript.append(tool_message)
                step_record["observations"].append({
                    "tool_call_id": tool_call.get("id", ""),
                    "tool_name": tool_result.get("tool_name", ""),
                    "arguments": tool_result.get("arguments", {}),
                    "result": tool_result.get("result", {}),
                })
            steps.append(step_record)
            continue

        final_text = render_content(assistant_message.get("content")).strip() or content.strip()
        steps.append(step_record)
        conversation.append({
            "role": "assistant",
            "content": assistant_message.get("content"),
        })
        break
    else:
        terminated_due_to_max_tool_rounds = True

    if not final_text and conversation and conversation[-1].get("role") == "assistant":
        final_text = render_content(conversation[-1].get("content")).strip()

    final_answer = extract_exact_answer(final_text)
    if not final_answer and final_text:
        final_answer = final_text.strip()

    return final_text, {
        "request": (last_trace or {}).get("request", {}),
        "response": (last_trace or {}).get("response", response_history[-1] if response_history else {}),
        "responses": response_history,
        "assistant_message": (last_trace or {}).get("assistant_message", transcript[-1] if transcript else {}),
        "started_at": (last_trace or {}).get("started_at", ""),
        "completed_at": (last_trace or {}).get("completed_at", utc_now()),
        "transcript": transcript,
        "steps": steps,
        "terminated_due_to_max_tool_rounds": terminated_due_to_max_tool_rounds,
        "used_search_tools": used_search_tools,
        "final_answer": final_answer,
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
    prompt = QUERY_TEMPLATE.format(Question=question)
    model_response, target_trace = run_target_sample(
        target_sampler,
        prompt=prompt,
        enable_search_tools=enable_search_tools,
        max_tool_rounds=max_tool_rounds,
    )
    extracted_final_answer = target_trace.get("final_answer") or extract_exact_answer(model_response)
    grader_input = extracted_final_answer or model_response
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
            "prompt": prompt,
            "extracted_final_answer": extracted_final_answer,
            "target": {
                **target_trace,
                "text": model_response,
            },
            "grader": {
                **grader_trace,
                "input_text": grader_input,
                "text": grader_response,
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
        extracted_final_answer=extracted_final_answer,
        grader_response=grader_response,
        raw_trajectory_path=raw_trajectory_path,
    )


def run_eval(
    *,
    target_sampler: ChatCompletionsSampler,
    grader_sampler: ChatCompletionsSampler,
    num_examples: int | None,
    raw_dir: str,
    enable_search_tools: bool,
    max_tool_rounds: int,
    n_concurrent: int,
) -> tuple[float, list[SampleResult]]:
    results: list[SampleResult] = []
    rows = load_examples(num_examples)

    if n_concurrent <= 1:
        for idx, row in enumerate(rows, start=1):
            result = evaluate_sample(
                idx=idx,
                row=row,
                target_sampler=target_sampler.clone(),
                grader_sampler=grader_sampler.clone(),
                raw_dir=raw_dir,
                enable_search_tools=enable_search_tools,
                max_tool_rounds=max_tool_rounds,
            )
            results.append(result)
            running_accuracy = sum(item.score for item in results) / len(results)
            print(f"[{len(results)}/{len(rows)}] accuracy={running_accuracy:.3f}")
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
                for idx, row in enumerate(rows, start=1)
            }
            for future in as_completed(future_to_idx):
                result = future.result()
                results.append(result)
                running_accuracy = sum(item.score for item in results) / len(results)
                print(f"[{len(results)}/{len(rows)}] accuracy={running_accuracy:.3f}")

    results.sort(key=lambda item: item.index)

    accuracy = sum(result.score for result in results) / len(results)
    return accuracy, results


def write_outputs(
    *,
    run_paths: dict[str, str],
    target_model: str,
    grader_model: str,
    target_base_url: str,
    grader_base_url: str | None,
    num_examples_requested: int | None,
    temperature: float,
    max_tokens: int,
    enable_search_tools: bool,
    search_engine: str,
    max_tool_rounds: int,
    n_concurrent: int,
    run_started_at: str,
    run_completed_at: str,
    accuracy: float,
    results: list[SampleResult],
) -> tuple[str, str, str]:
    summary_path = run_paths["summary_path"]
    details_path = run_paths["details_path"]
    manifest_path = run_paths["manifest_path"]
    raw_dir = run_paths["raw_dir"]

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_name": "browsecomp",
                "model_name": target_model,
                "grader_model": grader_model,
                "accuracy": accuracy,
                "num_examples": len(results),
                "raw_trajectory_dir": raw_dir,
                "search_tools_enabled": enable_search_tools,
                "search_engine": search_engine,
                "max_tool_rounds": max_tool_rounds,
                "n_concurrent": n_concurrent,
                "run_started_at": run_started_at,
                "run_completed_at": run_completed_at,
            },
            f,
            indent=2,
        )

    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_name": "browsecomp",
                "model_name": target_model,
                "accuracy": accuracy,
                "results": [
                    {
                        "index": result.index,
                        "task_id": result.task_id,
                        "score": result.score,
                        "question": result.question,
                        "correct_answer": result.correct_answer,
                        "model_response": result.model_response,
                        "extracted_final_answer": result.extracted_final_answer,
                        "grader_response": result.grader_response,
                        "raw_trajectory_path": result.raw_trajectory_path,
                    }
                    for result in results
                ],
            },
            f,
            indent=2,
        )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_name": "browsecomp",
                "run_prefix": run_paths["run_prefix"],
                "model_name": target_model,
                "grader_model": grader_model,
                "target_base_url": target_base_url,
                "grader_base_url": grader_base_url,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "search_tools_enabled": enable_search_tools,
                "search_engine": search_engine,
                "max_tool_rounds": max_tool_rounds,
                "n_concurrent": n_concurrent,
                "requested_examples": num_examples_requested,
                "completed_examples": len(results),
                "accuracy": accuracy,
                "run_started_at": run_started_at,
                "run_completed_at": run_completed_at,
                "summary_path": summary_path,
                "details_path": details_path,
                "raw_trajectory_dir": raw_dir,
                "task_ids": [result.task_id for result in results],
            },
            f,
            indent=2,
        )

    return summary_path, details_path, manifest_path


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
        help="Number of BrowseComp examples to run. Omit to run the full set.",
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
        help="Directory for summary, detailed results, and raw trajectory dumps.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature for the target model.",
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
        help="Disable the local search/open-url tools and query the model in a single shot.",
    )
    parser.add_argument(
        "--search-engine",
        choices=SEARCH_ENGINE_CHOICES,
        default=os.environ.get("BROWSECOMP_SEARCH_ENGINE", DEFAULT_SEARCH_ENGINE).lower(),
        help=(
            "Search provider for search_web. Use duckduckgo to avoid Serper credits, "
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
        system_message=DEFAULT_TARGET_SYSTEM_MESSAGE,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    grader_sampler = ChatCompletionsSampler(
        model=args.grader_model,
        api_key=args.grader_api_key,
        base_url=args.grader_base_url,
        system_message=DEFAULT_GRADER_SYSTEM_MESSAGE,
        temperature=0.0,
        max_tokens=2048,
    )

    accuracy, results = run_eval(
        target_sampler=target_sampler,
        grader_sampler=grader_sampler,
        num_examples=args.examples,
        raw_dir=run_paths["raw_dir"],
        enable_search_tools=not args.disable_search_tools,
        max_tool_rounds=args.max_tool_rounds,
        n_concurrent=args.n_concurrent,
    )
    run_completed_at = utc_now()
    summary_path, details_path, manifest_path = write_outputs(
        run_paths=run_paths,
        target_model=args.model,
        grader_model=args.grader_model,
        target_base_url=args.base_url,
        grader_base_url=args.grader_base_url,
        num_examples_requested=args.examples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_search_tools=not args.disable_search_tools,
        search_engine=args.search_engine,
        max_tool_rounds=args.max_tool_rounds,
        n_concurrent=args.n_concurrent,
        run_started_at=run_started_at,
        run_completed_at=run_completed_at,
        accuracy=accuracy,
        results=results,
    )

    print(f"Final accuracy: {accuracy:.3f}")
    print(f"Summary written to {summary_path}")
    print(f"Detailed results written to {details_path}")
    print(f"Run manifest written to {manifest_path}")
    print(f"Raw trajectories written to {run_paths['raw_dir']}")


if __name__ == "__main__":
    main()
