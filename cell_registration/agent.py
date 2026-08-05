"""Optional API-key agent for the TopoAlign CLI.

The agent can inspect the project and call a small, explicit set of local
TopoAlign operations.  It cannot execute arbitrary shell commands or write
source files.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import imageio.v3 as iio
import numpy as np
import pandas as pd

from .cli_config import TopoAlignConfig
from .i18n import normalize_language, tr
from .service import (
    _load_mask,
    _load_table,
    estimate_transform_from_matches,
    extract_features,
    match_features,
    register,
    segment_image,
)
from .settings import SettingsDocument, apply_agent_environment, load_settings
from .warp import load_transform, warp_array


MAX_READ_BYTES = 200_000
MAX_SEARCH_RESULTS = 80
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", "outputs", "external_methods", "benchmark"}


def _safe_path(workspace: Path, value: str | Path, *, must_exist: bool = False) -> Path:
    root = workspace.resolve()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path is outside the TopoAlign workspace: {value}") from exc
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def _json_result(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def _inspect_project(workspace: Path) -> dict[str, Any]:
    files: list[str] = []
    for path in workspace.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(str(path.relative_to(workspace)))
        if len(files) >= 500:
            break
    return {"workspace": str(workspace), "files": sorted(files), "file_count_sampled": len(files)}


def _search_code(workspace: Path, pattern: str, path_filter: str | None = None) -> dict[str, Any]:
    expression = re.compile(pattern, re.IGNORECASE)
    results: list[dict[str, Any]] = []
    for path in workspace.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path_filter and path_filter.lower() not in str(path.relative_to(workspace)).lower():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if expression.search(line):
                results.append({"path": str(path.relative_to(workspace)), "line": line_number, "text": line[:500]})
                if len(results) >= MAX_SEARCH_RESULTS:
                    return {"pattern": pattern, "results": results, "truncated": True}
    return {"pattern": pattern, "results": results, "truncated": False}


def _read_code_file(workspace: Path, path_value: str, start_line: int = 1, end_line: int = 200) -> dict[str, Any]:
    path = _safe_path(workspace, path_value, must_exist=True)
    if path.suffix.lower() not in {".py", ".md", ".toml", ".json", ".yaml", ".yml"}:
        raise ValueError("Only source and configuration files can be read by the agent.")
    data = path.read_bytes()
    if len(data) > MAX_READ_BYTES:
        raise ValueError(f"File is larger than the agent read limit ({MAX_READ_BYTES} bytes).")
    lines = data.decode("utf-8", errors="replace").splitlines()
    start = max(1, int(start_line))
    end = min(len(lines), max(start, int(end_line)))
    return {"path": str(path.relative_to(workspace)), "start_line": start, "end_line": end, "text": "\n".join(lines[start - 1:end])}


def _inspect_input(workspace: Path, path_value: str) -> dict[str, Any]:
    path = _safe_path(workspace, path_value, must_exist=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, nrows=5)
        return {"path": str(path.relative_to(workspace)), "kind": "csv", "columns": list(frame.columns), "preview": frame.to_dict(orient="records")}
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return {"path": str(path.relative_to(workspace)), "kind": "json", "payload": payload}
    array = np.asarray(iio.imread(path))
    return {"path": str(path.relative_to(workspace)), "kind": "array", "shape": list(array.shape), "dtype": str(array.dtype), "min": float(np.min(array)) if array.size else None, "max": float(np.max(array)) if array.size else None}


def _dispatch(name: str, arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    if name == "inspect_project":
        return _inspect_project(workspace)
    if name == "search_code":
        return _search_code(workspace, str(arguments["pattern"]), arguments.get("path_filter"))
    if name == "read_code_file":
        return _read_code_file(workspace, str(arguments["path"]), arguments.get("start_line", 1), arguments.get("end_line", 200))
    if name == "inspect_input":
        return _inspect_input(workspace, str(arguments["path"]))
    if name == "validate_config":
        config = TopoAlignConfig.from_json(_safe_path(workspace, arguments["config"], must_exist=True)) if arguments.get("config") else TopoAlignConfig.from_dict(arguments.get("payload", {}))
        return {"valid": True, "config": config.to_dict()}
    if name == "run_registration":
        payload = dict(arguments.get("config", {}))
        for key in ("fixed", "moving", "fixed_mask", "moving_mask", "fixed_features", "moving_features", "matches"):
            if payload.get(key):
                payload[key] = str(_safe_path(workspace, payload[key], must_exist=True))
        output = payload.get("output", {})
        if not isinstance(output, dict):
            raise ValueError("config.output must be an object")
        output = dict(output)
        output["output_dir"] = str(_safe_path(workspace, output.get("output_dir", "outputs/agent-run")))
        payload["output"] = output
        result = register(TopoAlignConfig.from_dict(payload))
        return result.to_dict()
    if name == "segment":
        image = _safe_path(workspace, arguments["image"], must_exist=True)
        output = _safe_path(workspace, arguments.get("output_dir", "outputs/agent-segment"))
        mask, metadata = segment_image(image, output, channel_axis=arguments.get("channel_axis", "auto"), registration_channel=int(arguments.get("registration_channel", -1)), gpu=bool(arguments.get("gpu", False)))
        return {"mask": str(mask.shape), "metadata": str(metadata), "output_dir": str(output)}
    if name == "extract_features":
        mask = _safe_path(workspace, arguments["mask"], must_exist=True)
        output = _safe_path(workspace, arguments.get("output_dir", "outputs/agent-features"))
        features, metadata = extract_features(mask, output)
        return {"cell_count": len(features), "metadata": str(metadata), "output_dir": str(output)}
    if name == "match":
        fixed = _load_table(_safe_path(workspace, arguments["fixed_features"], must_exist=True))
        moving = _load_table(_safe_path(workspace, arguments["moving_features"], must_exist=True))
        config = TopoAlignConfig.from_dict(arguments.get("config", {}))
        shape = tuple(int(v) for v in arguments["fixed_shape"])
        matches = match_features(fixed, moving, shape, config)
        output = _safe_path(workspace, arguments.get("output_dir", "outputs/agent-match"))
        output.mkdir(parents=True, exist_ok=True)
        matches.to_csv(output / "matches.csv", index=False)
        return {"match_count": len(matches), "matches": str(output / "matches.csv")}
    if name == "estimate_transform":
        fixed = _load_table(_safe_path(workspace, arguments["fixed_features"], must_exist=True))
        moving = _load_table(_safe_path(workspace, arguments["moving_features"], must_exist=True))
        matches = pd.read_csv(_safe_path(workspace, arguments["matches"], must_exist=True))
        transform, table = estimate_transform_from_matches(fixed, moving, matches, method=arguments.get("method", "rigid"))
        return {"direction": "moving_to_fixed", "matrix": np.asarray(transform.params).tolist(), "match_count": len(table)}
    if name == "warp":
        moving_path = _safe_path(workspace, arguments["moving"], must_exist=True)
        transform_path = _safe_path(workspace, arguments["transform"], must_exist=True)
        output = _safe_path(workspace, arguments.get("output_dir", "outputs/agent-warp"))
        output.mkdir(parents=True, exist_ok=True)
        moving = np.asarray(iio.imread(moving_path))
        warped = warp_array(moving, load_transform(transform_path), tuple(int(v) for v in arguments["fixed_shape"]), channel_axis=arguments.get("channel_axis", "auto"), order=0 if arguments.get("mask") else 1)
        target = output / "registered_moving.tif"
        import tifffile

        tifffile.imwrite(target, warped)
        return {"registered_moving": str(target)}
    if name == "read_result":
        path = _safe_path(workspace, arguments["path"], must_exist=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {"path": str(path.relative_to(workspace)), "result": payload}
    if name == "list_artifacts":
        directory = _safe_path(workspace, arguments["directory"], must_exist=True)
        return {"directory": str(directory.relative_to(workspace)), "files": [str(p.relative_to(workspace)) for p in sorted(directory.iterdir()) if p.is_file()]}
    raise ValueError(f"Agent tool is not allowlisted: {name}")


TOOL_NAMES = (
    "inspect_project", "search_code", "read_code_file", "inspect_input", "validate_config",
    "segment", "extract_features", "match", "estimate_transform", "warp", "run_registration",
    "read_result", "list_artifacts",
)


def _tool_schema(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        # Conversational tools intentionally have optional arguments. OpenAI
        # strict schemas require every property to be required or nullable.
        "strict": False,
    }


TOOLS = [
    _tool_schema("inspect_project", "List relevant TopoAlign project files.", {}, []),
    _tool_schema("search_code", "Search source code without modifying it.", {"pattern": {"type": "string"}, "path_filter": {"type": ["string", "null"]}}, ["pattern"]),
    _tool_schema("read_code_file", "Read a bounded source/config excerpt.", {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
    _tool_schema("inspect_input", "Inspect an image, CSV, or JSON input.", {"path": {"type": "string"}}, ["path"]),
    _tool_schema("validate_config", "Validate a TopoAlign JSON config or object.", {"config": {"type": ["string", "null"]}, "payload": {"type": ["object", "null"]}}, []),
    _tool_schema("run_registration", "Run a configured TopoAlign registration within the workspace.", {"config": {"type": "object"}}, ["config"]),
    _tool_schema("segment", "Segment an image into a label mask.", {"image": {"type": "string"}, "output_dir": {"type": "string"}, "channel_axis": {"type": "string"}, "registration_channel": {"type": "integer"}, "gpu": {"type": "boolean"}}, ["image"]),
    _tool_schema("extract_features", "Extract features from a label mask.", {"mask": {"type": "string"}, "output_dir": {"type": "string"}}, ["mask"]),
    _tool_schema("match", "Match two feature CSV tables.", {"fixed_features": {"type": "string"}, "moving_features": {"type": "string"}, "fixed_shape": {"type": "array", "items": {"type": "integer"}}, "output_dir": {"type": "string"}, "config": {"type": "object"}}, ["fixed_features", "moving_features", "fixed_shape"]),
    _tool_schema("estimate_transform", "Estimate a moving-to-fixed transform.", {"fixed_features": {"type": "string"}, "moving_features": {"type": "string"}, "matches": {"type": "string"}, "method": {"type": "string"}}, ["fixed_features", "moving_features", "matches"]),
    _tool_schema("warp", "Warp moving data into fixed coordinates.", {"moving": {"type": "string"}, "transform": {"type": "string"}, "fixed_shape": {"type": "array", "items": {"type": "integer"}}, "output_dir": {"type": "string"}, "channel_axis": {"type": "string"}, "mask": {"type": "boolean"}}, ["moving", "transform", "fixed_shape"]),
    _tool_schema("read_result", "Read a result manifest.", {"path": {"type": "string"}}, ["path"]),
    _tool_schema("list_artifacts", "List files in a workspace output directory.", {"directory": {"type": "string"}}, ["directory"]),
]


SYSTEM_PROMPT = """You are the TopoAlign CLI agent. Explain the project and operate only through the provided allowlisted tools. Never request or invent shell commands, never modify source files, and never access paths outside the workspace. Before registration, inspect or validate inputs when useful. Always state the fixed/reference space, moving/source space, transform direction, and output paths. Keep API keys private."""

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class _ActivityIndicator:
    """Show a neutral heartbeat while a blocking model turn is running."""

    def __init__(self, language: str) -> None:
        self._label = tr("agent_thinking", language)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._width = len(self._label) + 3
        self._enabled = bool(getattr(sys.stdout, "isatty", lambda: False)())

    def __enter__(self) -> "_ActivityIndicator":
        if not self._enabled:
            return self
        sys.stdout.write(f"\n{self._label}")
        sys.stdout.flush()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def _animate(self) -> None:
        dots = 0
        while not self._stop.wait(0.45):
            dots = dots % 3 + 1
            sys.stdout.write(f"\r{self._label}{'.' * dots}{' ' * (3 - dots)}")
            sys.stdout.flush()

    def __exit__(self, *_: object) -> None:
        if not self._enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        sys.stdout.write("\r" + " " * self._width + "\r")
        sys.stdout.flush()


def create_agent_client(*, api_key: str, base_url: str | None = None, timeout: float = 30.0) -> Any:
    """Create an OpenAI-compatible client without exposing credentials."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Agent requires the optional openai package; install it with `pip install openai`."
        ) from exc
    endpoint = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).rstrip("/")
    return OpenAI(api_key=api_key, base_url=endpoint, timeout=timeout, max_retries=0)


def check_agent_connection(*, api_key: str, base_url: str | None, model: str) -> dict[str, Any]:
    """Make a minimal Responses API call to validate the complete Agent path."""
    endpoint = (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
    try:
        client = create_agent_client(api_key=api_key, base_url=endpoint, timeout=15.0)
        response = client.responses.create(
            model=model,
            input="Reply with OK.",
            max_output_tokens=16,
        )
        return {
            "ok": True,
            "base_url": endpoint,
            "model": model,
            "response_id": getattr(response, "id", None),
        }
    except Exception as exc:
        return {
            "ok": False,
            "base_url": endpoint,
            "model": model,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


def _response_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    return ""


def _run_model_turn(
    client: Any,
    prompt: str,
    model: str,
    workspace: Path,
    previous_response_id: str | None = None,
    *,
    language: str = "en",
    max_tool_rounds: int = 8,
) -> tuple[str, str | None]:
    language_instruction = "Reply in Simplified Chinese." if normalize_language(language) == "zh" else "Reply in English."
    instructions = f"{SYSTEM_PROMPT}\n{language_instruction}"
    kwargs: dict[str, Any] = {"model": model, "instructions": instructions, "tools": TOOLS}
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id
        kwargs["input"] = prompt
    else:
        kwargs["input"] = prompt
    response = client.responses.create(**kwargs)
    for _ in range(max(1, int(max_tool_rounds))):
        calls = [item for item in getattr(response, "output", []) if getattr(item, "type", None) == "function_call"]
        if not calls:
            return _response_output_text(response), getattr(response, "id", None)
        outputs = []
        for call in calls:
            name = str(getattr(call, "name", ""))
            if name not in TOOL_NAMES:
                value = {"error": f"Tool is not allowlisted: {name}"}
            else:
                try:
                    arguments = json.loads(getattr(call, "arguments", "{}"))
                    value = _dispatch(name, arguments, workspace)
                except Exception as exc:  # tool failures are returned to the model for recovery
                    value = {"error": str(exc), "type": type(exc).__name__}
            outputs.append({"type": "function_call_output", "call_id": getattr(call, "call_id"), "output": _json_result(value)})
        response = client.responses.create(model=model, instructions=instructions, tools=TOOLS, previous_response_id=response.id, input=outputs)
    return "Agent stopped after the maximum tool-call rounds.", getattr(response, "id", None)


def _latest_result_path(workspace: Path) -> Path | None:
    candidates = [path for path in workspace.rglob("result.json") if path.is_file() and not any(part in SKIP_DIRS - {"outputs"} for part in path.parts)]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def _active_settings(workspace: Path, settings_path: Path | None) -> SettingsDocument:
    return load_settings(workspace, settings_path) if settings_path else load_settings(workspace)


def _print_latest_result(workspace: Path, *, artifacts_only: bool, language: str) -> None:
    result_path = _latest_result_path(workspace)
    if result_path is None:
        print(tr("no_result", language))
        return
    payload = json.loads(result_path.read_text(encoding="utf-8-sig"))
    value = payload.get("artifacts", {}) if artifacts_only and isinstance(payload, dict) else payload
    print(json.dumps({"path": str(result_path), "value": value}, ensure_ascii=False, indent=2))


def run_agent(
    *,
    prompt: str | None,
    api_key: str | None,
    workspace: Path,
    base_url: str | None = None,
    model: str | None = None,
    settings_path: str | Path | None = None,
    language: str | None = None,
) -> int:
    root = workspace.resolve()
    settings_file = Path(settings_path).resolve() if settings_path else None
    settings = _active_settings(root, settings_file)
    apply_agent_environment(settings)
    lang = normalize_language(language or settings.language)
    key = api_key or os.environ.get("OPENAI_API_KEY") or settings.api_key()
    if not key:
        print("Agent requires OPENAI_API_KEY or --api-key.")
        return 2
    endpoint = base_url or os.environ.get("OPENAI_BASE_URL") or str(settings.agent.get("base_url") or DEFAULT_OPENAI_BASE_URL)
    try:
        client = create_agent_client(api_key=key, base_url=endpoint)
    except ImportError as exc:
        print(exc)
        return 2
    selected_model = model or os.environ.get("TOPOALIGN_AGENT_MODEL") or str(settings.agent.get("model") or "gpt-5")
    max_tool_rounds = int(settings.agent.get("max_tool_rounds", 8))
    previous_response_id: str | None = None
    if prompt:
        try:
            with _ActivityIndicator(lang):
                text, _ = _run_model_turn(
                    client,
                    prompt,
                    selected_model,
                    root,
                    previous_response_id,
                    language=lang,
                    max_tool_rounds=max_tool_rounds,
                )
            print(text)
            return 0
        except Exception as exc:
            print(f"Agent error: {exc}")
            return 2
    print("\n" + tr("agent_session", lang, model=selected_model, endpoint=endpoint))
    print(tr("workspace", lang, workspace=root))
    print(tr("agent_exit_hint", lang))
    while True:
        try:
            user_prompt = input(tr("you_prompt", lang)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_prompt:
            continue
        lowered = user_prompt.lower()
        if lowered in {"exit", "quit", ":q", "/exit"}:
            print(tr("returning", lang))
            return 0

        if user_prompt.startswith("/"):
            command, _, argument = user_prompt.partition(" ")
            command = command.lower()
            argument = argument.strip()
            if command in {"/+", "/help"}:
                print(tr("slash_help", lang))
                continue
            if command == "/model":
                if not argument:
                    print(tr("current_model", lang, model=selected_model))
                    continue
                result = check_agent_connection(api_key=key, base_url=endpoint, model=argument)
                if not result["ok"]:
                    print(tr("model_failed", lang, error=result["error"]))
                    continue
                selected_model = argument
                os.environ["TOPOALIGN_AGENT_MODEL"] = selected_model
                settings.agent["model"] = selected_model
                settings.save()
                previous_response_id = None
                print(tr("model_switched", lang, model=selected_model))
                continue
            if command == "/api":
                result = check_agent_connection(api_key=key, base_url=endpoint, model=selected_model)
                if result["ok"]:
                    print(tr("api_ok", lang, endpoint=endpoint, model=selected_model))
                else:
                    print(tr("api_failed", lang, kind=result["error_type"], error=result["error"]))
                continue
            if command == "/config":
                print(json.dumps(settings.sanitized(), ensure_ascii=False, indent=2))
                continue
            if command == "/reload":
                settings = _active_settings(root, settings_file)
                apply_agent_environment(settings)
                lang = settings.language
                key = settings.api_key() or key
                endpoint = str(settings.agent.get("base_url") or endpoint)
                selected_model = str(settings.agent.get("model") or selected_model)
                max_tool_rounds = int(settings.agent.get("max_tool_rounds", 8))
                client = create_agent_client(api_key=key, base_url=endpoint)
                previous_response_id = None
                print(tr("config_reloaded", lang, path=settings.path))
                continue
            if command == "/tools":
                print("\n".join(f"- {name}" for name in TOOL_NAMES))
                continue
            if command == "/local":
                if not argument:
                    print(tr("local_usage", lang))
                    continue
                from .cli import _run_local_command

                _run_local_command(argument, lang)
                continue
            if command == "/result":
                _print_latest_result(root, artifacts_only=False, language=lang)
                continue
            if command == "/artifacts":
                _print_latest_result(root, artifacts_only=True, language=lang)
                continue
            if command == "/clear":
                previous_response_id = None
                print(tr("context_cleared", lang))
                continue
            print(tr("slash_help", lang))
            continue
        try:
            with _ActivityIndicator(lang):
                text, previous_response_id = _run_model_turn(
                    client,
                    user_prompt,
                    selected_model,
                    root,
                    previous_response_id,
                    language=lang,
                    max_tool_rounds=max_tool_rounds,
                )
            print(tr("agent_reply", lang, text=text))
        except Exception as exc:
            print(tr("agent_error", lang, error=exc))
