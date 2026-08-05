"""Persistent application settings for the TopoAlign CLI and Agent."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cli_config import TopoAlignConfig


PROJECT_CONFIG_NAME = "topoalign.config.json"
USER_CONFIG_PATH = Path.home() / ".topoalign" / "config.json"


def default_settings() -> dict[str, Any]:
    return {
        "version": 1,
        "cli": {
            "language": "en",
            "show_banner": True,
            "workspace": ".",
        },
        "agent": {
            "enabled": False,
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "api_key_env": "OPENAI_API_KEY",
            "model": "gpt-5",
            "timeout_seconds": 30,
            "max_tool_rounds": 8,
        },
        "registration": TopoAlignConfig().to_dict(),
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Settings file must contain a JSON object: {path}")
    return payload


@dataclass
class SettingsDocument:
    data: dict[str, Any]
    path: Path
    sources: list[Path]

    @property
    def language(self) -> str:
        value = str(self.data.get("cli", {}).get("language", "en")).lower()
        return "zh" if value.startswith("zh") else "en"

    @property
    def agent(self) -> dict[str, Any]:
        return self.data["agent"]

    @property
    def registration(self) -> dict[str, Any]:
        return self.data["registration"]

    def api_key(self) -> str | None:
        direct = str(self.agent.get("api_key") or "")
        if direct:
            return direct
        env_name = str(self.agent.get("api_key_env") or "OPENAI_API_KEY")
        return os.environ.get(env_name) or None

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.path not in self.sources:
            self.sources.append(self.path)
        return self.path

    def sanitized(self) -> dict[str, Any]:
        payload = deepcopy(self.data)
        agent = payload.setdefault("agent", {})
        agent["api_key"] = "configured" if self.api_key() else ""
        return payload


def load_settings(workspace: str | Path, explicit_path: str | Path | None = None) -> SettingsDocument:
    root = Path(workspace).resolve()
    project_path = root / PROJECT_CONFIG_NAME
    sources: list[Path] = []
    merged = default_settings()

    if explicit_path is not None:
        target = Path(explicit_path).resolve()
        if target.exists():
            merged = _deep_merge(merged, _read_json(target))
            sources.append(target)
        return SettingsDocument(data=merged, path=target, sources=sources)

    if USER_CONFIG_PATH.exists():
        merged = _deep_merge(merged, _read_json(USER_CONFIG_PATH))
        sources.append(USER_CONFIG_PATH)
    if project_path.exists():
        merged = _deep_merge(merged, _read_json(project_path))
        sources.append(project_path)

    target = project_path if project_path.exists() or not sources else USER_CONFIG_PATH
    return SettingsDocument(data=merged, path=target, sources=sources)


def apply_agent_environment(settings: SettingsDocument) -> None:
    agent = settings.agent
    key = settings.api_key()
    if key:
        os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_BASE_URL"] = str(agent.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    os.environ["TOPOALIGN_AGENT_MODEL"] = str(agent.get("model") or "gpt-5")
    os.environ["TOPOALIGN_LANGUAGE"] = settings.language
