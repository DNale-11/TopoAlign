"""Structured result and diagnostics objects for TopoAlign CLI runs."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageResult:
    name: str
    status: str = "completed"
    seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistrationResult:
    product: str = "TopoAlign"
    status: str = "completed"
    run_id: str | None = None
    stages: list[StageResult] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return target


class StageTimer:
    def __init__(self, result: RegistrationResult, name: str):
        self.result = result
        self.name = name
        self.started = 0.0
        self.stage: StageResult | None = None

    def __enter__(self) -> "StageTimer":
        self.started = time.perf_counter()
        self.stage = StageResult(name=self.name, status="running")
        self.result.stages.append(self.stage)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self.stage is not None
        self.stage.seconds = time.perf_counter() - self.started
        self.stage.status = "failed" if exc is not None else "completed"
        if exc is not None:
            self.stage.details["error"] = str(exc)
        return False
