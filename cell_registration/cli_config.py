"""Configuration objects used by the TopoAlign command-line interface.

This module intentionally stays separate from the napari plugin configuration.
The CLI can therefore evolve without changing the plugin's public behavior.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal


TransformMethod = Literal["rigid", "similarity", "affine"]
InputMode = Literal["auto", "image", "mask", "features"]


@dataclass
class SegmentationOptions:
    channel_axis: Literal["auto", "first", "last", "none"] = "auto"
    registration_channel: int = -1
    gpu: bool = False
    min_area: int | None = None
    max_area: int | None = None


@dataclass
class MatchingOptions:
    top_k: int = 50
    feature_weight: float = 1.0
    topology_weight: float = 0.35
    position_weight: float = 4.0
    distance_threshold: float | None = 2.0
    spatial_window_size: float | None = 100.0
    use_spatial_clusters: bool = False
    n_clusters: int = 9
    use_topology_filtering: bool = False
    k_pos_nei: int = 5
    k_neighbor: int = 5
    tau_pos: float = 0.5
    tau_nei: float = 0.3
    tau_map: float = 0.3


@dataclass
class TransformOptions:
    method: TransformMethod = "rigid"
    use_ransac: bool = False
    ransac_max_trials: int = 1000
    ransac_residual_threshold: float = 2.0
    residual_prune_quantile: float | None = None
    initial_coarse_transform: str | None = None


@dataclass
class OutputOptions:
    output_dir: str = "outputs/topoalign-run"
    save_overlay: bool = True
    save_registered_moving: bool = True
    save_features: bool = True
    save_matches: bool = True


@dataclass
class TopoAlignConfig:
    """Serializable configuration shared by CLI stages and the agent."""

    mode: InputMode = "auto"
    fixed: str | None = None
    moving: str | None = None
    fixed_mask: str | None = None
    moving_mask: str | None = None
    fixed_features: str | None = None
    moving_features: str | None = None
    matches: str | None = None
    fixed_shape: tuple[int, int] | None = None
    segmentation: SegmentationOptions = field(default_factory=SegmentationOptions)
    matching: MatchingOptions = field(default_factory=MatchingOptions)
    transform: TransformOptions = field(default_factory=TransformOptions)
    output: OutputOptions = field(default_factory=OutputOptions)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def from_json(cls, path: str | Path) -> "TopoAlignConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("TopoAlign config must be a JSON object.")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TopoAlignConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown TopoAlign config keys: {', '.join(unknown)}")

        def nested(name: str, typ: type[Any]) -> Any:
            value = payload.get(name)
            if value is None:
                return typ()
            if not isinstance(value, dict):
                raise ValueError(f"Config section '{name}' must be an object.")
            allowed = {f.name for f in fields(typ)}
            extra = sorted(set(value) - allowed)
            if extra:
                raise ValueError(f"Unknown keys in '{name}': {', '.join(extra)}")
            return typ(**value)

        kwargs = {key: payload[key] for key in known if key not in {"segmentation", "matching", "transform", "output"} and key in payload}
        kwargs["segmentation"] = nested("segmentation", SegmentationOptions)
        kwargs["matching"] = nested("matching", MatchingOptions)
        kwargs["transform"] = nested("transform", TransformOptions)
        kwargs["output"] = nested("output", OutputOptions)
        if kwargs.get("fixed_shape") is not None:
            kwargs["fixed_shape"] = tuple(int(v) for v in kwargs["fixed_shape"])
        return cls(**kwargs)

    def merged(self, **updates: Any) -> "TopoAlignConfig":
        """Return a copy with top-level values replaced."""
        return replace(self, **updates)
