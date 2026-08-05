"""TopoAlign command-line interface.

The CLI is intentionally separate from the legacy ``main.py`` entry point.
All commands return structured data first and only format a short human or
JSON summary at the boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pandas as pd
import tifffile

from .cli_config import TopoAlignConfig
from .config import DEFAULT_FEATURE_CONFIG
from .features import compute_cell_features
from .i18n import current_language, tr
from .results import RegistrationResult
from .settings import SettingsDocument, apply_agent_environment, load_settings
from .service import (
    _load_mask,
    _load_table,
    estimate_transform_from_matches,
    extract_features,
    match_features,
    register,
    segment_image,
)
from .warp import load_transform, valid_overlap_mask, warp_array


def _json_summary(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _print_brand(language: str = "en") -> None:
    """Render a terminal-native version of the TopoAlign logo."""
    logo = r"""
     .--------.  .--------.          .------------.
    / o----o /  / o----o /    >>>   |  o------o  |
   /  |\  /|/  /  |\  /|/           |  |\ /\ /|  |
  /   o-\/ o  /   o-\/ o            |  o--o--o  |
  '--------'   '--------'             '------------'
"""
    try:
        from rich.console import Console
        from rich import box
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        print(logo)
        print("TopoAlign CLI")
        print("Topology-guided cellular image registration\n")
        return

    console = Console()
    artwork = Text(logo, style="bold cyan")
    artwork.append("\n                     TOPO", style="bold white")
    artwork.append("ALIGN", style="bold bright_cyan")
    artwork.append(f"\n       {tr('tagline', language)}", style="dim white")
    console.print(
        Panel(
            artwork,
            border_style="cyan",
            box=box.ASCII,
            padding=(0, 2),
            width=min(console.width, 78),
        )
    )


LOCAL_COMMANDS = {"run", "segment", "features", "match", "transform", "warp", "inspect"}


def _split_interactive_command(command: str) -> list[str]:
    tokens = shlex.split(command, posix=False)
    return [token[1:-1] if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"} else token for token in tokens]


def _complete_run_settings(settings: SettingsDocument, language: str) -> bool:
    registration = settings.registration
    mode = str(registration.get("mode") or "auto")
    if mode == "auto":
        mode = "image"
    requirements = {
        "image": ("fixed", "moving"),
        "mask": ("fixed_mask", "moving_mask"),
        "features": ("fixed_features", "moving_features"),
    }
    required = requirements.get(mode, requirements["image"])
    if all(registration.get(key) for key in required):
        return True

    print(tr("run_setup", language, path=settings.path))
    selected = input(tr("input_mode", language, value=mode)).strip().lower() or mode
    mode = selected if selected in requirements else "image"
    registration["mode"] = mode
    prompt_keys = {
        "image": (("fixed", "fixed_image"), ("moving", "moving_image")),
        "mask": (("fixed_mask", "fixed_mask"), ("moving_mask", "moving_mask")),
        "features": (("fixed_features", "fixed_features"), ("moving_features", "moving_features")),
    }
    for key, message_key in prompt_keys[mode]:
        if not registration.get(key):
            registration[key] = input(tr(message_key, language)).strip() or None
    if not all(registration.get(key) for key, _ in prompt_keys[mode]):
        print(tr("missing_run_input", language))
        return False
    if mode == "features":
        registration["matches"] = input(tr("matches_optional", language)).strip() or registration.get("matches")
        if not registration.get("fixed_shape"):
            shape = input(tr("fixed_shape", language)).strip().split()
            if len(shape) == 2 and all(value.isdigit() for value in shape):
                registration["fixed_shape"] = [int(shape[0]), int(shape[1])]
    output = registration.setdefault("output", {})
    output["output_dir"] = input(tr("output_dir", language, value=output.get("output_dir", "outputs/topoalign-run"))).strip() or output.get("output_dir", "outputs/topoalign-run")
    transform = registration.setdefault("transform", {})
    transform["method"] = input(tr("method_prompt", language, value=transform.get("method", "rigid"))).strip() or transform.get("method", "rigid")
    settings.save()
    return True


def _run_local_command(
    command: str,
    language: str | None = None,
    settings: SettingsDocument | None = None,
) -> int:
    """Execute a normal TopoAlign subcommand without leaving the shell."""
    try:
        argv = _split_interactive_command(command)
    except ValueError as exc:
        print(tr("command_parse_failed", language, error=exc))
        return 2
    if not argv or argv[0] not in LOCAL_COMMANDS:
        print(tr("local_command_expected", language))
        return 2
    if argv == ["run"]:
        active_settings = settings or load_settings(Path.cwd())
        if not _complete_run_settings(active_settings, language or active_settings.language):
            return 2
    try:
        return main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


def _configure_agent_connection(settings: SettingsDocument, language: str) -> bool:
    """Collect and validate an optional Agent endpoint for this process."""
    from .agent import DEFAULT_OPENAI_BASE_URL

    current_url = str(settings.agent.get("base_url") or DEFAULT_OPENAI_BASE_URL)
    current_model = str(settings.agent.get("model") or "gpt-5")
    existing_key = settings.api_key()
    print("\n" + tr("config_title", language, path=settings.path))
    base_url = input(tr("api_url", language, value=current_url)).strip() or current_url
    key = input(tr("api_key_keep" if existing_key else "api_key", language)).strip() or existing_key
    if not key:
        print(tr("missing_key", language))
        return False
    model = input(tr("model", language, value=current_model)).strip() or current_model

    settings.agent.update(
        {
            "enabled": True,
            "base_url": base_url.rstrip("/"),
            "api_key": key,
            "model": model,
        }
    )
    settings.save()
    apply_agent_environment(settings)
    return _validate_current_agent_connection(language)


def _validate_current_agent_connection(language: str) -> bool:
    """Validate the Agent settings currently held in this process."""
    from .agent import DEFAULT_OPENAI_BASE_URL, check_agent_connection

    key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    model = os.environ.get("TOPOALIGN_AGENT_MODEL", "gpt-5")
    if not key:
        os.environ.pop("TOPOALIGN_AGENT_VALIDATED", None)
        return False
    print(tr("checking", language))
    result = check_agent_connection(api_key=key, base_url=base_url, model=model)
    if result["ok"]:
        os.environ["TOPOALIGN_AGENT_VALIDATED"] = "1"
        print(tr("check_ok", language, endpoint=result["base_url"], model=result["model"]))
        return True
    os.environ["TOPOALIGN_AGENT_VALIDATED"] = "0"
    print(tr("check_failed", language, kind=result["error_type"], error=result["error"]))
    print(tr("local_still_ready", language))
    return False


def _interactive_cli() -> int:
    """Start the persistent no-argument CLI shell."""
    settings = load_settings(Path.cwd())
    if not settings.sources:
        settings.save()
    apply_agent_environment(settings)
    language = settings.language
    if bool(settings.data.get("cli", {}).get("show_banner", True)):
        _print_brand(language)
    print(tr("config_loaded", language, path=settings.path))
    if settings.api_key():
        _validate_current_agent_connection(language)
    while True:
        configured = bool(os.environ.get("OPENAI_API_KEY"))
        validated = os.environ.get("TOPOALIGN_AGENT_VALIDATED") == "1"
        endpoint = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("TOPOALIGN_AGENT_MODEL", "gpt-5")
        status_key = "agent_valid" if validated else "agent_unvalidated" if configured else "agent_optional"
        print("\n" + tr("local_ready", language))
        print(tr("agent_status", language, status=tr(status_key, language), endpoint=endpoint, model=model))
        print(tr("direct_hint", language))
        print(tr("menu_agent", language))
        print(tr("menu_config", language))
        print(tr("menu_help", language))
        print(tr("menu_exit", language))
        print(tr("menu_language", language))
        try:
            choice = input("TopoAlign> ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            print("\n" + tr("exit", language))
            return 0

        if choice.lower() in {"agent", "/agent"}:
            choice = "1"
        elif choice.lower() in {"config", "/config"}:
            choice = "2"
        elif choice.lower() in {"help", "/help", "?"}:
            choice = "3"
        elif choice.lower() in {"language", "/language", "lang", "/lang"}:
            choice = "5"

        if choice == "1":
            if not configured and not _configure_agent_connection(settings, language):
                continue
            if configured and not validated and not _validate_current_agent_connection(language):
                continue
            from .agent import run_agent

            run_agent(
                prompt=None,
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
                model=None,
                workspace=Path.cwd(),
                settings_path=settings.path,
                language=language,
            )
            settings = load_settings(Path.cwd(), settings.path)
            apply_agent_environment(settings)
            language = settings.language
            print(tr("agent_return", language))
            continue

        if choice == "2":
            _configure_agent_connection(settings, language)
            continue

        if choice == "3":
            print()
            build_parser().print_help()
            continue

        if choice == "4" or choice.lower() in {"exit", "quit", ":q"}:
            print(tr("exit", language))
            return 0

        if choice == "5":
            selected = input(tr("language_prompt", language)).strip() or "1"
            language = "zh" if selected in {"2", "zh", "zh-cn", "中文"} else "en"
            settings.data.setdefault("cli", {})["language"] = language
            settings.save()
            apply_agent_environment(settings)
            print(tr("language_saved", language, selected="中文" if language == "zh" else "English"))
            continue

        if choice.split(maxsplit=1)[0].lower() in LOCAL_COMMANDS:
            _run_local_command(choice, language, settings)
            continue

        print(tr("unknown", language))


def _add_common_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fixed", type=Path, help="Fixed/reference image path.")
    parser.add_argument("--moving", type=Path, help="Moving/source image path.")
    parser.add_argument("--fixed-mask", type=Path, help="Existing fixed label mask.")
    parser.add_argument("--moving-mask", type=Path, help="Existing moving label mask.")
    parser.add_argument("--fixed-features", type=Path, help="Existing fixed feature CSV.")
    parser.add_argument("--moving-features", type=Path, help="Existing moving feature CSV.")
    parser.add_argument("--matches", type=Path, help="Existing match CSV.")
    parser.add_argument("--mode", choices=("auto", "image", "mask", "features"), default=None)


def _add_segmentation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--channel-axis", choices=("auto", "first", "last", "none"), default=None)
    parser.add_argument("--registration-channel", type=int, default=None)
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-area", type=int, default=None)
    parser.add_argument("--max-area", type=int, default=None)


def _add_matching_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--feature-weight", type=float, default=None)
    parser.add_argument("--topology-weight", type=float, default=None)
    parser.add_argument("--position-weight", type=float, default=None)
    parser.add_argument("--distance-threshold", type=float, default=None)
    parser.add_argument("--spatial-window-size", type=float, default=None)
    parser.add_argument("--use-spatial-clusters", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--use-topology-filtering", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--k-pos-nei", type=int, default=None)
    parser.add_argument("--k-neighbor", type=int, default=None)
    parser.add_argument("--tau-pos", type=float, default=None)
    parser.add_argument("--tau-nei", type=float, default=None)
    parser.add_argument("--tau-map", type=float, default=None)


def _add_transform_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=("rigid", "similarity", "affine"), default=None)
    parser.add_argument("--use-ransac", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ransac-max-trials", type=int, default=None)
    parser.add_argument("--ransac-residual-threshold", type=float, default=None)
    parser.add_argument("--residual-prune-quantile", type=float, default=None)
    parser.add_argument("--initial-coarse-transform", type=Path, default=None)


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print the structured result as JSON.")


def _label(english: str, chinese: str) -> str:
    return chinese if current_language() == "zh" else english


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="topoalign",
        description=_label(
            "TopoAlign: conversational agent and topology-guided cellular image registration.",
            "TopoAlign：对话式 Agent 与拓扑引导的细胞图像配准。",
        ),
        epilog=_label(
            "Run without a subcommand to enter the interactive TopoAlign shell.",
            "不带子命令运行即可进入 TopoAlign 交互界面。",
        ),
    )
    parser.add_argument("--version", action="version", version="TopoAlign 0.1.0")
    parser.add_argument("--prompt", dest="direct_prompt", help=_label("One-shot prompt for Agent mode.", "Agent 单次提示词。"))
    parser.add_argument("--api-key", dest="direct_api_key", help=_label("API key for Agent mode.", "Agent 使用的 API key。"))
    parser.add_argument("--base-url", dest="direct_base_url", help=_label("OpenAI-compatible API base URL.", "OpenAI-compatible API 基础地址。"))
    parser.add_argument("--model", dest="direct_model", help=_label("Model for Agent mode.", "Agent 使用的模型。"))
    parser.add_argument("--workspace", dest="direct_workspace", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=False)

    run_parser = sub.add_parser("run", help=_label("Run a complete registration pipeline.", "运行完整配准流程。"))
    _add_common_input_args(run_parser)
    _add_segmentation_args(run_parser)
    _add_matching_args(run_parser)
    _add_transform_args(run_parser)
    _add_output_args(run_parser)
    run_parser.add_argument("--config", type=Path, help=_label("JSON registration configuration file.", "JSON 配准配置文件。"))

    segment_parser = sub.add_parser("segment", help=_label("Segment an image into a label mask.", "将图像分割为标签 mask。"))
    segment_parser.add_argument("--image", type=Path, required=True)
    segment_parser.add_argument("--output-dir", type=Path, required=True)
    segment_parser.add_argument("--channel-axis", choices=("auto", "first", "last", "none"), default="auto")
    segment_parser.add_argument("--registration-channel", type=int, default=-1)
    segment_parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=False)

    features_parser = sub.add_parser("features", help=_label("Extract cell features from a label mask.", "从标签 mask 提取细胞特征。"))
    features_parser.add_argument("--mask", type=Path, required=True)
    features_parser.add_argument("--output-dir", type=Path, required=True)
    features_parser.add_argument("--min-area", type=int, default=None)
    features_parser.add_argument("--max-area", type=int, default=None)

    match_parser = sub.add_parser("match", help=_label("Match fixed and moving cell features.", "匹配 fixed 与 moving 细胞特征。"))
    match_parser.add_argument("--fixed-features", type=Path)
    match_parser.add_argument("--moving-features", type=Path)
    match_parser.add_argument("--fixed-mask", type=Path)
    match_parser.add_argument("--moving-mask", type=Path)
    match_parser.add_argument("--fixed-shape", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), help="Fixed image shape; inferred from a fixed mask/features when omitted.")
    match_parser.add_argument("--output-dir", type=Path, required=True)
    _add_matching_args(match_parser)
    _add_transform_args(match_parser)
    match_parser.add_argument("--json", action="store_true")

    transform_parser = sub.add_parser("transform", help=_label("Estimate a transform from features and matches.", "根据特征和匹配估计变换。"))
    transform_parser.add_argument("--fixed-features", type=Path, required=True)
    transform_parser.add_argument("--moving-features", type=Path, required=True)
    transform_parser.add_argument("--matches", type=Path, required=True)
    transform_parser.add_argument("--output-dir", type=Path, required=True)
    _add_transform_args(transform_parser)
    transform_parser.add_argument("--json", action="store_true")

    warp_parser = sub.add_parser("warp", help=_label("Warp a moving image or mask into fixed coordinates.", "将 moving 图像或 mask 变换到 fixed 坐标。"))
    warp_parser.add_argument("--moving", type=Path, required=True)
    warp_parser.add_argument("--transform", type=Path, required=True)
    warp_parser.add_argument("--fixed-shape", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), required=True)
    warp_parser.add_argument("--output-dir", type=Path, required=True)
    warp_parser.add_argument("--channel-axis", choices=("auto", "first", "last", "none"), default="auto")
    warp_parser.add_argument("--mask", action="store_true", help="Use nearest-neighbor interpolation for a label mask.")
    warp_parser.add_argument("--json", action="store_true")

    inspect_parser = sub.add_parser("inspect", help=_label("Inspect a config, result manifest, or input artifact.", "检查配置、结果清单或输入文件。"))
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.add_argument("--json", action="store_true")

    agent_parser = sub.add_parser("agent", help=_label("Use an API-key agent to understand and operate TopoAlign.", "使用 API-key Agent 理解并操作 TopoAlign。"))
    agent_parser.add_argument("--prompt", help="One-shot agent instruction; omit for interactive REPL.")
    agent_parser.add_argument("--api-key", help="OpenAI API key; prefer OPENAI_API_KEY.")
    agent_parser.add_argument("--base-url", help="OpenAI-compatible API base URL; defaults to OPENAI_BASE_URL.")
    agent_parser.add_argument("--model", default=None, help="Agent model; defaults to TOPOALIGN_AGENT_MODEL or gpt-5.")
    agent_parser.add_argument("--workspace", type=Path, default=Path.cwd())

    return parser


def _config_from_run_args(args: argparse.Namespace) -> TopoAlignConfig:
    if args.config:
        config = TopoAlignConfig.from_json(args.config)
    else:
        settings = load_settings(Path.cwd())
        config = TopoAlignConfig.from_dict(settings.registration)
    if args.mode is not None:
        config = replace(config, mode=args.mode)
    top = {
        "fixed": args.fixed,
        "moving": args.moving,
        "fixed_mask": args.fixed_mask,
        "moving_mask": args.moving_mask,
        "fixed_features": args.fixed_features,
        "moving_features": args.moving_features,
        "matches": args.matches,
    }
    top = {key: str(value) if isinstance(value, Path) else value for key, value in top.items() if value is not None}
    if args.output_dir is not None:
        top["output"] = replace(config.output, output_dir=str(args.output_dir))
    if top:
        config = replace(config, **top)

    seg = config.segmentation
    for name in ("channel_axis", "registration_channel", "gpu", "min_area", "max_area"):
        value = getattr(args, name)
        if value is not None:
            seg = replace(seg, **{name: value})
    matching = config.matching
    for name in (
        "top_k", "feature_weight", "topology_weight", "position_weight", "distance_threshold",
        "spatial_window_size", "use_spatial_clusters", "n_clusters", "use_topology_filtering",
        "k_pos_nei", "k_neighbor", "tau_pos", "tau_nei", "tau_map",
    ):
        value = getattr(args, name)
        if value is not None:
            matching = replace(matching, **{name: value})
    transform = config.transform
    for name, arg_name in (
        ("method", "method"), ("use_ransac", "use_ransac"), ("ransac_max_trials", "ransac_max_trials"),
        ("ransac_residual_threshold", "ransac_residual_threshold"), ("residual_prune_quantile", "residual_prune_quantile"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            transform = replace(transform, **{name: value})
    if args.initial_coarse_transform is not None:
        transform = replace(transform, initial_coarse_transform=str(args.initial_coarse_transform))
    return replace(config, segmentation=seg, matching=matching, transform=transform)


def _run_stage(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "segment":
        mask, metadata = segment_image(
            args.image,
            args.output_dir,
            channel_axis=args.channel_axis,
            registration_channel=args.registration_channel,
            gpu=args.gpu,
        )
        return {"product": "TopoAlign", "mask": str(Path(args.output_dir) / "mask.tif"), "metadata": str(metadata), "cell_pixels": int(np.count_nonzero(mask))}
    if args.command == "features":
        config = TopoAlignConfig()
        config = replace(config, segmentation=replace(config.segmentation, min_area=args.min_area, max_area=args.max_area))
        features, metadata = extract_features(args.mask, args.output_dir, config=config)
        return {"product": "TopoAlign", "features": str(Path(args.output_dir) / "features.csv"), "metadata": str(metadata), "cell_count": int(len(features))}
    if args.command == "match":
        config = TopoAlignConfig()
        matching = config.matching
        for name in (
            "top_k", "feature_weight", "topology_weight", "position_weight", "distance_threshold",
            "spatial_window_size", "use_spatial_clusters", "n_clusters", "use_topology_filtering",
            "k_pos_nei", "k_neighbor", "tau_pos", "tau_nei", "tau_map",
        ):
            value = getattr(args, name, None)
            if value is not None:
                matching = replace(matching, **{name: value})
        config = replace(config, matching=matching)
        if args.fixed_features:
            fixed = _load_table(args.fixed_features)
        elif args.fixed_mask:
            fixed = compute_cell_features(_load_mask(args.fixed_mask), DEFAULT_FEATURE_CONFIG)
        else:
            raise ValueError("match requires --fixed-features or --fixed-mask.")
        if args.moving_features:
            moving = _load_table(args.moving_features)
        elif args.moving_mask:
            moving = compute_cell_features(_load_mask(args.moving_mask), DEFAULT_FEATURE_CONFIG)
        else:
            raise ValueError("match requires --moving-features or --moving-mask.")
        if args.fixed_shape:
            fixed_shape = tuple(args.fixed_shape)
        elif args.fixed_mask:
            fixed_shape = tuple(_load_mask(args.fixed_mask).shape)
        else:
            fixed_shape = (
                max(1, int(np.ceil(pd.concat([fixed, moving])["centroid_y"].max())) + 1),
                max(1, int(np.ceil(pd.concat([fixed, moving])["centroid_x"].max())) + 1),
            )
        matches = match_features(fixed, moving, fixed_shape, config)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        matches_path = out / "matches.csv"
        matches.to_csv(matches_path, index=False)
        diagnostics = {"product": "TopoAlign", "match_count": int(len(matches)), "fixed_shape": list(fixed_shape)}
        diagnostics_path = out / "matching_diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        return {**diagnostics, "matches": str(matches_path), "diagnostics_path": str(diagnostics_path)}
    if args.command == "transform":
        fixed = _load_table(args.fixed_features)
        moving = _load_table(args.moving_features)
        matches = pd.read_csv(args.matches).reset_index(drop=True)
        transform, match_table = estimate_transform_from_matches(
            fixed, moving, matches, method=args.method or "rigid", use_ransac=bool(args.use_ransac),
            ransac_max_trials=args.ransac_max_trials or 1000,
            ransac_residual_threshold=args.ransac_residual_threshold or 2.0,
            residual_prune_quantile=args.residual_prune_quantile,
        )
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        matrix_path = out / "transform.moving_to_fixed.json"
        matrix_path.write_text(json.dumps({"product": "TopoAlign", "direction": "moving_to_fixed", "method": args.method or "rigid", "matrix": np.asarray(transform.params).tolist()}, indent=2), encoding="utf-8")
        matches_path = out / "matches.with_residuals.csv"
        match_table.to_csv(matches_path, index=False)
        return {"product": "TopoAlign", "transform": str(matrix_path), "matches": str(matches_path), "match_count": int(len(match_table))}
    if args.command == "warp":
        moving = np.asarray(iio.imread(args.moving))
        transform = load_transform(args.transform)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        warped = warp_array(moving, transform, tuple(args.fixed_shape), channel_axis=args.channel_axis, order=0 if args.mask else 1)
        registered_path = out / "registered_moving.tif"
        tifffile.imwrite(registered_path, warped)
        if moving.ndim == 2:
            source_shape = moving.shape
        elif args.channel_axis == "first":
            source_shape = moving.shape[1:]
        elif args.channel_axis in ("last", "auto") and moving.shape[-1] <= 4:
            source_shape = moving.shape[:2]
        elif args.channel_axis == "auto" and moving.ndim == 3:
            source_shape = moving.shape[1:]
        else:
            raise ValueError("Cannot infer moving spatial shape; set --channel-axis to first or last.")
        valid_path = out / "valid_overlap_mask.tif"
        tifffile.imwrite(valid_path, valid_overlap_mask(tuple(source_shape), transform, tuple(args.fixed_shape)).astype(np.uint8))
        return {"product": "TopoAlign", "registered_moving": str(registered_path), "valid_overlap_mask": str(valid_path)}
    if args.command == "inspect":
        path = args.path
        if path.suffix.lower() in {".py", ".md", ".toml", ".yaml", ".yml"}:
            data = path.read_text(encoding="utf-8")
            return {"path": str(path.resolve()), "kind": "text", "bytes": len(data.encode("utf-8")), "line_count": len(data.splitlines())}
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            return {"path": str(path.resolve()), "kind": "json", "keys": list(payload) if isinstance(payload, dict) else None, "payload": payload}
        if path.suffix.lower() == ".csv":
            table = pd.read_csv(path, nrows=5)
            return {"path": str(path.resolve()), "kind": "csv", "columns": list(table.columns), "preview_rows": len(table)}
        arr = np.asarray(iio.imread(path))
        return {"path": str(path.resolve()), "kind": "image", "shape": list(arr.shape), "dtype": str(arr.dtype)}
    raise ValueError(f"Unknown stage command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    try:
        startup_settings = load_settings(Path.cwd())
        apply_agent_environment(startup_settings)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"TopoAlign settings error: {exc}", file=sys.stderr)
        return 2
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            if any((args.direct_prompt, args.direct_api_key, args.direct_base_url, args.direct_model)):
                from .agent import run_agent

                return run_agent(
                    prompt=args.direct_prompt,
                    api_key=args.direct_api_key,
                    base_url=args.direct_base_url,
                    model=args.direct_model,
                    workspace=args.direct_workspace,
                )
            return _interactive_cli()
        if args.command == "run":
            result: RegistrationResult = register(_config_from_run_args(args))
            payload = result.to_dict()
        elif args.command == "agent":
            from .agent import run_agent

            return run_agent(
                prompt=args.prompt,
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                workspace=args.workspace,
            )
        else:
            payload = _run_stage(args)
        if getattr(args, "json", False):
            _json_summary(payload)
        else:
            print(json.dumps(payload, indent=2, default=str))
        return 0
    except (ValueError, FileNotFoundError, ImportError, OSError, json.JSONDecodeError) as exc:
        print(f"TopoAlign error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
