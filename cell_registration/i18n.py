"""Small English-first message catalog for interactive TopoAlign surfaces."""

from __future__ import annotations

import os


MESSAGES = {
    "en": {
        "tagline": "Topology-guided cellular image registration",
        "config_loaded": "Configuration: {path}",
        "local_ready": "Local CLI: ready (no API key required)",
        "agent_status": "Agent: {status}  api={endpoint}  model={model}",
        "agent_valid": "model call available",
        "agent_unvalidated": "configured but not validated",
        "agent_optional": "not configured (optional)",
        "direct_hint": "Enter a local command, for example: inspect image.tif or run --fixed a.tif --moving b.tif",
        "menu_agent": "[1] Enter optional Agent",
        "menu_config": "[2] Configure/test Agent API URL, API key, and model",
        "menu_help": "[3] Show command help",
        "menu_exit": "[4] Exit",
        "menu_language": "[5] Language / 语言",
        "exit": "Exited TopoAlign.",
        "agent_return": "Agent closed; returned to the main shell.",
        "unknown": "Unknown input. Enter a local command, help, agent, config, language, or exit.",
        "config_title": "Configure the optional Agent connection (saved to {path})",
        "api_url": "API URL [{value}]: ",
        "api_key": "API key: ",
        "api_key_keep": "API key [press Enter to keep the configured value]: ",
        "model": "Model [{value}]: ",
        "missing_key": "No API key entered; configuration was cancelled.",
        "checking": "Checking API URL, API key, and model access...",
        "check_ok": "Model call available: {endpoint}  model={model}",
        "check_failed": "Model call unavailable: {kind}: {error}",
        "local_still_ready": "Local commands remain available; use config to update Agent settings.",
        "language_prompt": "Select language: [1] English  [2] 中文: ",
        "language_saved": "Language saved: {selected}",
        "command_parse_failed": "Could not parse command: {error}",
        "local_command_expected": "Local commands start with run, segment, features, match, transform, warp, or inspect.",
        "run_setup": "Complete the missing run settings (saved to {path}).",
        "input_mode": "Input mode [image/mask/features] [{value}]: ",
        "fixed_image": "Fixed image path: ",
        "moving_image": "Moving image path: ",
        "fixed_mask": "Fixed mask path: ",
        "moving_mask": "Moving mask path: ",
        "fixed_features": "Fixed features CSV: ",
        "moving_features": "Moving features CSV: ",
        "matches_optional": "Matches CSV [optional]: ",
        "fixed_shape": "Fixed shape HEIGHT WIDTH: ",
        "output_dir": "Output directory [{value}]: ",
        "method_prompt": "Transform method [rigid/similarity/affine] [{value}]: ",
        "missing_run_input": "Required run input was not provided; returned to the main shell.",
        "agent_session": "Agent session | model={model} | api={endpoint}",
        "workspace": "Workspace: {workspace}",
        "agent_exit_hint": "Enter /+ for commands; use /exit to return to the main shell.",
        "you_prompt": "\nYou> ",
        "agent_reply": "\nAgent> {text}",
        "agent_thinking": "Agent> Thinking",
        "agent_error": "\nAgent error: {error}",
        "slash_help": """Agent commands:
  /+ or /help       Show this command list
  /model            Show the current model
  /model <id>       Validate and switch model
  /api              Test the configured URL, key, and model
  /config           Show the active configuration (key redacted)
  /reload           Reload the JSON configuration
  /tools            List allowlisted Agent tools
  /local <command>  Run a local TopoAlign command
  /result           Show the latest result.json
  /artifacts        List artifacts from the latest result
  /clear            Clear model conversation context
  /exit             Return to the main shell""",
        "current_model": "Current model: {model}",
        "model_switched": "Model switched to {model} and saved to configuration.",
        "model_failed": "Model switch failed: {error}",
        "api_ok": "Model call available: {endpoint}  model={model}",
        "api_failed": "Model call unavailable: {kind}: {error}",
        "config_reloaded": "Configuration reloaded: {path}",
        "context_cleared": "Conversation context cleared.",
        "local_usage": "Usage: /local <run|segment|features|match|transform|warp|inspect> ...",
        "no_result": "No result.json was found in the workspace.",
        "returning": "Returning to the main shell.",
    },
    "zh": {
        "tagline": "拓扑引导的细胞图像配准",
        "config_loaded": "配置文件：{path}",
        "local_ready": "本地 CLI：可用（不需要 API key）",
        "agent_status": "Agent：{status}  api={endpoint}  model={model}",
        "agent_valid": "模型可正常调用",
        "agent_unvalidated": "已配置但尚未验证",
        "agent_optional": "未配置（可选）",
        "direct_hint": "可直接输入本地命令，例如：inspect image.tif 或 run --fixed a.tif --moving b.tif",
        "menu_agent": "[1] 进入可选 Agent",
        "menu_config": "[2] 配置/测试 Agent 的 API URL、API key 和模型",
        "menu_help": "[3] 查看命令帮助",
        "menu_exit": "[4] 退出",
        "menu_language": "[5] Language / 语言",
        "exit": "已退出 TopoAlign。",
        "agent_return": "已退出 Agent，返回主界面。",
        "unknown": "未知输入。请输入本地命令、help、agent、config、language 或 exit。",
        "config_title": "配置可选 Agent 连接（保存到 {path}）",
        "api_url": "API URL [{value}]：",
        "api_key": "API key：",
        "api_key_keep": "API key [按 Enter 保留当前值]：",
        "model": "模型 [{value}]：",
        "missing_key": "未输入 API key，已取消配置。",
        "checking": "正在检查 API URL、API key 和模型访问权限……",
        "check_ok": "模型可正常调用：{endpoint}  model={model}",
        "check_failed": "模型无法调用：{kind}: {error}",
        "local_still_ready": "本地命令仍然可用；可使用 config 修改 Agent 设置。",
        "language_prompt": "选择语言：[1] English  [2] 中文：",
        "language_saved": "语言已保存：{selected}",
        "command_parse_failed": "命令解析失败：{error}",
        "local_command_expected": "本地命令应以 run、segment、features、match、transform、warp 或 inspect 开头。",
        "run_setup": "请补充缺少的运行设置（保存到 {path}）。",
        "input_mode": "输入模式 [image/mask/features] [{value}]：",
        "fixed_image": "Fixed 图像路径：",
        "moving_image": "Moving 图像路径：",
        "fixed_mask": "Fixed mask 路径：",
        "moving_mask": "Moving mask 路径：",
        "fixed_features": "Fixed features CSV：",
        "moving_features": "Moving features CSV：",
        "matches_optional": "Matches CSV [可选]：",
        "fixed_shape": "Fixed 尺寸 HEIGHT WIDTH：",
        "output_dir": "输出目录 [{value}]：",
        "method_prompt": "变换方法 [rigid/similarity/affine] [{value}]：",
        "missing_run_input": "未提供必需输入，已返回主界面。",
        "agent_session": "Agent 会话 | model={model} | api={endpoint}",
        "workspace": "工作目录：{workspace}",
        "agent_exit_hint": "输入 /+ 查看指令；使用 /exit 返回主界面。",
        "you_prompt": "\n你> ",
        "agent_reply": "\nAgent> {text}",
        "agent_thinking": "Agent> 正在处理",
        "agent_error": "\nAgent 错误：{error}",
        "slash_help": """Agent 指令：
  /+ 或 /help       查看指令列表
  /model            查看当前模型
  /model <id>       验证并切换模型
  /api              测试 URL、key 和模型
  /config           查看当前配置（key 已脱敏）
  /reload           重新读取 JSON 配置
  /tools            查看 Agent 白名单工具
  /local <command>  执行本地 TopoAlign 命令
  /result           查看最近的 result.json
  /artifacts        查看最近运行的输出文件
  /clear            清除模型对话上下文
  /exit             返回主界面""",
        "current_model": "当前模型：{model}",
        "model_switched": "模型已切换为 {model}，并写入配置文件。",
        "model_failed": "模型切换失败：{error}",
        "api_ok": "模型可正常调用：{endpoint}  model={model}",
        "api_failed": "模型无法调用：{kind}: {error}",
        "config_reloaded": "配置已重新载入：{path}",
        "context_cleared": "对话上下文已清除。",
        "local_usage": "用法：/local <run|segment|features|match|transform|warp|inspect> ...",
        "no_result": "工作目录中没有找到 result.json。",
        "returning": "正在返回主界面。",
    },
}


def normalize_language(value: str | None) -> str:
    return "zh" if str(value or "").lower().startswith("zh") else "en"


def current_language() -> str:
    return normalize_language(os.environ.get("TOPOALIGN_LANGUAGE", "en"))


def tr(key: str, language: str | None = None, **values: object) -> str:
    lang = normalize_language(language or current_language())
    template = MESSAGES.get(lang, MESSAGES["en"]).get(key, MESSAGES["en"].get(key, key))
    return template.format(**values)
