"""Command-line entry point for the standalone harness workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters import LlamaServerAdapter, LlamaServerConfig, LlamaServerProcess
from .api_layer import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLH small-model harness workbench")
    parser.add_argument("--tui", action="store_true", help="启动 TUI 工作台（等价于 koakumix-tui，不需要 --model）")
    parser.add_argument("--model", required=True, help="local GGUF or backend model path")
    parser.add_argument("--llama-server", default="llama-server", help="llama-server executable")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--mmproj")
    parser.add_argument("--no-jinja", action="store_true")
    parser.add_argument("--no-cache-prompt", action="store_true")
    parser.add_argument("--serve-host", default="127.0.0.1")
    parser.add_argument("--serve-port", type=int, default=8090)
    return parser


def _skill_main(argv: list[str]) -> int:
    """``koakumix skill list`` / ``koakumix skill run <name> --json '{...}'``。"""
    from .skills import SkillError, build_registry

    parser = argparse.ArgumentParser(prog="koakumix skill", description="Koakumix 内置技能")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="列出内置技能")
    run = sub.add_parser("run", help="执行内置技能")
    run.add_argument("name")
    run.add_argument("--json", default="{}", help="技能输入（JSON 字符串）")
    args = parser.parse_args(argv)

    registry = build_registry()  # CLI 不注入生图引擎：list 可用，run 会明确提示未配置
    if args.cmd == "list":
        print(json.dumps(registry.describe(), ensure_ascii=False, indent=2))
        return 0
    try:
        payload = json.loads(args.json)
        if not isinstance(payload, dict):
            raise ValueError("--json 必须是 JSON 对象")
        result = registry.run(args.name, payload)
    except (SkillError, ValueError) as exc:
        code = getattr(exc, "code", "invalid_json")
        print(f"{code}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["skill"]:
        return _skill_main(argv[1:])
    # 不带参数时默认进 TUI；`--tui` 等价于 `koakumix-tui`。两者都在 parse 之前分流，
    # 免得触发本解析器的 --model 必填校验。
    if not argv or "--tui" in argv:
        from .tui import main as tui_main  # noqa: PLC0415

        return tui_main([item for item in argv if item != "--tui"])
    args = build_parser().parse_args(argv)
    config = LlamaServerConfig(
        executable=args.llama_server,
        model=Path(args.model),
        host=args.host,
        port=args.port,
        context_size=args.ctx_size,
        max_new_tokens=args.max_new_tokens,
        mmproj=args.mmproj,
        enable_jinja=not args.no_jinja,
        cache_prompt=not args.no_cache_prompt,
    )
    process = LlamaServerProcess(config)
    adapter = LlamaServerAdapter(config, process=process)
    adapter.start()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - environment-dependent
        adapter.close()
        raise SystemExit("uvicorn is required to serve the harness API") from exc
    try:
        uvicorn.run(create_app(adapter), host=args.serve_host, port=args.serve_port)
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
