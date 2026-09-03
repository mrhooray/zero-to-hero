import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from infer.model_source import resolve_model_source
from infer.models import deepseek_v4_flash, glm53_flash
from infer.models.deepseek_v4_flash import model as deepseek_model
from infer.runtime import port as _port


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    checkpoint = _resolve_checkpoint(args)
    if args.model == deepseek_v4_flash.MODEL_ID:
        from infer.models.deepseek_v4_flash.launch import launch

        launch(
            checkpoint,
            args.host,
            args.port,
            args.parallelism,
            args.speculation,
            args.dspark_verify_width or deepseek_model.DSPARK_VERIFY_WIDTH,
        )
    else:
        from infer.models.glm53_flash.launch import launch

        launch(checkpoint, args.host, args.port, args.parallelism, args.speculation)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="infer")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("model", choices=(deepseek_v4_flash.MODEL_ID, glm53_flash.MODEL_ID))
    serve.add_argument("--checkpoint-dir", type=Path)
    serve.add_argument("--revision")
    serve.add_argument("--cache-dir", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_port, default=8000)
    serve.add_argument("--parallelism", choices=("dep4", "tep4"))
    serve.add_argument("--speculation", choices=("native", "none"), default="native")
    serve.add_argument(
        "--dspark-verify-width", type=int, choices=deepseek_model.DSPARK_VERIFY_WIDTHS
    )
    args = parser.parse_args(argv)
    if args.checkpoint_dir is not None and (
        args.revision is not None or args.cache_dir is not None
    ):
        parser.error("--revision and --cache-dir require Hugging Face loading")
    if args.parallelism is None:
        args.parallelism = "dep4"
    if args.dspark_verify_width is not None and (
        args.model != deepseek_v4_flash.MODEL_ID or args.speculation != "native"
    ):
        parser.error("--dspark-verify-width requires DeepSeek native speculation")
    return args


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir is not None:
        return resolve_model_source(args.checkpoint_dir)
    revision = args.revision or (
        deepseek_v4_flash.MODEL_REVISION
        if args.model == deepseek_v4_flash.MODEL_ID
        else glm53_flash.MODEL_REVISION
    )
    return resolve_model_source(
        repo_id=args.model,
        revision=revision,
        cache_dir=args.cache_dir,
        token=False,
    )
