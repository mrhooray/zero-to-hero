from functools import partial
from pathlib import Path

from infer.models.deepseek_v4_flash import model as deepseek_v4_flash
from infer.runtime import launch as launch_ranks
from infer.runtime import supervise


def launch(
    checkpoint: Path,
    host: str,
    port: int,
    parallelism: str,
    speculation: str = "native",
    dspark_verify_width: int = deepseek_v4_flash.DSPARK_VERIFY_WIDTH,
) -> None:
    if speculation not in {"native", "none"}:
        raise ValueError("DeepSeek speculation must be native or none")
    deepseek_v4_flash.validate_dspark_verify_width(dspark_verify_width)
    launch_ranks(
        partial(
            _serve_rank,
            parallelism=parallelism,
            speculation=speculation,
            dspark_verify_width=dspark_verify_width,
        ),
        checkpoint,
        host,
        port,
        "infer-deepseek-v4-flash-",
        supervise,
    )


def _serve_rank(
    rank: int,
    checkpoint: str,
    host: str,
    port: int,
    rendezvous: str,
    plan_pipes: tuple[tuple[object, object], ...],
    *,
    parallelism: str,
    speculation: str,
    dspark_verify_width: int,
) -> None:
    deepseek_v4_flash.configure_dspark_verify_width(dspark_verify_width)
    from infer.models.deepseek_v4_flash.serve import serve_rank

    serve_rank(
        rank,
        checkpoint,
        host,
        port,
        rendezvous,
        plan_pipes,
        parallelism=parallelism,
        speculation=speculation,
    )
