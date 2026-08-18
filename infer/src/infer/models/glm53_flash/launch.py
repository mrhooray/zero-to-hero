from functools import partial
from pathlib import Path

from infer.runtime import launch as launch_ranks
from infer.runtime import supervise


def launch(
    checkpoint: Path,
    host: str,
    port: int,
    parallelism: str,
    speculation: str = "native",
) -> None:
    launch_ranks(
        partial(
            _serve_rank,
            parallelism=parallelism,
            speculation=speculation,
        ),
        checkpoint,
        host,
        port,
        "infer-glm53_flash-",
        supervise,
    )


def _serve_rank(*args, **kwargs) -> None:
    from infer.models.glm53_flash.serve import serve_rank

    serve_rank(*args, **kwargs)
