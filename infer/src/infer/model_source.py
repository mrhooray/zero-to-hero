import re
from pathlib import Path

from huggingface_hub import snapshot_download

_COMMIT_REVISION = re.compile(r"[0-9a-fA-F]{40}")


def resolve_model_source(
    checkpoint_dir: str | Path | None = None,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> Path:
    if checkpoint_dir is not None:
        if repo_id is not None:
            raise ValueError("specify either checkpoint_dir or repo_id, not both")
        if revision is not None or cache_dir is not None or token is not None:
            raise ValueError(
                "revision, cache_dir, and token are only valid with repo_id"
            )
        path = Path(checkpoint_dir).expanduser()
        if not path.is_dir():
            raise ValueError("checkpoint_dir must be an existing directory")
        return path.resolve()

    if not repo_id:
        raise ValueError("checkpoint_dir or repo_id is required")
    if not isinstance(revision, str) or _COMMIT_REVISION.fullmatch(revision) is None:
        raise ValueError("repo_id requires an immutable 40-hex revision")

    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision.lower(),
            cache_dir=cache_dir,
            token=token,
        )
    )
    if not snapshot.is_dir():
        raise RuntimeError("snapshot_download did not return a local directory")
    return snapshot.resolve()
