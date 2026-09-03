"""Default collection policy: CPU-only.

`tests/gpu` (local CUDA numerics) is excluded unless `INFER_RUN_GPU_TESTS=1`.
Without CUDA the tier self-skips via `tests/gpu/conftest.py`; it never fails.

Modal job-spec/parser tests run here in root with `modal` mocked — nothing
in pytest executes remotely. Real `modal run` jobs live outside the suite.
"""

import os

collect_ignore = []
if os.environ.get("INFER_RUN_GPU_TESTS") != "1":
    collect_ignore.append("gpu")
if os.environ.get("INFER_RUN_MODAL_TESTS") != "1":
    collect_ignore.append("modal")
