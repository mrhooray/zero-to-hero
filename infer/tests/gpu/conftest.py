"""Local-CUDA tier. Self-skips without torch/CUDA; never fails collection.

Every module placed here must set `pytestmark = pytest.mark.gpu`.
Only genuinely numerical tests belong here — everything mockable runs in
`tests/` root instead (see `test_glm53_megafuse.py` for the pattern).

The compiled `flash_kda_C` extension is unavailable in unit-test contexts
(even on GPU; it is built separately by `tools/kernels`). It is stubbed
here so modules import; the stub raises on any call, so a test that truly
needs the extension fails loudly instead of passing on fakes. The
`flash_kda` version pin matches `ops/prefill_kda.py::FLASH_KDA_VERSION`.
"""

import importlib.metadata
import sys
import types

import pytest

torch = pytest.importorskip("torch", reason="GPU tests require torch")
if not torch.cuda.is_available():
    pytest.skip("GPU tests require CUDA", allow_module_level=True)


def _unavailable(*args: object, **kwargs: object) -> object:
    raise RuntimeError("flash_kda_C is unavailable in unit tests")


if "flash_kda_C" not in sys.modules:
    try:
        importlib.import_module("flash_kda_C")
    except ImportError:
        extension = types.ModuleType("flash_kda_C")
        extension.fwd = _unavailable
        extension.get_workspace_size = _unavailable
        sys.modules["flash_kda_C"] = extension

try:
    importlib.metadata.version("flash_kda")
except importlib.metadata.PackageNotFoundError:
    _real_version = importlib.metadata.version
    importlib.metadata.version = lambda name: (
        "0.0.1+1ce47ea.infer1" if name == "flash_kda" else _real_version(name)
    )
