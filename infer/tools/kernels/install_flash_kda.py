import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

FLASH_KDA_REPOSITORY = "https://github.com/MoonshotAI/FlashKDA.git"
FLASH_KDA_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
CUTLASS_REPOSITORY = "https://github.com/NVIDIA/cutlass.git"
CUTLASS_COMMIT = "5c149f52a436782210263fb2f19b354443a61c6a"
FLASH_KDA_VERSION = "0.0.1+1ce47ea.infer1"
FLASH_KDA_LICENSE = "MIT"
CUTLASS_LICENSE = "BSD-3-Clause"
API_SYMBOLS = ("fwd", "get_workspace_size")
FLASH_KDA_NOALLOC_PATCH = b"""diff --git a/csrc/flash_kda.cpp b/csrc/flash_kda.cpp
index 81f5483..dbfaee7 100644
--- a/csrc/flash_kda.cpp
+++ b/csrc/flash_kda.cpp
@@ -25 +25 @@ int64_t get_workspace_size(
-    return H * total_tiles * per_tile_bytes + tile_prefix_bytes;
+    return H * total_tiles * per_tile_bytes + tile_prefix_bytes + T_total * H * sizeof(at::BFloat16);
@@ -47,0 +48 @@ void fwd(
+    TORCH_CHECK(workspace.dtype() == torch::kUInt8, "workspace must be uint8");
@@ -130,4 +130,0 @@ void fwd(
-    // Transpose beta: [T_total, H] -> [H, T_total] (1D TMA, no T alignment constraint)
-    auto beta_t = beta_2d.t().contiguous();
-    auto beta_t_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(beta_t.data_ptr<at::BFloat16>());
-
@@ -161,0 +159,13 @@ void fwd(
+    int64_t beta_t_bytes = T_total * H * sizeof(at::BFloat16);
+    int64_t workspace_bytes = get_workspace_size(T_total, H, N_val);
+    TORCH_CHECK(workspace.numel() >= workspace_bytes, "workspace is too small");
+    auto beta_t = torch::from_blob(
+        workspace.data_ptr<uint8_t>() + workspace_bytes - beta_t_bytes,
+        {H, T_total},
+        beta.options()
+    );
+    beta_t.copy_(beta_2d.t());
+    auto beta_t_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(
+        beta_t.data_ptr<at::BFloat16>()
+    );
+
"""
FLASH_KDA_NOALLOC_PATCH_SHA256 = (
    "27c80df97cfa24d8b4a5bebc53e6e8cc906c16afc1ed7509cf9da9af9424e464"
)
PATCHED_SOURCE_SHA256 = {
    "csrc/flash_kda.cpp": "93382056dd78a5f1990f47656e3baaebe6b52b66d490345882d0e50a04b88c24",
}
PATCH_RECEIPT = {
    "path": "tools/kernels/install_flash_kda.py#FLASH_KDA_NOALLOC_PATCH",
    "sha256": FLASH_KDA_NOALLOC_PATCH_SHA256,
    "source_sha256": PATCHED_SOURCE_SHA256,
}
SOURCE_SHA256 = {
    ".gitmodules": "0780ba817a8113c89fbc3a07376f4d730f0e3529358c68e4d9a3d23382b71132",
    "LICENSE": "05f1750624d6ab5f6dd59dea79e3156f4fec6b3065c94aeeaf265df677f9e6e8",
    "README.md": "fc56ca9a3cd1786d0ff526a12be21500d79ee671e5f0eeb8352fad23d3ba8fb2",
    "csrc/flash_kda.cpp": "f736ab87c7c2e47fe416f8d04d62502a257a6ff32a0b23452c07e1bc65c87c04",
    "cutlass/LICENSE.txt": "80a7a18b73d41f64dd9ca881af35938f8de88b18c728703251f4c94d1299884d",
    "flash_kda/__init__.py": "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84",
    "setup.py": "c6810c772afda979e91e0337feca8ad63eac7c575f6eb2466681d96a18adb016",
}
BUILD_PACKAGE_VERSIONS = {
    "ninja": "1.13.0",
    "packaging": "26.3",
    "setuptools": "84.0.0",
    "torch": "2.13.0+cu129",
    "wheel": "0.48.0",
}
BUILD_ENV = {
    "CC": "/usr/bin/gcc",
    "CUDA_HOME": "/usr/local/cuda",
    "CUDAHOSTCXX": "/usr/bin/g++",
    "CXX": "/usr/bin/g++",
    "FLASH_KDA_CUDA_ARCHS": "100a",
    "FLASH_KDA_VERSION_SUFFIX": "+1ce47ea.infer1",
    "MAX_JOBS": "2",
    "NVCC_THREADS": "2",
}
BUILD_RECEIPT = Path("/opt/flash-kda-build.json")
UV_BINARY = Path("/.uv/uv")
UV_VERSION = "0.12.5"


def main() -> None:
    validate_build_environment()
    with tempfile.TemporaryDirectory(prefix="flash-kda-") as directory:
        source = Path(directory) / "source"
        checkout_source(source)
        source_receipt = validate_source_checkout(source)
        patch_receipt = apply_noalloc_patch(source)
        environment = os.environ.copy()
        environment.update(BUILD_ENV)
        subprocess.run(
            [
                str(UV_BINARY),
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-build-isolation",
                "--no-deps",
                str(source),
            ],
            check=True,
            cwd=source,
            env=environment,
        )

    receipt = {
        "artifact": validate_install(),
        "build_env": BUILD_ENV,
        "patch": patch_receipt,
        "source": source_receipt,
    }
    BUILD_RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("FLASH_KDA_BUILD=" + json.dumps(receipt, sort_keys=True))


def checkout_source(destination: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(destination)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "remote",
            "add",
            "origin",
            FLASH_KDA_REPOSITORY,
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "fetch",
            "--quiet",
            "--depth=1",
            "origin",
            FLASH_KDA_COMMIT,
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "checkout",
            "--quiet",
            "--detach",
            "FETCH_HEAD",
        ],
        check=True,
    )
    validate_root_checkout(destination)
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "submodule",
            "update",
            "--init",
            "--depth=1",
            "cutlass",
        ],
        check=True,
    )


def validate_build_environment() -> None:
    import torch

    actual_versions = {
        package: importlib.metadata.version(package)
        for package in BUILD_PACKAGE_VERSIONS
    }
    wrong = {
        package: {"actual": actual_versions[package], "expected": expected}
        for package, expected in BUILD_PACKAGE_VERSIONS.items()
        if actual_versions[package] != expected
    }
    if wrong:
        raise RuntimeError(f"build package mismatch: {wrong}")

    nvcc_output = subprocess.run(
        [f"{BUILD_ENV['CUDA_HOME']}/bin/nvcc", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    validate_runtime_versions(torch.__version__, torch.version.cuda, nvcc_output)
    uv_output = subprocess.run(
        [str(UV_BINARY), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if uv_output.split()[:2] != ["uv", UV_VERSION]:
        raise RuntimeError(f"{uv_output}, expected uv {UV_VERSION}")


def validate_root_checkout(source: Path) -> None:
    observed = {
        "cutlass_gitlink": gitlink(source, "cutlass"),
        "cutlass_repository": git_output(
            source,
            "config",
            "--file",
            ".gitmodules",
            "--get",
            "submodule.cutlass.url",
        ),
        "flash_kda_commit": git_output(source, "rev-parse", "HEAD"),
        "flash_kda_repository": git_output(source, "remote", "get-url", "origin"),
    }
    validate_source_identity(observed, require_cutlass_checkout=False)


def validate_source_checkout(source: Path) -> dict[str, object]:
    observed = {
        "cutlass_commit": git_output(source / "cutlass", "rev-parse", "HEAD"),
        "cutlass_gitlink": gitlink(source, "cutlass"),
        "cutlass_repository": git_output(
            source,
            "config",
            "--file",
            ".gitmodules",
            "--get",
            "submodule.cutlass.url",
        ),
        "flash_kda_commit": git_output(source, "rev-parse", "HEAD"),
        "flash_kda_repository": git_output(source, "remote", "get-url", "origin"),
    }
    validate_source_identity(observed)
    if git_output(source, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("FlashKDA source checkout is dirty")

    source_hashes = {
        relative: hashlib.sha256((source / relative).read_bytes()).hexdigest()
        for relative in SOURCE_SHA256
    }
    validate_source_hashes(source_hashes)
    return {
        "commits": {
            "cutlass": CUTLASS_COMMIT,
            "flash_kda": FLASH_KDA_COMMIT,
        },
        "licenses": {
            "cutlass": CUTLASS_LICENSE,
            "flash_kda": FLASH_KDA_LICENSE,
        },
        "repositories": {
            "cutlass": CUTLASS_REPOSITORY,
            "flash_kda": FLASH_KDA_REPOSITORY,
        },
        "sha256": source_hashes,
    }


def apply_noalloc_patch(source: Path) -> dict[str, object]:
    patch_sha256 = hashlib.sha256(FLASH_KDA_NOALLOC_PATCH).hexdigest()
    if patch_sha256 != FLASH_KDA_NOALLOC_PATCH_SHA256:
        raise RuntimeError("embedded FlashKDA no-allocation patch SHA-256 differs")
    subprocess.run(
        ["git", "apply", "--unidiff-zero", "--check", "-"],
        check=True,
        cwd=source,
        input=FLASH_KDA_NOALLOC_PATCH,
    )
    subprocess.run(
        ["git", "apply", "--unidiff-zero", "-"],
        check=True,
        cwd=source,
        input=FLASH_KDA_NOALLOC_PATCH,
    )
    subprocess.run(["git", "diff", "--check"], check=True, cwd=source)
    changed_source = git_output(source, "diff", "--name-status")
    patched_source_sha256 = {
        relative: hashlib.sha256((source / relative).read_bytes()).hexdigest()
        for relative in PATCHED_SOURCE_SHA256
    }
    if (
        changed_source != "M\tcsrc/flash_kda.cpp"
        or patched_source_sha256 != PATCHED_SOURCE_SHA256
    ):
        raise RuntimeError("embedded FlashKDA no-allocation patch identity differs")
    return {
        "path": PATCH_RECEIPT["path"],
        "sha256": patch_sha256,
        "source_sha256": patched_source_sha256,
    }


def validate_install() -> dict[str, object]:
    import flash_kda
    import flash_kda_C
    import torch

    version = importlib.metadata.version("flash-kda")
    if version != FLASH_KDA_VERSION:
        raise RuntimeError(f"flash-kda is {version}, expected {FLASH_KDA_VERSION}")
    validate_api_symbols(flash_kda, flash_kda_C)

    extension = Path(flash_kda_C.__file__).resolve()
    if not extension.name.startswith("flash_kda_C.") or extension.suffix != ".so":
        raise RuntimeError(f"unexpected FlashKDA extension artifact: {extension.name}")
    symbol_output = subprocess.run(
        ["nm", "-D", "--defined-only", str(extension)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    elf_output = subprocess.run(
        ["cuobjdump", "--list-elf", str(extension)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    validate_binary_artifact(symbol_output, elf_output)

    return {
        "api_symbols": list(API_SYMBOLS),
        "extension": str(extension),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "native_architecture": "sm_100a",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "version": version,
    }


def validate_runtime_versions(
    torch_version: str,
    torch_cuda: str | None,
    nvcc_output: str,
) -> None:
    if torch_version != BUILD_PACKAGE_VERSIONS["torch"] or torch_cuda != "12.9":
        raise RuntimeError(
            f"unexpected torch/CUDA versions: {torch_version}/{torch_cuda}"
        )
    match = re.search(r"\brelease\s+(\d+\.\d+)\b", nvcc_output)
    if match is None or match.group(1) != "12.9":
        raise RuntimeError("nvcc must be from CUDA 12.9")


def validate_source_identity(
    observed: dict[str, str],
    *,
    require_cutlass_checkout: bool = True,
) -> None:
    expected = {
        "cutlass_gitlink": CUTLASS_COMMIT,
        "cutlass_repository": CUTLASS_REPOSITORY,
        "flash_kda_commit": FLASH_KDA_COMMIT,
        "flash_kda_repository": FLASH_KDA_REPOSITORY,
    }
    if require_cutlass_checkout:
        expected["cutlass_commit"] = CUTLASS_COMMIT
    if observed != expected:
        raise RuntimeError(f"FlashKDA source identity mismatch: {observed}")


def validate_source_hashes(observed: dict[str, str]) -> None:
    if observed != SOURCE_SHA256:
        raise RuntimeError("FlashKDA verified source hashes differ")


def validate_api_symbols(package: ModuleType, extension: ModuleType) -> None:
    missing = {
        module.__name__: [
            symbol
            for symbol in API_SYMBOLS
            if not callable(getattr(module, symbol, None))
        ]
        for module in (package, extension)
    }
    missing = {module: symbols for module, symbols in missing.items() if symbols}
    if missing:
        raise RuntimeError(f"FlashKDA API symbols missing: {missing}")


def validate_binary_artifact(symbol_output: str, elf_output: str) -> None:
    if re.search(r"\bPyInit_flash_kda_C\b", symbol_output) is None:
        raise RuntimeError("FlashKDA extension has no PyInit_flash_kda_C symbol")
    architectures = set(re.findall(r"\bsm_\d+[af]?\b", elf_output))
    if architectures != {"sm_100a"}:
        raise RuntimeError(f"FlashKDA code objects are {sorted(architectures)}")


def gitlink(source: Path, path: str) -> str:
    fields = git_output(source, "ls-tree", "HEAD", path).split()
    if len(fields) != 4 or fields[:2] != ["160000", "commit"] or fields[3] != path:
        raise RuntimeError(f"{path} is not the expected gitlink")
    return fields[2]


def git_output(source: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    main()
