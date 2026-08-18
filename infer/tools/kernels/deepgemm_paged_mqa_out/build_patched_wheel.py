from __future__ import annotations

import argparse
import base64
import csv
import email.policy
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

SOURCE_REPOSITORY = "https://github.com/sgl-project/DeepGEMM.git"
SOURCE_COMMIT = "fa3a5ca07d768dd0f9089f70a445208b166c48d1"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
CUTLASS_REPOSITORY = "https://github.com/NVIDIA/cutlass.git"
CUTLASS_LICENSE = "BSD-3-Clause"
CUTLASS_LICENSE_SHA256 = (
    "80a7a18b73d41f64dd9ca881af35938f8de88b18c728703251f4c94d1299884d"
)
FMT_COMMIT = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
FMT_REPOSITORY = "https://github.com/fmtlib/fmt.git"
FMT_LICENSE = "MIT"
FMT_LICENSE_SHA256 = "07580f2a3b35709ce703d523f447b242f6dfec7582a8c0df102c7fa2849375f8"
PATCH_SHA256 = "ccd1867735a7c386cd2346fc7e2e848da853a8841a40751a6f09dc4f98cc8f98"
SOURCE_LICENSE = "MIT"
SOURCE_LICENSE_SHA256 = (
    "6c73a30c14a2938abaa6afe98ebb19e6d81dac5bd4ace0cb05c61c20afa37b85"
)
WHEEL_LICENSE = "Apache-2.0"
WHEEL_LICENSE_SHA256 = (
    "92964cac0c35f6d63fe51074ae3f1460d42d8eafa79c7060aa02599989b832aa"
)
EMBEDDED_LICENSES = {
    "LICENSE": (WHEEL_LICENSE, WHEEL_LICENSE_SHA256),
    "LICENSE.deepgemm": (SOURCE_LICENSE, SOURCE_LICENSE_SHA256),
    "LICENSE.cutlass": (CUTLASS_LICENSE, CUTLASS_LICENSE_SHA256),
    "LICENSE.fmt": (FMT_LICENSE, FMT_LICENSE_SHA256),
}
ADDED_LICENSE_SOURCES = {
    "LICENSE.deepgemm": "LICENSE",
    "LICENSE.cutlass": "third-party/cutlass/LICENSE.txt",
    "LICENSE.fmt": "third-party/fmt/LICENSE",
}
BASE_IMAGE = (
    "nvidia/cuda:12.9.1-devel-ubuntu22.04@"
    "sha256:38804006c937a83f28f63a959abcee688042072319c8614ad57b350958a30bd3"
)
EXPECTED_PACKAGES = {
    "apache-tvm-ffi": "0.1.11",
    "build": "1.3.0",
    "ninja": "1.13.0",
    "setuptools": "84.0.0",
    "torch": "2.13.0+cu129",
    "wheel": "0.48.0",
}
NINJA_BINARY_VERSION = "1.13.0.git.kitware.jobserver-pipe-1"
PATCHED_FILES = (
    "build_sgl_deep_gemm.sh",
    "csrc/apis/attention.hpp",
    "csrc/tvm_ffi_api.cpp",
    "sgl_deep_gemm/__init__.py",
)
ROOT = Path(__file__).resolve().parent
PATCH = ROOT / "deepgemm-paged-mqa-out.patch"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--expected-wheel-sha256")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()

    verify_patch()
    toolchain = verify_toolchain()
    source = prepare_source(args.work_dir)
    wheel = build_wheel(source, toolchain)
    wheel_sha256 = sha256(wheel)
    if (
        args.expected_wheel_sha256 is not None
        and wheel_sha256 != args.expected_wheel_sha256
    ):
        raise RuntimeError(
            f"wheel SHA-256 mismatch: {wheel_sha256} != {args.expected_wheel_sha256}"
        )
    verify_wheel_licenses(wheel)
    verify_wheel_surface(wheel)
    if args.install:
        install_wheel(wheel)

    receipt = args.work_dir / "deepgemm-paged-mqa-out-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "scope": {
                    "unblocks": "GLM FP8 and DeepSeek C4 MXFP4 paged scorers",
                    "excluded": (
                        "DeepSeek C4 ragged fp8_fp4_mqa_logits remains "
                        "allocation-backed"
                    ),
                },
                "abi": {
                    "deepseek_c4_mxfp4_cache": {
                        "logical_shape": ["P", 32, 1, 68],
                        "strides": [2176, 68, 68, 1],
                        "storage_offset": 0,
                        "base_alignment_bytes": 16,
                        "page_value_bytes": 2048,
                        "page_scale_bytes": 128,
                    }
                },
                "source": {
                    "repository": SOURCE_REPOSITORY,
                    "commit": SOURCE_COMMIT,
                    "license": SOURCE_LICENSE,
                    "license_sha256": SOURCE_LICENSE_SHA256,
                },
                "submodules": {
                    "third-party/cutlass": {
                        "repository": CUTLASS_REPOSITORY,
                        "commit": CUTLASS_COMMIT,
                        "license": CUTLASS_LICENSE,
                        "license_sha256": CUTLASS_LICENSE_SHA256,
                    },
                    "third-party/fmt": {
                        "repository": FMT_REPOSITORY,
                        "commit": FMT_COMMIT,
                        "license": FMT_LICENSE,
                        "license_sha256": FMT_LICENSE_SHA256,
                    },
                },
                "patch": {
                    "path": str(PATCH),
                    "sha256": PATCH_SHA256,
                    "files": PATCHED_FILES,
                },
                "toolchain": toolchain,
                "wheel": {
                    "path": str(wheel),
                    "sha256": wheel_sha256,
                    "sha256_scope": "artifact verification, not rebuild identity",
                    "tag": "py3-none-linux_x86_64",
                    "root_is_purelib": False,
                    "licenses": {
                        filename: {"license": license, "sha256": license_sha256}
                        for filename, (
                            license,
                            license_sha256,
                        ) in EMBEDDED_LICENSES.items()
                    },
                    "installed": args.install,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(receipt)


def verify_patch() -> None:
    if sha256(PATCH) != PATCH_SHA256:
        raise RuntimeError("patch SHA-256 mismatch")
    lines = PATCH.read_text().splitlines()
    additions = sum(
        line.startswith("+") and not line.startswith("+++") for line in lines
    )
    if additions > 240:
        raise RuntimeError(f"patch grew beyond the 240-line limit: {additions}")
    if any("deep_gemm/include/" in line for line in lines if line.startswith("+++")):
        raise RuntimeError("patch must not modify device kernels")


def verify_toolchain() -> dict[str, object]:
    import torch

    packages = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    if packages != EXPECTED_PACKAGES:
        raise RuntimeError(f"package mismatch: {packages} != {EXPECTED_PACKAGES}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 required, got {platform.python_version()}")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("the wheel is qualified only on Linux x86_64")
    if os.environ.get("INFER_BASE_IMAGE") != BASE_IMAGE:
        raise RuntimeError("INFER_BASE_IMAGE does not match the pinned CUDA image")
    if torch.version.cuda != "12.9":
        raise RuntimeError("the build requires PyTorch cu129")

    paths = pinned_tool_paths()
    nvcc = output(paths["nvcc"], "--version")
    cc_version = output(paths["cc"], "-dumpfullversion", "-dumpversion").strip()
    cxx_version = output(paths["cxx"], "-dumpfullversion", "-dumpversion").strip()
    ninja_version = output(paths["ninja"], "--version").strip()
    if (
        "release 12.9, V12.9.86" not in nvcc
        or cc_version != "11.4.0"
        or cxx_version != "11.4.0"
        or ninja_version != NINJA_BINARY_VERSION
    ):
        raise RuntimeError(
            "tool mismatch: "
            f"nvcc={nvcc!r}, gcc={cc_version!r}, g++={cxx_version!r}, "
            f"ninja={ninja_version!r}"
        )
    return {
        "base_image": BASE_IMAGE,
        "cxx11_abi": int(torch.compiled_with_cxx11_abi()),
        "gcc": cc_version,
        "g++": cxx_version,
        "ninja": ninja_version,
        "nvcc": "12.9.86",
        "packages": packages,
        "paths": paths,
        "python": platform.python_version(),
        "target_sm": "10.0",
    }


def pinned_tool_paths() -> dict[str, str]:
    interpreter = Path(sys.executable).resolve()
    ninja = interpreter.parent / "ninja"
    paths = {
        "cc": Path("/usr/bin/gcc"),
        "cxx": Path("/usr/bin/g++"),
        "ninja": ninja,
        "nvcc": Path("/usr/local/cuda/bin/nvcc"),
        "python": interpreter,
    }
    missing = tuple(str(path) for path in paths.values() if not path.is_file())
    if missing:
        raise RuntimeError(f"pinned tools are missing: {missing}")
    return {name: str(path.resolve()) for name, path in paths.items()}


def prepare_source(work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    source = work_dir / "DeepGEMM"
    if source.exists():
        raise RuntimeError(f"source path already exists: {source}")
    run(
        "git", "clone", "--filter=blob:none", "--no-checkout", SOURCE_REPOSITORY, source
    )
    run("git", "checkout", "--detach", SOURCE_COMMIT, cwd=source)
    run("git", "submodule", "update", "--init", "--recursive", cwd=source)
    require_revision(source, SOURCE_COMMIT)
    require_revision(source / "third-party/cutlass", CUTLASS_COMMIT)
    require_revision(source / "third-party/fmt", FMT_COMMIT)
    require_remote(source, SOURCE_REPOSITORY)
    require_remote(source / "third-party/cutlass", CUTLASS_REPOSITORY)
    require_remote(source / "third-party/fmt", FMT_REPOSITORY)
    if output("git", "status", "--porcelain", "--untracked-files=no", cwd=source):
        raise RuntimeError("source checkout is not clean before patching")
    if sha256(source / "LICENSE") != SOURCE_LICENSE_SHA256:
        raise RuntimeError("source license hash mismatch")
    if sha256(source / "sgl_deep_gemm/LICENSE") != WHEEL_LICENSE_SHA256:
        raise RuntimeError("wheel license hash mismatch")
    if sha256(source / "third-party/cutlass/LICENSE.txt") != CUTLASS_LICENSE_SHA256:
        raise RuntimeError("CUTLASS license hash mismatch")
    if sha256(source / "third-party/fmt/LICENSE") != FMT_LICENSE_SHA256:
        raise RuntimeError("fmt license hash mismatch")

    run("git", "apply", "--check", PATCH, cwd=source)
    run("git", "apply", PATCH, cwd=source)
    run("git", "diff", "--check", cwd=source)
    changed = tuple(
        sorted(output("git", "diff", "--name-only", cwd=source).splitlines())
    )
    if changed != tuple(sorted(PATCHED_FILES)):
        raise RuntimeError(f"unexpected patched files: {changed}")
    return source


def build_wheel(source: Path, toolchain: dict[str, object]) -> Path:
    paths = toolchain["paths"]
    assert isinstance(paths, dict)
    search_path = os.pathsep.join(
        dict.fromkeys(
            (
                str(Path(paths["python"]).parent),
                str(Path(paths["ninja"]).parent),
                "/usr/local/cuda/bin",
                "/usr/bin",
                "/bin",
            )
        )
    )
    env = os.environ.copy()
    env.update(
        {
            "CC": paths["cc"],
            "CUDA_HOME": "/usr/local/cuda",
            "CXX": paths["cxx"],
            "DG_JIT_USE_RUNTIME_API": "0",
            "MAX_JOBS": "1",
            "NVCC_THREADS": "2",
            "PATH": search_path,
            "PIP_NO_INDEX": "1",
            "PYTHON_EXE": paths["python"],
            "TVM_FFI_CUDA_ARCH_LIST": "10.0",
        }
    )
    subprocess.run(["bash", "build_sgl_deep_gemm.sh"], cwd=source, env=env, check=True)
    wheels = tuple((source / "dist").glob("sgl_deep_gemm-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one wheel, found {wheels}")
    tagged_name = output(
        paths["python"],
        "-m",
        "wheel",
        "tags",
        "--remove",
        "--platform-tag",
        "linux_x86_64",
        wheels[0],
        env=env,
    ).strip()
    wheel = (source / "dist" / tagged_name).resolve()
    mark_wheel_binary(wheel, source)
    verify_wheel_metadata(wheel)
    return wheel


def mark_wheel_binary(wheel: Path, source: Path) -> None:
    from wheel.wheelfile import WheelFile

    with tempfile.TemporaryDirectory(dir=wheel.parent) as temp_dir:
        rewritten = Path(temp_dir) / wheel.name
        with WheelFile(wheel) as archive, WheelFile(rewritten, "w") as target:
            target.comment = archive.comment
            for item in archive.infolist():
                if item.is_dir() or item.filename.endswith("/RECORD"):
                    continue
                payload = archive.read(item)
                if item.filename.endswith("/WHEEL"):
                    metadata = BytesParser(policy=email.policy.compat32).parsebytes(
                        payload
                    )
                    metadata.replace_header("Root-Is-Purelib", "false")
                    payload = metadata.as_bytes()
                target.writestr(item, payload)
            for filename, relative_source in ADDED_LICENSE_SOURCES.items():
                payload = (source / relative_source).read_bytes()
                expected_sha256 = EMBEDDED_LICENSES[filename][1]
                if hashlib.sha256(payload).hexdigest() != expected_sha256:
                    raise RuntimeError(
                        f"source license hash mismatch: {relative_source}"
                    )
                target.writestr(
                    f"{archive.dist_info_path}/licenses/{filename}", payload
                )
        os.replace(rewritten, wheel)


def verify_wheel_metadata(wheel: Path) -> None:
    if not wheel.name.endswith("-py3-none-linux_x86_64.whl"):
        raise RuntimeError(f"wheel has an unsafe platform tag: {wheel.name}")
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith("/WHEEL")]
        if len(names) != 1:
            raise RuntimeError(f"expected one WHEEL metadata file, found {names}")
        metadata = BytesParser(policy=email.policy.compat32).parsebytes(
            archive.read(names[0])
        )
    if metadata.get_all("Tag") != ["py3-none-linux_x86_64"]:
        raise RuntimeError(f"wrong internal wheel tags: {metadata.get_all('Tag')}")
    if metadata["Root-Is-Purelib"] != "false":
        raise RuntimeError("binary wheel must set Root-Is-Purelib to false")


def verify_wheel_licenses(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        records = [
            name for name in archive.namelist() if name.endswith(".dist-info/RECORD")
        ]
        if len(records) != 1:
            raise RuntimeError(f"expected one wheel RECORD, found {records}")
        dist_info = records[0].removesuffix("/RECORD")
        record = {
            path: (digest, size)
            for path, digest, size in csv.reader(
                io.StringIO(archive.read(records[0]).decode())
            )
        }
        for filename, (_, expected_sha256) in EMBEDDED_LICENSES.items():
            path = f"{dist_info}/licenses/{filename}"
            if archive.namelist().count(path) != 1:
                raise RuntimeError(f"expected one wheel license file: {path}")
            payload = archive.read(path)
            actual_sha256 = hashlib.sha256(payload).hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(f"wheel license hash mismatch: {path}")
            encoded_sha256 = (
                base64.urlsafe_b64encode(bytes.fromhex(expected_sha256))
                .rstrip(b"=")
                .decode()
            )
            if record.get(path) != (f"sha256={encoded_sha256}", str(len(payload))):
                raise RuntimeError(f"wheel RECORD mismatch: {path}")


def verify_wheel_surface(wheel: Path) -> None:
    apis = (
        b"get_paged_mqa_logits_metadata_out",
        b"fp8_paged_mqa_logits_out",
        b"fp8_fp4_paged_mqa_logits_out",
    )
    with zipfile.ZipFile(wheel) as archive:
        python = archive.read("deep_gemm/__init__.py")
        extension_name = next(
            name for name in archive.namelist() if name.endswith("deep_gemm/_C.so")
        )
        extension = archive.read(extension_name)
    if any(api not in python or api not in extension for api in apis):
        raise RuntimeError("built wheel is missing a caller-owned-output export")


def install_wheel(wheel: Path) -> None:
    run(sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", wheel)


def require_revision(repository: Path, expected: str) -> None:
    actual = output("git", "rev-parse", "HEAD", cwd=repository).strip()
    if actual != expected:
        raise RuntimeError(
            f"revision mismatch for {repository}: {actual} != {expected}"
        )


def require_remote(repository: Path, expected: str) -> None:
    actual = output("git", "remote", "get-url", "origin", cwd=repository).strip()
    if actual != expected:
        raise RuntimeError(f"remote mismatch for {repository}: {actual} != {expected}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output(
    *command: object,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    return subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout


def run(*command: object, cwd: Path | None = None) -> None:
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


if __name__ == "__main__":
    main()
