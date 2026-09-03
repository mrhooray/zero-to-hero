import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

PACKAGE = "infer-deepseek-v4-writer"
MODULE = "infer_deepseek_v4_writer"
VERSION = "0.3.0+f17b03ef"
TOKENSPEED_COMMIT = "f17b03efc1728875c586d848f49da5905032e87c"
CANDIDATE_SOURCE_SHA256 = (
    "173f5f63d01625cdf894103fd9e1917681cfd0d21f05ff5f42872fb83f917e1e"
)
V1_QUALIFIED_SOURCE_SHA256 = (
    "43b955988e5c53479daaaeafb912bebba2777fde6f1a5feba181fbbf74e575bf"
)
QUALIFICATION_STATUS = "qualified"
QUALIFIED_AOT_EXTENSION_SHA256: str | None = (
    "f49a2af7d66847f4a4ddb622cf6aa2cf6cbb945b098170ff2241abaab54f985f"
)
QUALIFIED_AOT_EXTENSION_SIZE: int | None = 279_432
QUALIFIED_BUILD_SHA256: str | None = (
    "59780c91b7a6a4cc881cf446d5689957bcd7b32062ed7721b41b4e8955edb4f2"
)
QUALIFICATION_ARTIFACT_SHA256: str | None = (
    "25f991c86b6fd169fbcd18c9db5c830bc964d1c3cc31524fd8c47104530c75b4"
)
QUALIFICATION_REVISION: str | None = "f8c075dcd7ca807ea564737629eba71669278942"
SOURCE_SHA256 = {
    "src/infer/models/deepseek_v4_flash/ops/_deepseek_v4_writer_bindings.cpp": (
        "1486b83e994e4c4991bb938478804f054bb00880da73711a9dd0f87b8f7b30cf"
    ),
    "src/infer/models/deepseek_v4_flash/ops/_deepseek_v4_writer.cu": (
        "1a75e9a5700af61a6f687d3a1a26ed4da62ac539e5479b8c2fc0b0a1ee9d7c9d"
    ),
    "src/infer/models/deepseek_v4_flash/ops/_deepseek_v4_writer.LICENSE.txt": (
        "9789cc95f9bc9aa9eedcbc4b17057e568a621c54e5db37ce7fc8c8e5ea5b5ee7"
    ),
}
STAGED_SOURCE_PATHS = {
    name: name.replace("models/deepseek_v4_flash/ops/", "ops/")
    for name in SOURCE_SHA256
}
ROOT = Path(__file__).resolve().parents[2]
BUILD_ROOT = Path("/opt/deepseek-v4-writer-build")
WHEEL_DIR = Path("/opt/deepseek-v4-writer-wheel")
BUILD_RECEIPT = Path("/opt/deepseek-v4-writer-build.json")
STAGED_SOURCE_ROOT = BUILD_ROOT / "source"
SOURCE_DATE_EPOCH = "315532800"
INTERNAL_BUILD_ARGUMENT = "--build-wheel"
DIAGNOSTICS_PREFIX = "DEEPSEEK_V4_WRITER_DIAGNOSTICS="
DIAGNOSTIC_ELF_SECTIONS = frozenset(
    {
        ".comment",
        ".note.gnu.build-id",
        ".nv_fatbin",
        ".rodata",
        ".strtab",
        ".symtab",
        ".text",
    }
)
STRIPPED_ELF_SECTIONS = frozenset({".strtab", ".symtab"})
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
    "LC_ALL": "C",
    "MAX_JOBS": "8",
    "NVCC_THREADS": "2",
    "PATH": "/usr/local/bin:/usr/local/cuda/bin:/usr/bin:/bin",
    "PYTHONHASHSEED": "0",
    "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH,
    "TMPDIR": str(BUILD_ROOT / "tmp"),
    "TZ": "UTC",
}
CLEARED_BUILD_ENV = (
    "AR",
    "CFLAGS",
    "CPPFLAGS",
    "CUDAARCHS",
    "CXXFLAGS",
    "LD",
    "LDFLAGS",
    "LDSHARED",
    "NVCC_APPEND_FLAGS",
    "NVCC_PREPEND_FLAGS",
    "PYTHONHOME",
    "PYTHONPATH",
    "RANLIB",
    "TORCH_CUDA_ARCH_LIST",
)
CXX_RANDOM_SEED = SOURCE_SHA256[
    "src/infer/models/deepseek_v4_flash/ops/_deepseek_v4_writer_bindings.cpp"
]
NVCC_RANDOM_SEED = SOURCE_SHA256[
    "src/infer/models/deepseek_v4_flash/ops/_deepseek_v4_writer.cu"
]
PATH_REMAP_FLAGS = (
    f"-ffile-prefix-map={ROOT}=.",
    f"-fdebug-prefix-map={ROOT}=.",
    f"-ffile-prefix-map={BUILD_ROOT}=./build",
    f"-fdebug-prefix-map={BUILD_ROOT}=./build",
)
LINK_FLAGS = ("-Wl,--strip-all",)
CXX_FLAGS = (
    "-O3",
    "-std=c++17",
    "-DNDEBUG",
    f"-frandom-seed={CXX_RANDOM_SEED}",
    *PATH_REMAP_FLAGS,
    "-g0",
)
NVCC_FLAGS = (
    "-O3",
    "-std=c++17",
    "-DNDEBUG",
    "--use_fast_math",
    "-lineinfo",
    f"--frandom-seed={NVCC_RANDOM_SEED}",
    *(f"-Xcompiler={flag}" for flag in PATH_REMAP_FLAGS),
    "-gencode=arch=compute_100a,code=sm_100a",
    "-gencode=arch=compute_100a,code=compute_100a",
)
EXPECTED_NVCC_VERSION = """nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_May_27_02:21:03_PDT_2025
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0"""
EXPECTED_LD_VERSION = "GNU ld (GNU Binutils for Ubuntu) 2.38"
EXPECTED_READELF_VERSION = "GNU readelf (GNU Binutils for Ubuntu) 2.38"
EXPECTED_NINJA_VERSION = "1.13.0.git.kitware.jobserver-pipe-1"
SYSCONFIG_KEYS = (
    "AR",
    "ARFLAGS",
    "CC",
    "CCSHARED",
    "CFLAGS",
    "CXX",
    "EXT_SUFFIX",
    "LDFLAGS",
    "LDSHARED",
    "SOABI",
)


def main() -> None:
    if sys.argv[1:] == [INTERNAL_BUILD_ARGUMENT]:
        validate_build_process_environment()
        build_wheel_in_environment()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    validate_qualification_mode(args.candidate)
    toolchain = validate_build_environment()
    sources = validate_sources(ROOT)
    wheel = build_wheel()
    install_wheel(wheel)
    artifact = validate_install(wheel, require_qualified=False)
    diagnostics = build_diagnostics(Path(artifact["extension"]))
    print(DIAGNOSTICS_PREFIX + json.dumps(diagnostics, sort_keys=True))
    build = build_provenance(toolchain)
    build_sha256 = canonical_sha256(build)
    build_identity_match = None
    if not args.candidate:
        validate_extension_identity(
            artifact["extension_sha256"],
            artifact["extension_size"],
            QUALIFIED_AOT_EXTENSION_SHA256,
            QUALIFIED_AOT_EXTENSION_SIZE,
        )
        build_identity_match = validate_build_identity(
            build_sha256, QUALIFIED_BUILD_SHA256
        )
    receipt = {
        "artifact": artifact,
        "build": build,
        "build_sha256": build_sha256,
        "build_identity_match": build_identity_match,
        "diagnostics": diagnostics,
        "license": {
            "name": "MIT",
            "sha256": SOURCE_SHA256[
                next(name for name in SOURCE_SHA256 if name.endswith("LICENSE.txt"))
            ],
        },
        "package": {"module": MODULE, "name": PACKAGE, "version": VERSION},
        "qualification": qualification_receipt(),
        "v1_oracle": {"source_sha256": V1_QUALIFIED_SOURCE_SHA256},
        "source": {
            "files": sources,
            "candidate_source_sha256": CANDIDATE_SOURCE_SHA256,
            "tokenspeed_commit": TOKENSPEED_COMMIT,
        },
    }
    BUILD_RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("DEEPSEEK_V4_WRITER_BUILD=" + json.dumps(receipt, sort_keys=True))


def validate_qualification_mode(candidate: bool) -> None:
    pins = (
        QUALIFIED_AOT_EXTENSION_SHA256,
        QUALIFIED_AOT_EXTENSION_SIZE,
        QUALIFIED_BUILD_SHA256,
        QUALIFICATION_ARTIFACT_SHA256,
        QUALIFICATION_REVISION,
    )
    if QUALIFICATION_STATUS == "pending":
        if any(pin is not None for pin in pins):
            raise RuntimeError("pending DeepSeek writer has stale qualification pins")
        if not candidate:
            raise RuntimeError("DeepSeek writer v3 qualification is pending")
        return
    if QUALIFICATION_STATUS != "qualified" or any(pin is None for pin in pins):
        raise RuntimeError("DeepSeek writer qualification state is invalid")
    if (
        re.fullmatch(r"[0-9a-f]{64}", QUALIFIED_AOT_EXTENSION_SHA256 or "") is None
        or type(QUALIFIED_AOT_EXTENSION_SIZE) is not int
        or QUALIFIED_AOT_EXTENSION_SIZE <= 0
        or re.fullmatch(r"[0-9a-f]{64}", QUALIFIED_BUILD_SHA256 or "") is None
        or re.fullmatch(r"[0-9a-f]{64}", QUALIFICATION_ARTIFACT_SHA256 or "") is None
        or re.fullmatch(r"[0-9a-f]{40}", QUALIFICATION_REVISION or "") is None
    ):
        raise RuntimeError("DeepSeek writer qualification pins are invalid")
    if candidate:
        raise RuntimeError("qualified DeepSeek writer cannot run in candidate mode")


def qualification_receipt() -> dict[str, object]:
    receipt: dict[str, object] = {
        "status": QUALIFICATION_STATUS,
        "version": "writer-v3",
    }
    if QUALIFICATION_STATUS == "qualified":
        receipt.update(
            artifact_sha256=QUALIFICATION_ARTIFACT_SHA256,
            build_sha256=QUALIFIED_BUILD_SHA256,
            extension_sha256=QUALIFIED_AOT_EXTENSION_SHA256,
            extension_size=QUALIFIED_AOT_EXTENSION_SIZE,
            git_sha=QUALIFICATION_REVISION,
        )
    return receipt


def build_provenance(toolchain: dict[str, object]) -> dict[str, object]:
    return {
        "cleared_environment": list(CLEARED_BUILD_ENV),
        "cxx_flags": list(CXX_FLAGS),
        "environment": BUILD_ENV,
        "environment_policy": "allowlist",
        "link_flags": list(LINK_FLAGS),
        "nvcc_flags": list(NVCC_FLAGS),
        "packages": BUILD_PACKAGE_VERSIONS,
        "paths": {
            "build_root": str(BUILD_ROOT),
            "source_root": str(ROOT),
            "staged_source_root": str(STAGED_SOURCE_ROOT),
            "wheel_dir": str(WHEEL_DIR),
        },
        "source_mtime_ns": int(SOURCE_DATE_EPOCH) * 1_000_000_000,
        "toolchain": toolchain,
    }


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_build_identity(digest: str, expected_digest: str | None) -> bool:
    if expected_digest is None:
        print(
            "DeepSeek writer build identity is unpinned; recording the observed digest",
            file=sys.stderr,
        )
        return False
    if digest != expected_digest:
        print(
            "DeepSeek writer build identity mismatch: "
            f"expected sha256={expected_digest}; observed sha256={digest}",
            file=sys.stderr,
        )
        return False
    return True


def validate_build_environment() -> dict[str, object]:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("the DeepSeek writer is qualified only on Linux x86_64")
    versions = {
        name: importlib.metadata.version(name) for name in BUILD_PACKAGE_VERSIONS
    }
    if versions != BUILD_PACKAGE_VERSIONS:
        raise RuntimeError(f"build package mismatch: {versions}")
    import torch

    ninja = Path(sys.executable).resolve().parent / "ninja"
    if not ninja.is_file():
        raise RuntimeError("pinned ninja executable is missing")
    tools = {
        "cc": executable_identity(BUILD_ENV["CC"], "-dumpfullversion"),
        "cxx": executable_identity(BUILD_ENV["CXX"], "-dumpfullversion"),
        "ld": executable_identity("/usr/bin/ld", "--version"),
        "ninja": executable_identity(str(ninja), "--version"),
        "nvcc": executable_identity(f"{BUILD_ENV['CUDA_HOME']}/bin/nvcc", "--version"),
        "python": executable_identity(sys.executable, "--version"),
        "readelf": executable_identity("/usr/bin/readelf", "--version"),
    }
    validate_toolchain(
        torch.__version__,
        torch.version.cuda,
        tools["nvcc"]["version"],
        tools["cc"]["version"],
        tools["cxx"]["version"],
        tools["ld"]["version"],
        tools["ninja"]["version"],
        tools["readelf"]["version"],
    )
    linker_name = command_output(BUILD_ENV["CXX"], "-print-prog-name=ld")
    linker_path = (
        linker_name if Path(linker_name).is_absolute() else shutil.which(linker_name)
    )
    if linker_path is None:
        raise RuntimeError(f"g++ linker is missing: {linker_name}")
    linker = Path(linker_path).resolve()
    if linker != Path(tools["ld"]["path"]):
        raise RuntimeError(f"g++ resolves an unexpected linker: {linker}")
    return {
        "executables": tools,
        "platform": {"machine": platform.machine(), "system": platform.system()},
        "python": platform.python_version(),
        "sysconfig": sysconfig_identity(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def validate_toolchain(
    torch_version: str,
    torch_cuda: str | None,
    nvcc: str,
    cc: str,
    cxx: str,
    ld: str,
    ninja: str,
    readelf: str,
) -> None:
    if torch_version != "2.13.0+cu129" or torch_cuda != "12.9":
        raise RuntimeError("DeepSeek writer requires torch 2.13.0+cu129/CUDA 12.9")
    if (
        nvcc != EXPECTED_NVCC_VERSION
        or cc != "11.4.0"
        or cxx != "11.4.0"
        or ld.splitlines()[0] != EXPECTED_LD_VERSION
        or ninja != EXPECTED_NINJA_VERSION
        or readelf.splitlines()[0] != EXPECTED_READELF_VERSION
    ):
        raise RuntimeError("DeepSeek writer toolchain identity mismatch")


def executable_identity(executable: str, *version_args: str) -> dict[str, object]:
    resolved = Path(executable).resolve()
    if not resolved.is_file():
        raise RuntimeError(f"DeepSeek writer tool is missing: {executable}")
    version = command_output(str(resolved), *version_args)
    if not version:
        raise RuntimeError(f"DeepSeek writer tool has no version: {executable}")
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "version": version,
    }


def sysconfig_identity() -> dict[str, str]:
    values = {name: sysconfig.get_config_var(name) for name in SYSCONFIG_KEYS}
    if any(not isinstance(value, str) for value in values.values()):
        raise RuntimeError(f"DeepSeek writer sysconfig is incomplete: {values}")
    return values


def validate_sources(root: Path) -> dict[str, str]:
    observed = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in SOURCE_SHA256
    }
    if observed != SOURCE_SHA256:
        raise RuntimeError(f"DeepSeek writer source hashes differ: {observed}")
    native = [root / name for name in SOURCE_SHA256 if not name.endswith(".txt")]
    validate_source_stems(native)
    cpp, cuda = native
    combined = hashlib.sha256(cpp.read_bytes() + b"\0" + cuda.read_bytes()).hexdigest()
    if combined != CANDIDATE_SOURCE_SHA256:
        raise RuntimeError("DeepSeek writer no longer matches the candidate source")
    return observed


def validate_source_stems(sources: list[Path]) -> None:
    stems = [source.stem for source in sources]
    if len(stems) != len(set(stems)):
        raise RuntimeError(f"DeepSeek writer source object stems collide: {stems}")


def build_wheel() -> Path:
    prepare_build_directories()
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), INTERNAL_BUILD_ARGUMENT],
        cwd=ROOT,
        env=BUILD_ENV,
        check=True,
    )
    return built_wheel()


def prepare_build_directories() -> None:
    for directory in (BUILD_ROOT, WHEEL_DIR):
        if directory.exists():
            raise RuntimeError(f"build directory already exists: {directory}")
        directory.mkdir(parents=True)
    (BUILD_ROOT / "tmp").mkdir()


def validate_build_process_environment() -> None:
    mismatched = {
        name: os.environ.get(name)
        for name, expected in BUILD_ENV.items()
        if os.environ.get(name) != expected
    }
    uncleared = [name for name in CLEARED_BUILD_ENV if name in os.environ]
    prepared = (
        BUILD_ROOT.is_dir()
        and set(BUILD_ROOT.iterdir()) == {BUILD_ROOT / "tmp"}
        and (BUILD_ROOT / "tmp").is_dir()
        and WHEEL_DIR.is_dir()
        and not any(WHEEL_DIR.iterdir())
    )
    if (
        mismatched
        or uncleared
        or sys.argv[1:] != [INTERNAL_BUILD_ARGUMENT]
        or not prepared
    ):
        raise RuntimeError(
            "DeepSeek writer build process is not canonical: "
            f"mismatched={mismatched}, uncleared={uncleared}, "
            f"argv={sys.argv[1:]}, prepared={prepared}"
        )


def build_wheel_in_environment() -> Path:
    from setuptools import Distribution
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    source_paths = [ROOT / name for name in SOURCE_SHA256 if not name.endswith(".txt")]
    validate_source_stems(source_paths)
    source_paths = stage_sources(source_paths)
    extension = CUDAExtension(
        MODULE,
        sources=[str(path) for path in source_paths],
        extra_compile_args={"cxx": list(CXX_FLAGS), "nvcc": list(NVCC_FLAGS)},
        extra_link_args=list(LINK_FLAGS),
    )
    distribution = Distribution(
        {
            "description": "Pinned DeepSeek V4 raw-Q/KV writer",
            "ext_modules": [extension],
            "name": PACKAGE,
            "version": VERSION,
        }
    )
    distribution.cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=True)}
    distribution.metadata.license = "MIT"
    distribution.metadata.license_files = [
        "src/infer/models/deepseek_v4_flash/ops/_deepseek_v4_writer.LICENSE.txt"
    ]
    distribution.command_options = {
        "build": {"build_base": ("build helper", str(BUILD_ROOT / "build"))},
        "bdist_wheel": {
            "bdist_dir": ("build helper", str(BUILD_ROOT / "wheel")),
            "dist_dir": ("build helper", str(WHEEL_DIR)),
        },
        "egg_info": {"egg_base": ("build helper", str(BUILD_ROOT))},
    }
    distribution.script_name = "setup.py"
    working_directory = Path.cwd()
    os.chdir(ROOT)
    try:
        distribution.run_command("bdist_wheel")
    finally:
        os.chdir(working_directory)
    return built_wheel()


def stage_sources(sources: list[Path]) -> list[Path]:
    epoch_ns = int(SOURCE_DATE_EPOCH) * 1_000_000_000
    staged = []
    for source in sources:
        relative = source.relative_to(ROOT)
        target = STAGED_SOURCE_ROOT / STAGED_SOURCE_PATHS[str(relative)]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o644)
        os.utime(target, ns=(epoch_ns, epoch_ns))
        payload = target.read_bytes()
        stat = target.stat()
        if (
            hashlib.sha256(payload).hexdigest() != SOURCE_SHA256[str(relative)]
            or stat.st_mtime_ns != epoch_ns
            or stat.st_mode & 0o777 != 0o644
        ):
            raise RuntimeError(f"staged DeepSeek writer source differs: {target}")
        staged.append(target)
    return staged


def built_wheel() -> Path:
    wheels = tuple(WHEEL_DIR.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one DeepSeek writer wheel, found {wheels}")
    return wheels[0]


def build_diagnostics(installed_extension: Path) -> dict[str, object]:
    objects = sorted((BUILD_ROOT / "build").rglob("*.o"))
    linked = sorted((BUILD_ROOT / "build").rglob(f"{MODULE}*.so"))
    if len(objects) != 2 or len(linked) != 1:
        raise RuntimeError(
            f"unexpected DeepSeek writer build outputs: objects={objects}, linked={linked}"
        )
    diagnostics = {
        "objects": {
            str(path.relative_to(BUILD_ROOT)): elf_file_receipt(path)
            for path in objects
        },
        "linked_extension": elf_file_receipt(linked[0], require_stripped=True),
        "installed_extension": elf_file_receipt(
            installed_extension, require_stripped=True
        ),
        "staged_sources": staged_source_receipts(),
    }
    if (
        diagnostics["linked_extension"]["sha256"]
        != diagnostics["installed_extension"]["sha256"]
    ):
        raise RuntimeError("linked and installed DeepSeek writer extensions differ")
    return diagnostics


def staged_source_receipts() -> dict[str, dict[str, object]]:
    epoch_ns = int(SOURCE_DATE_EPOCH) * 1_000_000_000
    receipts = {}
    for name, expected_hash in SOURCE_SHA256.items():
        if name.endswith(".txt"):
            continue
        path = STAGED_SOURCE_ROOT / STAGED_SOURCE_PATHS[name]
        payload = path.read_bytes()
        stat = path.stat()
        receipt = {
            "mode": stat.st_mode & 0o777,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }
        if (
            receipt["sha256"] != expected_hash
            or receipt["mtime_ns"] != epoch_ns
            or receipt["mode"] != 0o644
        ):
            raise RuntimeError(
                f"staged DeepSeek writer source receipt differs: {receipt}"
            )
        receipts[name] = receipt
    return receipts


def elf_file_receipt(
    path: Path, *, require_stripped: bool = False
) -> dict[str, object]:
    payload = path.read_bytes()
    listing = command_output("/usr/bin/readelf", "--wide", "--sections", str(path))
    return {
        "sections": elf_section_receipts(
            payload, listing, require_stripped=require_stripped
        ),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def elf_section_receipts(
    payload: bytes, listing: str, *, require_stripped: bool = False
) -> list[dict[str, object]]:
    pattern = re.compile(
        r"^\s*\[\s*(\d+)\]\s+(\S+)\s+(\S+)\s+[0-9a-fA-F]+\s+"
        r"([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+",
        re.MULTILINE,
    )
    matches = pattern.findall(listing)
    retained = STRIPPED_ELF_SECTIONS.intersection(name for _, name, _, _, _ in matches)
    if require_stripped and retained:
        raise RuntimeError(
            f"DeepSeek writer final ELF retains stripped sections: {sorted(retained)}"
        )
    sections = []
    for index, name, section_type, offset_hex, size_hex in matches:
        if index == "0" or name not in DIAGNOSTIC_ELF_SECTIONS:
            continue
        offset = int(offset_hex, 16)
        size = int(size_hex, 16)
        if section_type == "NOBITS":
            digest = None
        elif offset + size <= len(payload):
            digest = hashlib.sha256(payload[offset : offset + size]).hexdigest()
        else:
            raise RuntimeError(f"ELF section exceeds its file: {name}")
        sections.append(
            {
                "index": int(index),
                "name": name,
                "offset": offset,
                "sha256": digest,
                "size": size,
                "type": section_type,
            }
        )
    names = [section["name"] for section in sections]
    if ".text" not in names or len(names) != len(set(names)):
        raise RuntimeError("DeepSeek writer ELF section table is incomplete")
    return sections


def install_wheel(wheel: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            str(wheel),
        ],
        check=True,
    )


def validate_install(
    wheel: Path, *, require_qualified: bool = True
) -> dict[str, object]:
    module = importlib.import_module(MODULE)
    if importlib.metadata.version(PACKAGE) != VERSION or not callable(
        getattr(module, "fused_qnorm_rope_kv_insert", None)
    ):
        raise RuntimeError("DeepSeek writer package surface mismatch")
    extension = Path(module.__file__).resolve()
    symbols = command_output("nm", "-D", str(extension))
    elf = command_output("cuobjdump", "--list-elf", str(extension))
    ptx = command_output("cuobjdump", "--dump-ptx", str(extension))
    validate_binary(symbols, elf, ptx)
    extension_bytes = extension.read_bytes()
    extension_sha256 = hashlib.sha256(extension_bytes).hexdigest()
    if require_qualified:
        validate_qualification_mode(candidate=False)
        validate_extension_identity(
            extension_sha256,
            len(extension_bytes),
            QUALIFIED_AOT_EXTENSION_SHA256,
            QUALIFIED_AOT_EXTENSION_SIZE,
        )
    with zipfile.ZipFile(wheel) as archive:
        binaries = [name for name in archive.namelist() if name.endswith(".so")]
        licenses = [name for name in archive.namelist() if name.endswith("LICENSE.txt")]
        if len(binaries) != 1 or len(licenses) != 1:
            raise RuntimeError("DeepSeek writer wheel surface mismatch")
        if archive.read(binaries[0]) != extension_bytes:
            raise RuntimeError("installed DeepSeek writer differs from its wheel")
        license_hash = hashlib.sha256(archive.read(licenses[0])).hexdigest()
        if (
            license_hash
            != SOURCE_SHA256[
                "src/infer/models/deepseek_v4_flash/ops/"
                "_deepseek_v4_writer.LICENSE.txt"
            ]
        ):
            raise RuntimeError("DeepSeek writer wheel license mismatch")
    return {
        "code_objects": {"ptx_sm100a": True, "sass_sm100a": True},
        "extension": str(extension),
        "extension_sha256": extension_sha256,
        "extension_size": len(extension_bytes),
        "wheel": str(wheel),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }


def validate_extension_identity(
    digest: str,
    size: int,
    expected_digest: str | None,
    expected_size: int | None,
) -> None:
    if expected_digest is None or expected_size is None:
        raise RuntimeError("DeepSeek writer AOT qualification is pending")
    if digest != expected_digest or size != expected_size:
        raise RuntimeError(
            "DeepSeek writer AOT extension identity mismatch: "
            f"expected sha256={expected_digest} size={expected_size}; "
            f"observed sha256={digest} size={size}"
        )


def validate_binary(symbols: str, elf: str, ptx: str, module: str = MODULE) -> None:
    if f"PyInit_{module}" not in symbols:
        raise RuntimeError("DeepSeek writer extension export is missing")
    architectures = set(re.findall(r"sm_[0-9]+a?", elf))
    if architectures != {"sm_100a"}:
        raise RuntimeError(f"unexpected DeepSeek writer code objects: {architectures}")
    if re.search(r"\.target\s+sm_100a\b", ptx) is None:
        raise RuntimeError("DeepSeek writer compute_100a PTX is missing")


def command_output(*command: str) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()


if __name__ == "__main__":
    main()
