import ast
import hashlib
import unittest
from pathlib import Path

from tools.benchmark.profiles import glm53_flash
from tools.kernels import install_flash_kda

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_WRAPPER = (
    ROOT / "src/infer/models/glm53_flash/ops/prefill_kda.py"
)
PROVENANCE_NOTICE = (
    ROOT / "src/infer/models/glm53_flash/ops/prefill_kda.LICENSE.txt"
)


def wrapper_version() -> str:
    tree = ast.parse(RUNTIME_WRAPPER.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            if (
                len(targets) == 1
                and isinstance(targets[0], ast.Name)
                and targets[0].id == "FLASH_KDA_VERSION"
                and isinstance(node.value, ast.Constant)
            ):
                return node.value.value
    raise AssertionError("FLASH_KDA_VERSION not found in runtime wrapper")


class FlashKDAProvenanceTest(unittest.TestCase):
    def test_profile_reports_the_installer_revision(self) -> None:
        self.assertEqual(
            glm53_flash.FLASH_KDA_COMMIT, install_flash_kda.FLASH_KDA_COMMIT
        )
        self.assertEqual(
            glm53_flash.FLASH_KDA_VERSION, install_flash_kda.FLASH_KDA_VERSION
        )
        self.assertEqual(
            glm53_flash.CUTLASS_COMMIT, install_flash_kda.CUTLASS_COMMIT
        )

    def test_runtime_wrapper_matches_the_installer(self) -> None:
        self.assertEqual(
            wrapper_version(), install_flash_kda.FLASH_KDA_VERSION
        )

    def test_provenance_notice_matches_the_installer(self) -> None:
        notice = PROVENANCE_NOTICE.read_text()
        for pinned in (
            install_flash_kda.FLASH_KDA_COMMIT,
            install_flash_kda.CUTLASS_COMMIT,
            install_flash_kda.FLASH_KDA_VERSION,
            install_flash_kda.FLASH_KDA_NOALLOC_PATCH_SHA256,
            install_flash_kda.PATCHED_SOURCE_SHA256["csrc/flash_kda.cpp"],
            install_flash_kda.SOURCE_SHA256["csrc/flash_kda.cpp"],
            "tools/kernels/install_flash_kda.py#FLASH_KDA_NOALLOC_PATCH",
        ):
            self.assertIn(pinned, notice)

    def test_embedded_patch_self_hash(self) -> None:
        self.assertEqual(
            hashlib.sha256(install_flash_kda.FLASH_KDA_NOALLOC_PATCH).hexdigest(),
            install_flash_kda.FLASH_KDA_NOALLOC_PATCH_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
