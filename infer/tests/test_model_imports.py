import sys
import unittest

DEEPSEEK_MODULES = (
    "infer.models.deepseek_v4_flash.attention",
    "infer.models.deepseek_v4_flash.checkpoint",
    "infer.models.deepseek_v4_flash.dspark",
    "infer.models.deepseek_v4_flash.megamoe",
    "infer.models.deepseek_v4_flash.target",
)
GLM_MODULES = (
    "infer.models.glm53_flash.checkpoint",
    "infer.models.glm53_flash.nextn",
    "infer.models.glm53_flash.target",
)


class ModelImportTest(unittest.TestCase):
    def test_deepseek_modules_import_directly(self) -> None:
        for name in DEEPSEEK_MODULES:
            with self.subTest(module=name):
                __import__(name)
                self.assertIn(name, sys.modules)

    def test_glm_modules_import_directly(self) -> None:
        for name in GLM_MODULES:
            with self.subTest(module=name):
                __import__(name)
                self.assertIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
