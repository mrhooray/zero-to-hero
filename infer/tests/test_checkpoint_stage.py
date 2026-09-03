import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.checkpoint import launch_modal as modal_stage
from tools.checkpoint import stage


class CheckpointStageTest(unittest.TestCase):
    def test_stages_and_reuses_a_pinned_glm_checkpoint(self) -> None:
        model_id = "zai-org/GLM-5.3-Flash"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calls = []

            def download(**kwargs):
                calls.append(kwargs)
                root = Path(kwargs["local_dir"])
                (root / "config.json").write_text("{}")
                (root / "tokenizer.json").write_text("tokenizer")
                (root / "chat_template.jinja").write_text("template")
                (root / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": {"weight": "model-00001.safetensors"}})
                )
                (root / "model-00001.safetensors").write_bytes(b"weights")
                return str(root)

            first = stage.stage_checkpoint(model_id, root, downloader=download)
            second = stage.stage_checkpoint(
                model_id,
                root,
                downloader=Mock(side_effect=AssertionError("unexpected download")),
            )

        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["files"], 5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["revision"], stage.MODEL_SPECS[model_id].revision)
        self.assertIn("*.safetensors", calls[0]["allow_patterns"])

    def test_requires_every_shard_named_by_the_index(self) -> None:
        spec = stage.MODEL_SPECS["zai-org/GLM-5.3-Flash"]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "missing.safetensors"}})
            )
            for name in spec.required_files:
                if name == "model.safetensors.index.json":
                    continue
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"file")

            with self.assertRaisesRegex(ValueError, "missing.safetensors"):
                stage.stage_checkpoint(
                    spec.model_id,
                    root,
                    downloader=lambda **_: str(root),
                )


class ModalCheckpointStageTest(unittest.TestCase):
    @patch("tools.checkpoint.launch_modal.modal.Sandbox.create")
    def test_transfer_sandbox_has_no_gpu(self, create: Mock) -> None:
        modal_stage.create_stage_sandbox(
            object(),
            object(),
            object(),
            "zai-org/GLM-5.3-Flash",
            "0123456789abcdef",
        )
        _, kwargs = create.call_args
        self.assertEqual(kwargs["cpu"], 8)
        self.assertEqual(kwargs["memory"], 8192)
        self.assertNotIn("gpu", kwargs)


if __name__ == "__main__":
    unittest.main()
