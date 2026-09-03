import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.benchmark import benchmark_decode, decode
from tools.benchmark.profiles import deepseek_v4_flash, glm53_flash


class BenchmarkRunnerTest(unittest.TestCase):
    def test_workload_builder_is_model_agnostic(self) -> None:
        class Codec:
            def encode_messages(self, messages):
                content = messages[0]["content"]
                group = int(content.split("benchmark group ", 1)[1][0])
                return (group, *([10] * 8_191), 100 + group)

        workload = decode.build_workload(
            Codec(),
            model_key="example_model",
            checkpoint_revision="example_revision",
        )

        validation = decode.validate_workload(workload, Codec())
        self.assertEqual(workload["model"]["key"], "example_model")
        self.assertEqual(validation["rendered_prompt_tokens"], 8_193)
        self.assertEqual(validation["within_group_lcp_tokens"], 8_192)

    def test_server_validation_does_not_assume_an_accelerator_stack(self) -> None:
        config = {
            "schema": decode.SERVER_CONFIG_SCHEMA,
            "engine": "example-engine",
            "model_key": "example-model",
            "model_id": "example/model",
            "endpoint": "http://127.0.0.1:18181",
            "launch_command": ["example-server"],
            "resolved_server_config": {"device": "example-accelerator"},
            "checkpoint": {"revision": "example-revision"},
        }

        validation = decode.validate_server_config(config)

        self.assertEqual(validation["engine"], "example-engine")

    def test_runner_launches_the_profile_server_command(self) -> None:
        workload = {
            "contract": {},
            "model": {
                "key": deepseek_v4_flash.MODEL_KEY,
                "checkpoint_revision": deepseek_v4_flash.MODEL_REVISION,
            },
        }
        config = {
            "model_key": deepseek_v4_flash.MODEL_KEY,
            "model_id": deepseek_v4_flash.MODEL_ID,
            "endpoint": "http://127.0.0.1:18181",
            "launch_command": ["python", "-m", "infer", "serve"],
            "checkpoint": {"revision": deepseek_v4_flash.MODEL_REVISION},
        }
        server = Mock(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            server_log = Path(directory) / "server.log"
            with (
                patch.object(
                    benchmark_decode,
                    "_read_json",
                    side_effect=(workload, config),
                ),
                patch.object(decode, "validate_workload", return_value={}),
                patch.object(decode, "validate_server_config", return_value={}),
                patch.object(decode, "run_decode_benchmark") as run_benchmark,
                patch.object(benchmark_decode, "SERVER_LOG", server_log),
                patch.object(
                    benchmark_decode.subprocess, "Popen", return_value=server
                ) as popen,
                patch.object(benchmark_decode, "wait_for_server") as wait,
                patch.object(benchmark_decode, "stop_server"),
            ):
                returncode = benchmark_decode.main(
                    [
                        "--workload",
                        "workload.json",
                        "--server-config",
                        "server.json",
                        "--working-directory",
                        directory,
                        "--concurrency",
                        "4",
                    ]
                )

        self.assertEqual(returncode, 0)
        popen.assert_called_once()
        self.assertEqual(popen.call_args.args[0], config["launch_command"])
        self.assertEqual(popen.call_args.kwargs["cwd"], Path(directory))
        wait.assert_called_once_with(server, config["endpoint"])
        self.assertEqual(run_benchmark.call_args.args[3], (4,))

    def test_profile_owns_the_deepseek_server_configuration(self) -> None:
        checkpoint = {
            "revision": deepseek_v4_flash.MODEL_REVISION,
            "sha256": "b" * 64,
        }
        config = deepseek_v4_flash.server_config(checkpoint)

        validation = decode.validate_server_config(config)
        self.assertEqual(validation["status"], "validated")
        self.assertEqual(config["model_id"], deepseek_v4_flash.MODEL_ID)
        self.assertIn("--checkpoint-dir", config["launch_command"])
        self.assertEqual(config["resolved_server_config"]["data_parallel"], 4)

    def test_every_profile_exposes_the_launcher_interface(self) -> None:
        for profile in (deepseek_v4_flash, glm53_flash):
            with self.subTest(profile=profile.MODEL_KEY):
                for attribute in (
                    "NAME",
                    "MODEL_KEY",
                    "MODEL_ID",
                    "MODEL_REVISION",
                    "CHECKPOINT_VOLUME",
                    "CHECKPOINT_MOUNT",
                    "REMOTE_ROOT",
                    "GPU",
                    "CPU",
                    "MEMORY_MIB",
                    "FUNCTION_TIMEOUT_SECONDS",
                    "ENVIRONMENT",
                ):
                    self.assertTrue(hasattr(profile, attribute), attribute)
                for function in (
                    "build_image",
                    "launcher_metadata",
                    "benchmark_inputs",
                    "checkpoint_manifest",
                    "server_config",
                    "gpu_topology",
                ):
                    self.assertTrue(
                        callable(getattr(profile, function, None)), function
                    )
                config = profile.server_config(
                    {"revision": profile.MODEL_REVISION, "sha256": "b" * 64}
                )
                self.assertEqual(
                    decode.validate_server_config(config)["status"], "validated"
                )
                command = config["launch_command"]
                self.assertEqual(command[1:4], ["-m", "infer", "serve"])
                self.assertIn(profile.MODEL_ID, command)

    def test_profile_validates_the_checkpoint_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "schema": 1,
                "model_id": deepseek_v4_flash.MODEL_ID,
                "revision": deepseek_v4_flash.MODEL_REVISION,
                "files": [],
            }
            (root / "infer-checkpoint-manifest.json").write_text(json.dumps(manifest))
            with patch.object(deepseek_v4_flash, "CHECKPOINT_MOUNT", root):
                receipt = deepseek_v4_flash.checkpoint_manifest()

        self.assertEqual(receipt["model_id"], deepseek_v4_flash.MODEL_ID)
        self.assertEqual(receipt["revision"], deepseek_v4_flash.MODEL_REVISION)
        self.assertEqual(receipt["files"], [])


if __name__ == "__main__":
    unittest.main()
