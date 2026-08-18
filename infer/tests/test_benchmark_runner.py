import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.benchmark import benchmark_decode, decode, launch_modal
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

    def test_tokenspeed_readiness_uses_chat_completion(self) -> None:
        response = Mock(status=200)
        response.read.return_value = b"{}"
        connection = Mock()
        connection.getresponse.return_value = response
        server = Mock(returncode=None)
        server.poll.return_value = None

        with patch.object(
            benchmark_decode.http.client,
            "HTTPConnection",
            return_value=connection,
        ):
            validation = benchmark_decode.wait_for_tokenizer(
                server,
                "http://127.0.0.1:8000",
                "model",
            )

        body = json.loads(connection.request.call_args.args[2])
        self.assertEqual(validation["chat_completion"], "ready")
        self.assertEqual(body["model"], "model")
        self.assertEqual(body["max_tokens"], 1)

    def test_success_receipt_drops_per_request_details(self) -> None:
        result = {
            "status": "passed",
            "priming": [{"requests": [{"ordinal": 0}], "cached_tokens": [0]}],
            "benchmarks": [
                {
                    "concurrency": 4,
                    "content_sha256": ["a" * 64, "b" * 64],
                    "requests": [{"ordinal": 0}, {"ordinal": 1}],
                }
            ],
        }

        benchmark_decode.compact_success_receipt(result)

        self.assertNotIn("requests", result["priming"][0])
        point = result["benchmarks"][0]
        self.assertNotIn("requests", point)
        self.assertNotIn("content_sha256", point)
        self.assertEqual(point["distinct_measured_content_count"], 2)
        self.assertEqual(
            point["measured_content_set_sha256"],
            "fa0dafbf43f1f551e536353e9d1a942a8e86e41a0b58dfeaf264ef217f6b862a",
        )

    def test_failed_receipt_keeps_per_request_details(self) -> None:
        result = {
            "status": "failed",
            "benchmarks": [{"requests": [{"ordinal": 0}]}],
        }

        benchmark_decode.compact_success_receipt(result)

        self.assertEqual(result["benchmarks"][0]["requests"], [{"ordinal": 0}])

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

    def test_deepseek_profile_selects_native_speculation(self) -> None:
        checkpoint = {
            "revision": deepseek_v4_flash.MODEL_REVISION,
            "sha256": "b" * 64,
        }

        config = deepseek_v4_flash.server_config(checkpoint, "native")

        command = config["launch_command"]
        self.assertEqual(command[command.index("--speculation") + 1], "native")
        self.assertEqual(config["resolved_server_config"]["speculation"], "native")

    def test_result_records_the_resolved_speculation_mode(self) -> None:
        receipt = {}
        workload = {"groups": [], "requests": []}
        config = {
            "endpoint": "http://127.0.0.1:18181",
            "resolved_server_config": {"speculation": "native"},
        }

        with (
            patch.object(decode, "priming_waves", return_value=()),
            patch.object(decode, "qualify_priming", return_value={"primed": True}),
        ):
            decode.run_decode_benchmark(receipt, workload, config, ())

        self.assertEqual(receipt["qualification"]["speculation"], "native")

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

    def test_external_profiles_preserve_original_runtime_parameters(self) -> None:
        checkpoint = {"revision": "revision", "sha256": "b" * 64}
        for profile in (deepseek_v4_flash, glm53_flash):
            for engine in ("tokenspeed", "sglang"):
                with self.subTest(profile=profile.MODEL_KEY, engine=engine):
                    config = profile.server_config(checkpoint, engine=engine)
                    command = config["launch_command"]
                    self.assertEqual(config["engine"], engine)
                    self.assertEqual(
                        config["resolved_server_config"]["speculation"], "none"
                    )
                    self.assertEqual(
                        config["resolved_server_config"][
                            "max_num_seqs_global"
                            if engine == "tokenspeed"
                            else "max_running_requests_global"
                        ],
                        128,
                    )
                    self.assertEqual(
                        config["resolved_server_config"]["cuda_graph_local_batch_max"],
                        32,
                    )
                    if engine == "tokenspeed":
                        self.assertIn("--enable-prefix-caching", command)
                        self.assertIn("--dp-aware", command)
                    else:
                        self.assertIn("--enable-dp-attention", command)
                        self.assertEqual(
                            config["resolved_server_config"]["prefix_routing"],
                            "client_group_to_dp_rank",
                        )
                        if profile is deepseek_v4_flash:
                            self.assertEqual(
                                command[:2],
                                [
                                    "env",
                                    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096",
                                ],
                            )
                            self.assertEqual(
                                command[command.index("--stream-interval") + 1],
                                "3",
                            )
                            self.assertEqual(
                                config["resolved_server_config"][
                                    "mega_moe_max_tokens_per_rank"
                                ],
                                4096,
                            )
                        elif profile is glm53_flash:
                            self.assertEqual(
                                command[command.index("--max-mamba-cache-size") + 1],
                                "640",
                            )
                            self.assertEqual(
                                config["resolved_server_config"][
                                    "max_mamba_cache_size_global"
                                ],
                                640,
                            )

    def test_sglang_request_routes_prefix_group_to_dp_rank(self) -> None:
        body = json.loads(
            decode.request_body(
                model_id="model",
                messages=[{"role": "user", "content": "prompt"}],
                routed_dp_rank=3,
            )
        )

        self.assertEqual(body["routed_dp_rank"], 3)

    def test_external_engine_runs_in_named_sandbox(self) -> None:
        sandbox = Mock(returncode=0)
        sandbox.stdout.read.return_value = "truncated receipt"
        sandbox.stderr.read.return_value = ""
        sandbox.filesystem.read_text.return_value = '{"status":"passed"}\n'
        receipt = sandbox.exec.return_value
        profile = SimpleNamespace(
            REMOTE_ROOT=Path("/opt/infer"),
            GPU="B200:4",
            FUNCTION_TIMEOUT_SECONDS=60,
            CHECKPOINT_MOUNT=Path("/checkpoint"),
            ENVIRONMENT={},
        )

        with patch.object(
            launch_modal.modal.Sandbox, "create", return_value=sandbox
        ) as create:
            result = launch_modal.run_sandbox(
                Mock(),
                Mock(),
                Mock(),
                profile,
                "glm53_flash",
                "tokenspeed",
                [4],
                "none",
                "0123456789abcdef",
            )

        self.assertEqual(result["result"]["status"], "passed")
        sandbox.filesystem.read_text.assert_called_once_with(
            launch_modal.SANDBOX_RESULT
        )
        receipt.wait.assert_called_once_with()
        sandbox.filesystem.write_text.assert_called_once_with(
            "received\n", launch_modal.SANDBOX_RESULT_ACK
        )
        self.assertEqual(create.call_args.kwargs["name"], "0123456789abcdef")
        sandbox.detach.assert_called_once_with()

    def test_cleanup_accepts_an_already_stopped_app(self) -> None:
        result = {}
        completed = SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="App is already stopped.\n",
        )

        with patch.object(launch_modal.subprocess, "run", return_value=completed):
            launch_modal.stop_app("ap-example", result)

        self.assertTrue(result["app_stopped"])
        self.assertNotIn("cleanup_error", result)

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

    def test_stream_uses_finish_event_for_last_token_timing(self) -> None:
        events = (
            {
                "id": "completion",
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "completion",
                "choices": [{"delta": {"content": "a"}, "finish_reason": None}],
            },
            {
                "id": "completion",
                "choices": [{"delta": {}, "finish_reason": "length"}],
            },
            {
                "id": "completion",
                "choices": [],
                "usage": {},
            },
        )
        response = io.BytesIO(
            b"".join(
                b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
            )
            + b"data: [DONE]\n\n"
        )

        with patch.object(decode.time, "perf_counter", side_effect=(10.0, 20.0)):
            parsed = decode.parse_stream(response)

        self.assertEqual(parsed["first_content_at"], 10.0)
        self.assertEqual(parsed["last_token_at"], 20.0)

    @patch("tools.benchmark.launch_modal.subprocess.run")
    def test_cleanup_targets_the_selected_modal_environment(self, run: Mock) -> None:
        run.return_value.returncode = 0
        result = {}

        launch_modal.stop_app("ap-123", result, "test-environment")

        self.assertEqual(
            run.call_args.args[0],
            [
                launch_modal.sys.executable,
                "-m",
                "modal",
                "app",
                "stop",
                "-y",
                "-e",
                "test-environment",
                "ap-123",
            ],
        )
        self.assertTrue(result["app_stopped"])


if __name__ == "__main__":
    unittest.main()
