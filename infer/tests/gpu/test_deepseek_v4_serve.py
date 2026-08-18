import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import pytest

pytestmark = pytest.mark.gpu

import torch

from infer import runtime as shared_serve
from infer.models.deepseek_v4_flash import MODEL_ID, serve
from infer.models.deepseek_v4_flash import model as deepseek_v4
from infer.models.deepseek_v4_flash.codec import DeepSeekV4Codec


class FakeVerify:
    def __init__(self, key, events) -> None:
        self.key = key
        self.events = events

    def stage(self, controls) -> None:
        self.events.append(("stage", self.key, controls))

    def reset_attention_metadata(self) -> None:
        self.events.append(("reset_metadata", self.key))


class FakeRuntime:
    def __init__(self, events, *, target_only=False) -> None:
        self.events = events
        decode = {
            (bucket, parity): FakeVerify((bucket, parity), events)
            for bucket in (0, *deepseek_v4.DECODE_BATCH_SIZES)
            for parity in range(deepseek_v4.DECODE_DESCRIPTOR_PARITIES)
        }
        self.verify = {} if target_only else decode
        self.decode = decode if target_only else {}

    def reset_state(self, slot) -> None:
        self.events.append(("reset", slot))

    def stage_prefill(self, *descriptors):
        self.events.append(("stage_prefill", descriptors))
        return SimpleNamespace(
            token_count=sum(map(len, descriptors[1])),
            dspark_seed_count=sum(map(len, descriptors[1])),
        )

    def publish_prefill(self, batch) -> None:
        self.events.append(("publish_prefill", batch.token_count))


class FakeDSparkRuntime:
    def __init__(self, events) -> None:
        self.events = events
        self.batches = {
            (bucket, parity): SimpleNamespace(
                reset_attention_metadata=lambda key=(bucket, parity): events.append(
                    ("reset_dspark_metadata", key)
                )
            )
            for bucket in (0, *deepseek_v4.DECODE_BATCH_SIZES)
            for parity in range(deepseek_v4.DECODE_DESCRIPTOR_PARITIES)
        }

    def batch(self, *key):
        return self.batches[key]

    def reset_state(self, slot) -> None:
        self.events.append(("dspark_reset", slot))


class FakeDSparkModel:
    def __init__(self, events) -> None:
        self.events = events

    def seed_prefill(self, _runtime, _target, batch) -> None:
        self.events.append(("dspark_seed", batch.dspark_seed_count))


def _fake_ops_modules():
    """Fake the lazily-imported ops bundles (vendor stack unavailable).

    Mirrors test_tep4_build_shards_target_but_replicates_dspark_endpoints:
    serve wiring is pinned here; ops construction belongs to the B200
    acceptance env, not unit tests.
    """
    return patch.dict(
        sys.modules,
        {
            "infer.models.deepseek_v4_flash.ops.target": SimpleNamespace(
                DeepSeekV4TargetOps=Mock(return_value="target-ops"),
                DeepSeekV4TEP4TargetOps=Mock(return_value="tep-ops"),
            ),
            "infer.models.deepseek_v4_flash.ops.dspark": SimpleNamespace(
                DeepSeekV4DSparkOps=Mock(return_value="dspark-ops"),
            ),
        },
    )


class DeepSeekV4ServeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._ops_patcher = _fake_ops_modules()
        cls._ops_patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._ops_patcher.stop()

    def test_rank_zero_runs_the_shared_frontend_then_broadcasts_idle(self) -> None:
        engine = Mock()
        engine.tick.return_value = None
        codec = Mock(spec=DeepSeekV4Codec)
        service = object()
        app = object()

        with (
            patch.object(serve, "Service", return_value=service) as make_service,
            patch.object(serve, "create_app", return_value=app) as make_app,
            patch.object(serve, "drive_service") as drive,
        ):
            serve._drive_rank_zero(engine, codec, "0.0.0.0", 9000, "dep4")

        make_service.assert_called_once_with(
            engine,
            capacity=(deepseek_v4.DEP_SIZE * serve._PROVISIONAL_LIVE_SLOTS_PER_RANK),
            vocab_size=deepseek_v4.VOCAB_SIZE,
            max_context_tokens=deepseek_v4.MAX_CONTEXT_TOKENS,
            eos_token_ids=serve.EOS_TOKEN_IDS,
        )
        make_app.assert_called_once_with(
            codec=codec,
            service=service,
            model_name=MODEL_ID,
        )
        drive.assert_called_once_with(
            service,
            app,
            "0.0.0.0",
            9000,
            "deepseek-v4-http",
        )
        engine.tick.assert_called_once_with()

    @patch.object(serve, "require_memory_reserve")
    def test_builds_the_provisional_dep4_recipe_and_every_graph(
        self, require_memory_reserve: Mock
    ) -> None:
        events = []
        runtime = FakeRuntime(events)
        dspark_runtime = FakeDSparkRuntime(events)
        model = Mock()
        model.prefill.side_effect = lambda _, batch: events.append(
            ("prefill", batch.token_count)
        )
        dspark_model = FakeDSparkModel(events)
        world = object()
        dist = SimpleNamespace(
            group=SimpleNamespace(WORLD=world),
            barrier=lambda **_kwargs: events.append(("barrier",)),
        )
        stream = Mock()
        stream.wait_stream.side_effect = lambda _: events.append(("wait",))
        stream.synchronize.side_effect = lambda: events.append(("stream_sync",))
        stream_context = MagicMock()
        graph_context = MagicMock()
        graphs = tuple(Mock() for _ in runtime.verify)
        state_manager = object()
        scheduler = object()
        engine = object()
        target_weights = SimpleNamespace(embedding="embedding", head="head")

        with (
            patch.object(
                serve, "open_deepseek_v4_checkpoint", return_value="view"
            ) as open_checkpoint,
            patch.object(
                serve, "load_dep4_target_weights", return_value=target_weights
            ) as load_weights,
            patch.object(serve, "DeepSeekV4TargetModel", return_value=model),
            patch.object(
                serve, "load_dep4_dspark_weights", return_value="dspark-weights"
            ) as load_dspark_weights,
            patch.object(
                serve, "DeepSeekV4DSparkModel", return_value=dspark_model
            ) as make_dspark_model,
            patch.object(
                serve, "allocate_target_runtime", return_value=runtime
            ) as allocate_runtime,
            patch.object(
                serve, "allocate_dep4_dspark_runtime", return_value=dspark_runtime
            ) as allocate_dspark_runtime,
            patch.object(
                serve,
                "_run_decode",
                side_effect=lambda *_args: events.append(("run_decode", _args[-1])),
            ),
            patch.object(serve, "_record_memory") as record_memory,
            patch.object(
                serve, "DeepSeekV4TargetWorker", return_value="worker"
            ) as make_worker,
            patch.object(
                serve, "StateManager", return_value=state_manager
            ) as make_state_manager,
            patch.object(serve, "Scheduler", return_value=scheduler) as make_scheduler,
            patch.object(serve, "DistributedEngine", return_value=engine),
            patch("torch.cuda.Stream", return_value=stream),
            patch("torch.cuda.current_stream", return_value="current"),
            patch("torch.cuda.stream", return_value=stream_context),
            patch("torch.cuda.graph_pool_handle", return_value="pool"),
            patch("torch.cuda.CUDAGraph", side_effect=graphs),
            patch("torch.cuda.graph", return_value=graph_context) as capture,
            patch("torch.cuda.Event", side_effect=("event-0", "event-1")),
            patch(
                "torch.cuda.synchronize", side_effect=lambda: events.append(("sync",))
            ),
        ):
            returned = serve._build_engine(
                0,
                Path("/checkpoint"),
                "cuda:0",
                dist,
                ("sender-1", "sender-2", "sender-3"),
                "dep4",
            )

        self.assertIs(returned, engine)
        open_checkpoint.assert_called_once_with(Path("/checkpoint"))
        load_weights.assert_called_once_with("view", 0, "cuda:0")
        load_dspark_weights.assert_called_once_with("view", 0, "cuda:0")
        make_dspark_model.assert_called_once_with(
            "dspark-weights", "embedding", "head", unittest.mock.ANY
        )
        allocate_runtime.assert_called_once_with(
            target_weights,
            serve._PROVISIONAL_LIVE_SLOTS_PER_RANK,
            serve._PROVISIONAL_HISTORY_BLOCKS_PER_RANK,
            "cuda:0",
            speculative=True,
            tensor_parallel=False,
            snapshot_slot_count=serve._PREFIX_SNAPSHOT_SLOTS_PER_RANK,
        )
        allocate_dspark_runtime.assert_called_once_with(runtime, "cuda:0")
        self.assertEqual(
            record_memory.call_args_list,
            [
                call(unittest.mock.ANY, "cuda:0", 0, "start"),
                call(unittest.mock.ANY, "cuda:0", 0, "weights"),
                call(unittest.mock.ANY, "cuda:0", 0, "runtime"),
                call(unittest.mock.ANY, "cuda:0", 0, "graphs"),
            ],
        )
        require_memory_reserve.assert_called_once_with(unittest.mock.ANY, "cuda:0", 0)
        keys = tuple(sorted(runtime.verify))
        for key in keys:
            self.assertEqual(
                [event for event in events if event == ("stage", key, ())],
                [("stage", key, ())] * 4,
            )
            self.assertEqual(
                [event for event in events if event == ("run_decode", key)],
                [("run_decode", key)] * 5,
            )
            self.assertEqual(
                [event for event in events if event == ("reset_metadata", key)],
                [("reset_metadata", key)] * 5,
            )
            self.assertEqual(
                [event for event in events if event == ("reset_dspark_metadata", key)],
                [("reset_dspark_metadata", key)] * 5,
            )
        self.assertEqual(capture.call_count, len(keys))
        self.assertTrue(
            all(
                entry.kwargs == {"pool": "pool", "stream": stream}
                for entry in capture.call_args_list
            )
        )
        prefill = [event for event in events if event[0] == "stage_prefill"]
        self.assertEqual(len(prefill), deepseek_v4.DEP_SIZE)
        self.assertEqual(
            tuple(map(len, prefill[0][1][1])),
            (deepseek_v4.PREFILL_CHUNK_TOKENS // deepseek_v4.MAX_PREFILL_REQUESTS,)
            * deepseek_v4.MAX_PREFILL_REQUESTS,
        )
        self.assertTrue(all(not event[1][1] for event in prefill[1:]))
        self.assertEqual(
            [event for event in events if event[0] == "reset"],
            [("reset", slot) for slot in range(deepseek_v4.MAX_PREFILL_REQUESTS)],
        )
        self.assertEqual(
            [event for event in events if event[0] == "dspark_seed"],
            [
                (
                    "dspark_seed",
                    deepseek_v4.PREFILL_CHUNK_TOKENS if rank == 0 else 0,
                )
                for rank in range(deepseek_v4.DEP_SIZE)
            ],
        )
        self.assertEqual(
            [event for event in events if event[0] == "barrier"],
            [("barrier",), ("barrier",)],
        )
        graph_map = {key: graph for key, graph in zip(keys, graphs, strict=True)}
        make_worker.assert_called_once_with(
            model,
            runtime,
            dspark_model,
            dspark_runtime,
            graph_map,
            0,
            world,
            dist,
            ("event-0", "event-1"),
            "dep4",
        )

        make_state_manager.assert_called_once()
        self.assertEqual(make_state_manager.call_args.args, ())
        self.assertEqual(
            make_state_manager.call_args.kwargs,
            {
                "history_block_count": serve._PROVISIONAL_HISTORY_BLOCKS_PER_RANK,
                "live_slot_count": serve._PROVISIONAL_LIVE_SLOTS_PER_RANK,
                "snapshot_slot_count": serve._PREFIX_SNAPSHOT_SLOTS_PER_RANK,
                "lane_count": deepseek_v4.DEP_SIZE,
                "history_block_tokens": (
                    deepseek_v4.COMPRESSED_HISTORY_CANDIDATE_BLOCK_TOKENS
                ),
            },
        )
        make_scheduler.assert_called_once_with(
            state_manager,
            token_budget=(
                deepseek_v4.PREFILL_CHUNK_TOKENS
                + max(deepseek_v4.DECODE_BATCH_SIZES) * deepseek_v4.DSPARK_VERIFY_WIDTH
            ),
            prefill_chunk_size=deepseek_v4.PREFILL_CHUNK_TOKENS,
            decode_width=deepseek_v4.DSPARK_VERIFY_WIDTH,
            max_batch_size=max(deepseek_v4.DECODE_BATCH_SIZES),
            max_prefill_rows=deepseek_v4.MAX_PREFILL_REQUESTS,
            max_decode_ticks_between_prefills=16,
            graph_buckets=deepseek_v4.DECODE_BATCH_SIZES,
            max_queued_requests=(
                deepseek_v4.DEP_SIZE * serve._PROVISIONAL_LIVE_SLOTS_PER_RANK
            ),
        )

    @patch.object(serve, "require_memory_reserve")
    def test_target_only_build_omits_dspark_and_uses_one_token_scheduling(
        self, _require_memory_reserve: Mock
    ) -> None:
        runtime = FakeRuntime([], target_only=True)
        graphs = {key: object() for key in runtime.decode}
        weights = SimpleNamespace(embedding="embedding", head="head")
        dist = SimpleNamespace(group=SimpleNamespace(WORLD="world"))
        state_manager = object()

        with (
            patch.object(serve, "open_deepseek_v4_checkpoint", return_value="view"),
            patch.object(serve, "load_dep4_target_weights", return_value=weights),
            patch.object(serve, "DeepSeekV4TargetModel", return_value="model"),
            patch.object(serve, "load_dep4_dspark_weights") as load_dspark,
            patch.object(serve, "DeepSeekV4DSparkModel") as make_dspark,
            patch.object(
                serve, "allocate_target_runtime", return_value=runtime
            ) as allocate_runtime,
            patch.object(serve, "allocate_dep4_dspark_runtime") as allocate_dspark,
            patch.object(
                serve, "_warm_and_capture", return_value=graphs
            ) as warm_and_capture,
            patch.object(serve, "_record_memory"),
            patch.object(
                serve, "DeepSeekV4TargetWorker", return_value="worker"
            ) as make_worker,
            patch.object(serve, "StateManager", return_value=state_manager),
            patch.object(serve, "Scheduler", return_value="scheduler") as scheduler,
            patch.object(serve, "DistributedEngine", return_value="engine"),
            patch("torch.cuda.Event", side_effect=("event-0", "event-1")),
        ):
            result = serve._build_engine(
                0,
                Path("/checkpoint"),
                "cuda:0",
                dist,
                ("sender-1", "sender-2", "sender-3"),
                "dep4",
                "none",
            )

        self.assertEqual(result, "engine")
        load_dspark.assert_not_called()
        make_dspark.assert_not_called()
        allocate_dspark.assert_not_called()
        allocate_runtime.assert_called_once_with(
            weights,
            serve._PROVISIONAL_LIVE_SLOTS_PER_RANK,
            serve._PROVISIONAL_HISTORY_BLOCKS_PER_RANK,
            "cuda:0",
            speculative=False,
            tensor_parallel=False,
            snapshot_slot_count=serve._PREFIX_SNAPSHOT_SLOTS_PER_RANK,
        )
        warm_and_capture.assert_called_once_with(
            0, "model", runtime, None, None, dist, unittest.mock.ANY, "dep4"
        )
        self.assertIsNone(make_worker.call_args.args[2])
        self.assertIsNone(make_worker.call_args.args[3])
        scheduler.assert_called_once_with(
            state_manager,
            token_budget=(
                deepseek_v4.PREFILL_CHUNK_TOKENS + max(deepseek_v4.DECODE_BATCH_SIZES)
            ),
            prefill_chunk_size=deepseek_v4.PREFILL_CHUNK_TOKENS,
            decode_width=1,
            max_batch_size=max(deepseek_v4.DECODE_BATCH_SIZES),
            max_prefill_rows=deepseek_v4.MAX_PREFILL_REQUESTS,
            max_decode_ticks_between_prefills=16,
            graph_buckets=deepseek_v4.DECODE_BATCH_SIZES,
            max_queued_requests=(
                deepseek_v4.DEP_SIZE * serve._PROVISIONAL_LIVE_SLOTS_PER_RANK
            ),
        )

    def test_target_only_warmup_and_capture_use_target_decode_runtimes(self) -> None:
        events = []
        runtime = FakeRuntime(events, target_only=True)
        model = Mock()
        dist = SimpleNamespace(
            group=SimpleNamespace(WORLD="world"),
            barrier=lambda **_kwargs: events.append(("barrier",)),
        )
        stream = Mock()
        graphs = tuple(Mock() for _ in runtime.decode)

        with (
            patch("torch.cuda.Stream", return_value=stream),
            patch("torch.cuda.current_stream", return_value="current"),
            patch("torch.cuda.stream", return_value=MagicMock()),
            patch("torch.cuda.graph_pool_handle", return_value="pool"),
            patch("torch.cuda.CUDAGraph", side_effect=graphs),
            patch("torch.cuda.graph", return_value=MagicMock()),
            patch("torch.cuda.synchronize"),
            patch.object(serve, "_run_target_decode") as run_target,
            patch.object(serve, "_run_decode") as run_native,
        ):
            captured = serve._warm_and_capture(
                0, model, runtime, None, None, dist, torch, "dep4"
            )

        keys = tuple(sorted(runtime.decode))
        self.assertEqual(captured, dict(zip(keys, graphs, strict=True)))
        self.assertEqual(run_target.call_count, 5 * len(keys))
        run_native.assert_not_called()
        for key in keys:
            self.assertEqual(
                [event for event in events if event == ("stage", key, ())],
                [("stage", key, ())] * 4,
            )
            self.assertEqual(
                [event for event in events if event == ("reset_metadata", key)],
                [("reset_metadata", key)] * 5,
            )

    @patch.object(serve, "require_memory_reserve")
    def test_nonzero_rank_has_no_scheduler(self, _require_memory_reserve: Mock) -> None:
        runtime = FakeRuntime([])
        dspark_runtime = FakeDSparkRuntime([])
        dist = SimpleNamespace(group=SimpleNamespace(WORLD="world"))
        target_weights = SimpleNamespace(embedding="embedding", head="head")
        with (
            patch.object(serve, "open_deepseek_v4_checkpoint", return_value="view"),
            patch.object(
                serve, "load_dep4_target_weights", return_value=target_weights
            ),
            patch.object(serve, "DeepSeekV4TargetModel", return_value="model"),
            patch.object(
                serve, "load_dep4_dspark_weights", return_value="dspark-weights"
            ),
            patch.object(serve, "DeepSeekV4DSparkModel", return_value="dspark-model"),
            patch.object(serve, "allocate_target_runtime", return_value=runtime),
            patch.object(
                serve, "allocate_dep4_dspark_runtime", return_value=dspark_runtime
            ),
            patch.object(serve, "_warm_and_capture", return_value="graphs"),
            patch.object(serve, "_record_memory"),
            patch.object(serve, "DeepSeekV4TargetWorker", return_value="worker"),
            patch.object(serve, "StateManager") as make_state_manager,
            patch.object(serve, "DistributedEngine", return_value="engine") as engine,
            patch("torch.cuda.Event", side_effect=("event-0", "event-1")),
        ):
            self.assertEqual(
                serve._build_engine(
                    2, Path("/checkpoint"), "cuda:2", dist, ("in",), "dep4"
                ),
                "engine",
            )
        make_state_manager.assert_not_called()
        engine.assert_called_once_with(2, None, "worker", ("in",))

    @patch.object(serve, "require_memory_reserve")
    def test_tep4_build_shards_target_but_replicates_dspark_endpoints(
        self, _require_memory_reserve: Mock
    ) -> None:
        runtime = FakeRuntime([])
        dspark_runtime = FakeDSparkRuntime([])
        weights = SimpleNamespace(embedding="local-embedding", head="local-head")
        view = Mock()
        view.load_target_tensor.side_effect = ("full-embedding", "full-head")
        dist = SimpleNamespace(group=SimpleNamespace(WORLD="world"))
        tep_ops = SimpleNamespace(
            DeepSeekV4TargetOps=Mock(),
            DeepSeekV4TEP4TargetOps=Mock(return_value="tep-ops"),
        )
        dspark_ops = SimpleNamespace(
            DeepSeekV4DSparkOps=Mock(return_value="dspark-ops"),
        )

        with (
            patch.dict(
                sys.modules,
                {
                    "infer.models.deepseek_v4_flash.ops.target": tep_ops,
                    "infer.models.deepseek_v4_flash.ops.dspark": dspark_ops,
                },
            ),
            patch.object(serve, "open_deepseek_v4_checkpoint", return_value=view),
            patch.object(
                serve, "load_tep4_target_weights", return_value=weights
            ) as load_weights,
            patch.object(serve, "load_dep4_target_weights") as load_dep_weights,
            patch.object(serve, "DeepSeekV4TargetModel", return_value="model") as model,
            patch.object(
                serve, "load_dep4_dspark_weights", return_value="dspark-weights"
            ),
            patch.object(
                serve, "DeepSeekV4DSparkModel", return_value="dspark-model"
            ) as dspark_model,
            patch.object(
                serve, "allocate_target_runtime", return_value=runtime
            ) as allocate_runtime,
            patch.object(
                serve, "allocate_dep4_dspark_runtime", return_value=dspark_runtime
            ),
            patch.object(serve, "_warm_and_capture", return_value="graphs"),
            patch.object(serve, "_record_memory"),
            patch.object(
                serve, "DeepSeekV4TargetWorker", return_value="worker"
            ) as worker,
            patch.object(serve, "DistributedEngine", return_value="engine"),
            patch("torch.cuda.Event", side_effect=("event-0", "event-1")),
        ):
            result = serve._build_engine(
                2, Path("/checkpoint"), "cuda:2", dist, ("in",), "tep4"
            )

        self.assertEqual(result, "engine")
        load_weights.assert_called_once_with(view, 2, "cuda:2")
        load_dep_weights.assert_not_called()
        model.assert_called_once_with(weights, "tep-ops")
        self.assertEqual(
            view.load_target_tensor.call_args_list,
            [
                call("embed.weight", 2, "cuda:2", sharded=False),
                call("head.weight", 2, "cuda:2", sharded=False),
            ],
        )
        dspark_model.assert_called_once_with(
            "dspark-weights", "full-embedding", "full-head", "dspark-ops"
        )
        self.assertTrue(allocate_runtime.call_args.kwargs["tensor_parallel"])
        self.assertEqual(worker.call_args.args[-1], "tep4")

    def test_memory_ledger_logs_allocator_state(self) -> None:
        cuda = SimpleNamespace(
            mem_get_info=lambda _device: (30, 180),
            memory_allocated=lambda _device: 10,
            memory_reserved=lambda _device: 20,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            serve._record_memory(SimpleNamespace(cuda=cuda), "cuda:1", 1, "runtime")
        self.assertEqual(
            output.getvalue(),
            "rank=1 stage=runtime allocated=10 reserved=20 free=30 total=180\n",
        )

    def test_memory_reserve_is_ten_percent_of_total_hbm(self) -> None:
        for free_bytes, total_bytes, error in (
            (19, 181, None),
            (18, 181, "free=18 required=19 total=181"),
        ):
            with self.subTest(free_bytes=free_bytes, total_bytes=total_bytes):
                cuda = SimpleNamespace(
                    empty_cache=Mock(),
                    mem_get_info=Mock(return_value=(free_bytes, total_bytes)),
                )
                torch = SimpleNamespace(cuda=cuda)
                if error is None:
                    shared_serve.require_memory_reserve(torch, "cuda:1", 1)
                else:
                    with self.assertRaisesRegex(RuntimeError, error):
                        shared_serve.require_memory_reserve(torch, "cuda:1", 1)
                cuda.empty_cache.assert_called_once_with()
                cuda.mem_get_info.assert_called_once_with("cuda:1")


if __name__ == "__main__":
    unittest.main()
