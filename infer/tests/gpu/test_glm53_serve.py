import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

pytestmark = pytest.mark.gpu

import torch

from infer.models.glm53_flash import serve
from infer.models.glm53_flash import model as glm53_flash
from infer.models.glm53_flash.codec import GLM53Codec


class GLM53ServeTest(unittest.TestCase):
    def test_rank_zero_warms_dep4_buckets_and_builds_four_lane_scheduler(
        self,
    ) -> None:
        events = []
        world = object()
        dist = SimpleNamespace(group=SimpleNamespace(WORLD=world), barrier=Mock())
        model = SimpleNamespace(weights=SimpleNamespace(endpoint="endpoint"))
        nextn_model = object()
        runtime = Mock()
        runtime.stage_decode.side_effect = lambda slots, _index: SimpleNamespace(
            token_ids=f"tokens-{len(slots)}",
            state_indices_int64=f"states-{len(slots)}",
        )
        runtime.decode_staging = (object(), object())
        nextn_runtime = Mock()
        staging = object()
        worker = Mock()
        worker.dispatch.side_effect = lambda plan: events.append(("dispatch", plan))
        stream = Mock()
        stream_context = MagicMock()
        scheduler = object()
        engine = object()
        all_reduce = SimpleNamespace(decode="decode_reduce", verify="verify_reduce")
        buckets = glm53_flash.GLM53_DEP4_DECODE_BATCH_SIZES
        width = glm53_flash.GLM53_TARGET_VERIFY_WIDTH

        with (
            patch.object(serve, "load_glm53_target_weights", return_value="weights"),
            patch.object(serve, "GLM53TargetModel", return_value=model) as make_target,
            patch.object(
                serve, "allocate_glm53_target_runtime", return_value=runtime
            ) as allocate_target,
            patch.object(
                serve, "load_glm53_nextn_weights", return_value="nextn_weights"
            ) as load_nextn,
            patch.object(
                serve, "GLM53NextNModel", return_value=nextn_model
            ) as make_nextn,
            patch.object(
                serve, "allocate_glm53_nextn_runtime", return_value=nextn_runtime
            ) as allocate_nextn,
            patch.object(
                serve, "allocate_glm53_worker_staging", return_value=staging
            ) as allocate_staging,
            patch.object(serve, "GLM53Worker", return_value=worker) as make_worker,
            patch.object(
                serve, "StateManager", return_value="state_manager"
            ) as make_state_manager,
            patch.object(serve, "Scheduler", return_value=scheduler) as make_scheduler,
            patch.object(serve, "DistributedEngine", return_value=engine),
            patch.object(
                serve, "require_memory_reserve"
            ) as require_memory_reserve,
            patch("torch.cuda.Stream", return_value=stream),
            patch("torch.cuda.current_stream", return_value="current_stream"),
            patch("torch.cuda.stream", return_value=stream_context),
            patch("torch.cuda.Event", side_effect=("event-0", "event-1")),
        ):
            returned = serve._build_engine(
                0,
                Path("/checkpoint"),
                "cuda:0",
                dist,
                all_reduce,
                ("sender-1", "sender-2", "sender-3"),
                "dep4",
            )

        self.assertIs(returned, engine)
        dispatches = [event[1] for event in events if event[0] == "dispatch"]
        self.assertEqual(len(dispatches), 2 * len(buckets) + 1)
        decode_plans = dispatches[:-1]
        expected_keys = []
        for plan in decode_plans:
            bucket = plan.graph_bucket
            self.assertEqual(len(plan.lanes), glm53_flash.TP_SIZE)
            self.assertEqual(len(plan.lanes[0]), bucket)
            self.assertTrue(
                all(row.max_accept_tokens == width for row in plan.lanes[0])
            )
            expected_keys.append((bucket, plan.staging_index))
        self.assertEqual(
            expected_keys,
            [(bucket, parity) for bucket in buckets for parity in range(2)],
        )
        self.assertIsNone(dispatches[-1].graph_bucket)
        self.assertEqual(
            len(dispatches[-1].lanes[0][0].token_ids), glm53_flash.KDA_CHUNK_SIZE
        )
        allocate_target.assert_called_once_with(
            0,
            "cuda:0",
            live_slot_count=serve._DEP4_NATIVE_LIVE_SLOTS,
            snapshot_slot_count=2 * serve._DEP4_NATIVE_LIVE_SLOTS,
            attention_tp_size=1,
            speculative=True,
        )
        ops = make_target.call_args.args[1]
        self.assertEqual(make_target.call_args.args[0], "weights")
        load_nextn.assert_called_once_with(Path("/checkpoint"), 0, "cuda:0", 1)
        make_nextn.assert_called_once_with("nextn_weights", ops)
        allocate_nextn.assert_called_once_with(runtime)
        allocate_staging.assert_called_once_with(
            "cuda:0",
            serve._DEP4_NATIVE_LIVE_SLOTS,
            glm53_flash.TP_SIZE,
            speculative=True,
        )
        make_worker.assert_called_once_with(
            model,
            runtime,
            nextn_model,
            nextn_runtime,
            world,
            all_reduce,
            staging,
            ("event-0", "event-1"),
            rank=0,
            collectives=dist,
            parallelism="dep4",
        )
        self.assertEqual(
            nextn_runtime.seed_candidates.call_count, 2 * len(buckets)
        )
        for index, (bucket, parity) in enumerate(expected_keys):
            self.assertEqual(
                nextn_runtime.seed_candidates.call_args_list[index].args,
                (
                    nextn_model,
                    f"tokens-{bucket}",
                    f"states-{bucket}",
                    bucket,
                    parity,
                    "endpoint",
                    world,
                    "decode_reduce",
                ),
            )
        worker.capture_decode_graphs.assert_called_once()
        captured_plans, captured_stream, captured_torch = (
            worker.capture_decode_graphs.call_args.args
        )
        self.assertEqual(captured_plans, decode_plans)
        self.assertIs(captured_stream, stream)
        self.assertIs(captured_torch, torch)
        stream.wait_stream.assert_called_once_with("current_stream")
        self.assertEqual(stream.synchronize.call_count, 2)
        self.assertEqual(dist.barrier.call_count, 2)
        require_memory_reserve.assert_called_once_with(
            unittest.mock.ANY, "cuda:0", 0
        )
        self.assertEqual(runtime.reset_state.call_count, max(buckets))
        self.assertEqual(nextn_runtime.reset_slot.call_count, max(buckets))
        self.assertEqual(make_state_manager.call_args.args, ())
        self.assertEqual(
            make_state_manager.call_args.kwargs,
            {
                "history_block_count": serve.GLM53_TARGET_HISTORY_BLOCKS,
                "live_slot_count": serve._DEP4_NATIVE_LIVE_SLOTS,
                "snapshot_slot_count": 2 * serve._DEP4_NATIVE_LIVE_SLOTS,
                "lane_count": glm53_flash.TP_SIZE,
                "history_block_tokens": (
                    glm53_flash.SPARSE_MLA_HISTORY_BLOCK_TOKENS
                ),
            },
        )
        self.assertEqual(make_scheduler.call_args.args, ("state_manager",))
        self.assertEqual(
            make_scheduler.call_args.kwargs,
            {
                "token_budget": (
                    glm53_flash.KDA_CHUNK_SIZE + max(buckets) * width
                ),
                "prefill_chunk_size": glm53_flash.KDA_CHUNK_SIZE,
                "decode_width": width,
                "max_batch_size": max(buckets),
                "max_prefill_rows": max(buckets),
                "max_decode_ticks_between_prefills": 8,
                "graph_buckets": buckets,
                "max_queued_requests": (
                    glm53_flash.TP_SIZE * serve._DEP4_NATIVE_LIVE_SLOTS
                ),
                "single_resident_prefill_chunk_size": (
                    serve._DEP4_SINGLE_RESIDENT_PREFILL_CHUNK_SIZE
                ),
            },
        )

    def test_target_only_build_skips_nextn_and_uses_single_token_decode(
        self,
    ) -> None:
        events = []
        world = object()
        dist = SimpleNamespace(group=SimpleNamespace(WORLD=world), barrier=Mock())
        all_reduce = SimpleNamespace(decode="decode_reduce", verify="verify_reduce")
        model = SimpleNamespace(weights=SimpleNamespace(endpoint="endpoint"))
        runtime = Mock()
        runtime.stage_decode.side_effect = lambda slots, _index: SimpleNamespace(
            token_ids=f"tokens-{len(slots)}",
            state_indices_int64=f"states-{len(slots)}",
        )
        runtime.decode_staging = (object(), object())
        worker = Mock()
        worker.dispatch.side_effect = lambda plan: events.append(("dispatch", plan))
        stream = Mock()
        stream_context = MagicMock()
        buckets = glm53_flash.GLM53_DEP4_TARGET_ONLY_DECODE_BATCH_SIZES

        with (
            patch.object(serve, "load_glm53_target_weights", return_value="weights"),
            patch.object(serve, "GLM53TargetModel", return_value=model),
            patch.object(
                serve, "allocate_glm53_target_runtime", return_value=runtime
            ) as allocate_target,
            patch.object(serve, "load_glm53_nextn_weights") as load_nextn,
            patch.object(serve, "GLM53NextNModel") as make_nextn,
            patch.object(serve, "allocate_glm53_nextn_runtime") as allocate_nextn,
            patch.object(
                serve, "allocate_glm53_worker_staging", return_value="staging"
            ) as allocate_staging,
            patch.object(serve, "GLM53Worker", return_value=worker) as make_worker,
            patch.object(serve, "StateManager", return_value="state-manager"),
            patch.object(serve, "Scheduler", return_value="scheduler") as make_scheduler,
            patch.object(serve, "DistributedEngine", return_value="engine"),
            patch.object(serve, "require_memory_reserve"),
            patch("torch.cuda.Stream", return_value=stream),
            patch("torch.cuda.current_stream", return_value="current-stream"),
            patch("torch.cuda.stream", return_value=stream_context),
            patch("torch.cuda.Event", side_effect=("event-0", "event-1")),
        ):
            engine = serve._build_engine(
                0,
                Path("/checkpoint"),
                "cuda:0",
                dist,
                all_reduce,
                ("sender-1", "sender-2", "sender-3"),
                "dep4",
                "none",
            )

        self.assertEqual(engine, "engine")
        load_nextn.assert_not_called()
        make_nextn.assert_not_called()
        allocate_nextn.assert_not_called()
        allocate_target.assert_called_once_with(
            0,
            "cuda:0",
            live_slot_count=serve._DEP4_TARGET_ONLY_LIVE_SLOTS,
            snapshot_slot_count=2 * serve._DEP4_TARGET_ONLY_LIVE_SLOTS,
            attention_tp_size=1,
            speculative=False,
        )
        allocate_staging.assert_called_once_with(
            "cuda:0",
            serve._DEP4_TARGET_ONLY_LIVE_SLOTS,
            glm53_flash.TP_SIZE,
            speculative=False,
        )
        make_worker.assert_called_once_with(
            model,
            runtime,
            None,
            None,
            world,
            all_reduce,
            "staging",
            ("event-0", "event-1"),
            rank=0,
            collectives=dist,
            parallelism="dep4",
        )
        dispatches = [event[1] for event in events if event[0] == "dispatch"]
        decode_plans = dispatches[:-1]
        self.assertEqual(len(decode_plans), 2 * len(buckets))
        self.assertTrue(
            all(
                row.max_accept_tokens == 1
                for plan in decode_plans
                for row in plan.lanes[0]
            )
        )
        worker.capture_decode_graphs.assert_called_once()
        self.assertEqual(make_scheduler.call_args.kwargs["decode_width"], 1)
        self.assertEqual(
            make_scheduler.call_args.kwargs["token_budget"],
            glm53_flash.KDA_CHUNK_SIZE + max(buckets),
        )
        self.assertEqual(
            make_scheduler.call_args.kwargs["graph_buckets"],
            buckets,
        )
        self.assertEqual(
            make_scheduler.call_args.kwargs["max_batch_size"], max(buckets)
        )
        self.assertEqual(
            make_scheduler.call_args.kwargs["max_prefill_rows"], max(buckets)
        )
        self.assertEqual(
            make_scheduler.call_args.kwargs["max_queued_requests"],
            glm53_flash.TP_SIZE * serve._DEP4_TARGET_ONLY_LIVE_SLOTS,
        )
        self.assertEqual(runtime.reset_state.call_count, max(buckets))

    def test_rank_startup_gates_serving_on_all_rank_readiness(self) -> None:
        engine = object()
        codec = object()
        events = []
        pipes = tuple((Mock(), Mock()) for _ in range(glm53_flash.TP_SIZE - 1))
        all_reduce = Mock()

        with (
            patch.object(
                serve, "set_rank_env", side_effect=lambda *_: events.append("env")
            ),
            patch.object(
                serve,
                "_bind_b200",
                side_effect=lambda *_: events.append("bind") or "cuda:0",
            ),
            patch.object(
                serve,
                "_create_all_reduce",
                side_effect=lambda *_: events.append("all_reduce_create")
                or all_reduce,
            ),
            patch.object(
                serve,
                "_build_engine",
                side_effect=lambda *_args: events.append("build") or engine,
            ),
            patch.object(
                serve,
                "GLM53Codec",
                side_effect=lambda *_: events.append("codec") or codec,
            ),
            patch.object(
                serve,
                "wait_until_ready",
                side_effect=lambda *_: events.append("wait_ready"),
            ) as wait_ready,
            patch.object(
                serve,
                "_drive_rank_zero",
                side_effect=lambda *args: events.append(("serve", args)),
            ),
            patch(
                "torch.distributed.init_process_group",
                side_effect=lambda *_args, **_kwargs: events.append("init"),
            ),
            patch(
                "torch.distributed.barrier",
                side_effect=lambda *_args, **_kwargs: events.append("barrier"),
            ),
            patch(
                "torch.distributed.destroy_process_group",
                side_effect=lambda *_args, **_kwargs: events.append("destroy"),
            ),
        ):
            serve.serve_rank(
                0,
                "/checkpoint",
                "127.0.0.1",
                8000,
                "file:///store",
                pipes,
                parallelism="dep4",
            )

        self.assertEqual(
            events[:7],
            [
                "env",
                "bind",
                "init",
                "all_reduce_create",
                "build",
                "codec",
                "wait_ready",
            ],
        )
        self.assertEqual(events[7][0], "serve")
        self.assertEqual(
            events[7][1], (engine, codec, "127.0.0.1", 8000, "dep4", "native")
        )
        self.assertEqual(events[8:], ["barrier", "destroy"])
        self.assertEqual(wait_ready.call_args.args[1:], (0, "GLM"))
        all_reduce.destroy.assert_called_once_with()
        self.assertTrue(all(receiver.close.called for receiver, _ in pipes))
        self.assertTrue(all(sender.close.called for _, sender in pipes))

    def test_dep4_service_capacity_follows_speculation(self) -> None:
        for speculation, local_slots in (
            ("native", serve._DEP4_NATIVE_LIVE_SLOTS),
            ("none", serve._DEP4_TARGET_ONLY_LIVE_SLOTS),
        ):
            with (
                self.subTest(speculation=speculation),
                patch.object(serve, "Service", return_value=Mock()) as make_service,
                patch.object(serve, "create_app"),
                patch.object(serve, "drive_service"),
            ):
                engine = Mock()
                engine.tick.return_value = None

                serve._drive_rank_zero(
                    engine,
                    Mock(spec=GLM53Codec),
                    "127.0.0.1",
                    8000,
                    "dep4",
                    speculation,
                )

            self.assertEqual(
                make_service.call_args.kwargs["capacity"],
                glm53_flash.TP_SIZE * local_slots,
            )


if __name__ == "__main__":
    unittest.main()
