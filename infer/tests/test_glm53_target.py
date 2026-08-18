import inspect
import sys
import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from infer.models.glm53_flash import checkpoint as glm53_checkpoint
from infer.models.glm53_flash import model as glm53_flash
from infer.models.glm53_flash import target as glm53_target


def torch_index_select_stub() -> ModuleType:
    """CPU stand-in for torch.index_select (torch is not installed here)."""
    stub = ModuleType("torch")
    stub.index_select = Mock(return_value=None)
    return stub


def make_weights(tp_rank: int = 0) -> glm53_target.GLM53TargetWeights[object]:
    return glm53_target.GLM53TargetWeights(
        tp_rank=tp_rank,
        endpoint="endpoint",
        dense_layers=tuple(
            ("dense", layer_id) for layer_id in glm53_flash.DENSE_LAYER_IDS
        ),
        sparse_kda_layers=tuple(
            ("sparse_kda", layer_id) for layer_id in glm53_flash.SPARSE_KDA_LAYER_IDS
        ),
        sparse_mla_layers=tuple(
            ("sparse_mla", layer_id)
            for layer_id in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS
        ),
    )


def make_state() -> glm53_target.GLM53TargetState[object]:
    return glm53_target.GLM53TargetState(
        kda_layers=tuple(
            ("kda_state", layer_id)
            for layer_id in glm53_target.GLM53_TARGET_KDA_LAYER_IDS
        ),
        sparse_mla_layers=tuple(
            ("mla_history", layer_id)
            for layer_id in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS
        ),
    )


def make_workspace() -> glm53_target.GLM53TargetDecodeWorkspace[object]:
    return glm53_target.GLM53TargetDecodeWorkspace(
        endpoint=SimpleNamespace(normalized=Streams("normalized")),
        layer="layer_workspace",
        kda="kda_workspace",
        sparse_ffn="sparse_ffn_workspace",
        sparse_mla_projection="mla_projection_workspace",
        sparse_mla_decode="mla_decode_workspace",
        sparse_mla_output="mla_output_workspace",
    )


def make_transaction() -> glm53_target.GLM53TargetTransaction[object]:
    return glm53_target.GLM53TargetTransaction(
        kda_layers=tuple(
            glm53_flash.KDAState(
                recurrent=("kda_transaction", layer_id),
                conv=("conv_transaction", layer_id),
            )
            for layer_id in glm53_target.GLM53_TARGET_KDA_LAYER_IDS
        ),
        sparse_mla_layers=tuple(
            glm53_target.GLM53TargetSparseMLATransaction(
                tail_key=("tail_key", layer_id),
                tail_gate=("tail_gate", layer_id),
            )
            for layer_id in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS
        ),
    )


def make_batch() -> glm53_target.GLM53TargetDecodeBatch[object]:
    return glm53_target.GLM53TargetDecodeBatch(
        token_ids="token_ids",
        state_indices="state_indices",
        state_indices_int64="state_indices_int64",
        cu_seqlens="cu_seqlens",
        sparse_mla="sparse_mla_batch",
    )


class Streams:
    def __init__(self, label: str) -> None:
        self.label = label

    def __getitem__(self, rows: slice):
        return self.label, rows.start, rows.stop


def make_prefill_workspace() -> glm53_target.GLM53TargetPrefillWorkspace[object]:
    return glm53_target.GLM53TargetPrefillWorkspace(
        endpoint="prefill_endpoint_workspace",
        layer=SimpleNamespace(streams_out=Streams("layer_streams")),
        kda="prefill_kda_workspace",
        sparse_ffn="prefill_sparse_ffn_workspace",
        sparse_mla_projection="prefill_mla_projection_workspace",
        sparse_mla_decode="prefill_mla_decode_workspace",
        sparse_mla_output="prefill_mla_output_workspace",
    )


def make_prefill_batch(
    total_tokens: int,
    start_token: int = 0,
    sample: bool = False,
) -> glm53_target.GLM53TargetPrefillBatch[object]:
    metadata = SimpleNamespace(
        total_tokens=total_tokens,
        batch_size=1,
        validate_unchanged=lambda: None,
    )
    staged_query = SimpleNamespace(
        total_tokens=total_tokens,
        batch_size=1,
        validate_unchanged=lambda: None,
    )
    return glm53_target.GLM53TargetPrefillBatch(
        token_ids=SimpleNamespace(shape=(total_tokens,)),
        kda=SimpleNamespace(metadata=metadata),
        sparse_mla=staged_query,
        sample_indices=SimpleNamespace(shape=(1,)) if sample else None,
        sample_state_indices=SimpleNamespace(shape=(1,)) if sample else None,
        sample_count=int(sample),
    )


class RecordingOps:
    def __init__(self) -> None:
        self.calls = []
        self.collectives = []
        self.all_reduces = []

    def decode_embedding(
        self, token_ids, weights, workspace, process_group, all_reduce
    ):
        self.calls.append(("embedding", token_ids, weights, workspace, process_group))
        self.collectives.append(("all_reduce", "embedding"))
        self.all_reduces.append(all_reduce)
        return ("streams", -1)

    def decode_dense_layer(
        self,
        streams,
        weights,
        state,
        state_indices,
        cu_seqlens,
        kda_workspace,
        layer_workspace,
        all_reduce,
    ):
        layer_id = weights[1]
        self._record_layer(
            "dense",
            layer_id,
            streams,
            state,
            state_indices,
            cu_seqlens,
            kda_workspace,
            None,
            layer_workspace,
            None,
            all_reduce,
        )
        return ("streams", layer_id)

    def decode_sparse_kda_layer(
        self,
        streams,
        weights,
        state,
        state_indices,
        cu_seqlens,
        kda_workspace,
        ffn_workspace,
        layer_workspace,
        process_group,
        all_reduce,
    ):
        layer_id = weights[1]
        self._record_layer(
            "sparse_kda",
            layer_id,
            streams,
            state,
            state_indices,
            cu_seqlens,
            kda_workspace,
            ffn_workspace,
            layer_workspace,
            process_group,
            all_reduce,
        )
        return ("streams", layer_id)

    def decode_sparse_mla_layer(
        self,
        streams,
        weights,
        batch,
        history,
        projection_workspace,
        decode_workspace,
        output_workspace,
        ffn_workspace,
        layer_workspace,
        process_group,
        all_reduce,
    ):
        layer_id = weights[1]
        self.calls.append(
            (
                "sparse_mla",
                layer_id,
                streams,
                batch,
                history,
                projection_workspace,
                decode_workspace,
                output_workspace,
                ffn_workspace,
                layer_workspace,
                process_group,
            )
        )
        self.collectives.extend(
            (("all_reduce", layer_id, "attention"), ("all_reduce", layer_id, "ffn"))
        )
        self.all_reduces.append(all_reduce)
        return ("streams", layer_id)

    def decode_head(self, streams, weights, workspace, process_group):
        self.calls.append(("head", streams, weights, workspace, process_group))
        self.collectives.append(("all_gather", "head"))
        return "token"

    def _record_layer(
        self,
        kind,
        layer_id,
        streams,
        state,
        state_indices,
        cu_seqlens,
        kda_workspace,
        ffn_workspace,
        layer_workspace,
        process_group,
        all_reduce,
    ) -> None:
        self.calls.append(
            (
                kind,
                layer_id,
                streams,
                state,
                state_indices,
                cu_seqlens,
                kda_workspace,
                ffn_workspace,
                layer_workspace,
                process_group,
            )
        )
        self.collectives.extend(
            (("all_reduce", layer_id, "attention"), ("all_reduce", layer_id, "ffn"))
        )
        self.all_reduces.append(all_reduce)


class PrefillRecordingOps:
    def __init__(self) -> None:
        self.calls = []
        self.collectives = []

    def prefill_embedding(self, token_ids, weights, workspace, process_group):
        self.calls.append(("embedding", token_ids, weights, workspace, process_group))
        self.collectives.append(("all_reduce", "embedding"))
        self.total_tokens = token_ids.shape[0]
        return "endpoint_streams", None, self.total_tokens

    def prefill_dense_layer(
        self, streams, weights, state, batch, kda_workspace, layer_workspace, all_reduce
    ):
        layer_id = weights[1]
        self.calls.append(
            (
                "dense",
                layer_id,
                streams,
                state,
                batch,
                kda_workspace,
                layer_workspace,
            )
        )
        self.collectives.extend(
            (("all_reduce", layer_id, "attention"), ("all_reduce", layer_id, "ffn"))
        )
        return layer_workspace.streams_out[: batch.metadata.total_tokens]

    def prefill_sparse_kda_layer(
        self,
        streams,
        weights,
        state,
        batch,
        kda_workspace,
        ffn_workspace,
        layer_workspace,
        process_group,
        all_reduce,
    ):
        layer_id = weights[1]
        self.calls.append(
            (
                "sparse_kda",
                layer_id,
                streams,
                state,
                batch,
                kda_workspace,
                ffn_workspace,
                layer_workspace,
                process_group,
            )
        )
        self.collectives.extend(
            (("all_reduce", layer_id, "attention"), ("all_reduce", layer_id, "ffn"))
        )
        return layer_workspace.streams_out

    def prefill_sparse_mla_layer(
        self,
        streams,
        weights,
        staged_query,
        history,
        projection_workspace,
        decode_workspace,
        output_workspace,
        ffn_workspace,
        layer_workspace,
        process_group,
        all_reduce,
    ):
        layer_id = weights[1]
        self.calls.append(
            (
                "sparse_mla",
                layer_id,
                streams,
                staged_query,
                history,
                projection_workspace,
                decode_workspace,
                output_workspace,
                ffn_workspace,
                layer_workspace,
                process_group,
            )
        )
        self.collectives.extend(
            (("all_reduce", layer_id, "attention"), ("all_reduce", layer_id, "ffn"))
        )
        return layer_workspace.streams_out

    def normalize_head(self, streams, weights, workspace, process_group):
        self.calls.append(("normalize", streams, weights, workspace, process_group))
        return "normalized", None, self.total_tokens

    def decode_head(self, streams, weights, workspace, process_group):
        self.calls.append(("head", streams, weights, workspace, process_group))
        self.collectives.append(("all_gather", "head"))
        return "token"


class VerifyRecordingOps:
    def __init__(self) -> None:
        self.calls = []
        self.all_reduces = []

    def decode_embedding(
        self, token_ids, weights, workspace, process_group, all_reduce
    ):
        self.calls.append(("embedding", token_ids, weights, workspace, process_group))
        self.all_reduces.append(all_reduce)
        return ("streams", -1)

    def verify_dense_layer(
        self,
        streams,
        weights,
        state,
        transaction,
        *_args,
    ):
        self.all_reduces.append(_args[-1])
        layer_id = weights[1]
        self.calls.append(("dense", layer_id, streams, state, transaction))
        return ("streams", layer_id)

    def verify_sparse_kda_layer(
        self,
        streams,
        weights,
        state,
        transaction,
        *_args,
    ):
        self.all_reduces.append(_args[-1])
        layer_id = weights[1]
        self.calls.append(("sparse_kda", layer_id, streams, state, transaction))
        return ("streams", layer_id)

    def verify_sparse_mla_layer(
        self,
        streams,
        weights,
        batch,
        history,
        transaction_tail_key,
        transaction_tail_gate,
        *_args,
    ):
        self.all_reduces.append(_args[-1])
        layer_id = weights[1]
        self.calls.append(
            (
                "sparse_mla",
                layer_id,
                streams,
                history,
                (transaction_tail_key, transaction_tail_gate),
                batch,
            )
        )
        return ("streams", layer_id)

    def decode_head(self, streams, weights, workspace, process_group):
        self.calls.append(("head", streams, weights, workspace, process_group))
        return "tokens"


class GLM53TargetRecordsTest(unittest.TestCase):
    def test_records_and_fixed_layer_sets_are_exact(self) -> None:
        self.assertEqual(glm53_target.GLM53_TARGET_LAYER_IDS, tuple(range(45)))
        self.assertEqual(glm53_target.GLM53_TARGET_VERIFY_WIDTH, 4)
        self.assertEqual(
            glm53_target.GLM53_TARGET_KDA_LAYER_IDS,
            tuple(
                layer_id
                for layer_id in glm53_target.GLM53_TARGET_LAYER_IDS
                if layer_id not in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS
            ),
        )
        self.assertEqual(len(glm53_target.GLM53_TARGET_KDA_LAYER_IDS), 34)
        self.assertEqual(
            tuple(field.name for field in fields(glm53_target.GLM53TargetOutput)),
            ("normalized", "token"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(glm53_target.GLM53TargetWeights)),
            (
                "tp_rank",
                "endpoint",
                "dense_layers",
                "sparse_kda_layers",
                "sparse_mla_layers",
                "attention_tp_size",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(glm53_target.GLM53TargetState)),
            ("kda_layers", "sparse_mla_layers"),
        )
        self.assertEqual(
            tuple(
                field.name for field in fields(glm53_target.GLM53TargetDecodeWorkspace)
            ),
            (
                "endpoint",
                "layer",
                "kda",
                "sparse_ffn",
                "sparse_mla_projection",
                "sparse_mla_decode",
                "sparse_mla_output",
            ),
        )
        self.assertEqual(
            tuple(
                field.name for field in fields(glm53_target.GLM53TargetPrefillWorkspace)
            ),
            (
                "endpoint",
                "layer",
                "kda",
                "sparse_ffn",
                "sparse_mla_projection",
                "sparse_mla_decode",
                "sparse_mla_output",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(glm53_target.GLM53TargetDecodeBatch)),
            (
                "token_ids",
                "state_indices",
                "state_indices_int64",
                "cu_seqlens",
                "sparse_mla",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(glm53_target.GLM53TargetPrefillBatch)),
            (
                "token_ids",
                "kda",
                "sparse_mla",
                "sample_indices",
                "sample_state_indices",
                "sample_count",
            ),
        )


class GLM53TargetLoaderTest(unittest.TestCase):
    def test_inventory_precedes_exact_layer_order(self) -> None:
        events = []
        weight_map = {"pinned": "map"}

        def read_index(root):
            events.append(("index", root))
            return weight_map

        def validate_inventory(actual):
            events.append(("inventory", actual))

        def load_endpoint(root, rank, device, _attention_tp_size):
            events.append(("endpoint", root, rank, device))
            return "endpoint"

        def load_layer(kind):
            def load(root, layer_id, rank, device, _attention_tp_size):
                events.append((kind, layer_id, root, rank, device))
                return (kind, layer_id)

            return load

        with (
            patch.object(
                glm53_checkpoint, "_read_checkpoint_weight_map", side_effect=read_index
            ),
            patch.object(
                glm53_checkpoint,
                "validate_glm53_checkpoint_inventory",
                side_effect=validate_inventory,
            ),
            patch.object(
                glm53_checkpoint,
                "load_glm53_endpoint_weights",
                side_effect=load_endpoint,
            ),
            patch.object(
                glm53_checkpoint,
                "load_glm53_dense_layer_weights",
                side_effect=load_layer("dense"),
            ),
            patch.object(
                glm53_checkpoint,
                "load_glm53_sparse_kda_layer_weights",
                side_effect=load_layer("sparse_kda"),
            ),
            patch.object(
                glm53_checkpoint,
                "load_glm53_sparse_mla_layer_weights",
                side_effect=load_layer("sparse_mla"),
            ),
        ):
            weights = glm53_target.load_glm53_target_weights("/checkpoint", 2, "cuda:2")

        root = Path("/checkpoint")
        self.assertEqual(
            events[:3],
            [
                ("index", root),
                ("inventory", weight_map),
                ("endpoint", root, 2, "cuda:2"),
            ],
        )
        layer_events = events[3:]
        self.assertEqual([event[1] for event in layer_events], list(range(45)))
        self.assertEqual(
            [event[0] for event in layer_events],
            [
                "dense"
                if layer_id in glm53_flash.DENSE_LAYER_IDS
                else "sparse_mla"
                if layer_id in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS
                else "sparse_kda"
                for layer_id in range(45)
            ],
        )
        self.assertEqual(weights.tp_rank, 2)
        self.assertEqual(len(weights.dense_layers), 3)
        self.assertEqual(len(weights.sparse_kda_layers), 31)
        self.assertEqual(len(weights.sparse_mla_layers), 11)

    def test_invalid_rank_fails_before_index_io(self) -> None:
        with (
            patch.object(glm53_checkpoint, "_read_checkpoint_weight_map") as read_index,
            self.assertRaisesRegex(ValueError, "tp_rank"),
        ):
            glm53_target.load_glm53_target_weights("/checkpoint", 4, "cuda:0")
        read_index.assert_not_called()


class GLM53TargetPrefillTest(unittest.TestCase):
    def run_prefill(self, total_tokens: int, sample_output: bool):
        ops = PrefillRecordingOps()
        weights = make_weights()
        state = make_state()
        workspace = make_prefill_workspace()
        batch = make_prefill_batch(total_tokens, sample=sample_output)
        group = object()
        model = glm53_target.GLM53TargetModel(weights, ops)
        head_workspace = (
            SimpleNamespace(streams="sampled_streams") if sample_output else None
        )

        with (
            patch.object(glm53_target, "_validate_tep4_world", return_value=0),
            patch.object(
                glm53_target, "_target_prefill_workspace_view", return_value=workspace
            ),
            patch.dict(sys.modules, {"torch": torch_index_select_stub()}),
        ):
            result = model.prefill_tokens(
                batch,
                state,
                workspace,
                head_workspace,
                group,
            )
        return result, ops, state, workspace, batch, group

    def test_prefill_owns_all_45_layers_and_skips_intermediate_head(self) -> None:
        result, ops, state, workspace, batch, group = self.run_prefill(63, False)

        self.assertEqual(
            result,
            glm53_target.GLM53TargetOutput(("normalized", None, 63), None),
        )
        self.assertEqual(
            [entry[0] for entry in ops.calls],
            [
                "embedding",
                *(
                    "dense"
                    if layer_id in glm53_flash.DENSE_LAYER_IDS
                    else "sparse_mla"
                    if layer_id in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS
                    else "sparse_kda"
                    for layer_id in range(45)
                ),
                "normalize",
            ],
        )
        layer_calls = ops.calls[1:-1]
        self.assertEqual([entry[1] for entry in layer_calls], list(range(45)))
        self.assertEqual(layer_calls[0][2], ("endpoint_streams", None, 63))
        self.assertEqual(layer_calls[1][2], ("layer_streams", None, 63))
        self.assertTrue(
            all(
                entry[2] is workspace.layer.streams_out
                for entry in layer_calls[len(glm53_flash.DENSE_LAYER_IDS) :]
            )
        )
        self.assertEqual(len(ops.collectives), 91)
        self.assertEqual(ops.collectives[0], ("all_reduce", "embedding"))
        self.assertEqual(
            ops.collectives[1:],
            [
                collective
                for layer_id in range(45)
                for collective in (
                    ("all_reduce", layer_id, "attention"),
                    ("all_reduce", layer_id, "ffn"),
                )
            ],
        )

        kda_calls = [entry for entry in layer_calls if entry[0] != "sparse_mla"]
        self.assertEqual([entry[3] for entry in kda_calls], list(state.kda_layers))
        self.assertTrue(all(entry[4] is batch.kda for entry in kda_calls))
        mla_calls = [entry for entry in layer_calls if entry[0] == "sparse_mla"]
        self.assertEqual(
            [entry[4] for entry in mla_calls], list(state.sparse_mla_layers)
        )
        self.assertTrue(all(entry[3] is batch.sparse_mla for entry in mla_calls))
        self.assertTrue(all(entry[-1] is group for entry in layer_calls[3:]))
        self.assertEqual(
            ops.calls[-1],
            (
                "normalize",
                workspace.layer.streams_out,
                "endpoint",
                workspace.endpoint,
                group,
            ),
        )

    def test_final_prefill_returns_all_hidden_rows_and_heads_only_the_last(
        self,
    ) -> None:
        result, ops, _, _, _, group = self.run_prefill(7, True)

        self.assertEqual(
            result,
            glm53_target.GLM53TargetOutput(("normalized", None, 7), "token"),
        )
        self.assertEqual(len(ops.collectives), 92)
        self.assertEqual(ops.collectives[-1], ("all_gather", "head"))
        self.assertEqual(
            ops.calls[-1],
            (
                "head",
                "sampled_streams",
                "endpoint",
                SimpleNamespace(streams="sampled_streams"),
                group,
            ),
        )

    def test_descriptor_disagreement_fails_before_embedding(self) -> None:
        ops = PrefillRecordingOps()
        model = glm53_target.GLM53TargetModel(make_weights(), ops)
        batch = make_prefill_batch(7)
        batch.token_ids.shape = (6,)

        with (
            patch.object(glm53_target, "_validate_tep4_world", return_value=0),
            self.assertRaisesRegex(ValueError, "descriptors disagree"),
        ):
            model.prefill_tokens(
                batch,
                make_state(),
                make_prefill_workspace(),
                None,
                object(),
            )
        self.assertEqual(ops.calls, [])

    def test_startup_owns_one_prefill_arena_and_hot_path_allocates_none(
        self,
    ) -> None:
        allocation = inspect.getsource(glm53_target._allocate_prefill_staging)
        self.assertIn("capacity=glm53_flash.KDA_CHUNK_SIZE", allocation)
        self.assertNotIn("range(1, glm53_flash.KDA_CHUNK_SIZE + 1)", allocation)
        runtime_allocation = inspect.getsource(
            glm53_target.allocate_glm53_target_runtime
        )
        self.assertNotIn("_allocate_kda_prefill_batches", runtime_allocation)
        self.assertIn("_allocate_prefill_staging", runtime_allocation)
        prefill = inspect.getsource(glm53_target.GLM53TargetRuntime.stage_prefill)
        self.assertNotIn("for offset, token_id in enumerate(sequence)", prefill)
        self.assertIn("staging.token_ids_cpu[rows].copy_", prefill)
        self.assertIn("staging.segments_cpu[: len(segments)].copy_", prefill)

        for method in (
            glm53_target.GLM53TargetRuntime.stage_prefill,
            glm53_target.GLM53TargetModel.prefill_tokens,
        ):
            source = inspect.getsource(method)
            for forbidden in (
                "torch.empty",
                "torch.zeros",
                "torch.tensor",
                ".new_empty",
                ".new_zeros",
                ".clone(",
            ):
                self.assertNotIn(forbidden, source)


class GLM53TargetDecodeTest(unittest.TestCase):
    def test_decode_mutates_target_state_without_a_verify_transaction(self) -> None:
        ops = RecordingOps()
        weights = make_weights()
        state = make_state()
        workspace = make_workspace()
        batch = make_batch()
        group = object()
        all_reduce = object()
        model = glm53_target.GLM53TargetModel(weights, ops)

        with patch.object(glm53_target, "_validate_tep4_world", return_value=0):
            result = model.decode_tokens(
                batch,
                state,
                workspace,
                group,
                all_reduce,
            )

        self.assertIs(result.normalized, workspace.endpoint.normalized)
        self.assertEqual(result.token, "token")
        self.assertEqual(
            [entry[0] for entry in ops.calls],
            [
                "embedding",
                *(
                    "dense"
                    if layer_id in glm53_flash.DENSE_LAYER_IDS
                    else "sparse_mla"
                    if layer_id in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS
                    else "sparse_kda"
                    for layer_id in range(45)
                ),
                "head",
            ],
        )
        layer_calls = ops.calls[1:-1]
        self.assertEqual([entry[1] for entry in layer_calls], list(range(45)))
        kda_calls = [entry for entry in layer_calls if entry[0] != "sparse_mla"]
        self.assertEqual([entry[3] for entry in kda_calls], list(state.kda_layers))
        mla_calls = [entry for entry in layer_calls if entry[0] == "sparse_mla"]
        self.assertEqual(
            [entry[4] for entry in mla_calls],
            list(state.sparse_mla_layers),
        )
        self.assertTrue(all(entry[3] is batch.sparse_mla for entry in mla_calls))
        self.assertEqual(ops.all_reduces, [all_reduce] * 46)


class GLM53TargetVerifyTest(unittest.TestCase):
    def test_verify_owns_fixed_width_rows_all_layers_and_concrete_transactions(
        self,
    ) -> None:
        ops = VerifyRecordingOps()
        weights = make_weights()
        state = make_state()
        transaction = make_transaction()
        workspace = make_workspace()
        batch = make_batch()
        group = object()
        all_reduce = object()
        model = glm53_target.GLM53TargetModel(weights, ops)

        with patch.object(glm53_target, "_validate_tep4_world", return_value=0):
            result = model.verify_tokens(
                batch, state, transaction, workspace, group, all_reduce
            )

        self.assertIs(result.normalized, workspace.endpoint.normalized)
        self.assertEqual(result.token, "tokens")
        self.assertEqual(
            [entry[0] for entry in ops.calls],
            [
                "embedding",
                *(
                    "dense"
                    if layer_id in glm53_flash.DENSE_LAYER_IDS
                    else "sparse_mla"
                    if layer_id in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS
                    else "sparse_kda"
                    for layer_id in range(45)
                ),
                "head",
            ],
        )
        layer_calls = ops.calls[1:-1]
        self.assertEqual([entry[1] for entry in layer_calls], list(range(45)))
        self.assertEqual(
            [entry[2] for entry in layer_calls],
            [("streams", layer_id - 1) for layer_id in range(45)],
        )
        kda_calls = [entry for entry in layer_calls if entry[0] != "sparse_mla"]
        self.assertEqual(
            [(entry[3], entry[4]) for entry in kda_calls],
            list(zip(state.kda_layers, transaction.kda_layers, strict=True)),
        )
        mla_calls = [entry for entry in layer_calls if entry[0] == "sparse_mla"]
        self.assertEqual(
            [(entry[3], entry[4]) for entry in mla_calls],
            [
                (history, (staged.tail_key, staged.tail_gate))
                for history, staged in zip(
                    state.sparse_mla_layers,
                    transaction.sparse_mla_layers,
                    strict=True,
                )
            ],
        )
        self.assertTrue(all(entry[5] is batch.sparse_mla for entry in mla_calls))
        self.assertEqual(ops.all_reduces, [all_reduce] * 46)


class GLM53TargetModelTest(unittest.TestCase):
    def test_wrong_weight_cardinality_fails_at_construction(self) -> None:
        weights = replace(
            make_weights(), sparse_mla_layers=make_weights().sparse_mla_layers[:-1]
        )
        with self.assertRaisesRegex(ValueError, "sparse MLA weights must contain 11"):
            glm53_target.GLM53TargetModel(weights, RecordingOps())


if __name__ == "__main__":
    unittest.main()
