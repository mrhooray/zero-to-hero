import inspect
import unittest
from unittest.mock import patch

import pytest
import torch

from infer.models.glm53_flash.model import (
    LOCAL_VOCAB_SIZE,
    TOKENIZER_VOCAB_SIZE,
    TP_SIZE,
    VOCAB_SIZE,
)
from infer.models.glm53_flash.ops import distributed_argmax

pytestmark = pytest.mark.gpu


class _FakeTP4:
    def __init__(self, rank: int, candidates: torch.Tensor) -> None:
        self.rank = rank
        self.candidates = candidates
        self.calls: list[tuple[torch.Size, torch.Size, torch.dtype, object]] = []

    def all_gather_into_tensor(
        self,
        output: torch.Tensor,
        local: torch.Tensor,
        *,
        group: object,
    ) -> None:
        torch.testing.assert_close(local, self.candidates[self.rank])
        self.calls.append((local.shape, output.shape, local.dtype, group))
        output.copy_(self.candidates.view(-1, 2))


class GLM53DistributedArgmaxTest(unittest.TestCase):
    def test_workspace_is_55_bytes_per_row(self) -> None:
        capacity = 64
        workspace = distributed_argmax.allocate_glm53_distributed_argmax_workspace(
            "cpu", capacity
        )
        tensors = (
            workspace.local_values,
            workspace.local_indices,
            workspace.local_candidates,
            workspace.gathered_candidates,
            workspace.best_values,
            workspace.select,
        )

        self.assertEqual(
            tuple(tensor.shape for tensor in tensors),
            (
                (capacity,),
                (capacity,),
                (capacity, 2),
                (TP_SIZE * capacity, 2),
                (capacity,),
                (capacity,),
            ),
        )
        workspace_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in tensors
        )
        full_vocab_bytes = (
            capacity * VOCAB_SIZE * (torch.bfloat16.itemsize + torch.float32.itemsize)
        )
        self.assertEqual(workspace_bytes, 55 * capacity)
        self.assertEqual(full_vocab_bytes - workspace_bytes, 59_470_400)

    def test_matches_full_vocab_argmax_for_graph_buckets_and_ties(self) -> None:
        workspace = distributed_argmax.allocate_glm53_distributed_argmax_workspace(
            "cpu", 64
        )
        workspace_pointers = tuple(
            tensor.data_ptr()
            for tensor in (
                workspace.local_values,
                workspace.local_indices,
                workspace.local_candidates,
                workspace.gathered_candidates,
                workspace.best_values,
                workspace.select,
            )
        )
        group = object()
        rank = 3

        for batch_size in (1, 7, 17, 64):
            shards = self._logits(batch_size)
            candidates = self._candidates(shards)
            fake = _FakeTP4(rank, candidates)
            output = torch.empty(batch_size, dtype=torch.int64)
            full_logits = shards.transpose(0, 1).reshape(batch_size, -1)
            expected = torch.argmax(
                full_logits[:, :TOKENIZER_VOCAB_SIZE],
                dim=1,
            )

            with (
                patch.object(
                    distributed_argmax.dist,
                    "get_world_size",
                    return_value=TP_SIZE,
                ),
                patch.object(
                    distributed_argmax.dist,
                    "get_rank",
                    return_value=rank,
                ),
                patch.object(
                    distributed_argmax.dist,
                    "all_gather_into_tensor",
                    side_effect=fake.all_gather_into_tensor,
                ),
            ):
                result = distributed_argmax.glm53_distributed_argmax(
                    shards[rank],
                    output,
                    workspace,
                    group,
                )

            torch.testing.assert_close(result, expected)
            self.assertEqual(result.data_ptr(), output.data_ptr())
            self.assertEqual(
                fake.calls,
                [
                    (
                        torch.Size((batch_size, 2)),
                        torch.Size((TP_SIZE * batch_size, 2)),
                        torch.float32,
                        group,
                    )
                ],
            )
            self.assertEqual(
                tuple(
                    tensor.data_ptr()
                    for tensor in (
                        workspace.local_values,
                        workspace.local_indices,
                        workspace.local_candidates,
                        workspace.gathered_candidates,
                        workspace.best_values,
                        workspace.select,
                    )
                ),
                workspace_pointers,
            )

        self.assertEqual(
            output[:3].tolist(),
            [LOCAL_VOCAB_SIZE + 9, 3 * LOCAL_VOCAB_SIZE + 3, 5],
        )

    def test_call_path_has_one_gather_and_no_storage_allocation(self) -> None:
        source = inspect.getsource(distributed_argmax.glm53_distributed_argmax)
        self.assertEqual(source.count("all_gather_into_tensor"), 1)
        self.assertEqual(source.count("torch.max("), 1)
        self.assertEqual(source.count("torch.where("), 2)
        self.assertIn("torch.gt(", source)
        for allocation in ("torch.empty", "torch.zeros", "torch.full", "torch.cat"):
            self.assertNotIn(allocation, source)

    def test_rejects_non_tp4_and_non_bf16_logits(self) -> None:
        workspace = distributed_argmax.allocate_glm53_distributed_argmax_workspace(
            "cpu", 1
        )
        logits = torch.empty((1, LOCAL_VOCAB_SIZE), dtype=torch.bfloat16)
        output = torch.empty(1, dtype=torch.int64)
        with (
            patch.object(
                distributed_argmax.dist,
                "get_world_size",
                return_value=3,
            ),
            self.assertRaisesRegex(ValueError, "TP4"),
        ):
            distributed_argmax.glm53_distributed_argmax(
                logits,
                output,
                workspace,
                object(),
            )

        with (
            patch.object(
                distributed_argmax.dist,
                "get_world_size",
                return_value=TP_SIZE,
            ),
            patch.object(distributed_argmax.dist, "get_rank", return_value=0),
            self.assertRaisesRegex(ValueError, "wrong shape or dtype"),
        ):
            distributed_argmax.glm53_distributed_argmax(
                logits.float(),
                output,
                workspace,
                object(),
            )

    @staticmethod
    def _logits(batch_size: int) -> torch.Tensor:
        shards = torch.full(
            (TP_SIZE, batch_size, LOCAL_VOCAB_SIZE),
            -100,
            dtype=torch.bfloat16,
        )
        valid_last_rank = TOKENIZER_VOCAB_SIZE - (TP_SIZE - 1) * LOCAL_VOCAB_SIZE
        shards[TP_SIZE - 1, :, valid_last_rank:].fill_(100)
        for row in range(batch_size):
            for rank in range(TP_SIZE):
                shards[rank, row, 7 + rank] = rank
            if row % 3 == 0:
                shards[1, row, 9] = 20
                shards[2, row, 0] = 20
            elif row % 3 == 1:
                shards[3, row, 3] = 20
            else:
                shards[0, row, 11] = 20
                shards[0, row, 5] = 20
        return shards

    @staticmethod
    def _candidates(shards: torch.Tensor) -> torch.Tensor:
        rank_candidates = []
        for rank in range(TP_SIZE):
            valid_width = min(
                LOCAL_VOCAB_SIZE,
                TOKENIZER_VOCAB_SIZE - rank * LOCAL_VOCAB_SIZE,
            )
            values, indices = torch.max(shards[rank, :, :valid_width], dim=1)
            indices.add_(rank * LOCAL_VOCAB_SIZE)
            rank_candidates.append(
                torch.stack((values.float(), indices.float()), dim=1)
            )
        return torch.stack(rank_candidates)


if __name__ == "__main__":
    unittest.main()
