import importlib.util
import unittest

import pytest

import torch

from infer.models.glm53_flash import model as model
from infer.models.glm53_flash.ops import core as glm53_ops

pytestmark = pytest.mark.gpu

HAS_GPU_RUNTIME = (
    importlib.util.find_spec("triton") is not None and torch.cuda.is_available()
)


@unittest.skipUnless(HAS_GPU_RUNTIME, "requires CUDA and Triton")
class GLM53RouterGPUTest(unittest.TestCase):
    def test_fused_boundaries_match_torch_route(self) -> None:
        torch.manual_seed(7)
        for rows in (1, 4, 16, 32):
            raw_scores = torch.randn(
                rows, model.NUM_ROUTED_EXPERTS, dtype=torch.float32, device="cuda"
            )
            correction_bias = torch.randn(
                model.NUM_ROUTED_EXPERTS, dtype=torch.float32, device="cuda"
            )
            scores = raw_scores.clone()
            selection = torch.empty_like(scores)
            glm53_ops._route_sparse_ffn_pre_topk[(rows,)](
                scores,
                correction_bias,
                selection,
                NUM_EXPERTS=model.NUM_ROUTED_EXPERTS,
                BLOCK=512,
                num_warps=4,
            )

            expected_scores = torch.sigmoid(raw_scores)
            expected_selection = expected_scores + correction_bias
            torch.testing.assert_close(scores, expected_scores, rtol=1e-5, atol=2e-7)
            torch.testing.assert_close(
                selection, expected_selection, rtol=1e-5, atol=2e-7
            )
            _, topk_ids64 = torch.topk(selection, model.TOP_K, dim=1, sorted=False)
            topk_values = torch.empty(
                rows, model.TOP_K, dtype=torch.float32, device="cuda"
            )
            topk_ids = torch.empty(rows, model.TOP_K, dtype=torch.int32, device="cuda")
            glm53_ops._route_sparse_ffn_post_topk[(rows,)](
                scores,
                topk_ids64,
                topk_values,
                topk_ids,
                NUM_EXPERTS=model.NUM_ROUTED_EXPERTS,
                TOPK=model.TOP_K,
                ROUTE_SCALE=model.SPARSE_FFN_ROUTE_SCALE,
                num_warps=1,
            )

            expected_values = torch.gather(expected_scores, 1, topk_ids64)
            expected_values /= expected_values.sum(dim=1, keepdim=True) + 1e-20
            expected_values *= model.SPARSE_FFN_ROUTE_SCALE
            self.assertTrue(torch.equal(topk_ids, topk_ids64.to(torch.int32)))
            torch.testing.assert_close(
                topk_values, expected_values, rtol=1e-5, atol=2e-7
            )


if __name__ == "__main__":
    unittest.main()
