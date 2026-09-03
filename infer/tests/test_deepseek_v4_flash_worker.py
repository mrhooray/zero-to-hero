import unittest
from unittest.mock import Mock, call

from infer.models.deepseek_v4_flash.worker import _run_decode, _run_target_decode


class DeepSeekV4WorkerTest(unittest.TestCase):
    def test_run_decode_orders_draft_verify_and_commit(self) -> None:
        model = Mock()
        dspark_model = Mock()
        runtime = Mock()
        dspark_runtime = Mock()
        dspark_runtime.batch.return_value = "batch"
        runtime.verify = {(2, 0): "verify"}

        _run_decode(model, runtime, dspark_model, dspark_runtime, (2, 0))

        dspark_runtime.batch.assert_called_once_with(2, 0)
        dspark_model.draft.assert_called_once_with(
            dspark_runtime, "batch", "verify"
        )
        dspark_model.commit.assert_called_once_with(
            dspark_runtime, "batch", "verify"
        )
        model.assert_has_calls(
            [
                call.verify("verify"),
                call.accept_verified("verify"),
                call.publish_verified("verify"),
            ]
        )

    def test_run_target_decode_forwards_the_bucket_runtime(self) -> None:
        model = Mock()
        runtime = Mock()
        runtime.decode = {(4, 1): "decode"}

        _run_target_decode(model, runtime, (4, 1))

        model.decode_target.assert_called_once_with("decode")


if __name__ == "__main__":
    unittest.main()
