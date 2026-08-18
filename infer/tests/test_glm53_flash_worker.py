import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from infer.models.glm53_flash.worker import GLM53Worker


class GLM53WorkerTest(unittest.TestCase):
    def test_worker_owns_decode_graph_capture(self) -> None:
        worker = object.__new__(GLM53Worker)
        worker._rank = 1
        worker._lane_count = 4
        worker._stage_decode = Mock(return_value=("batch", 2))
        worker._run_decode = Mock()
        worker._decode_graphs = None
        plan = SimpleNamespace(
            lanes=("lane-0", "lane-1", "lane-2", "lane-3"),
            staging_index=1,
            graph_bucket=8,
        )
        graph = object()
        stream = Mock()
        cuda = SimpleNamespace(
            graph_pool_handle=Mock(return_value="pool"),
            stream=Mock(return_value=nullcontext()),
            CUDAGraph=Mock(return_value=graph),
            graph=Mock(return_value=nullcontext()),
            synchronize=Mock(),
        )

        worker.capture_decode_graphs((plan,), stream, SimpleNamespace(cuda=cuda))

        worker._stage_decode.assert_called_once_with("lane-1", 1, 8)
        worker._run_decode.assert_called_once_with("batch", 2, 1)
        stream.synchronize.assert_called_once_with()
        cuda.graph.assert_called_once_with(graph, pool="pool", stream=stream)
        cuda.synchronize.assert_called_once_with()
        self.assertEqual(worker._decode_graphs, {(2, 1): graph})


if __name__ == "__main__":
    unittest.main()
