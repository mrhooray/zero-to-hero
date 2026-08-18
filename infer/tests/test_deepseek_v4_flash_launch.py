import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from infer.models.deepseek_v4_flash import launch as launcher


class DeepSeekV4LaunchTest(unittest.TestCase):
    def test_launch_forwards_one_startup_specialization(self) -> None:
        with patch.object(launcher, "launch_ranks") as launch_ranks:
            launcher.launch(
                Path("/checkpoint"),
                "127.0.0.1",
                8000,
                "dep4",
                "native",
                4,
            )

        entrypoint = launch_ranks.call_args.args[0]
        self.assertEqual(
            entrypoint.keywords,
            {
                "parallelism": "dep4",
                "speculation": "native",
                "dspark_verify_width": 4,
            },
        )
        self.assertEqual(
            launch_ranks.call_args.args[1:],
            (
                Path("/checkpoint"),
                "127.0.0.1",
                8000,
                "infer-deepseek-v4-flash-",
                launcher.supervise,
            ),
        )

    def test_rank_specializes_shapes_before_importing_serving_code(self) -> None:
        script = """
import importlib.util
import json
import sys

from infer.models.deepseek_v4_flash import launch as launcher
from infer.models.deepseek_v4_flash import model as deepseek_v4_flash

events = {
    "serve_loaded_before_call": "infer.models.deepseek_v4_flash.serve" in sys.modules
}

class Loader:
    def create_module(self, _spec):
        return None

    def exec_module(self, module):
        events["import_block_size"] = deepseek_v4_flash.DSPARK_BLOCK_SIZE
        events["import_verify_width"] = deepseek_v4_flash.DSPARK_VERIFY_WIDTH
        module.serve_rank = lambda *_args, **_kwargs: events.update(called=True)

class Finder:
    def find_spec(self, fullname, _path, _target=None):
        if fullname == "infer.models.deepseek_v4_flash.serve":
            return importlib.util.spec_from_loader(fullname, Loader())
        return None

sys.meta_path.insert(0, Finder())
launcher._serve_rank(
    2,
    "/checkpoint",
    "127.0.0.1",
    8000,
    "file:///rendezvous",
    (("receiver", "sender"),),
    parallelism="dep4",
    speculation="native",
    dspark_verify_width=4,
)
print(json.dumps(events))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "serve_loaded_before_call": False,
                "import_block_size": 3,
                "import_verify_width": 4,
                "called": True,
            },
        )

    def test_rejects_other_widths(self) -> None:
        for width in (3, 5, 7):
            with (
                self.subTest(width=width),
                self.assertRaisesRegex(ValueError, "must be 4 or 6"),
            ):
                launcher.launch(
                    Path("/checkpoint"), "127.0.0.1", 8000, "dep4", "native", width
                )


if __name__ == "__main__":
    unittest.main()
