import unittest
from pathlib import Path
from unittest.mock import patch

from infer.models.glm53_flash import launch as launcher


class GLM53LaunchTest(unittest.TestCase):
    def test_launch_forwards_server_configuration(self) -> None:
        with patch.object(launcher, "launch_ranks") as launch_ranks:
            launcher.launch(
                Path("/checkpoint"),
                "127.0.0.1",
                8000,
                "tep4",
                "native",
            )

        entrypoint = launch_ranks.call_args.args[0]
        self.assertEqual(
            entrypoint.keywords,
            {"parallelism": "tep4", "speculation": "native"},
        )
        self.assertEqual(
            launch_ranks.call_args.args[1:],
            (
                Path("/checkpoint"),
                "127.0.0.1",
                8000,
                "infer-glm53_flash-",
                launcher.supervise,
            ),
        )


if __name__ == "__main__":
    unittest.main()
