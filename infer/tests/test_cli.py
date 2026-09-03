import unittest
from pathlib import Path
from unittest.mock import patch

from infer import cli
from infer.models import deepseek_v4_flash, glm53_flash


class CLITest(unittest.TestCase):
    def test_local_checkpoint_routes_to_the_selected_model(self) -> None:
        checkpoint = Path("/resolved/checkpoint")
        with (
            patch.object(
                cli, "resolve_model_source", return_value=checkpoint
            ) as resolve,
            patch("infer.models.glm53_flash.launch.launch") as launch,
        ):
            cli.main(
                [
                    "serve",
                    glm53_flash.MODEL_ID,
                    "--checkpoint-dir",
                    "/checkpoint",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "9000",
                ]
            )

        resolve.assert_called_once_with(Path("/checkpoint"))
        launch.assert_called_once_with(checkpoint, "0.0.0.0", 9000, "dep4", "native")

    def test_hugging_face_uses_the_selected_models_pinned_revision(self) -> None:
        checkpoint = Path("/resolved/checkpoint")
        with (
            patch.object(
                cli, "resolve_model_source", return_value=checkpoint
            ) as resolve,
            patch("infer.models.deepseek_v4_flash.launch.launch") as launch,
        ):
            cli.main(["serve", deepseek_v4_flash.MODEL_ID, "--cache-dir", "/cache"])

        resolve.assert_called_once_with(
            repo_id=deepseek_v4_flash.MODEL_ID,
            revision=deepseek_v4_flash.MODEL_REVISION,
            cache_dir=Path("/cache"),
            token=False,
        )
        launch.assert_called_once_with(
            checkpoint, "127.0.0.1", 8000, "dep4", "native", 6
        )

    def test_parallelism_defaults_to_dep4_for_both_models(self) -> None:
        checkpoint = Path("/resolved/checkpoint")
        cases = (
            (
                deepseek_v4_flash.MODEL_ID,
                "infer.models.deepseek_v4_flash.launch.launch",
                (checkpoint, "127.0.0.1", 8000, "dep4", "native", 6),
            ),
            (
                glm53_flash.MODEL_ID,
                "infer.models.glm53_flash.launch.launch",
                (checkpoint, "127.0.0.1", 8000, "dep4", "native"),
            ),
        )
        for model, launcher, expected in cases:
            with (
                self.subTest(model=model),
                patch.object(cli, "resolve_model_source", return_value=checkpoint),
                patch(launcher) as launch,
            ):
                cli.main(["serve", model, "--checkpoint-dir", "/checkpoint"])
            launch.assert_called_once_with(*expected)

    def test_both_models_accept_explicit_parallelism(self) -> None:
        checkpoint = Path("/resolved/checkpoint")
        with (
            patch.object(cli, "resolve_model_source", return_value=checkpoint),
            patch("infer.models.deepseek_v4_flash.launch.launch") as launch,
        ):
            cli.main(["serve", deepseek_v4_flash.MODEL_ID, "--parallelism", "tep4"])
        launch.assert_called_once_with(
            checkpoint, "127.0.0.1", 8000, "tep4", "native", 6
        )

        with (
            patch.object(cli, "resolve_model_source", return_value=checkpoint),
            patch("infer.models.glm53_flash.launch.launch") as launch,
        ):
            cli.main(["serve", glm53_flash.MODEL_ID, "--parallelism", "dep4"])
        launch.assert_called_once_with(checkpoint, "127.0.0.1", 8000, "dep4", "native")

    def test_speculation_mode_is_explicitly_forwarded(self) -> None:
        checkpoint = Path("/resolved/checkpoint")
        with (
            patch.object(cli, "resolve_model_source", return_value=checkpoint),
            patch("infer.models.deepseek_v4_flash.launch.launch") as launch,
        ):
            cli.main(["serve", deepseek_v4_flash.MODEL_ID, "--speculation", "none"])

        launch.assert_called_once_with(checkpoint, "127.0.0.1", 8000, "dep4", "none", 6)

    def test_dspark_verify_width_is_explicitly_forwarded(self) -> None:
        checkpoint = Path("/resolved/checkpoint")
        with (
            patch.object(cli, "resolve_model_source", return_value=checkpoint),
            patch("infer.models.deepseek_v4_flash.launch.launch") as launch,
        ):
            cli.main(["serve", deepseek_v4_flash.MODEL_ID, "--dspark-verify-width", "4"])

        launch.assert_called_once_with(
            checkpoint, "127.0.0.1", 8000, "dep4", "native", 4
        )

    def test_dspark_verify_width_requires_native_deepseek(self) -> None:
        for model, extra in (
            (deepseek_v4_flash.MODEL_ID, ["--speculation", "none"]),
            (glm53_flash.MODEL_ID, []),
        ):
            with (
                self.subTest(model=model),
                self.assertRaises(SystemExit),
                patch.object(cli, "resolve_model_source") as resolve,
            ):
                cli.main(["serve", model, "--dspark-verify-width", "4", *extra])
            resolve.assert_not_called()

    def test_local_checkpoint_rejects_hugging_face_options(self) -> None:
        for option in ("--revision", "--cache-dir"):
            with (
                self.subTest(option=option),
                self.assertRaises(SystemExit),
                patch.object(cli, "resolve_model_source") as resolve,
            ):
                cli.main(
                    [
                        "serve",
                        glm53_flash.MODEL_ID,
                        "--checkpoint-dir",
                        "/checkpoint",
                        option,
                        "value",
                    ]
                )
            resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
