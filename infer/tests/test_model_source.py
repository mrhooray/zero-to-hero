import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from infer import model_source

REVISION = "0123456789abcdef0123456789abcdef01234567"


class LocalModelSourceTest(unittest.TestCase):
    def test_existing_directory_is_resolved_without_hugging_face(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            checkpoint.mkdir()

            with patch.object(model_source, "snapshot_download") as download:
                resolved = model_source.resolve_model_source(checkpoint)

            self.assertEqual(resolved, checkpoint.resolve())
            download.assert_not_called()

    def test_nonexistent_path_is_not_inferred_to_be_a_repo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "org" / "model"

            with (
                self.assertRaisesRegex(ValueError, "existing directory"),
                patch.object(model_source, "snapshot_download") as download,
            ):
                model_source.resolve_model_source(missing)

            download.assert_not_called()

    def test_file_is_not_a_checkpoint_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            checkpoint.touch()

            with self.assertRaisesRegex(ValueError, "existing directory"):
                model_source.resolve_model_source(checkpoint)

    def test_hugging_face_options_are_rejected_for_local_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = (
                {"revision": REVISION},
                {"cache_dir": directory},
                {"token": "do-not-render-this-token"},
                {"repo_id": "org/model"},
            )
            for option in options:
                with (
                    self.subTest(option=next(iter(option))),
                    self.assertRaises(ValueError),
                ):
                    model_source.resolve_model_source(directory, **option)


class HuggingFaceModelSourceTest(unittest.TestCase):
    def test_explicit_repo_and_commit_are_downloaded_once(self) -> None:
        token = "private-token"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            cache = root / "cache"

            with patch.object(
                model_source, "snapshot_download", return_value=str(snapshot)
            ) as download:
                resolved = model_source.resolve_model_source(
                    repo_id="org/model",
                    revision=REVISION.upper(),
                    cache_dir=cache,
                    token=token,
                )

            self.assertEqual(resolved, snapshot.resolve())
            download.assert_called_once()
            self.assertEqual(download.call_args.kwargs["repo_id"], "org/model")
            self.assertEqual(download.call_args.kwargs["revision"], REVISION)
            self.assertEqual(download.call_args.kwargs["cache_dir"], cache)
            self.assertIs(download.call_args.kwargs["token"], token)

    def test_missing_and_mutable_revisions_are_rejected_before_download(self) -> None:
        revisions = (None, "", "main", "v1.0", "0" * 39, "0" * 41, "g" * 40)
        for revision in revisions:
            with (
                self.subTest(revision=revision),
                self.assertRaisesRegex(ValueError, "immutable 40-hex"),
                patch.object(model_source, "snapshot_download") as download,
            ):
                model_source.resolve_model_source(
                    repo_id="org/model", revision=revision
                )
            download.assert_not_called()

    def test_download_result_must_be_a_local_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            downloaded_file = Path(directory) / "snapshot"
            downloaded_file.touch()

            with (
                self.assertRaisesRegex(RuntimeError, "local directory"),
                patch.object(
                    model_source,
                    "snapshot_download",
                    return_value=str(downloaded_file),
                ),
            ):
                model_source.resolve_model_source(
                    repo_id="org/model", revision=REVISION
                )

    def test_a_source_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "required"):
            model_source.resolve_model_source()


if __name__ == "__main__":
    unittest.main()
