from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import shutil
import sys
import tarfile
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_script(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ASSETS = load_script(
    "v110_release_assets", ROOT / "scripts/v110-release-assets.py"
)


class V110ReleaseAssetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.evidence = self.root / "evidence"
        (self.evidence / "nested").mkdir(parents=True)
        (self.evidence / "terminal.json").write_text(
            '{"state":"PASS"}\n', encoding="utf-8"
        )
        (self.evidence / "nested/sample.txt").write_text(
            "measured seconds: 3.5\n", encoding="utf-8"
        )
        self.authority = self.root / "final-receipt.json"
        self.authority.write_text('{"authorization":true}\n', encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_bundle_is_deterministic_integrity_only_and_exactly_replayed(self) -> None:
        first = self.root / "first.tar.gz"
        second = self.root / "second.tar.gz"
        manifest = ASSETS.create_bundle(
            {"final": self.evidence}, {"final_receipt": self.authority}, first
        )
        ASSETS.create_bundle(
            {"final": self.evidence}, {"final_receipt": self.authority}, second
        )
        self.assertEqual(first.read_bytes(), second.read_bytes())
        relocated = self.root / "relocated.tar.gz"
        shutil.copyfile(first, relocated)
        self.assertEqual(ASSETS.verify_bundle(relocated), manifest)
        self.assertTrue(manifest["portable_integrity_only"])
        self.assertFalse(manifest["semantic_replay"])
        self.assertFalse(manifest["release_authority"])
        (self.evidence / "nested/sample.txt").write_text(
            "measured seconds: 3.6\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            ASSETS.ReleaseAssetError, "differs from current release inputs"
        ):
            ASSETS.verify_bundle_inputs(
                relocated,
                {"final": self.evidence},
                {"final_receipt": self.authority},
            )

    def test_bundle_rejects_symlinked_evidence(self) -> None:
        (self.evidence / "link").symlink_to(self.authority)
        with self.assertRaisesRegex(ASSETS.ReleaseAssetError, "symlink"):
            ASSETS.create_bundle(
                {"final": self.evidence},
                {"final_receipt": self.authority},
                self.root / "unsafe.tar.gz",
            )

    def test_arbitrary_self_declared_manifest_cannot_pass(self) -> None:
        unsigned = {
            "schema": ASSETS.BUNDLE_SCHEMA,
            "portable_integrity_only": True,
            "semantic_replay": False,
            "release_authority": False,
            "files": [
                {
                    "path": "evidence/claimed-pass.json",
                    "role": "evidence",
                    "size_bytes": 2,
                    "sha256": ASSETS._sha256_bytes(b"{}"),
                },
                {
                    "path": "authority/final_receipt/final.json",
                    "role": "final_receipt",
                    "size_bytes": 2,
                    "sha256": ASSETS._sha256_bytes(b"{}"),
                },
            ],
        }
        manifest = dict(unsigned)
        manifest["bundle_id"] = ASSETS._canonical_sha256(unsigned)
        path = self.root / "declaration-only.tar.gz"
        with tarfile.open(path, mode="w:gz") as archive:
            payload = ASSETS._canonical_bytes(manifest) + b"\n"
            info = tarfile.TarInfo("bundle-manifest.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        with self.assertRaisesRegex(
            ASSETS.ReleaseAssetError, "bundled file is missing"
        ):
            ASSETS.verify_bundle(path)

    def test_bundle_rejects_path_traversal_even_with_manifest(self) -> None:
        path = self.root / "traversal.tar.gz"
        with tarfile.open(path, mode="w:gz") as archive:
            info = tarfile.TarInfo("../escape")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        with self.assertRaisesRegex(ASSETS.ReleaseAssetError, "unsafe archive"):
            ASSETS.verify_bundle(path)

    def test_asset_manifest_replays_all_downloaded_bytes_and_no_extras(self) -> None:
        package = self.root / "gpmeep.conda"
        receipt = self.root / "receipt.json"
        package.write_bytes(b"package")
        receipt.write_bytes(b"receipt")
        manifest_path = self.root / "release-assets.json"
        manifest = ASSETS.create_asset_manifest(
            {"package": package, "final_receipt": receipt}, manifest_path
        )
        download = self.root / "download"
        download.mkdir()
        for source in (package, receipt, manifest_path):
            shutil.copyfile(source, download / source.name)
        replayed = ASSETS.verify_asset_directory(
            download / manifest_path.name, download, manifest["manifest_id"]
        )
        self.assertEqual(replayed, manifest)
        ASSETS.verify_asset_sources(
            manifest_path, {"package": package, "final_receipt": receipt}
        )
        (download / "receipt.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ASSETS.ReleaseAssetError, "asset (size|hash) differs"):
            ASSETS.verify_asset_directory(
                download / manifest_path.name, download, manifest["manifest_id"]
            )
        with self.assertRaisesRegex(ASSETS.ReleaseAssetError, "tag-bound manifest ID"):
            ASSETS.verify_asset_directory(
                download / manifest_path.name, download, "0" * 64
            )

    def test_asset_manifest_refuses_duplicate_basenames(self) -> None:
        left = self.root / "left"
        right = self.root / "right"
        left.mkdir()
        right.mkdir()
        (left / "same.json").write_text("left", encoding="utf-8")
        (right / "same.json").write_text("right", encoding="utf-8")
        with self.assertRaisesRegex(ASSETS.ReleaseAssetError, "duplicate"):
            ASSETS.create_asset_manifest(
                {
                    "candidate_receipt": left / "same.json",
                    "final_receipt": right / "same.json",
                },
                self.root / "duplicates.json",
            )


if __name__ == "__main__":
    unittest.main()
