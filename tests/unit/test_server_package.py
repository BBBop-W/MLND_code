"""Packaging safety and independent verification; stdlib only, no solver."""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts import package_server as package


class ServerPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mlnd-package-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        names = list(package.FILES)
        names += [f"data/benchmarks/tsl_2l20h50od/{name}" for name in package.BENCHMARK_FILES]
        names += [f"examples/tsl_2l20h50od/{name}" for name in package.EXAMPLE_FILES]
        for name in names:
            self.put(name, f"fixture: {name}\n".encode())
        for tree, suffixes in package.TREES.items():
            self.put(f"{tree}/fixture{sorted(suffixes)[0]}", b"fixture\n")
        self.output = self.root / "delivery"

    def put(self, name, data=b"excluded sentinel"):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def create(self, **kwargs):
        return package.create_package(self.source, self.output, **kwargs)

    def test_allowlist_and_zip_roundtrip_preserve_frozen_bytes(self):
        excluded = ["outputs/do_not_read.py", "data/raw_orders.csv", "Meituan-TSL/private.csv",
                    "cpp/build/mlnd_cpp", "cpp/src/object.o", "cpp/include/license.hpp",
                    "src/__pycache__/cached.py", "src/outputs/do_not_read.py", ".git/config",
                    "scripts/cache.tmp", "docs/old_temporary.txt"]
        for name in excluded:
            self.put(name)
        self.put(package.OPTIONAL_FILES[0], b"current audit")
        report = self.create(make_zip=True)
        manifest = json.loads((self.output / package.MANIFEST).read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertTrue(set(excluded).isdisjoint(manifest["files"]))
        self.assertIn(package.OPTIONAL_FILES[0], manifest["files"])
        for name in manifest["files"]:
            self.assertEqual((self.source / name).read_bytes(), (self.output / name).read_bytes())
        self.assertEqual(package.verify_package(Path(report["archive"]))["manifest_sha256"],
                         report["manifest_sha256"])
        extracted = self.root / "extracted"
        with zipfile.ZipFile(report["archive"]) as archive:
            archive.extractall(extracted)
        self.assertEqual(package.verify_package(extracted)["files"], report["files"])

    def test_missing_required_source_fails_before_output_creation(self):
        (self.source / "cpp/CMakeLists.txt").unlink()
        with self.assertRaisesRegex(ValueError, "Required regular file missing"):
            self.create()
        self.assertFalse(self.output.exists())

    def test_existing_directory_or_archive_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep"
        sentinel.write_bytes(b"untouched")
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(sentinel.read_bytes(), b"untouched")
        another = self.root / "another"
        another.with_suffix(".zip").write_bytes(b"keep archive")
        with self.assertRaises(FileExistsError):
            package.create_package(self.source, another, make_zip=True)
        self.assertFalse(another.exists())

    def test_output_must_not_be_nested_in_source(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            package.create_package(self.source, self.source / "delivery")
        self.assertFalse((self.source / "delivery").exists())

    def test_tamper_missing_and_extra_files_are_detected(self):
        self.create()
        extra = self.output / "unexpected.txt"
        extra.write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "file set differs"):
            package.verify_package(self.output)
        self.assertTrue(package.verify_package(self.output, allow_extra_files=True)["passed"])
        extra.unlink()
        victim = self.output / "cpp/CMakeLists.txt"
        original = victim.read_bytes()
        victim.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            package.verify_package(self.output, allow_extra_files=True)
        victim.write_bytes(original)
        victim.unlink()
        with self.assertRaisesRegex(ValueError, "missing="):
            package.verify_package(self.output)

    def test_symlink_cannot_import_external_file(self):
        outside = self.root / "external.py"
        outside.write_bytes(b"not allowed")
        link = self.source / "src/external.py"
        try:
            link.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Symlinks unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.create()
        self.assertFalse(self.output.exists())

    def test_unsafe_manifest_and_windows_collisions_are_rejected(self):
        self.create()
        path = self.output / package.MANIFEST
        initial = json.loads(path.read_text(encoding="utf-8"))
        for unsafe in ("../escape", "/absolute", "C:/outside", "folder\\file", "aux.txt", "trailing."):
            changed = {**initial, "files": {unsafe: package.fingerprint(b"")}}
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError, msg=unsafe):
                package.verify_package(self.output)
        changed = {**initial, "files": {"A.py": package.fingerprint(b""), "a.py": package.fingerprint(b"")}}
        path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "collision"):
            package.verify_package(self.output)

    def test_zip_traversal_and_corruption_are_rejected_without_extraction(self):
        report = self.create(make_zip=True)
        archive = Path(report["archive"])
        with zipfile.ZipFile(archive, "a") as stream:
            stream.writestr("../escape", b"outside")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            package.verify_package(archive)
        self.assertFalse((self.root / "escape").exists())
        corrupted = self.root / "corrupted.zip"
        with zipfile.ZipFile(report["archive"]) as original, zipfile.ZipFile(corrupted, "x") as target:
            for name in original.namelist():
                if name != "../escape":
                    target.writestr(name, b"changed" if name == "README.md" else original.read(name))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            package.verify_package(corrupted)


if __name__ == "__main__":
    unittest.main()
