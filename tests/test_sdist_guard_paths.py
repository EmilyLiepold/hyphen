"""Filesystem contracts for path preflight; no package or artifact execution."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools/guard_sdist_smoke.py"
SPEC = importlib.util.spec_from_file_location("sdist_guard_paths", TOOL)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class RootValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "source-parent/source"
        self.backend = self.root / "backend-parent/backend"
        for directory in (self.source, self.backend):
            directory.mkdir(parents=True)
            (directory / "input.txt").write_bytes(b"preserve\x00\xff")
        self.output = self.root / "new-run"
        (self.root / "occupied").write_text("preserve")
        (self.root / "existing-directory").mkdir()
        (self.root / "dangling").symlink_to(self.root / "absent")
        (self.root / "source-alias").symlink_to(self.source, target_is_directory=True)
        (self.root / "backend-alias").symlink_to(self.backend, target_is_directory=True)

    def inventory(self):
        return {str(path.relative_to(self.root)):
                ("link", os.readlink(path)) if path.is_symlink() else
                ("directory", None) if path.is_dir() else ("file", path.read_bytes())
                for path in self.root.rglob("*")}

    def check(self, arguments, error=None):
        before = self.inventory()
        try:
            if error:
                with self.assertRaisesRegex(ValueError, error):
                    guard.validate_roots(*arguments)
            else:
                self.assertEqual((self.source, self.output, self.backend),
                                 guard.validate_roots(*arguments))
        finally:
            self.assertEqual(before, self.inventory())

    def test_fresh_disjoint_paths_and_aliases_resolve_without_writes(self):
        self.check((self.source, self.output, self.backend))
        self.check(tuple(os.path.relpath(path) for path in
                         (self.root / "source-alias", self.output, self.root / "backend-alias")))

    def test_overlapping_roots_are_rejected_before_existing_output(self):
        for directory in (self.source, self.backend):
            for output in (directory, directory.parent, directory / "new-run"):
                with self.subTest(output=output):
                    self.check((self.source, output, self.backend), "disjoint")
        for alias in ("source-alias", "backend-alias"):
            for suffix in ("", "..", "new-run"):
                with self.subTest(alias=alias, suffix=suffix):
                    self.check((self.source, self.root / alias / suffix, self.backend), "disjoint")

    def test_existing_outputs_and_invalid_directories_are_rejected(self):
        for name in ("occupied", "existing-directory", "dangling"):
            with self.subTest(output=name):
                self.check((self.source, self.root / name, self.backend), "already exist")
        for invalid in (self.root / "absent", self.root / "occupied"):
            with self.subTest(invalid=invalid):
                self.check((invalid, self.output, self.backend), "existing directories")
                self.check((self.source, self.output, invalid), "existing directories")
                self.check((self.source, invalid / "run", self.backend), "parent")

    def test_direct_execution_refuses_without_creating_outputs(self):
        before = self.inventory()
        result = subprocess.run([sys.executable, "-I", "-B", str(TOOL),
                                 str(self.source), str(self.output), str(self.backend)],
                                cwd=self.root, capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Artifact guard not implemented", result.stderr)
        self.assertEqual(before, self.inventory())
