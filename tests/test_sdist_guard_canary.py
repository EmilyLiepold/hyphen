"""Controlled canary protocol faults; actual macOS confinement is checked separately."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

TOOL = Path(__file__).resolve().parents[1] / 'tools/guard_sdist_smoke.py'
SPEC = importlib.util.spec_from_file_location('sdist_guard_canary', TOOL)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class CanaryGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.backend = self.root / 'source', self.root / 'backend'
        for path in (self.source / 'tools/sdist_smoke.py', self.backend / 'setuptools/__init__.py'):
            path.parent.mkdir(parents=True)
            path.write_bytes(b'preserve\x00\xff')
        self.good = {key: True for key in ('work_write_allowed', 'source_write_denied',
                     'backend_write_denied', 'evidence_write_denied', 'socket_denied')}

    def attempt(self, label, stdout=None, returncode=0, error=None, success=False):
        output = self.root / label
        completed = subprocess.CompletedProcess([], returncode, stdout, 'recorded stderr')
        reached = False
        with patch.object(guard.sys, 'platform', 'darwin'), patch.object(guard.os, 'access', return_value=True):
            with patch.object(guard.subprocess, 'run', side_effect=error, return_value=completed) as run:
                try:
                    result = guard.confinement_preflight(self.source, output, self.backend)
                    reached = True
                    self.assertEqual(output / 'work', result)
                except RuntimeError:
                    self.assertFalse(success)
        self.assertEqual(success, reached)
        record = json.loads((output / 'canary.json').read_text())
        self.assertEqual('passed' if success else 'failed', record['status'])
        self.assertEqual(10, run.call_args.kwargs['timeout'])
        self.assertEqual(record['argv'], run.call_args.args[0])
        self.assertGreaterEqual(record['elapsed_seconds'], 0)
        self.assertEqual('preserve\n', (output / 'protected-canary').read_text())
        for path in (self.source / 'tools/sdist_smoke.py', self.backend / 'setuptools/__init__.py'):
            self.assertEqual(b'preserve\x00\xff', path.read_bytes())
        return record

    def test_only_exact_success_returns_to_caller(self):
        record = self.attempt('good', json.dumps(self.good), success=True)
        self.assertEqual(0, record['returncode'])
        self.assertIsNone(record['error'])
        invalid = ['not json', '[]', 'null', json.dumps({}), json.dumps({**self.good, 'extra': True}),
                   json.dumps(self.good) + '\nextra', json.dumps(self.good)[:-1] + ',"socket_denied":true}']
        for value in (False, 1, 'true', [], None):
            invalid.append(json.dumps({**self.good, 'socket_denied': value}))
        for index, stdout in enumerate(invalid):
            with self.subTest(stdout=stdout):
                record = self.attempt(f'invalid-{index}', stdout)
                self.assertEqual(stdout, record['stdout'])
                self.assertIsNotNone(record['error'])
        self.attempt('nonzero', json.dumps(self.good), returncode=7)

    def test_launch_error_and_timeout_are_retained_before_refusal(self):
        errors = [OSError('controlled launch failure'),
                  subprocess.TimeoutExpired(['controlled'], 10, output=b'partial output', stderr=b'partial error')]
        for index, error in enumerate(errors):
            record = self.attempt(f'error-{index}', error=error)
            self.assertEqual(type(error).__name__, record['error']['type'])
            self.assertIsNone(record['returncode'])
        self.assertEqual('partial output', record['stdout'])
        self.assertEqual('partial error', record['stderr'])

    def test_platform_and_missing_probe_refuse_before_writes(self):
        output = self.root / 'unused'
        with patch.object(guard.sys, 'platform', 'linux'):
            with self.assertRaises(RuntimeError):
                guard.confinement_preflight(self.source, output, self.backend)
        with patch.object(guard.sys, 'platform', 'darwin'), patch.object(guard.os, 'access', return_value=False):
            with self.assertRaises(RuntimeError):
                guard.confinement_preflight(self.source, output, self.backend)
        with patch.object(guard.sys, 'platform', 'darwin'), patch.object(guard.os, 'access', return_value=True):
            (self.backend / 'setuptools/__init__.py').unlink()
            with self.assertRaises(RuntimeError):
                guard.confinement_preflight(self.source, output, self.backend)
        self.assertFalse(output.exists())
