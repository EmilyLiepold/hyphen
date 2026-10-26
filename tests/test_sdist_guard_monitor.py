"""Strict one-shot process-table protocol and canary-before-monitor contract."""
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

TOOL = Path(__file__).resolve().parents[1] / 'tools/guard_sdist_smoke.py'
SPEC = importlib.util.spec_from_file_location('sdist_guard_monitor', TOOL)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class ProcessTableTests(unittest.TestCase):
    def test_valid_table_preserves_zero_fields_and_reported_kib(self):
        self.assertEqual([[0, 0, 0, 0], [17, 0, 4, 123]],
                         guard._parse_process_table('\n0 0 0 0\n 17 0 4 123\n', 17, 4))

    def test_every_nonblank_row_and_self_identity_are_required(self):
        invalid = ['', ' \n', None, b'17 0 4 123', '17 0 4', '17 0 4 123 9',
                   '17 0 4 123\n17 0 4 456', '9 0 4 123', '17 0 5 123',
                   '17 0 4 123\nbad row', '17 0 4 123\n0 0 0 -1']
        invalid += [f'17 0 4 {value}' for value in ('+1', '-1', '1.0', '1e3', '１２', 'nan')]
        for stdout in invalid:
            with self.subTest(stdout=stdout), self.assertRaises(ValueError):
                guard._parse_process_table(stdout, 17, 4)

    def attempt(self, stdout='17 0 4 123\n', returncode=0, error=None, success=False):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.CompletedProcess([], returncode, stdout, 'retained stderr')
            with patch.object(guard.os, 'getpid', return_value=17), patch.object(guard.os, 'getpgrp', return_value=4):
                with patch.object(guard.subprocess, 'run', side_effect=error, return_value=result) as run:
                    if success:
                        self.assertEqual([[17, 0, 4, 123]], guard.process_table_preflight(temporary))
                    else:
                        with self.assertRaises(RuntimeError):
                            guard.process_table_preflight(temporary)
            record = json.loads((Path(temporary) / 'monitor.json').read_text())
        run.assert_called_once_with(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,rss='],
                                    capture_output=True, text=True, timeout=2)
        self.assertEqual('passed' if success else 'failed', record['status'])
        self.assertEqual(2, record['timeout_seconds'])
        self.assertGreaterEqual(record['elapsed_seconds'], 0)
        self.assertEqual((17, 4, 'KiB'), (record['self_pid'], record['self_pgid'], record['rss_unit']))
        return record

    def test_success_records_raw_command_output(self):
        record = self.attempt(success=True)
        self.assertEqual(('17 0 4 123\n', 'retained stderr', 0, None),
                         (record['stdout'], record['stderr'], record['returncode'], record['error']))

    def test_unavailable_nonzero_timeout_and_malformed_fail_with_evidence(self):
        for error in (FileNotFoundError('/bin/ps'), PermissionError('denied'),
                      subprocess.TimeoutExpired(['/bin/ps'], 2, output=b'partial\xff', stderr=b'timeout')):
            with self.subTest(error=error):
                record = self.attempt(error=error)
                self.assertEqual(type(error).__name__, record['error']['type'])
                self.assertIsNone(record['returncode'])
        self.assertEqual(('partial\ufffd', 'timeout'), (record['stdout'], record['stderr']))
        record = self.attempt(returncode=7)
        self.assertEqual(7, record['returncode'])
        for stdout in ('bad row', '', None, b'17 0 4 123'):
            self.assertEqual('ValueError', self.attempt(stdout=stdout)['error']['type'])

    def test_evidence_write_failure_cannot_return_success(self):
        result = subprocess.CompletedProcess([], 0, f'{guard.os.getpid()} 0 {guard.os.getpgrp()} 0', '')
        with patch.object(guard.subprocess, 'run', return_value=result):
            with patch.object(guard.Path, 'write_text', side_effect=OSError('unwritable')):
                with self.assertRaisesRegex(OSError, 'unwritable'):
                    guard.process_table_preflight('unused')

    def test_sequence_stops_at_each_failed_gate(self):
        for failed in ('canary', 'monitor', None):
            with self.subTest(failed=failed):
                events = Mock()
                canary = Mock(side_effect=RuntimeError('canary') if failed == 'canary' else None, return_value=Path('canonical/work'))
                monitor = Mock(side_effect=RuntimeError('monitor') if failed == 'monitor' else None)
                events.attach_mock(canary, 'canary')
                events.attach_mock(monitor, 'monitor')
                with patch.object(guard, 'confinement_preflight', canary), patch.object(guard, 'process_table_preflight', monitor):
                    if failed:
                        with self.assertRaisesRegex(RuntimeError, failed):
                            guard.preflight('source', 'output', 'backend')
                    else:
                        self.assertEqual(Path('canonical/work'), guard.preflight('source', 'output', 'backend'))
                self.assertEqual(['canary'] if failed == 'canary' else ['canary', 'monitor'],
                                 [event[0] for event in events.mock_calls])
                if failed == 'canary':
                    monitor.assert_not_called()
                else:
                    monitor.assert_called_once_with(Path('canonical'))

    def test_sequence_keeps_canonical_canary_evidence_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            canonical = Path(temporary) / 'new-run'
            canonical.mkdir()
            original = Path(temporary) / 'missing/../new-run'
            work = canonical / 'work'
            result = subprocess.CompletedProcess([], 0, f'{guard.os.getpid()} 0 {guard.os.getpgrp()} 0', '')
            with patch.object(guard, 'confinement_preflight', return_value=work):
                with patch.object(guard.subprocess, 'run', return_value=result):
                    self.assertEqual(work, guard.preflight('source', original, 'backend'))
            self.assertEqual('passed', json.loads((canonical / 'monitor.json').read_text())['status'])
            self.assertFalse((Path(temporary) / 'missing').exists())

    def test_direct_cli_still_refuses(self):
        with self.assertRaisesRegex(SystemExit, 'Artifact guard not implemented'):
            runpy.run_path(str(TOOL), run_name='__main__')
