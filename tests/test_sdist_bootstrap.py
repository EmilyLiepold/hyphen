"""Bootstrap wiring only; the separately guarded smoke exercises real artifacts."""
import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock
import sys


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.run = Path(scratch.name)
        self.events = []

        def create(prefix):
            prefix.mkdir()
            (prefix / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')

        self.builder = Mock(return_value=SimpleNamespace(create=create))
        self.execute = Mock(side_effect=lambda argv: self.events.append(('bootstrap', argv)))
        self.copy = Mock(side_effect=lambda *args, **kwargs: self.events.append(('copy', args)))
        self.scope = dict(run=self.run, sys=sys, backend=self.run / 'backend',
                          venv=SimpleNamespace(EnvBuilder=self.builder), execute=self.execute,
                          shutil=SimpleNamespace(copytree=self.copy, ignore_patterns=Mock()))
        script = Path(__file__).resolve().parents[1] / 'tools/sdist_smoke.py'
        tree = ast.parse(script.read_text())
        function, = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == 'environment']
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(script), 'exec'), self.scope)

    def test_requested_pip_uses_recorded_helper_before_backend_copy(self):
        python = self.scope['environment']('install-env', pip=True)
        self.builder.assert_called_once_with(with_pip=False)
        self.execute.assert_called_once_with(
            [python, '-I', '-B', '-m', 'ensurepip', '--upgrade', '--default-pip'])
        self.assertEqual([event[0] for event in self.events], ['bootstrap', 'copy'])
        self.assertEqual(python, self.run / 'install-env/bin/python')

    def test_builder_environment_does_not_bootstrap_pip(self):
        self.scope['environment']('build-env')
        self.builder.assert_called_once_with(with_pip=False)
        self.execute.assert_not_called()
        self.copy.assert_called_once()

    def test_bootstrap_error_prevents_backend_copy(self):
        self.execute.side_effect = RuntimeError('controlled bootstrap failure')
        with self.assertRaisesRegex(RuntimeError, 'controlled bootstrap failure'):
            self.scope['environment']('install-env', pip=True)
        self.copy.assert_not_called()


if __name__ == '__main__':
    unittest.main()
