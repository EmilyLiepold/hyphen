"""Check an installed Hyphen artifact and its complete runtime dependencies.

Prerequisites: install the reviewed artifact and dependencies into a disposable
venv; redirect config/cache writes into writable scratch and set MPLBACKEND=Agg.
Run outside the checkout under the agreed resource/write/network guard:
  /work/env/bin/python -I -B tools/installed_environment_smoke.py /work/env /work/report.json
Replace both /work paths and the tool path. The report must not already exist.
Exit 0 and status=pass require pip check, all twelve Hyphen module imports from
the requested venv, its CLI entry point and --help. Other distributions must also
live inside that venv. Failures remain in the report and produce exit 1.
Imports may write redirected caches and set NumPy's global seed/warning filters.
This is an installation smoke check; inspect artifact bytes/provenance separately.
It does not execute IFU/merger science workflows or establish MSA support, other
Python/platform compatibility, numerical correctness or release acceptance.
"""
import argparse
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('prefix', type=Path)
parser.add_argument('report', type=Path)
args = parser.parse_args()
prefix, report = args.prefix.resolve(), args.report.resolve()
if not sys.flags.isolated:
    parser.error('Run with Python -I to exclude checkout and user-site imports')
if prefix != Path(sys.prefix).resolve() or sys.prefix == sys.base_prefix:
    parser.error('PREFIX must identify this interpreter\'s virtual environment')
if report.exists() or not report.parent.is_dir():
    parser.error('REPORT must be new and its parent directory must exist')
errors, imports = [], {}
modules = ['hyphen'] + ['hyphen.' + name for name in (
    'batch_dewiggle_ifu_step', 'batch_dewiggle_msa_step', 'dewiggle_lambda_space',
    'dewiggler_stack', 'frame_merger', 'helpers', 'hyphenate',
    'single_dewiggle_ifu_step', 'single_dewiggle_msa_step', 'tracer_helpers', 'yaml_helpers')]
for name in modules:
    try:
        origin = Path(importlib.import_module(name).__file__).resolve()
        imports[name] = str(origin)
        if not origin.is_relative_to(prefix):
            errors.append(f'{name} imported outside PREFIX: {origin}')
    except Exception as error:
        errors.append(f'{name}: {error!r}')
distributions = {}
for dist in metadata.distributions():
    origin = Path(dist.locate_file('')).resolve()
    distributions[dist.metadata['Name']] = dict(version=dist.version, origin=str(origin))
    if not origin.is_relative_to(prefix):
        errors.append(f'{dist.metadata["Name"]} metadata outside PREFIX: {origin}')
try:
    dist = metadata.distribution('hyphen')
    entry, = [e for e in dist.entry_points if e.group == 'console_scripts' and e.name == 'hyphenate']
    if entry.value != 'hyphen.hyphenate:hyphenate' or not callable(entry.load()):
        errors.append('Unexpected hyphenate entry point')
    project = dict(version=dist.version, requires=dist.requires, entry_point=entry.value)
except Exception as error:
    errors.append(f'Hyphen metadata/entry point: {error!r}')
    project = None
commands = {}
for name, argv in [('pip_check', [sys.executable, '-I', '-B', '-m', 'pip', 'check']),
                   ('cli_help', [sys.executable, '-I', '-B', str(prefix / 'bin/hyphenate'), '--help'])]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        commands[name] = dict(argv=argv, returncode=result.returncode,
                              stdout=result.stdout, stderr=result.stderr)
        if result.returncode or (name == 'cli_help' and 'yaml_file' not in result.stdout):
            errors.append(f'{name} failed; see command output')
    except Exception as error:
        errors.append(f'{name}: {error!r}')
payload = dict(status='fail' if errors else 'pass', errors=errors,
    python=sys.version, executable=sys.executable, prefix=str(prefix),
    platform=platform.platform(), machine=platform.machine(), cwd=str(Path.cwd()),
    imports=imports, distributions=distributions, hyphen=project, commands=commands)
with report.open('x') as output:
    output.write(json.dumps(payload, indent=2) + '\n')
print(f'{"FAIL" if errors else "PASS"}: {report}')
raise SystemExit(bool(errors))
