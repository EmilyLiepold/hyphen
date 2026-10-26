"""Offline sdist file/install smoke; --no-deps leaves runtime validation incomplete.

Usage: python sdist_smoke.py SOURCE_EXPORT NEW_RUN UNPACKED_SETUPTOOLS
Run inside the separately documented resource/write/network guard. SOURCE_EXPORT
must identify the reviewed source; the setuptools directory is read-only input.
Each subprocess has a 120-second timeout. Existing run directories are refused.
An optimized outer interpreter is refused; omit -O/-OO and unset any honored
PYTHONOPTIMIZE. NEW_RUN must be outside both SOURCE_EXPORT and the backend,
including through symlinks; these checks precede run-directory creation.
"""
import sys

if sys.flags.optimize:
    raise SystemExit("Use unoptimized Python: omit -O/-OO and unset PYTHONOPTIMIZE")

import email
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tomllib
import venv

source, run, backend = (Path(p).resolve() for p in sys.argv[1:])
if run.is_relative_to(backend):
    raise SystemExit("NEW_RUN must be outside the setuptools backend directory")
assert source.is_dir() and backend.is_dir()
assert not run.exists() and not source.is_relative_to(run) and not run.is_relative_to(source)
run.mkdir()
commands = []
sha = lambda data: hashlib.sha256(data).hexdigest()


def execute(argv, cwd=run, required=True):
    result = subprocess.run(list(map(str, argv)), cwd=cwd, capture_output=True,
                            text=True, timeout=120)
    commands.append(dict(argv=list(map(str, argv)), cwd=str(cwd),
                         returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    (run / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    if required and result.returncode:
        raise RuntimeError(f'Command failed; preserve {run}/commands.json')
    return result


def environment(name, pip=False):
    prefix = run / name
    venv.EnvBuilder(with_pip=False).create(prefix)
    python = prefix / 'bin/python'
    if pip:
        execute([python, '-I', '-B', '-m', 'ensurepip', '--upgrade', '--default-pip'])
    site = prefix / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    shutil.copytree(backend, site, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    assert 'include-system-site-packages = false' in (prefix / 'pyvenv.cfg').read_text()
    return python


expected = {p.relative_to(source).as_posix(): p.read_bytes() for p in (source / 'hyphen').glob('*.py')}
assert len(expected) == 12, 'Review the expected runtime module inventory before changing it'
build_source = run / 'source'
shutil.copytree(source, build_source)
dist = run / 'dist'
dist.mkdir()
builder = environment('build-env')
execute([builder, '-I', '-B', '-c',
         'import setuptools.build_meta as b,sys; print(b.build_sdist(sys.argv[1]))', dist], build_source)
artifact, = dist.glob('*.tar.gz')
with tarfile.open(artifact) as archive:
    members = archive.getnames()
    root = members[0].split('/')[0]
    assert all(archive.extractfile(f'{root}/{name}').read() == data for name, data in expected.items())
    metadata = email.message_from_bytes(archive.extractfile(f'{root}/PKG-INFO').read())
project = tomllib.loads((source / 'pyproject.toml').read_text())['project']
assert sorted(metadata.get_all('Requires-Dist')) == sorted(project['dependencies'])
python = environment('install-env', pip=True)
execute([python, '-I', '-B', '-m', 'pip', 'install', '--no-index', '--no-cache-dir',
         '--no-deps', '--no-build-isolation', artifact])
smoke = execute([python, '-I', '-B', '-c', """import hashlib,importlib.metadata as m,json,pathlib,platform,sys
import hyphen
prefix=pathlib.Path(sys.prefix).resolve(); origin=pathlib.Path(hyphen.__file__).resolve()
assert origin.is_relative_to(prefix)
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in origin.parent.glob('*.py')}
for p in origin.parent.glob('*.py'): compile(p.read_bytes(),str(p),'exec')
d=m.distribution('hyphen')
print(json.dumps(dict(origin=str(origin),prefix=str(prefix),modules_sha256=files,
 python=sys.version,executable=sys.executable,platform=platform.platform(),
 version=d.version,requires=d.requires,entry_points=[str(e) for e in d.entry_points],
 distributions={d.metadata['Name'].lower():d.version for d in m.distributions()})))"""])
installed = json.loads(smoke.stdout)
assert set(installed['distributions']) == {'hyphen','pip','setuptools'}
assert installed['modules_sha256'] == {Path(n).name: sha(b) for n, b in expected.items()}
assert sorted(installed['requires']) == sorted(project['dependencies'])
check = execute([python, '-I', '-B', '-m', 'pip', 'check'], required=False)
report = dict(status='sdist file/install smoke passed; runtime dependencies not validated',
 artifact=str(artifact), artifact_sha256=sha(artifact.read_bytes()), members=members,
 metadata={k:metadata.get_all(k) for k in ['Name','Version','Requires-Python','Requires-Dist',
 'Classifier','Project-URL','License']}, installed=installed,
 pip_check=dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr),
 dependency_complete=False, source_modules_sha256={n:sha(b) for n,b in expected.items()})
(run / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
