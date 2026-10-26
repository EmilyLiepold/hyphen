"""Path, confinement and process-table preflight; no artifact launcher yet."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def validate_roots(source, output, backend):
    """Return resolved paths for directory inputs and a fresh, disjoint output.

    The output's parent must exist. This function performs no writes; callers
    must keep the filesystem stable between validation and subsequent use.
    """
    requested_output = Path(output)
    try:
        source, output, backend = map(Path.resolve, map(Path, (source, output, backend)))
    except (OSError, RuntimeError) as error:
        raise ValueError("Cannot resolve source, output or backend") from error
    if not source.is_dir() or not backend.is_dir():
        raise ValueError("Source and backend must be existing directories")
    for name, directory in (("source", source), ("backend", backend)):
        if output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError(f"Output must be disjoint from {name}")
    if requested_output.is_symlink() or output.exists():
        raise ValueError("Output must not already exist, including as a symlink")
    if not output.parent.is_dir():
        raise ValueError("Output parent must be an existing directory")
    return source, output, backend


CANARY = """import errno,json,os,socket,sys
from pathlib import Path
work,source,backend,evidence=map(Path,sys.argv[1:])
work.write_text('allowed\\n')
checks={'work_write_allowed':work.read_text()=='allowed\\n'}
for name,path in [('source_write_denied',source),('backend_write_denied',backend),('evidence_write_denied',evidence)]:
 try:
  descriptor=os.open(path,os.O_WRONLY)
 except OSError as error:
  checks[name]=error.errno in (errno.EPERM,errno.EACCES)
 else:
  os.close(descriptor)
  checks[name]=False
try:
 with socket.create_connection(('127.0.0.1',9),timeout=1):
  checks['socket_denied']=False
except OSError as error:
 checks['socket_denied']=error.errno in (errno.EPERM,errno.EACCES)
print(json.dumps(checks))
"""
CANARY_KEYS = {'work_write_allowed', 'source_write_denied', 'backend_write_denied',
               'evidence_write_denied', 'socket_denied'}


def _unique_object(pairs):
    if len(pairs) != len(dict(pairs)):
        raise ValueError('Duplicate canary keys')
    return dict(pairs)


def confinement_preflight(source, output, backend):
    """Return a fresh work directory only after a strict macOS sandbox canary.

    Subprocess/protocol failures retain canary.json if it can be written. Root,
    platform and required-file refusals precede writes. A pass establishes neither a
    working process monitor nor permission to launch the artifact utility.
    """
    source, output, backend = validate_roots(source, output, backend)
    probes = [source / 'tools/sdist_smoke.py', backend / 'setuptools/__init__.py']
    if sys.platform != 'darwin' or not all(os.access(p, os.X_OK) for p in
                                         ('/usr/bin/sandbox-exec', sys.executable)):
        raise RuntimeError('macOS, executable sandbox-exec and Python are required')
    if not all(path.is_file() for path in probes):
        raise RuntimeError('Source smoke tool and backend setuptools module are required')
    output.mkdir()
    work = output / 'work'
    work.mkdir()
    protected = output / 'protected-canary'
    protected.write_text('preserve\n')
    profile = output / 'protection.sb'
    profile.write_text('(version 1)\n(allow default)\n(deny network*)\n'
                       '(deny file-write* (require-all (require-not (subpath '
                       + json.dumps(str(work)) + ')) (require-not (literal "/dev/null"))))\n')
    argv = ['/usr/bin/sandbox-exec', '-f', str(profile), sys.executable, '-I', '-B',
            '-c', CANARY, str(work / 'allowed-canary'), *map(str, probes), str(protected)]
    record = dict(status='failed', argv=argv, timeout_seconds=10, returncode=None,
                  stdout=None, stderr=None, error=None)
    started = time.monotonic()
    try:
        result = subprocess.run(argv, cwd=work, capture_output=True, text=True, timeout=10)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode != 0:
            raise RuntimeError('Canary process failed')
        checks = json.loads(result.stdout, object_pairs_hook=_unique_object)
        if (type(checks) is not dict or set(checks) != CANARY_KEYS or
                not all(type(value) is bool and value for value in checks.values())):
            raise ValueError('Canary requires exactly five Boolean true checks')
        record['status'] = 'passed'
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
        record['error'] = dict(type=type(error).__name__, reason=str(error))
        if isinstance(error, subprocess.TimeoutExpired):
            for name in ('stdout', 'stderr'):
                value = getattr(error, name)
                record[name] = value.decode(errors='replace') if isinstance(value, bytes) else value
    record['elapsed_seconds'] = time.monotonic() - started
    (output / 'canary.json').write_text(json.dumps(record, indent=2) + '\n')
    if record['status'] != 'passed':
        raise RuntimeError(f'Confinement preflight failed; preserve {output}/canary.json')
    return work


def _parse_process_table(stdout, self_pid, self_pgid):
    """Return [pid, ppid, pgid, rss] rows; ps reports RSS in KiB on macOS."""
    if type(stdout) is not str:
        raise ValueError('Process table must be text')
    rows, pids = [], set()
    for line in stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 4 or not all(field.isascii() and field.isdecimal() for field in fields):
            raise ValueError('Process table requires four unsigned decimal fields per row')
        row = list(map(int, fields))
        if row[0] in pids:
            raise ValueError('Process table contains duplicate PIDs')
        pids.add(row[0])
        rows.append(row)
    if not any(row[0] == self_pid and row[2] == self_pgid for row in rows):
        raise ValueError('Process table must include this process and its actual group')
    return rows


def process_table_preflight(output):
    """Record one bounded ps snapshot in an existing evidence directory.

    Return validated rows only on success. This prerequisite checks availability
    and parsing; it provides no continuous monitoring or resource enforcement.
    Evidence-writing errors propagate, including after an otherwise valid check.
    """
    argv = ['/bin/ps', '-axo', 'pid=,ppid=,pgid=,rss=']
    self_pid, self_pgid = os.getpid(), os.getpgrp()
    record = dict(status='failed', argv=argv, timeout_seconds=2, returncode=None,
                  stdout=None, stderr=None, error=None, self_pid=self_pid,
                  self_pgid=self_pgid, rss_unit='KiB')
    started = time.monotonic()
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=2)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode != 0:
            raise RuntimeError('Process-table command failed')
        rows = _parse_process_table(result.stdout, self_pid, self_pgid)
        record['status'] = 'passed'
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
        record['error'] = dict(type=type(error).__name__, reason=str(error))
        if isinstance(error, subprocess.TimeoutExpired):
            record.update(stdout=error.stdout, stderr=error.stderr)
    for name in ('stdout', 'stderr'):
        if isinstance(record[name], bytes):
            record[name] = record[name].decode(errors='replace')
    record['elapsed_seconds'] = time.monotonic() - started
    (Path(output) / 'monitor.json').write_text(json.dumps(record, indent=2) + '\n')
    if record['status'] != 'passed':
        raise RuntimeError(f'Process-table preflight failed; preserve {output}/monitor.json')
    return rows


def preflight(source, output, backend):
    """Return work only after confinement and process-table prerequisites pass.

    A passed canary.json alone does not establish complete preflight success.
    This sequence does not launch the artifact utility.
    """
    work = confinement_preflight(source, output, backend)
    process_table_preflight(work.parent)
    return work


if __name__ == "__main__":
    raise SystemExit("Artifact guard not implemented; this module provides preflight only")
