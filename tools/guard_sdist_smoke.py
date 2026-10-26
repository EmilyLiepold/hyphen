"""Path preflight for a future macOS sdist guard; no artifact launcher yet."""
from pathlib import Path


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


if __name__ == "__main__":
    raise SystemExit("Artifact guard not implemented; this module provides path preflight only")
