"""Project-anchored paths, independent of the caller's working directory."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def project_path(value):
    path=Path(value).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def project_relative(value):
    path=project_path(value)
    try:return path.relative_to(ROOT).as_posix()
    except ValueError:return str(path)
