"""Project-anchored paths, independent of the caller's working directory."""
# 터미널 위치가 달라도 프로젝트 기준으로 같은 파일을 찾기 위한 경로 도우미입니다.
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


# 사용자 홈 기호를 확장하고 상대경로에는 프로젝트 루트를 붙입니다.
def project_path(value):
    path=Path(value).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


# 프로젝트 내부는 슬래시 형식 상대경로로, 외부 경로는 절대경로 문자열로 반환합니다.
def project_relative(value):
    path=project_path(value)
    try:return path.relative_to(ROOT).as_posix()
    except ValueError:return str(path)
