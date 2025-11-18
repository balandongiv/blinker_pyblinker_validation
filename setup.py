from __future__ import annotations

from pathlib import Path
import tomllib
from setuptools import find_packages, setup


def load_project_metadata() -> tuple[str, str, str]:
    """Load project metadata from ``pyproject.toml``."""

    pyproject_path = Path(__file__).parent / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf8"))
    project = data.get("project", {})
    name = project.get("name", "blinker-pyblinker-validation")
    version = project.get("version", "0.1.10")
    description = project.get("description", "")
    return name, version, description


NAME, VERSION, DESCRIPTION = load_project_metadata()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
