from pathlib import Path
from setuptools import setup

CWD = Path(__file__).absolute().parent


def get_version():
    # Gets the version
    path = CWD / "corridor_grid" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")

setup(
    name="corridor_grid",
    version=get_version(),
    description="A set of corridor environments.",
)
