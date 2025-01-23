import pathlib
import subprocess
import sys

import toml

dirname = pathlib.Path(__file__).resolve().parent

print(f"load {dirname.parent.parent / 'pyproject.toml'}")
with open(dirname.parent.parent / "pyproject.toml") as pyproject_file:
    pyproject = toml.load(pyproject_file)

dependencies = []
if "project" in pyproject:
    project = pyproject["project"]
    if "dependencies" in project:
        dependencies += project["dependencies"]
    if "optional-dependencies" in project:
        optional_dependencies = project["optional-dependencies"]
        if "tests" in optional_dependencies:
            dependencies += optional_dependencies["tests"]

if len(dependencies) > 0:
    subprocess.run([sys.executable, "-m", "pip", "install"] + dependencies, check=True)
