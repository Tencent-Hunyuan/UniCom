from pathlib import Path

from setuptools import find_packages, setup


def parse_requirements(filename: str):
    requirements = []
    try:
        with open(filename, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    except FileNotFoundError:
        pass
    return requirements


ROOT_DIR = Path(__file__).resolve().parent

try:
    with (ROOT_DIR / "README.md").open("r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""


setup(
    name="unicom",
    version="1.0.0",
    description="UniCom - Unified Multimodal Model with SigLIP2 Compressed Features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where=str(ROOT_DIR / "UniCom"), include=["unicom", "unicom.*", "utils", "utils.*"]),
    package_dir={"": "UniCom"},
    python_requires=">=3.10",
    install_requires=parse_requirements(str(ROOT_DIR / "requirements.txt")),
    include_package_data=True,
)
