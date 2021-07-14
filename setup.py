import pathlib

from setuptools import find_packages, setup

import versioneer


REPO_ROOT = pathlib.Path(__file__).parent

with open(REPO_ROOT / "README.md", encoding="utf-8") as f:
    README = f.read()

REQUIREMENTS = [
    "numpy",
    "scipy",
    "jupyter",
]

setup(
    name="pachinko",
    version=versioneer.get_version(),
    description="One-shot optimal decision-making agents and algorithms",
    long_description=README,
    long_description_content_type="text/markdown",
    author="datavaluepeople",
    author_email="opensource@datavaluepeople.com",
    url="https://github.com/datavaluepeople/pachinko",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=REQUIREMENTS,
    python_requires=">=3.7",
    cmdclass=versioneer.get_cmdclass(),
)
