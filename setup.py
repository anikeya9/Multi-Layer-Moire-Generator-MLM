from setuptools import setup, find_packages

setup(
    name="MLM",
    version="0.1",
    author="Anikeya Aditya, Sampad Mohanty",
    license="MIT",
    url="https://github.com/anikeya9/Multi-Layer-Moire-Generator-MLM",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)

