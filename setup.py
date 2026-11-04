from setuptools import setup, find_packages

setup(
    name="MLM",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    # Runtime dependencies. Keep in sync with requirements.txt.
    install_requires=[
        "numpy",
        "pandas",
        "polars",
        "numba",
        "tqdm",
        "icecream",
        "ase",
        "matplotlib",
        "scipy",  # optional speed-up used by structure_writer.select_atoms_fractional
        "pyyaml",
    ],
    extras_require={
        "test": ["pytest"],
        "gui": ["ipywidgets", "py3Dmol", "voila"],
        "checks": ["spglib"],
        "dev": ["pytest", "ruff", "mypy", "pre-commit", "nbconvert", "ipykernel"],
    },
    # `mlm run|find|show|write|gui ...` (same as `python -m MLM ...`)
    entry_points={"console_scripts": ["mlm = MLM.cli:main"]},
)
