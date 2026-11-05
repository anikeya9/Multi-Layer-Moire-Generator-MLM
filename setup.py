from setuptools import find_packages, setup

setup(
    name="MLM",
    version="0.2.0",
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
        # the GUI runs in a Jupyter kernel (in a notebook, or as a web page through Voila)
        "gui": ["ipywidgets", "py3Dmol", "voila", "ipykernel"],
        "checks": ["spglib"],
        "notebooks": ["jupyterlab", "ipykernel", "nbconvert"],
        "dev": ["pytest", "ruff", "mypy", "pre-commit", "nbconvert", "ipykernel"],
        # everything a user needs: `pip install -e ".[all]"`
        "all": ["ipywidgets", "py3Dmol", "voila", "ipykernel", "spglib", "jupyterlab", "nbconvert", "pytest"],
    },
    # `mlm run|find|show|write|gui ...` (same as `python -m MLM ...`)
    entry_points={"console_scripts": ["mlm = MLM.cli:main"]},
)
