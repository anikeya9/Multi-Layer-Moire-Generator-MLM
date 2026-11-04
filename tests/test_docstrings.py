"""Run the docstring examples of the new modules, so the documentation stays correct.

Examples that need a run directory or a long search are marked ``# doctest: +SKIP``.
"""

import doctest
import importlib

import pytest

MODULES = ["MLM.io", "MLM.structure", "MLM.strain", "MLM.search", "MLM.index", "MLM.spec", "MLM.build", "MLM.cli"]


@pytest.mark.parametrize("name", MODULES)
def test_docstring_examples(name):
    module = importlib.import_module(name)
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    result = doctest.testmod(module, optionflags=flags, verbose=False)
    assert result.failed == 0, f"{result.failed} docstring example(s) failed in {name}"
