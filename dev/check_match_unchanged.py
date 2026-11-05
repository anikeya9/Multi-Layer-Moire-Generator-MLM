"""Fail if src/MLM/match.py differs from the reference implementation (it must not change).

Used by pre-commit and CI. The hash is that of match.py on the original `dev` branch.
"""

import hashlib
import sys
from pathlib import Path

REFERENCE_SHA256 = "6b278a1b24bb41323b7a0a01a3b8072164fe5fc534207fd3f29b2e19342137d7"

path = Path(__file__).resolve().parents[1] / "src" / "MLM" / "match.py"
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != REFERENCE_SHA256:
    sys.exit(f"{path} was modified (sha256 {actual}); it is the reference implementation and must not change.")
print("match.py unchanged")
