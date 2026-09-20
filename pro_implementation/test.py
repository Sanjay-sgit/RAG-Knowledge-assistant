"""
Kept for backward compatibility - the smoke test now lives in tests/smoke_test.py.
Preferred (from the project root):  python -m tests.smoke_test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.smoke_test import main  # noqa: E402

if __name__ == "__main__":
    main()
