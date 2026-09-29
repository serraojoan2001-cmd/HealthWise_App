"""
===============================================================================
 FILE : tests/smoke_test.py   (optional check - run:  python tests/smoke_test.py)
 ROLE : Opens every dashboard page in Streamlit's headless test runner and
        fails loudly if any page raises an error.
===============================================================================
"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["boardroom_brief", "data_detective", "risk_sorter", "price_engine",
         "price_lab", "batch_scorer", "guardrails"]

failed = False
for name in PAGES:
    script = (f"import sys; sys.path.insert(0, r'{ROOT}')\n"
              f"from views import {name} as m\nm.render()\n")
    at = AppTest.from_string(script, default_timeout=90).run()
    ok = not at.exception
    failed |= not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    for e in at.exception:
        print("      ", e.value)
sys.exit(1 if failed else 0)
