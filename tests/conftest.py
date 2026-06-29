"""
pytest session-level configuration for test reporting.

- Ensures the reports/ directory tree exists before any test runs.
- Prints a DeepEval report summary after the full session finishes,
  listing every JSON file saved to reports/deepeval/.
"""

import os
import json
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
DEEPEVAL_REPORTS_DIR = REPORTS_DIR / "deepeval"


def pytest_configure(config):
    """Create the reports directories before test collection starts."""
    DEEPEVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def pytest_sessionfinish(session, exitstatus):
    """After all tests finish, print a summary of any DeepEval JSON reports."""
    if not DEEPEVAL_REPORTS_DIR.is_dir():
        return

    json_files = sorted(DEEPEVAL_REPORTS_DIR.glob("*.json"))
    if not json_files:
        return

    print(f"\n{'=' * 62}")
    print("  DeepEval Evaluation Reports")
    print(f"  Saved to: {DEEPEVAL_REPORTS_DIR}")
    print(f"{'=' * 62}")

    for path in json_files:
        print(f"\n  📄 {path.name}")
        try:
            data = json.loads(path.read_text())
            test_results = data.get(
                "testResults", data.get("test_results", []))
            for tr in test_results:
                name = tr.get("name", "unnamed")
                success = tr.get("success", None)
                icon = "✅" if success else "❌"
                print(f"     {icon}  {name}")
                for md in tr.get("metricsData", tr.get("metrics_data", [])):
                    mname = md.get("name", "")
                    score = md.get("score", "?")
                    threshold = md.get("threshold", "?")
                    ok = md.get("success", None)
                    mk = "✓" if ok else "✗"
                    reason = md.get("reason") or ""
                    print(
                        f"          {mk}  {mname}: score={score:.2f} threshold={threshold}  {reason}")
        except Exception as exc:
            print(f"     (could not parse: {exc})")

    print(f"\n{'=' * 62}\n")
