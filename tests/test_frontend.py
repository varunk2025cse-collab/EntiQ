"""Frontend static checks: every ES module must parse (catches bracket errors before a page fails in the browser)."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = sorted((ROOT / "frontend" / "js").rglob("*.js"))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("path", JS, ids=[p.name for p in JS])
def test_module_parses(path):
    r = subprocess.run(["node", "--input-type=module", "--check"], input=path.read_text(encoding="utf-8"),
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr


def test_no_innerhtml_with_data():
    """UI renders data via textContent only (XSS safety)."""
    import re

    for p in JS:
        assert not re.search(r"\.(innerHTML|outerHTML)\s*=|insertAdjacentHTML", p.read_text(encoding="utf-8")), p.name
