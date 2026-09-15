"""Importing spherical must not contact remote services.

``astroquery.gaia`` queries the Gaia archive status when it is imported and prints the
response, so a module-level import made every spherical import, and every worker process,
hit the network (#158). Checked in a fresh interpreter, since this test session may already
have imported it.
"""
from __future__ import annotations

import subprocess
import sys

_CHECK = """
import importlib, pkgutil, sys
import spherical.database
for info in pkgutil.iter_modules(spherical.database.__path__, "spherical.database."):
    importlib.import_module(info.name)
print(sorted(m for m in sys.modules if m.startswith("astroquery.gaia")))
"""


def test_importing_the_database_does_not_import_astroquery_gaia():
    result = subprocess.run([sys.executable, "-c", _CHECK], capture_output=True, text=True, check=True)
    assert result.stdout.strip().splitlines()[-1] == "[]", result.stdout
