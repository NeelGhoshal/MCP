"""Swap in pysqlite3 where the system Python's sqlite3 can't load extensions (e.g. Render's Python),
since sqlite-vec is a loadable extension. Import this before anything that imports sqlite3."""

import sqlite3
import sys

if not hasattr(sqlite3.Connection, "enable_load_extension"):
    import pysqlite3

    sys.modules["sqlite3"] = pysqlite3
