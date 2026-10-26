import sqlite3

import pymupdf
import sqlite_vec

from librarian import __version__
from librarian.server import create_server


def test_package_imports():
    assert __version__ == "0.1.0"
    assert create_server().name == "paper-librarian"


def test_pymupdf_available():
    assert pymupdf.open() is not None


def test_sqlite_vec_loads():
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    (version,) = db.execute("select vec_version()").fetchone()
    assert version.startswith("v")
