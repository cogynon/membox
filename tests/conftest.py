"""Shared test fixtures."""

import os
import tempfile

import pytest


@pytest.fixture
def tmp_db():
    """Provide a temporary database path, cleaned up after the test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for ext in ("", "-wal", "-shm"):
        try:
            os.remove(path + ext)
        except FileNotFoundError:
            pass
