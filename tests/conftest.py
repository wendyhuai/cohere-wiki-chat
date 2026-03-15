"""Shared fixtures for tests."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    """Ensure a fake API key is set so Settings can be constructed."""
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setenv("COHERE_MODEL", "command-r7b-12-2024")
