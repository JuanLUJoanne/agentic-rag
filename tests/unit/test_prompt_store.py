"""
Tests for prompt versioning store (src/gateway/prompt_store.py).
"""
from __future__ import annotations

import pytest

from src.gateway.prompt_store import PromptStore


@pytest.fixture
def store():
    """Fresh in-memory store per test."""
    return PromptStore(db_path=":memory:")


class TestPromptStore:
    def test_save_increments_version(self, store):
        """First save returns 1, second returns 2."""
        v1 = store.save("my_prompt", "template one")
        v2 = store.save("my_prompt", "template two")
        assert v1 == 1
        assert v2 == 2

    def test_get_latest_returns_newest(self, store):
        """get(name) with no version arg returns the highest version."""
        store.save("p", "version 1 template")
        store.save("p", "version 2 template")
        result = store.get("p")
        assert result is not None
        assert result["version"] == 2
        assert result["template"] == "version 2 template"

    def test_get_specific_version(self, store):
        """get(name, version=1) returns v1 even after v2 is saved."""
        store.save("p", "first")
        store.save("p", "second")
        result = store.get("p", version=1)
        assert result is not None
        assert result["version"] == 1
        assert result["template"] == "first"

    def test_rollback_creates_new_version(self, store):
        """Rollback to v1 creates v3 with the same template as v1."""
        store.save("p", "original")
        store.save("p", "updated")
        rolled = store.rollback("p", version=1)
        assert rolled["version"] == 3
        assert rolled["template"] == "original"

    def test_list_versions_returns_all(self, store):
        """list_versions returns all saved versions in order."""
        store.save("p", "t1")
        store.save("p", "t2")
        store.save("p", "t3")
        versions = store.list_versions("p")
        assert len(versions) == 3
        assert [v["version"] for v in versions] == [1, 2, 3]

    def test_unknown_prompt_returns_none(self, store):
        """get returns None for a prompt name that has never been saved."""
        result = store.get("nonexistent_prompt")
        assert result is None

    def test_different_prompts_have_independent_versions(self, store):
        """Version numbers are independent per prompt name."""
        store.save("prompt_a", "a-v1")
        store.save("prompt_a", "a-v2")
        v = store.save("prompt_b", "b-v1")
        assert v == 1, "prompt_b should start at version 1 independently"

    def test_description_stored(self, store):
        """Description field is stored and retrievable."""
        store.save("p", "template", description="initial release")
        result = store.get("p")
        assert result is not None
        assert result["description"] == "initial release"
