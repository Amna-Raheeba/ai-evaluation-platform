"""The FastAPI layer is optional: the package must work without it,
and asking for the app without it must produce a clear error."""
from __future__ import annotations

import importlib.util

import pytest

from eval_platform import api


def test_api_module_imports_without_fastapi():
    assert hasattr(api, "create_app")


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is not None,
                    reason="fastapi installed here; the missing-dependency path is CI-relevant only")
def test_create_app_without_fastapi_is_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="fastapi is not installed"):
        api.create_app(str(tmp_path / "x.db"))


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None,
                    reason="fastapi not installed")
def test_create_app_with_fastapi(tmp_path):
    app = api.create_app(str(tmp_path / "x.db"))
    assert app.title.startswith("AI Evaluation")
