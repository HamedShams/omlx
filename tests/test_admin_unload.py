# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe admin-triggered model unload."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from omlx import server
from omlx.admin import routes as admin_routes


@pytest.mark.asyncio
async def test_active_model_unload_returns_accepted_until_quiescent():
    entry = MagicMock()
    entry.engine = object()
    entry.is_loading = False
    pool = MagicMock()
    pool.get_entry.return_value = entry
    pool.request_unload = AsyncMock(return_value=False)

    with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
        response = await admin_routes.unload_model("model-a", is_admin=True)

    assert response.status_code == 202
    assert json.loads(response.body) == {
        "status": "unloading",
        "model_id": "model-a",
        "message": "Aborting active requests before unloading model-a",
    }
    pool.request_unload.assert_awaited_once_with(
        "model-a", reason="manual admin unload"
    )


@pytest.mark.asyncio
async def test_idle_model_unload_returns_completed():
    entry = MagicMock()
    entry.engine = object()
    entry.is_loading = False
    pool = MagicMock()
    pool.get_entry.return_value = entry
    pool.request_unload = AsyncMock(return_value=True)

    with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
        response = await admin_routes.unload_model("model-a", is_admin=True)

    assert response == {
        "status": "ok",
        "model_id": "model-a",
        "message": "Unloaded model-a",
    }


@pytest.mark.asyncio
async def test_lease_rejected_during_manual_unload_uses_unload_error():
    pool = MagicMock()
    pool.get_abort_requested_reason.return_value = "manual admin unload"
    lease = server._LLMEngineLease(model_id="model-a")

    with (
        patch.object(server._server_state, "engine_pool", pool),
        pytest.raises(HTTPException) as exc_info,
    ):
        await server._raise_if_llm_lease_abort_requested(lease)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == (
        "Request aborted because this model is being unloaded."
    )


@pytest.mark.asyncio
async def test_public_unload_of_active_model_aborts_requests_first():
    # /v1/models/{id}/unload used to call _unload_engine directly, which
    # stopped the engine without finishing the active requests' collectors,
    # so a streaming client hung forever (#4378).
    entry = MagicMock()
    entry.engine = object()
    entry.is_loading = False
    pool = MagicMock()
    pool.get_entry.return_value = entry
    pool.request_unload = AsyncMock(return_value=False)

    with patch.object(server._server_state, "engine_pool", pool):
        response = await server.unload_model("model-a", _=True)

    assert response.status_code == 202
    assert json.loads(response.body) == {
        "status": "unloading",
        "model_id": "model-a",
        "message": "Aborting active requests before unloading model-a",
    }
    pool.request_unload.assert_awaited_once_with("model-a", reason="manual unload")
    pool._unload_engine.assert_not_called()


@pytest.mark.asyncio
async def test_public_unload_of_idle_model_returns_ok():
    entry = MagicMock()
    entry.engine = object()
    entry.is_loading = False
    pool = MagicMock()
    pool.get_entry.return_value = entry
    pool.request_unload = AsyncMock(return_value=True)

    with patch.object(server._server_state, "engine_pool", pool):
        response = await server.unload_model("model-a", _=True)

    assert response == {"status": "ok", "model_id": "model-a"}
    pool.request_unload.assert_awaited_once_with("model-a", reason="manual unload")


@pytest.mark.asyncio
async def test_public_unload_rejects_a_model_that_is_still_loading():
    entry = MagicMock()
    entry.engine = object()
    entry.is_loading = True
    pool = MagicMock()
    pool.get_entry.return_value = entry
    pool.request_unload = AsyncMock()

    with (
        patch.object(server._server_state, "engine_pool", pool),
        pytest.raises(HTTPException) as exc_info,
    ):
        await server.unload_model("model-a", _=True)

    assert exc_info.value.status_code == 409
    pool.request_unload.assert_not_awaited()
