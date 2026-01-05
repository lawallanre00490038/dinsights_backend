import sys
import types
import json
from types import SimpleNamespace
from fastapi.testclient import TestClient

# Provide a lightweight fake `duckdb` module for tests so imports don't fail in this environment
import pandas as pd
fake_duck = types.SimpleNamespace(connect=lambda path: types.SimpleNamespace(execute=lambda q: types.SimpleNamespace(df=lambda: pd.DataFrame())))
sys.modules['duckdb'] = fake_duck

import main


def test_upload_and_session_persistence(monkeypatch):
    client = TestClient(main.app)
    session_id = "sess-1"

    # Mock S3 download and store_file behavior
    monkeypatch.setattr(main, "download_file_from_s3", lambda url: b"col1,col2\n1,2\n")
    monkeypatch.setattr(main, "store_file_and_register_dataset", lambda file_like: "dataset_123")

    # Mock graph.invoke to return plots but no AIMessage (so message_content falls back)
    def fake_invoke(state):
        return {"messages": [], "plots": [{"type": "chart", "id": "p1"}]}

    monkeypatch.setattr(main, "graph", SimpleNamespace(invoke=fake_invoke))

    payload = {
        "user_query": "Analyze",
        "session_id": session_id,
        "input_data": [{"data_path": "s3://bucket/file.csv"}]
    }

    r = client.post("/api/chat", json=payload)
    assert r.status_code == 200
    data = r.json()

    assert data["sessionId"] == session_id
    # ensure session store was updated
    assert main.SESSION_STORE.get(session_id)["dataset_id"] == "dataset_123"

    # messages shape
    assert len(data["messages"]) == 2
    user = data["messages"][0]
    assistant = data["messages"][1]

    assert set(user.keys()) == {"id", "role", "content", "createdAt"}
    assert assistant.get("charts") == [{"type": "chart", "id": "p1"}]


def test_restore_dataset_without_input(monkeypatch):
    client = TestClient(main.app)
    session_id = "sess-2"

    # pre-populate session store
    main.SESSION_STORE[session_id] = {"dataset_id": "dataset_456", "updated_at": "now"}

    invoked = {}

    def fake_invoke(state):
        invoked["state"] = state
        return {"messages": [], "plots": []}

    monkeypatch.setattr(main, "graph", SimpleNamespace(invoke=fake_invoke))

    payload = {"user_query": "Status?", "session_id": session_id}
    r = client.post("/api/chat", json=payload)
    assert r.status_code == 200

    # graph.invoke should receive the restored dataset_id
    assert invoked.get("state") is not None
    assert invoked["state"].get("dataset_id") == "dataset_456"
