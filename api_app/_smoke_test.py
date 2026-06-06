"""Lightweight API smoke test.

This test checks routing without loading MN-BERT weights. It is intended to run
quickly before a defense demo.
"""

from fastapi.testclient import TestClient

from api_app.main import app


def main() -> None:
    client = TestClient(app)
    root = client.get("/")
    assert root.status_code == 200, root.text
    health = client.get("/health")
    assert health.status_code == 200, health.text
    payload = health.json()
    assert payload["status"] == "ok"
    assert "model_paths_exist" in payload
    final_model = payload["final_model"]
    assert final_model["architecture"] == "flat"
    assert final_model["checkpoint"].endswith("best_mnbert_sota_corrected_model")
    assert final_model["checkpoint_exists"] is True
    assert final_model["max_length"] == 256
    assert final_model["source_prefix"] == "[news.mn] "
    assert final_model["reported_metrics"]["test_accuracy"] == 0.8326
    assert final_model["reported_metrics"]["test_macro_f1"] == 0.8062
    print("API smoke test passed")


if __name__ == "__main__":
    main()
