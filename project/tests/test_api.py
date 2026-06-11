def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root_redirect(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (200, 307, 404)


def test_openapi_json(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    data = resp.json()
    assert data["info"]["title"] == "mailkb backend"
    assert "/health" in str(data["paths"])
    assert "/pipeline/init-db" in str(data["paths"])


def test_cors_headers(client):
    resp = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"


def test_method_not_allowed(client):
    resp = client.put("/health")
    assert resp.status_code == 405 or resp.status_code == 405


class TestInit:
    def test_init_db(self, client):
        resp = client.post("/pipeline/init-db")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "results" in data
        for r in data["results"]:
            assert r["status"] == "ok", f"{r['file']}: {r['status']}"

    def test_init_db_wrong_method(self, client):
        resp = client.get("/pipeline/init-db")
        assert resp.status_code == 405


class TestPipeline:
    def test_import_mbox(self, client, mock_pipeline):
        resp = client.post("/pipeline/import-mbox")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert mock_pipeline["import_mbox"] is True

    def test_import_mbox_twice(self, client, mock_pipeline):
        client.post("/pipeline/import-mbox")
        resp = client.post("/pipeline/import-mbox")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_dedup(self, client, mock_pipeline):
        resp = client.post("/pipeline/dedup")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert mock_pipeline["dedup"] is True

    def test_clean_bodies_defaults(self, client, mock_pipeline):
        resp = client.post("/pipeline/clean-bodies", json={})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert mock_pipeline["clean"] == {"fetch_batch": 30, "llm_batch": 5}

    def test_clean_bodies_custom(self, client, mock_pipeline):
        resp = client.post("/pipeline/clean-bodies", json={"fetch_batch": 10, "llm_batch": 2})
        assert resp.status_code == 200
        assert mock_pipeline["clean"] == {"fetch_batch": 10, "llm_batch": 2}

    def test_clean_bodies_zero_values(self, client, mock_pipeline):
        resp = client.post("/pipeline/clean-bodies", json={"fetch_batch": 0, "llm_batch": 0})
        assert resp.status_code == 200
        assert mock_pipeline["clean"] == {"fetch_batch": 0, "llm_batch": 0}

    def test_clean_bodies_negative_values(self, client, mock_pipeline):
        resp = client.post("/pipeline/clean-bodies", json={"fetch_batch": -1, "llm_batch": -5})
        assert resp.status_code == 200
        assert mock_pipeline["clean"] == {"fetch_batch": -1, "llm_batch": -5}

    def test_parse_defaults(self, client, mock_pipeline):
        resp = client.post("/pipeline/parse", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["result"]["success_count"] == 5
        assert mock_pipeline["parse"] == {"limit": 50, "batch_size": 3, "max_workers": 6}

    def test_parse_custom(self, client, mock_pipeline):
        resp = client.post("/pipeline/parse", json={"limit": 10, "batch_size": 5, "max_workers": 2})
        assert resp.status_code == 200
        assert mock_pipeline["parse"] == {"limit": 10, "batch_size": 5, "max_workers": 2}

    def test_parse_large_values(self, client, mock_pipeline):
        resp = client.post("/pipeline/parse", json={"limit": 99999, "batch_size": 100, "max_workers": 50})
        assert resp.status_code == 200
        assert mock_pipeline["parse"] == {"limit": 99999, "batch_size": 100, "max_workers": 50}

    def test_parse_empty_body(self, client):
        resp = client.post("/pipeline/parse", content=b"{}", headers={"Content-Type": "application/json"})
        assert resp.status_code == 200

    def test_index_messages_defaults(self, client, mock_pipeline):
        resp = client.post("/pipeline/index-messages", json={})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert mock_pipeline["index"] == {"batch_size": 1000, "recreate": False}

    def test_index_messages_custom(self, client, mock_pipeline):
        resp = client.post("/pipeline/index-messages", json={"batch_size": 500, "recreate": True})
        assert resp.status_code == 200
        assert mock_pipeline["index"] == {"batch_size": 500, "recreate": True}

    def test_index_messages_recreate_false(self, client, mock_pipeline):
        resp = client.post("/pipeline/index-messages", json={"batch_size": 100, "recreate": False})
        assert resp.status_code == 200
        assert mock_pipeline["index"]["recreate"] is False


class TestSearch:
    def test_search_threads(self, client, mock_retrieval):
        resp = client.post("/search/threads", json={"project_hint": "segezha"})
        assert resp.status_code == 200
        assert resp.json() == {"threads": [], "project_hint": "segezha"}
        assert mock_retrieval["search_threads"] == {"project_hint": "segezha", "limit": 10}

    def test_search_threads_custom_limit(self, client, mock_retrieval):
        resp = client.post("/search/threads", json={"project_hint": "sibur", "limit": 5})
        assert resp.status_code == 200
        assert mock_retrieval["search_threads"]["limit"] == 5

    def test_search_threads_zero_limit(self, client, mock_retrieval):
        resp = client.post("/search/threads", json={"project_hint": "segezha", "limit": 0})
        assert resp.status_code == 200
        assert mock_retrieval["search_threads"]["limit"] == 0

    def test_search_threads_empty_hint_value(self, client, mock_retrieval):
        resp = client.post("/search/threads", json={"project_hint": ""})
        assert resp.status_code == 200

    def test_corpus_batch(self, client, mock_retrieval):
        resp = client.post("/search/corpus-batch", json={"project_hint": "segezha"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_hint"] == "segezha"
        assert data["has_more"] is False
        assert mock_retrieval["corpus_batch"]["project_hint"] == "segezha"

    def test_corpus_batch_custom(self, client, mock_retrieval):
        resp = client.post(
            "/search/corpus-batch",
            json={"project_hint": "sibur", "offset": 10, "batch_size": 20, "thread_limit": 15},
        )
        assert resp.status_code == 200
        p = mock_retrieval["corpus_batch"]
        assert p["project_hint"] == "sibur"
        assert p["offset"] == 10
        assert p["batch_size"] == 20
        assert p["thread_limit"] == 15

    def test_corpus_batch_zero_offset(self, client, mock_retrieval):
        resp = client.post("/search/corpus-batch", json={"project_hint": "test", "offset": 0})
        assert resp.status_code == 200
        assert mock_retrieval["corpus_batch"]["offset"] == 0


class TestAnalysis:
    def test_batch_analysis(self, client, mock_retrieval):
        resp = client.post("/analysis/batch", json={"project_hint": "segezha"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["result"] == "batch summary done"
        assert mock_retrieval["batch_analysis"] == "segezha"

    def test_global_analysis(self, client, mock_retrieval):
        resp = client.post("/analysis/global", json={"project_hint": "sibur"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["result"] == "global report done"
        assert mock_retrieval["global_analysis"] == "sibur"

    def test_batch_analysis_long_name(self, client, mock_retrieval):
        long_name = "a" * 1000
        resp = client.post("/analysis/batch", json={"project_hint": long_name})
        assert resp.status_code == 200
        assert mock_retrieval["batch_analysis"] == long_name

    def test_global_analysis_unicode(self, client, mock_retrieval):
        resp = client.post("/analysis/global", json={"project_hint": "проект"})
        assert resp.status_code == 200
        assert mock_retrieval["global_analysis"] == "проект"

    def test_clear_summaries(self, client, mock_retrieval):
        resp = client.post("/summaries/clear")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert mock_retrieval["clear_summaries"] is True

    def test_clear_summaries_idempotent(self, client, mock_retrieval):
        client.post("/summaries/clear")
        resp = client.post("/summaries/clear")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestValidation:
    def test_search_threads_missing_hint(self, client):
        resp = client.post("/search/threads", json={})
        assert resp.status_code == 422

    def test_search_threads_wrong_type(self, client):
        resp = client.post("/search/threads", json={"project_hint": 123})
        assert resp.status_code == 422

    def test_search_threads_null_hint(self, client):
        resp = client.post("/search/threads", json={"project_hint": None})
        assert resp.status_code == 422

    def test_analysis_missing_hint(self, client):
        resp = client.post("/analysis/batch", json={})
        assert resp.status_code == 422

    def test_analysis_extra_fields(self, client, mock_retrieval):
        resp = client.post("/analysis/batch", json={"project_hint": "test", "extra": "ignored"})
        assert resp.status_code == 200
        assert mock_retrieval["batch_analysis"] == "test"

    def test_corpus_batch_missing_hint(self, client):
        resp = client.post("/search/corpus-batch", json={})
        assert resp.status_code == 422

    def test_corpus_batch_negative_offset(self, client, mock_retrieval):
        resp = client.post("/search/corpus-batch", json={"project_hint": "test", "offset": -1})
        assert resp.status_code == 200

    def test_parse_invalid_json(self, client):
        resp = client.post("/pipeline/parse", content=b"not json", headers={"Content-Type": "application/json"})
        assert resp.status_code == 422

    def test_clean_bodies_string_instead_of_number(self, client):
        resp = client.post("/pipeline/clean-bodies", json={"fetch_batch": "abc", "llm_batch": 5})
        assert resp.status_code == 422

    def test_index_messages_string_recreate(self, client, mock_pipeline):
        resp = client.post("/pipeline/index-messages", json={"recreate": "yes"})
        assert resp.status_code == 200
        assert mock_pipeline["index"]["recreate"] is True


class TestInitEdgeCases:
    def test_init_db_no_sql_dir(self, client, monkeypatch):
        class FakePath(str):
            def is_dir(self): return False
            def glob(self, pat): return []
            def __truediv__(self, other): return FakePath(f"{self}/{other}")
        monkeypatch.setattr("app.SQL_DIR", FakePath("nonexistent_dir"))
        monkeypatch.setattr("app.Path", FakePath)
        resp = client.post("/pipeline/init-db")
        assert resp.status_code == 404

    def test_init_db_sql_error(self, client, monkeypatch):
        def mock_run(path):
            return {"file": path.name, "status": "error: syntax error"}

        monkeypatch.setattr("app._run_sql_file", mock_run)
        from pathlib import Path
        monkeypatch.setattr("app.Path.glob", lambda self, pat: [Path("test.sql")])
        monkeypatch.setattr("app.Path.is_dir", lambda self: True)
        resp = client.post("/pipeline/init-db")
        assert resp.status_code == 200
        data = resp.json()
        assert any(r["status"].startswith("error") for r in data["results"])
