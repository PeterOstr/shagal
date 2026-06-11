import json
import sys
import types
from unittest.mock import MagicMock

import pytest


def _build_mock_module(name, attrs=None):
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod


@pytest.fixture(scope="session", autouse=True)
def _mock_heavy_deps():
    modules = {}

    mock_langchain_agents = _build_mock_module("langchain.agents")
    mock_langchain_agents.create_agent = MagicMock()

    mock_langchain_chat = _build_mock_module("langchain.chat_models")
    mock_langchain_chat.init_chat_model = MagicMock()

    mock_langchain_core_documents = _build_mock_module("langchain_core.documents")
    mock_langchain_core_documents.Document = MagicMock()

    mock_langchain_openai = _build_mock_module("langchain_openai")
    mock_langchain_openai.OpenAIEmbeddings = MagicMock()

    mock_langchain_qdrant = _build_mock_module("langchain_qdrant")
    mock_langchain_qdrant.QdrantVectorStore = MagicMock()

    mock_qdrant_client = _build_mock_module("qdrant_client")
    mock_qdrant_client.QdrantClient = MagicMock()

    mock_qdrant_models = _build_mock_module("qdrant_client.models")
    mock_qdrant_models.Distance = MagicMock()
    mock_qdrant_models.VectorParams = MagicMock()

    mock_clickhouse = _build_mock_module("clickhouse_connect")
    mock_clickhouse.get_client = MagicMock()

    mock_langsmith = _build_mock_module("langsmith")

    mock_bs4 = _build_mock_module("bs4")
    mock_bs4.BeautifulSoup = MagicMock()

    mock_pandas = _build_mock_module("pandas")
    mock_pandas.DataFrame = MagicMock()

    mock_dotenv = _build_mock_module("dotenv")
    mock_dotenv.load_dotenv = MagicMock()

    mock_tqdm = _build_mock_module("tqdm")
    mock_tqdm.tqdm = MagicMock()

    mock_mailbox = _build_mock_module("mailbox")
    mock_mailbox.mbox = MagicMock()

    mock_langchain_agents_middleware = _build_mock_module("langchain.agents.middleware")
    mock_langchain_agents_middleware.PIIMiddleware = MagicMock()
    mock_langchain_agents_middleware.SummarizationMiddleware = MagicMock()

    mock_langchain_tools = _build_mock_module("langchain.tools")
    mock_langchain_tools.tool = lambda f: f

    mock_langchain_agents_tool_calling = _build_mock_module("langchain.agents.tool_calling_agent")

    installed = [
        ("langchain", _build_mock_module("langchain")),
        ("langchain_core", _build_mock_module("langchain_core")),
        ("langchain_openai", mock_langchain_openai),
        ("langchain_qdrant", mock_langchain_qdrant),
        ("qdrant_client", mock_qdrant_client),
        ("qdrant_client.models", mock_qdrant_models),
        ("clickhouse_connect", mock_clickhouse),
        ("bs4", mock_bs4),
        ("pandas", mock_pandas),
        ("dotenv", mock_dotenv),
        ("tqdm", mock_tqdm),
        ("mailbox", mock_mailbox),
        ("langchain.agents", mock_langchain_agents),
        ("langchain.agents.middleware", mock_langchain_agents_middleware),
        ("langchain.agents.tool_calling_agent", mock_langchain_agents_tool_calling),
        ("langchain.tools", mock_langchain_tools),
        ("langchain.chat_models", mock_langchain_chat),
        ("langchain_core.documents", mock_langchain_core_documents),
        ("langsmith", mock_langsmith),
    ]

    for name, mod in installed:
        if name not in sys.modules:
            modules[name] = mod
            sys.modules[name] = mod

    yield

    for name in modules:
        sys.modules.pop(name, None)


def _make_tool(func):
    """Wrap a function so it looks like a LangChain tool with .invoke()."""
    obj = MagicMock()
    obj.invoke = func
    return obj


@pytest.fixture
def client():
    from app import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_pipeline(monkeypatch):
    calls = {}

    def mock_import():
        calls["import_mbox"] = True

    def mock_dedup():
        calls["dedup"] = True

    def mock_clean(fetch_batch=30, llm_batch=5):
        calls["clean"] = {"fetch_batch": fetch_batch, "llm_batch": llm_batch}

    def mock_parse(limit=50, batch_size=3, max_workers=6):
        calls["parse"] = {"limit": limit, "batch_size": batch_size, "max_workers": max_workers}
        return {"success_count": 5, "error_count": 0, "errors": []}

    def mock_index(batch_size=1000, recreate=False):
        calls["index"] = {"batch_size": batch_size, "recreate": recreate}

    monkeypatch.setattr("app.import_mbox_to_clickhouse", mock_import)
    monkeypatch.setattr("app.deduplicate_emails", mock_dedup)
    monkeypatch.setattr("app.clean_email_bodies_from_db", mock_clean)
    monkeypatch.setattr("app.parse_emails_from_db", mock_parse)
    monkeypatch.setattr("app.index_messages", mock_index)

    return calls


@pytest.fixture
def mock_retrieval(monkeypatch):
    calls = {}

    def mock_search_invoke(payload):
        calls["search_threads"] = payload
        return json.dumps({"threads": [], "project_hint": payload["project_hint"]})

    def mock_corpus_invoke(payload):
        calls["corpus_batch"] = payload
        return json.dumps({"batch": [], "project_hint": payload["project_hint"], "has_more": False})

    def mock_batch_analysis(project_hint):
        calls["batch_analysis"] = project_hint
        return "batch summary done"

    def mock_global_analysis(project_hint):
        calls["global_analysis"] = project_hint
        return "global report done"

    def mock_clear():
        calls["clear_summaries"] = True

    monkeypatch.setattr("app.search_project_threads", _make_tool(mock_search_invoke))
    monkeypatch.setattr("app.get_project_corpus_batch", _make_tool(mock_corpus_invoke))
    monkeypatch.setattr("app.run_batch_analysis", mock_batch_analysis)
    monkeypatch.setattr("app.run_global_analysis", mock_global_analysis)
    monkeypatch.setattr("app.clear_summaries", mock_clear)

    return calls
