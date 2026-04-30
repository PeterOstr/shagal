import argparse

from .agents import build_batch_agent, build_global_agent
from .config import DEFAULT_AGENT_MODEL, DEFAULT_INGEST_BATCH_SIZE
from .ingestion import ingest_messages, ingest_threads, load_all_joined_df
from .infra import get_embedding_dim
from .tools import clear_summaries, get_project_corpus_batch, search_project_threads


def cmd_info(_: argparse.Namespace) -> None:
    print("Embedding dim:", get_embedding_dim())


def cmd_ingest_messages(args: argparse.Namespace) -> None:
    ingest_messages(batch_size=args.batch_size, recreate=args.recreate)


def cmd_ingest_threads(args: argparse.Namespace) -> None:
    ingest_threads(recreate=args.recreate)


def cmd_ingest_all(args: argparse.Namespace) -> None:
    ingest_messages(batch_size=args.batch_size, recreate=args.recreate)
    ingest_threads(recreate=args.recreate)


def cmd_df_test(_: argparse.Namespace) -> None:
    df = load_all_joined_df()
    print(df.columns.tolist())
    print(df[["id", "thread_key", "subject"]].head())


def cmd_search_threads(args: argparse.Namespace) -> None:
    result = search_project_threads.invoke(
        {"project_hint": args.project_hint, "limit": args.limit}
    )
    print(result)


def cmd_get_corpus_batch(args: argparse.Namespace) -> None:
    result = get_project_corpus_batch.invoke(
        {
            "project_hint": args.project_hint,
            "offset": args.offset,
            "batch_size": args.batch_size,
            "thread_limit": args.thread_limit,
        }
    )
    print(result)


def cmd_run_batch_agent(args: argparse.Namespace) -> None:
    agent = build_batch_agent(model=args.model)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Начни обработку проекта {args.project_hint} батчами. "
                        "Делай summary каждого батча и сохраняй их через save_summary. "
                        "Итоговый отчёт не делай."
                    ),
                }
            ]
        }
    )
    print(result["messages"][-1].content)


def cmd_run_global_agent(args: argparse.Namespace) -> None:
    agent = build_global_agent(model=args.model)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Сделай итоговый отчёт по проекту {args.project_hint} "
                        "на основе сохранённых batch summaries."
                    ),
                }
            ]
        }
    )
    print(result["messages"][-1].content)


def cmd_clear_summaries(_: argparse.Namespace) -> None:
    clear_summaries()
    print("Old summaries cleared.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mailkb MVP CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("info")
    p.set_defaults(func=cmd_info)

    p = subparsers.add_parser("ingest-messages")
    p.add_argument("--batch-size", type=int, default=DEFAULT_INGEST_BATCH_SIZE)
    p.add_argument("--recreate", action="store_true")
    p.set_defaults(func=cmd_ingest_messages)

    p = subparsers.add_parser("ingest-threads")
    p.add_argument("--recreate", action="store_true")
    p.set_defaults(func=cmd_ingest_threads)

    p = subparsers.add_parser("ingest-all")
    p.add_argument("--batch-size", type=int, default=DEFAULT_INGEST_BATCH_SIZE)
    p.add_argument("--recreate", action="store_true")
    p.set_defaults(func=cmd_ingest_all)

    p = subparsers.add_parser("df-test")
    p.set_defaults(func=cmd_df_test)

    p = subparsers.add_parser("search-threads")
    p.add_argument("project_hint", type=str)
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_search_threads)

    p = subparsers.add_parser("get-corpus-batch")
    p.add_argument("project_hint", type=str)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--thread-limit", type=int, default=10)
    p.set_defaults(func=cmd_get_corpus_batch)

    p = subparsers.add_parser("run-batch-agent")
    p.add_argument("project_hint", type=str)
    p.add_argument("--model", type=str, default=DEFAULT_AGENT_MODEL)
    p.set_defaults(func=cmd_run_batch_agent)

    p = subparsers.add_parser("run-global-agent")
    p.add_argument("project_hint", type=str)
    p.add_argument("--model", type=str, default=DEFAULT_AGENT_MODEL)
    p.set_defaults(func=cmd_run_global_agent)

    p = subparsers.add_parser("clear-summaries")
    p.set_defaults(func=cmd_clear_summaries)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

