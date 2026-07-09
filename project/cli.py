import argparse

from pipeline import (
    clean_email_bodies_from_db,
    deduplicate_emails,
    import_mbox_to_clickhouse,
    index_messages,
    parse_emails_from_db,
)
from retrieval import clear_summaries, run_batch_analysis, run_global_analysis


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import-mbox")
    import_parser.add_argument("--max-emails", type=int, default=0)
    subparsers.add_parser("dedup")
    subparsers.add_parser("clear-summaries")

    clean_parser = subparsers.add_parser("clean-bodies")
    clean_parser.add_argument("--fetch-batch", type=int, default=30)
    clean_parser.add_argument("--llm-batch", type=int, default=5)

    parse_parser = subparsers.add_parser("parse")
    parse_parser.add_argument("--limit", type=int, default=50)
    parse_parser.add_argument("--batch-size", type=int, default=3)
    parse_parser.add_argument("--max-workers", type=int, default=6)

    index_parser = subparsers.add_parser("index-messages")
    index_parser.add_argument("--batch-size", type=int, default=1000)
    index_parser.add_argument("--recreate", action="store_true")

    batch_analysis = subparsers.add_parser("batch-analysis")
    batch_analysis.add_argument("project_hint", type=str)
    batch_analysis.add_argument("--max-batches", type=int, default=0)

    global_analysis = subparsers.add_parser("global-analysis")
    global_analysis.add_argument("project_hint", type=str)

    args = parser.parse_args()

    if args.command == "import-mbox":
        import_mbox_to_clickhouse(max_emails=args.max_emails)
    elif args.command == "dedup":
        deduplicate_emails()
    elif args.command == "clean-bodies":
        clean_email_bodies_from_db(
            fetch_batch=args.fetch_batch,
            llm_batch=args.llm_batch,
        )
    elif args.command == "parse":
        result = parse_emails_from_db(
            limit=args.limit,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
        )
        print(result)
    elif args.command == "index-messages":
        index_messages(
            batch_size=args.batch_size,
            recreate=args.recreate,
        )
    elif args.command == "batch-analysis":
        print(run_batch_analysis(args.project_hint, max_batches=args.max_batches))
    elif args.command == "global-analysis":
        print(run_global_analysis(args.project_hint))
    elif args.command == "clear-summaries":
        clear_summaries()
        print("Old summaries cleared.")


main()