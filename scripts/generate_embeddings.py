"""Slice 6 batch embedding backfill.

Walks every trace in the DB that has an ATIF blob, derives the
embedding-input text via
:func:`trace_marketplace.search.text_extraction.extract_searchable_text`,
and stores one ``text-embedding-3-small`` vector per row in the
``trace_embeddings`` table.

Idempotency
-----------

Each candidate's source text is hashed (SHA-256). We look up the
hash in ``trace_embeddings.source_text_hash`` and skip rows whose
hash already matches. Re-running the script after slice 5 uploads
new traces costs only the embed calls for the *new* rows; re-running
without changes is a free no-op.

Cost / safety
-------------

* ``--dry-run`` prints the projected call count + dollar cost
  without hitting the API. Run this first.
* ``--limit N`` caps the work to N rows (handy for a $0.001 smoke
  test before the full pass).
* ``--model`` lets you swap embedding models; the column is
  recorded so a future hybrid setup can disambiguate.
* The script aborts with a clear message if ``OPENAI_API_KEY`` is
  unset rather than letting the SDK leak a stack trace.

Usage:
    uv run python scripts/generate_embeddings.py --dry-run
    uv run python scripts/generate_embeddings.py --limit 5
    uv run python scripts/generate_embeddings.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

from trace_marketplace.enrich.embeddings import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_MODEL,
    embed_batch,
    estimate_cost_usd,
    existing_source_hashes,
    hash_source_text,
    require_api_key,
    store_embedding,
)
from trace_marketplace.search.text_extraction import extract_searchable_text
from trace_marketplace.storage.db import DEFAULT_DB_PATH, connect, init_schema

log = logging.getLogger("generate_embeddings")

CHARS_PER_TOKEN_ESTIMATE: float = 4.0
"""Rough OpenAI tokenizer ratio. Used only for dry-run cost
projection; the live path uses the actual ``usage.prompt_tokens``
returned by the API."""


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=EMBEDDING_MODEL,
        help=f"OpenAI embedding model (default: {EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"OpenAI requests per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on traces to process (default: all eligible rows).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print projected call count + cost, do not call the API.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-embed every eligible row, ignoring the source-text-hash "
            "skip check. Useful after a text-extraction logic change."
        ),
    )
    return parser.parse_args(argv)


def _build_candidate_text(
    trace_id: str,
    atif_blob: str | None,
    failure_label: str | None,
    failure_label_reasoning: str | None,
) -> str | None:
    """Return the embed-input text for one trace, or ``None`` to skip.

    Skip-able cases:
    * ``atif`` is NULL (raw-only upload from slice 5).
    * ``atif`` is non-parseable JSON (we log and move on rather than
      raising; one bad row mustn't tank the batch).
    """
    if not atif_blob:
        return None
    try:
        atif = json.loads(atif_blob)
    except json.JSONDecodeError as exc:
        log.warning("trace %s: skipping, atif is not valid JSON (%s)", trace_id, exc)
        return None
    text = extract_searchable_text(
        atif,
        failure_label=failure_label,
        failure_label_reasoning=failure_label_reasoning,
    )
    return text


def _select_targets(
    conn: sqlite3.Connection, *, limit: int | None
) -> list[tuple[str, str, str | None, str | None]]:
    """Return ``(id, atif, failure_label, failure_label_reasoning)`` rows."""
    sql = (
        "SELECT id, atif, failure_label, failure_label_reasoning "
        "FROM traces WHERE atif IS NOT NULL ORDER BY id"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [
        (row["id"], row["atif"], row["failure_label"], row["failure_label_reasoning"])
        for row in conn.execute(sql)
    ]


def _classify_candidates(
    rows: list[tuple[str, str, str | None, str | None]],
    *,
    existing: dict[str, str],
    force: bool,
) -> tuple[list[tuple[str, str]], int, int]:
    """Bucket rows into (to_embed, skipped_fresh, skipped_unparseable).

    Returns ``(pairs, skipped_fresh, skipped_unparseable)`` where
    ``pairs`` is the ``[(trace_id, text)]`` list the embedder will
    actually send to OpenAI.
    """
    pairs: list[tuple[str, str]] = []
    skipped_fresh = 0
    skipped_unparseable = 0
    for trace_id, atif_blob, label, reasoning in rows:
        text = _build_candidate_text(trace_id, atif_blob, label, reasoning)
        if text is None:
            skipped_unparseable += 1
            continue
        if not force:
            current_hash = hash_source_text(text)
            if existing.get(trace_id) == current_hash:
                skipped_fresh += 1
                continue
        pairs.append((trace_id, text))
    return pairs, skipped_fresh, skipped_unparseable


def _print_dry_run_summary(
    pairs: list[tuple[str, str]],
    *,
    batch_size: int,
    skipped_fresh: int,
    skipped_unparseable: int,
) -> None:
    char_total = sum(len(text) for _, text in pairs)
    est_tokens = int(char_total / CHARS_PER_TOKEN_ESTIMATE)
    est_calls = (len(pairs) + batch_size - 1) // batch_size
    print()
    print("=" * 60)
    print("DRY RUN -- no API calls will be made")
    print("-" * 60)
    print(f"  Rows to embed:            {len(pairs)}")
    print(f"  Rows skipped (fresh):     {skipped_fresh}")
    print(f"  Rows skipped (no/bad atif): {skipped_unparseable}")
    print(f"  Total chars to embed:     {char_total:,}")
    print(f"  Est. input tokens:        {est_tokens:,}")
    print(f"  Est. API calls:           {est_calls} (batch_size={batch_size})")
    print(f"  Est. cost:                ${estimate_cost_usd(est_tokens):.4f}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if not args.db.exists():
        log.error("Database not found at %s", args.db)
        return 1

    with connect(args.db) as conn:
        init_schema(conn)
        rows = _select_targets(conn, limit=args.limit)
        if not rows:
            print("No traces with atif. Nothing to embed.")
            return 0

        ids = [r[0] for r in rows]
        existing = (
            existing_source_hashes(conn, ids, model=args.model)
            if not args.force
            else {}
        )
        pairs, skipped_fresh, skipped_unparseable = _classify_candidates(
            rows, existing=existing, force=args.force
        )

        if args.dry_run:
            _print_dry_run_summary(
                pairs,
                batch_size=args.batch_size,
                skipped_fresh=skipped_fresh,
                skipped_unparseable=skipped_unparseable,
            )
            return 0

        if not pairs:
            print(
                f"All {len(rows)} traces already up to date "
                f"({skipped_fresh} fresh, {skipped_unparseable} skipped)."
            )
            return 0

        require_api_key()

        print(
            f"Embedding {len(pairs)} traces with {args.model} "
            f"(batch_size={args.batch_size}, "
            f"fresh-skipped={skipped_fresh}, unparseable={skipped_unparseable})..."
        )
        ids_only = [tid for tid, _ in pairs]
        texts = [text for _, text in pairs]
        vectors = embed_batch(
            texts,
            model=args.model,
            batch_size=args.batch_size,
        )
        for trace_id, text, vec in zip(ids_only, texts, vectors):
            store_embedding(
                conn,
                trace_id=trace_id,
                embedding=vec,
                source_text_hash=hash_source_text(text),
                model=args.model,
            )
        print(f"Stored {len(vectors)} embeddings.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
