#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import requests

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.config import Settings 

settings = Settings()

API_URL = settings.RAG_API_URL
REQUEST_TIMEOUT = 60    # seconds to wait for a single response
DELAY_BETWEEN_REQUESTS = 0.5   # seconds between questions (throttle)
MAX_RETRIES = 2     # extra retries on transient failures
RETRY_DELAY = 3     # seconds to wait before each retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# API

def query_rag(question: str) -> dict:
    payload = {
        "session_id": str(uuid.uuid4()),  # fresh session per question
        "question": question,
    }
    resp = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    if "answer" not in data:
        raise ValueError(
            f"Unexpected response shape — keys found: {list(data.keys())}"
        )
    return data


def parse_response(data: dict) -> dict:
    answer = data.get("answer", "")
    chunks = data.get("source_chunks", [])

    contexts = [c.get("text", "").strip() for c in chunks]
    scores = [round(float(c.get("score", 0.0)), 6) for c in chunks]
    pages = [c.get("metadata", {}).get("page", "") for c in chunks]
    sources = [
        Path(c.get("metadata", {}).get("source", "")).name
        for c in chunks
    ]

    return {
        "answer": answer,
        "contexts": json.dumps(contexts, ensure_ascii=False),
        "retrieval_scores": json.dumps(scores),
        "retrieval_pages": json.dumps(pages),
        "source_files": json.dumps(sources),
    }


# Main

def populate(csv_path: str, question_col: str, gt_col: str) -> None:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info(f"Loaded {len(df)} rows from '{path}'")
    logger.info(f"Columns detected: {df.columns.tolist()}")

    # Validate required input columns
    for col in [question_col, gt_col]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. "
                f"Available columns: {df.columns.tolist()}\n"
                f"Tip: pass the exact name with --question-col / --gt-col"
            )

    # Add output columns if they don't exist yet
    for col in ["answer", "contexts", "retrieval_scores",
                "retrieval_pages", "source_files", "status"]:
        if col not in df.columns:
            df[col] = ""

    backup = path.with_stem(path.stem + "_backup")
    if not backup.exists():
        df.to_csv(backup, index=False, encoding="utf-8-sig")
        logger.info(f"Backup created → '{backup}'")

    total = len(df)
    success = skipped = errors = 0

    for i, row in df.iterrows():
        # Skip rows that already have an answer (resume-safe)
        existing = str(df.at[i, "answer"]).strip()
        if existing and existing not in ("", "nan"):
            skipped += 1
            logger.info(
                f"[{i+1:>4}/{total}] ⏭  Already done — "
                f"{str(row[question_col])[:60]}"
            )
            continue

        question = str(row[question_col]).strip()
        logger.info(f"[{i+1:>4}/{total}]   {question[:80]}")

        last_err = None
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                raw = query_rag(question)
                fields = parse_response(raw)

                for key, val in fields.items():
                    df.at[i, key] = val
                df.at[i, "status"] = "ok"
                success += 1

                n_chunks = len(json.loads(fields["contexts"]))
                logger.info(
                    f"           {n_chunks} chunk(s) | "
                    f"{len(fields['answer'])} chars in answer"
                )
                last_err = None
                break   # success — exit retry loop

            except Exception as exc:
                last_err = exc
                if attempt <= MAX_RETRIES:
                    logger.warning(
                        f"           Attempt {attempt} failed: {exc}. "
                        f"Retrying in {RETRY_DELAY}s…"
                    )
                    time.sleep(RETRY_DELAY)

        if last_err:
            err_msg = f"error: {str(last_err)[:150]}"
            df.at[i, "status"] = err_msg
            errors += 1
            logger.error(
                f"           Gave up after {MAX_RETRIES + 1} attempt(s): "
                f"{last_err}"
            )

        # Save after EVERY row — progress is never lost
        df.to_csv(path, index=False, encoding="utf-8-sig")

        if i < total - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    # Summary
    bar = "=" * 55
    logger.info(
        f"\n{bar}\n"
        f"  Finished processing '{path.name}'\n"
        f"  Total : {total:>4}\n"
        f"  OK    : {success:>4}\n"
        f"  Skipped: {skipped:>4}  (already had answers)\n"
        f"  Errors : {errors:>4}\n"
        f"{bar}"
    )
    if errors:
        failed = df[df["status"].str.startswith("error", na=False)]
        logger.info("Rows with errors:")
        for idx, r in failed.iterrows():
            logger.info(f"  Row {idx+1}: {str(r[question_col])[:60]}  →  {r['status']}")


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate a QA CSV dataset with RAG answers and contexts."
    )
    parser.add_argument(
        "csv_file",
        help="Path to the .csv QA dataset (e.g. qa_dataset.csv)"
    )
    parser.add_argument(
        "--question-col",
        default="question",
        help="Exact column name for questions (default: 'question')"
    )
    parser.add_argument(
        "--gt-col",
        default="ground_truth",
        help="Exact column name for ground truth (default: 'ground_truth')"
    )
    args = parser.parse_args()

    populate(args.csv_file, args.question_col, args.gt_col)
