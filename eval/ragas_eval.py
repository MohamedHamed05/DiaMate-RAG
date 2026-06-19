import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Standalone: add src/ to sys.path so we can import utils.config
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.config import Settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration

settings = Settings()

GROQ_API_KEY = settings.GROQ_API_KEY
GROQ_MODEL = settings.LLM_MODEL or "meta-llama/llama-4-scout-17b-16e-instruct"
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
EMBED_MODEL = settings.EMBEDDING_MODEL or "embeddinggemma:latest"

# RAGAS metrics to run.
# Each metric requires certain fields — see comments below.
METRICS = [
    "faithfulness", # needs: response, retrieved_contexts
    "answer_relevancy", # needs: user_input, response, retrieved_contexts
    "context_precision", # needs: user_input, retrieved_contexts, reference
    "context_recall", # needs: retrieved_contexts, reference
]


def build_llm():
    """Wrap Groq in a RAGAS-compatible LLM."""
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set in src/.env. Add GROQ_API_KEY='gsk_...' to src/.env."
        )
    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)
    return LangchainLLMWrapper(llm)


def build_embeddings():
    """
    Connect to the bge-m3 model served by Ollama (Docker container).
    Ollama exposes an OpenAI-compatible /api/embeddings endpoint at OLLAMA_BASE_URL.
    """
    from langchain_ollama import OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    model = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return LangchainEmbeddingsWrapper(model)


def load_metrics(names: list[str], ragas_llm, ragas_embeddings):
    """
    Import and instantiate RAGAS metric objects.
    Injects llm + embeddings so every metric uses the same models.
    """
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )

    name_to_cls = {
        "faithfulness": Faithfulness,
        "answer_relevancy": AnswerRelevancy,
        "context_precision": ContextPrecision,
        "context_recall": ContextRecall,
    }

    # Groq only supports n=1 per request.
    # AnswerRelevancy defaults to strictness=3 (sends n=3), which causes a 400.
    # strictness=1 fixes this — one generation is enough for reliable scoring.
    METRIC_KWARGS = {
        "answer_relevancy": {"strictness": 1},
    }

    objects = []
    for name in names:
        cls = name_to_cls.get(name)
        if cls is None:
            logger.warning(f"Unknown metric '{name}' — skipping.")
            continue
        kwargs = METRIC_KWARGS.get(name, {})
        m = cls(**kwargs)
        m.llm = ragas_llm
        m.embeddings = ragas_embeddings
        objects.append(m)
    return objects


# Data preparation

def load_csv(path: Path, question_col: str, gt_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info(f"Loaded {len(df)} rows  |  columns: {df.columns.tolist()}")

    required = [question_col, gt_col, "answer", "contexts", "status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            "Make sure you ran populate_rag_dataset.py first.\n"
            "Also check --question-col / --gt-col if your column names differ."
        )

    # Only keep rows that were successfully populated
    n_total = len(df)
    df = df[df["status"] == "ok"].copy().reset_index(drop=True)
    logger.info(
        f"Kept {len(df)} / {n_total} rows with status='ok'  "
        f"({n_total - len(df)} skipped)"
    )
    if df.empty:
        raise ValueError("No rows with status='ok'. Run populate_rag_dataset.py first.")
    return df


def parse_contexts(raw) -> list[str]:
    """
    Convert the contexts column value (JSON string) to a Python list of strings.
    Falls back gracefully if parsing fails.
    """
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(c) for c in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return [str(raw)]  # last-resort: treat whole cell as one context


def prepare_ragas_dataset(df: pd.DataFrame, question_col: str, gt_col: str):
    """
    Build a RAGAS EvaluationDataset from the DataFrame.

    Column mapping:
        question      → user_input
        ground_truth  → reference
        answer        → response
        contexts      → retrieved_contexts  (JSON string → list[str])
    """
    from ragas import EvaluationDataset, SingleTurnSample

    samples = []
    for i, row in df.iterrows():
        contexts = parse_contexts(row["contexts"])
        samples.append(
            SingleTurnSample(
                user_input=str(row[question_col]),
                response=str(row["answer"]),
                retrieved_contexts=contexts,
                reference=str(row[gt_col]),
            )
        )

    logger.info(f"Built EvaluationDataset with {len(samples)} samples.")
    return EvaluationDataset(samples=samples)


# Evaluation

def run_eval(
    csv_path: str,
    question_col: str = "question",
    gt_col: str = "ground_truth",
    output_path: str = None,
) -> None:
    from ragas import evaluate
    from ragas.run_config import RunConfig

    path = Path(csv_path)
    df = load_csv(path, question_col, gt_col)

    logger.info("Loading LLM (Groq)…")
    ragas_llm = build_llm()

    logger.info(f"Connecting to Ollama embeddings ({EMBED_MODEL}) at {OLLAMA_BASE_URL}…")
    ragas_embeddings = build_embeddings()

    metrics = load_metrics(METRICS, ragas_llm, ragas_embeddings)
    logger.info(f"Metrics: {[type(m).__name__ for m in metrics]}")

    dataset = prepare_ragas_dataset(df, question_col, gt_col)

    logger.info("Running RAGAS evaluate()  (this will make LLM calls per row)…")
    results = evaluate(dataset=dataset, metrics=metrics, run_config=RunConfig(max_workers=1, max_retries=100, max_wait=180, timeout=180))

    # Merge per-row scores back into the DataFrame
    scores_df = results.to_pandas()
    metric_cols = [c for c in scores_df.columns if c in METRICS]
    for col in metric_cols:
        df[col] = scores_df[col].values

    # Print summary
    bar = "=" * 55
    logger.info(f"\n{bar}\n  RAGAS Results — {path.name}\n{bar}")
    for col in metric_cols:
        vals = df[col].dropna()
        logger.info(
            f"  {col:<28}  mean={vals.mean():.4f}  "
            f"min={vals.min():.4f}  max={vals.max():.4f}"
        )
    logger.info(bar)

    # Save results
    out = Path(output_path) if output_path else path.with_stem(path.stem + "_ragas_results")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info(f"Results saved → '{out}'")


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on a populated QA CSV dataset."
    )
    parser.add_argument("csv_file", help="Populated CSV (output of populate_rag_dataset.py)")
    parser.add_argument("--question-col", default="question", help="Column name for questions")
    parser.add_argument("--gt-col", default="ground_truth", help="Column name for ground truth")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    run_eval(args.csv_file, args.question_col, args.gt_col, args.output)