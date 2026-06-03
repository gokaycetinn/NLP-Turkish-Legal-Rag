"""
Download Turkish legal datasets from HuggingFace Hub.

Datasets used:
  - Renicames/turkish-lawchatbot          QA chatbot pairs       SFT
  - yeniguno/turkish-law-eqa              Legal extractive QA    SFT
  - OrionCAF/turkish_law_qa_dataset       Legal QA pairs         SFT
  - erdem-erdem/Turkish-Law-Documents-700k-clustered  Court decisions  retrieval-augment

The Kaggle dataset (batuhankalem/turkishlaw-dataset-for-llm-finetuning) is downloaded
separately via the Kaggle API — see notebooks/download_kaggle.ipynb.

Run:
    python -m src.ingest.download_hf_datasets [--include court]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "finetune"

# (hf_path, output_filename, sample_n_or_None, description)
SFT_DATASETS = [
    ("Renicames/turkish-lawchatbot",       "renicames_lawchatbot.jsonl",  None, "QA chatbot pairs"),
    ("yeniguno/turkish-law-eqa",           "yeniguno_law_eqa.jsonl",      None, "Extractive QA"),
    ("OrionCAF/turkish_law_qa_dataset",    "orioncaf_law_qa.jsonl",       None, "Legal QA"),
]

# Court decisions corpus — large; downsample to keep manageable
COURT_DATASET = (
    "erdem-erdem/Turkish-Law-Documents-700k-clustered",
    "erdem_court_decisions.jsonl",
    50_000,  # subsample
    "Yargıtay + Danıştay court decisions",
)

log = logging.getLogger("download_hf")


def download_dataset(hf_path: str, output_name: str, sample_n: int | None, desc: str):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install: pip install datasets")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / output_name

    if out_path.exists():
        log.info("Already exists, skipping: %s", out_path.name)
        return

    log.info("Loading %s (%s) ...", hf_path, desc)
    ds = load_dataset(hf_path, split="train")
    log.info("  Loaded %d rows", len(ds))

    if sample_n and len(ds) > sample_n:
        ds = ds.shuffle(seed=42).select(range(sample_n))
        log.info("  Downsampled to %d rows", sample_n)

    log.info("  Writing to %s", out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log.info("  Done: %s (%d rows)", out_path.name, len(ds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include",
        nargs="+",
        choices=["sft", "court", "all"],
        default=["sft"],
        help="Which dataset groups to download (default: sft only)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    include = set(args.include)
    if "all" in include:
        include = {"sft", "court"}

    if "sft" in include:
        for hf_path, name, n, desc in SFT_DATASETS:
            try:
                download_dataset(hf_path, name, n, desc)
            except Exception as e:
                log.error("Failed %s: %s", hf_path, e)

    if "court" in include:
        try:
            download_dataset(*COURT_DATASET)
        except Exception as e:
            log.error("Failed court dataset: %s", e)


if __name__ == "__main__":
    main()
