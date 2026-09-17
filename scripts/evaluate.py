"""Measure zero-shot accuracy against the labeled ticket dataset.

Usage: uv run scripts/evaluate.py --csv all_tickets_processed_improved_v3.csv --sample 200
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from collections import Counter
from pathlib import Path

from ticketag.exceptions import TickeTagError
from ticketag.models import ClassifiedTicket
from ticketag.pipeline import TicketClassificationPipeline
from ticketag.zeroshot import classifier_from_env

logger = logging.getLogger("evaluate")


def load_sample(
    path: Path, text_column: str, label_column: str, size: int, seed: int
) -> list[tuple[str, str]]:
    """Reservoir-sample (text, gold_label) pairs so the full CSV never hits memory."""
    rng = random.Random(seed)
    sample: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            pair = (row.get(text_column, ""), row.get(label_column, ""))
            if not pair[0].strip() or not pair[1].strip():
                continue
            if len(sample) < size:
                sample.append(pair)
                continue
            slot = rng.randint(0, index)
            if slot < size:
                sample[slot] = pair
    return sample


def score(pairs: list[tuple[str, str]]) -> tuple[float, dict[str, tuple[float, float, float]]]:
    """Compute overall accuracy plus per-class precision, recall and F1."""
    hits = Counter[str]()
    predicted = Counter[str]()
    actual = Counter[str]()
    for gold, prediction in pairs:
        actual[gold] += 1
        predicted[prediction] += 1
        if gold == prediction:
            hits[gold] += 1
    accuracy = sum(hits.values()) / len(pairs) if pairs else 0.0
    report: dict[str, tuple[float, float, float]] = {}
    for label in sorted(actual | predicted):
        precision = hits[label] / predicted[label] if predicted[label] else 0.0
        recall = hits[label] / actual[label] if actual[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = (precision, recall, f1)
    return accuracy, report


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and print a per-class report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--text-column", default="Document")
    parser.add_argument("--label-column", default="Topic_group")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
    sample = load_sample(args.csv, args.text_column, args.label_column, args.sample, args.seed)
    if not sample:
        logger.error("No usable rows found in %s", args.csv)
        return 1
    logger.info("Classifying %d tickets", len(sample))

    try:
        pipeline = TicketClassificationPipeline(classifier_from_env())
        # The gold label rides along as ticket_id so results stay aligned after concurrency.
        results = list(
            pipeline.run_batch(((gold, text) for text, gold in sample), max_workers=args.workers)
        )
    except TickeTagError as error:
        logger.error("Evaluation aborted: %s", error)
        return 1

    scored = [
        (r.ticket_id, r.classification.category) for r in results if isinstance(r, ClassifiedTicket)
    ]
    failures = len(results) - len(scored)
    accuracy, report = score(scored)

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold", "predicted", "confidence", "justification"])
        for result in results:
            if isinstance(result, ClassifiedTicket):
                writer.writerow(
                    [
                        result.ticket_id,
                        result.classification.category,
                        f"{result.classification.confidence:.2f}",
                        result.classification.justification,
                    ]
                )

    print(f"\nScored {len(scored)} tickets ({failures} failed) | accuracy {accuracy:.1%}\n")
    print(f"{'category':<24}{'precision':>10}{'recall':>10}{'f1':>10}")
    for label, (precision, recall, f1) in report.items():
        print(f"{label:<24}{precision:>10.2f}{recall:>10.2f}{f1:>10.2f}")
    print(f"\nPredictions written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
