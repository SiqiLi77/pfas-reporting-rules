#!/usr/bin/env python3
"""Summarize co-reporting conditional on at least one reported shared PFAS."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "outputs" / "ucmr_cross_cycle_count_mrl_strict_event_20260815"
RATE_INPUT = ANALYSIS_DIR / "tail_rate_table.csv"
DECISION_INPUT = ANALYSIS_DIR / "decision.json"
CSV_OUTPUT = ANALYSIS_DIR / "conditional_coreporting_literature_alignment.csv"
JSON_OUTPUT = ANALYSIS_DIR / "conditional_coreporting_literature_alignment.json"

REGIMES = (
    ("ucmr3_native", "UCMR 3 at native MRLs"),
    ("ucmr5_native", "UCMR 5 at native MRLs"),
    ("ucmr5_common_c1", "UCMR 5 at UCMR 3 numeric cutoffs"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    decision = json.loads(DECISION_INPUT.read_text(encoding="utf-8"))
    pairs = int(decision["subsets"]["identifier_matched"]["pairs"])

    rates: dict[tuple[str, int], float] = {}
    with RATE_INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["subset"] != "identifier_matched" or row["estimand"] != "event_equal":
                continue
            if row["regime"] not in {regime for regime, _ in REGIMES}:
                continue
            k = int(row["k"])
            if k in (1, 2):
                rates[(row["regime"], k)] = float(row["rate"])

    rows: list[dict[str, object]] = []
    for regime, label in REGIMES:
        rate_k1 = rates[(regime, 1)]
        rate_k2 = rates[(regime, 2)]
        count_k1 = round(rate_k1 * pairs)
        count_k2 = round(rate_k2 * pairs)
        if not 0 <= count_k2 <= count_k1 <= pairs:
            raise ValueError(f"Invalid nested tail counts for {regime}")
        if abs(count_k1 / pairs - rate_k1) > 1e-12 or abs(count_k2 / pairs - rate_k2) > 1e-12:
            raise ValueError(f"Rates do not reconstruct integer event counts for {regime}")
        conditional_rate = count_k2 / count_k1
        rows.append(
            {
                "subset": "identifier_matched",
                "estimand": "event_equal_descriptive",
                "regime": regime,
                "regime_label": label,
                "matched_event_count": pairs,
                "events_k_ge_1": count_k1,
                "events_k_ge_2": count_k2,
                "conditional_rate_k_ge_2_given_k_ge_1": f"{conditional_rate:.12f}",
                "conditional_percent": f"{100 * conditional_rate:.1f}",
            }
        )

    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    record = {
        "status": "DESCRIPTIVE LITERATURE-ALIGNMENT SUMMARY COMPLETE",
        "formula": "P(K6>=2 | K6>=1) = n(K6>=2) / n(K6>=1)",
        "scope": "Event-equal strict identifier-matched SE1 events; descriptive, not the primary jurisdiction/PWS estimand.",
        "matched_event_count": pairs,
        "inputs": {
            str(RATE_INPUT.relative_to(ROOT)): sha256(RATE_INPUT),
            str(DECISION_INPUT.relative_to(ROOT)): sha256(DECISION_INPUT),
        },
        "rows": rows,
    }
    JSON_OUTPUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
