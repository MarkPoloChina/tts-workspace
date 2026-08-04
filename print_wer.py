#!/usr/bin/env python3
"""Print Seed-TTS WER eval items whose WER is not zero."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_items(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("seed_tts_wer_eval_items")
    if not isinstance(items, list):
        raise ValueError(f"{json_path} does not contain a list field: seed_tts_wer_eval_items")

    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter WER eval items and print entries whose WER is not zero."
    )
    parser.add_argument(
        "--json-path",
        default=Path(__file__).with_name("1784688020.json"),
        type=Path,
        help="Path to the JSON file. Defaults to 1784688020.json in this script's directory.",
    )
    args = parser.parse_args()

    items = load_items(args.json_path)
    nonzero_items = [item for item in items if float(item.get("wer", 0.0)) >= 0.2]

    for index, item in enumerate(nonzero_items, start=1):
        print(f"[{index}] utterance_id: {item.get('utterance_id', '')}")
        print(f"WER: {item.get('wer')}")
        print(f"Ground Truth: {item.get('reference_raw', '')}")
        print(f"Prediction: {item.get('asr_raw', '')}")
        print()

    print(f"Total non-zero WER items: {len(nonzero_items)}")


if __name__ == "__main__":
    main()
