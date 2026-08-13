#!/usr/bin/env python
"""Build and validate the evaluation-only retail-bank counterfactual suite."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hello_slm.banking_counterfactual_eval_data import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    CounterfactualEvalDataError,
    write_counterfactual_benchmark,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = write_counterfactual_benchmark(args.output_dir)
    except (OSError, CounterfactualEvalDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "manifest.json"),
                "record_count": manifest["report"]["record_count"],
                "counterfactual_pair_count": manifest["report"]["counterfactual_pair_count"],
                "contamination_audit": manifest["report"]["audit"]["status"],
                "training_allowed": manifest["training_allowed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
