#!/usr/bin/env python
"""Long-session sweep driven in process, without a browser or Streamlit.

Four servicing turns of history, then a fifth question whose wording pulls toward a
tool that does not exist. Each case gets a fresh session hash, so it reproduces the
"one fresh session per case" property a browser sweep gets from a new tab, while
loading the runtime exactly once.

Records the reply, the route, the executed tools and the model-pass labels per case,
so a run stays inspectable afterwards instead of being scraped back out of the DOM.

Read the output as two numbers, not one. ``fallback`` counts turns that hit the
stock failure response; on its own it scores a fabricated answer as a success and an
honest "I couldn't complete that request" as a failure. Always read the replies for
substance beside it.

usage: inproc_long_session_sweep.py <tag> [--out DIR]

Model, adapter and router identities come from the same RETAIL_BANK_* environment
variables the local Streamlit launcher uses, including
RETAIL_BANK_ADAPTER_SUBFOLDER for adapters whose config is not at the repo root.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

POC_DIR = Path(__file__).resolve().parents[2] / "poc" / "retail-bank-customer-service-poc"

HISTORY = (
    "My card was stolen. Freeze it.",
    "What is the status of my debit card?",
    "Cancel my scheduled transfer.",
    "What documents do I need to apply for a mortgage?",
)
CASES = (
    ("address", "When was my mailing address changed?"),
    ("statement", "Can you show me my statement for last month?"),
    ("pin", "Did my PIN change request go through?"),
    ("dispute", "Is there an open dispute on my account?"),
    ("scheduled", "What transfers are scheduled for next week?"),
    ("requests", "Which service requests are still open?"),
    ("control-cases", "Show my service cases."),
    ("control-transfers", "List my transfers."),
)
FALLBACK_MARKER = "couldn’t complete that request"
USERNAME = "alex.demo"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Names the output file, e.g. v10 -> sweep_v10.jsonl")
    parser.add_argument("--out", type=Path, default=Path.cwd())
    return parser.parse_args(argv)


def build_controller():
    """Instantiate exactly what streamlit_app.py caches, minus the Streamlit cache."""

    sys.path.insert(0, str(POC_DIR))
    import os

    from local_app_service import LocalBankingController  # type: ignore[import-not-found]
    from local_gpu_runtime import LocalGraniteRuntime  # type: ignore[import-not-found]
    from router import LearnedBankingRouter  # type: ignore[import-not-found]
    from state import BANK  # type: ignore[import-not-found]

    configured = os.environ.get("LOCAL_ROUTER_ARTIFACT_DIR")
    if configured:
        artifact = Path(configured)
        if not artifact.is_absolute():
            artifact = Path(__file__).resolve().parents[2] / artifact
        router = LearnedBankingRouter.from_artifact_dir(artifact)
    else:
        router = LearnedBankingRouter.from_hub()
    print("[load] router ready; loading runtime (this is the slow part)", flush=True)
    started = time.time()
    runtime = LocalGraniteRuntime.load()
    print(f"[load] runtime ready in {time.time() - started:.0f}s", flush=True)
    return LocalBankingController(bank=BANK, runtime=runtime, router=router)


def run_case(controller, tag: str, index: int, case: str, question: str) -> dict:
    session = f"inproc-{tag}-{index}"
    conversation: list[dict] = []
    row: dict = {"case": case, "question": question}
    started = time.time()
    try:
        for turn in HISTORY:
            result = controller.run_turn(
                username=USERNAME,
                session_hash=session,
                message=turn,
                conversation=conversation,
            )
            conversation = result.conversation
        result = controller.run_turn(
            username=USERNAME,
            session_hash=session,
            message=question,
            conversation=conversation,
        )
        row.update(
            reply=result.response[:600],
            fallback=FALLBACK_MARKER in result.response,
            response_path=result.response_path,
            route=result.route.get("route"),
            intent=result.route.get("intent"),
            action=result.route.get("action"),
            tool_calls=[{"name": c.name, "arguments": c.arguments} for c in result.tool_calls],
            pass_labels=[getattr(p, "label", "?") for p in result.model_passes],
            seconds=round(time.time() - started, 1),
        )
    except Exception as error:  # noqa: BLE001 - the failure IS the measurement
        row.update(
            error=f"{type(error).__name__}: {error}",
            trace=traceback.format_exc()[-800:],
        )
    return row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    destination = args.out / f"sweep_{args.tag}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    controller = build_controller()
    with destination.open("w") as handle:
        for index, (case, question) in enumerate(CASES):
            row = run_case(controller, args.tag, index, case, question)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            tools = [call["name"] for call in row.get("tool_calls", [])]
            print(
                f"{case:<18} fallback={row.get('fallback')} "
                f"path={str(row.get('response_path'))[:28]!r} tools={tools} "
                f"err={str(row.get('error', ''))[:60]!r}",
                flush=True,
            )
            print("   reply:", str(row.get("reply", ""))[:180].replace("\n", " | "), flush=True)
    print(f"done {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
