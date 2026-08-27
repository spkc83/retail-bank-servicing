"""Deictic reference suite: the behaviour the long-session sweep never tests.

Turn 1 surfaces a table; turn 2 refers back into it ("that card", "the one to X").
Success requires the model to bind the reference and emit the right tool ARGUMENTS —
the harness validates arguments but cannot invent them, so this isolates what
fine-tuning contributes beyond the harness.

usage: deictic_suite.py <tag>   (model comes from RETAIL_BANK_* env, like the sweep)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POC = REPO / "poc" / "retail-bank-customer-service-poc"
sys.path.insert(0, str(POC))

TAG = sys.argv[1]
OUT = Path(os.environ.get("DEICTIC_OUT_DIR", ".")) / f"deictic_{TAG}.jsonl"
USERNAME = "alex.demo"

# (case, setup turns, deictic turn, want-tool, want-args-contains)
FLOWS = [
    ("freeze_that_card", ["Show me my cards."], "Freeze that card.",
     "freeze_card", "4821"),
    ("replace_shown_card", ["What is the status of my debit card?"], "Please replace that one.",
     "replace_card", "4821"),
    ("cancel_named_transfer", ["List my transfers."], "Cancel the one to River Consulting.",
     "cancel_transfer", "River Consulting"),
    ("dispute_named_purchase", ["Show my five most recent transactions."],
     "I did not make the North Harbor Market purchase. Dispute it.",
     "dispute_transaction", "North Harbor"),
]


def main() -> int:
    from local_app_service import LocalBankingController  # type: ignore[import-not-found]
    from local_gpu_runtime import LocalGraniteRuntime  # type: ignore[import-not-found]
    from router import LearnedBankingRouter  # type: ignore[import-not-found]
    from state import BANK

    router = LearnedBankingRouter.from_artifact_dir(
        REPO / "artifacts" / "banking-conversation-router-v8-first-turn-mutation"
    )
    print("[load] router ready; loading runtime", flush=True)
    t0 = time.time()
    runtime = LocalGraniteRuntime.load()
    print(f"[load] runtime ready in {time.time() - t0:.0f}s", flush=True)
    controller = LocalBankingController(bank=BANK, runtime=runtime, router=router)

    with OUT.open("w") as out:
        for index, (case, setup, deictic, want_tool, want_arg) in enumerate(FLOWS):
            session = f"deictic-{TAG}-{index}"
            conversation: list[dict] = []
            row: dict = {"case": case, "deictic": deictic, "want_tool": want_tool}
            try:
                for turn in setup:
                    res = controller.run_turn(
                        username=USERNAME, session_hash=session,
                        message=turn, conversation=conversation,
                    )
                    conversation = res.conversation
                t1 = time.time()
                res = controller.run_turn(
                    username=USERNAME, session_hash=session,
                    message=deictic, conversation=conversation,
                )
                calls = [
                    {"name": c.name, "arguments": c.arguments} for c in res.tool_calls
                ]
                hit = any(
                    c["name"] == want_tool
                    and want_arg.lower() in json.dumps(c["arguments"]).lower()
                    for c in calls
                )
                row.update(
                    reply=res.response[:400], response_path=res.response_path,
                    tool_calls=calls, correct_binding=hit,
                    fallback="couldn’t complete that request" in res.response,
                    seconds=round(time.time() - t1, 1),
                )
            except Exception as error:  # noqa: BLE001
                row.update(error=f"{type(error).__name__}: {error}"[:200], correct_binding=False)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{case:<24} binding={row.get('correct_binding')} calls={row.get('tool_calls')} "
                  f"fallback={row.get('fallback')} err={row.get('error','')[:60]}", flush=True)
    print(f"done {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
