"""Author prompt realizations for the records whose questions shadow the eval splits.

One rewrite per (record family, split). The finals are untouched: each new
question asks for the same thing in different words, so the grounded answer
still answers it. Test wording is never touched -- it is the held-out side.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from hello_slm.banking_servicing_alignment_data import (  # noqa: E402
    _current_user_text,
    build_servicing_alignment_splits,
)
from hello_slm.banking_tool_sft_data import _immutable_record_hash  # noqa: E402

# record-id prefix -> {split: new question}. The trailing `_suffix` the builders
# append is preserved by construction, since we rewrite the whole rendered turn.
REWRITES = {
    "svc_case_created": {
        "train": "What date was the address update logged",
        "validation": "Remind me when the address change was filed",
    },
    "svc_case_status": {
        "train": "Where does the address update stand",
        "validation": "Give me the current state of that filing",
    },
    "card_replace_that_one": {
        "train": "Send out a replacement for that listed card",
        "validation": "I would like a new copy of the shown card",
    },
    "card_freeze_that_one": {
        "train": "Put a hold on it while I am away",
        "validation": "Lock that card until I get back",
    },
    "banking_to_external_ood": {
        "train": "Any chance of rain around Cedar Point",
        "validation": "How hot will it be in Lake City",
    },
    "external_to_banking": {
        "train": "Anyway, pull up my open requests",
        "validation": "Let us return to my support tickets",
    },
    "policy_detour": {
        "train": "Before that, explain the interest rules",
        "validation": "Quick question on how interest accrues",
    },
    "policy_resume": {
        # The resume turn must keep an explicit continuation cue: the runtime's
        # resume gate keys on it, and a repository test pins it.
        "train": "Got it. Continue the charge challenge",
        "validation": "Understood. Continue the claim review",
    },
}


# Families whose validation rows echo their own train rows. These carry
# entities the final and the tool arguments depend on -- a card's last four,
# a payee name -- so the rewrite substitutes the opening clause and leaves
# everything after it untouched. None of these families has test rows: fixing
# them makes the dev gate honest rather than the published test numbers.
VALIDATION_PREFIX_REWRITES = {
    "history_entity_action": (
        "Cancel the pending payment you identified",
        "Stop the payment that came up above",
    ),
    "tool_outcome_consistency": (
        "Cancel the pending transfer to",
        # Not "scheduled transfer": that phrase belongs to a held-out row, and
        # the isolation invariant rejected the first attempt at this line.
        "Stop the upcoming payment to",
    ),
    "clarification_answer": ("It is", "The digits are"),
    "agent_repair": (
        "Do not repeat that; check the actual service case",
        "That is not right; look up the real service case",
    ),
    "history_entity_ambiguity": (
        "Replace the card we were discussing",
        "Swap out the card from that exchange",
    ),
    "long_context_tool_fidelity": (
        "Before we wrap this up I still need you to",
        "One more thing before we finish: please",
    ),
}


def validation_rewrite(record, text: str) -> str | None:
    """Substitute the opening clause of a validation prompt, keeping entities."""
    family = str(record.get("metadata", {}).get("scenario_family", ""))
    pair = VALIDATION_PREFIX_REWRITES.get(family)
    if pair is None:
        return None
    old, new = pair
    if not text.startswith(old):
        return None
    tail = text[len(old) :]
    # The builders concatenate with no space in places ("It is 4821from this
    # chat"); newly authored text does not inherit that.
    if tail and not tail[0].isspace():
        tail = " " + tail
    return new + tail


SUFFIX = {"train": "on my account", "validation": "from this chat"}


def stem_and_tail(text: str, split: str) -> tuple[str, str]:
    """Split a rendered turn into its authored stem and everything appended after."""
    marker = SUFFIX[split]
    index = text.find(marker)
    if index == -1:
        return text, ""
    return text[:index], text[index:]


def main() -> int:
    splits, _ = build_servicing_alignment_splits()
    rows = []
    for split in ("train", "validation"):
        for record in splits[split]:
            record_id = str(record["record_id"])
            prefix = next(
                (p for p in REWRITES if record_id.startswith(f"{p}_{split}")),
                None,
            )
            if prefix is None:
                if split == "validation":
                    rewritten = validation_rewrite(record, _current_user_text(record))
                    if rewritten and rewritten != _current_user_text(record):
                        rows.append(
                            {
                                "record_id": record_id,
                                "immutable_hash": _immutable_record_hash(record),
                                "user_content": rewritten,
                            }
                        )
                continue
            original = _current_user_text(record)
            _, tail = stem_and_tail(original, split)
            # The builders concatenate stem and suffix with no space, which
            # trains 'createdfrom my profile' into the model. Newly authored
            # prompts do not reproduce that; the frozen test rows keep it.
            rewritten = f"{REWRITES[prefix][split]} {tail}" if tail else REWRITES[prefix][split]
            if rewritten == original:
                continue
            rows.append(
                {
                    "record_id": record_id,
                    "immutable_hash": _immutable_record_hash(record),
                    "user_content": rewritten,
                }
            )
    out = (
        pathlib.Path(__file__).resolve().parents[2]
        / "data/sources/banking-servicing-alignment-v5-prompt-realizations.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows, key=lambda r: r["record_id"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} prompt realizations to {out}")
    for row in rows[:3]:
        print("   ", row["record_id"], "->", row["user_content"][:80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
