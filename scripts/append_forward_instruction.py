"""Append explicit forwarding instruction to behavior text where the role
implies forwarding but the behavior text omits it.

Pattern: object has a single declared peer, role text contains a forwarding
verb (forward/route/dispatch/pass/feed/send to), but behavior text doesn't
mention the peer or any dispatch verb. The planner reads behavior as the
prescriptive text and ends up emitting a `tool` step (using whatever tool
is in scope) that bypasses the orchestration peer entirely.

Fix: append one sentence to behavior — "Then forward the [event] to
<peer_id>." Conservative addition that preserves existing behavior text
and uses the peer's own object_id so the planner can latch onto it.

Modifies data/zapier/workflows-mods.jsonl in place. Backup at
data/zapier/workflows-mods.jsonl.bak_pre_forward_fix (this script
creates it).
"""
import json
import os
import shutil


FORWARDING_VERBS = (
    "forward", "rout", "dispatch", "pass to", "passes", "pass it", "pass each",
    "pass on", "pass through", "send to", "send it", "feeds", "feeding",
    "feed into", "feed each", "pipe ",
)

BEHAVIOR_DISPATCH_VERBS = (
    "forward", "rout", "dispatch", "pass to", "pass on", "send to",
    "feed", "tell ", "ask ", "notify", "deliver to",
)


def needs_forward_instruction(obj: dict) -> bool:
    """True iff this object's role implies forwarding but its behavior
    text omits the forwarding instruction.
    """
    role = (obj.get("role") or "").lower()
    behavior = (obj.get("behavior") or "").lower()
    peers = [p.get("object_id") for p in obj.get("peers", [])]
    if not peers:
        return False
    if not any(v in role for v in FORWARDING_VERBS):
        return False
    # Behavior must NOT mention any peer by name OR any dispatch verb
    if any(p in behavior for p in peers):
        return False
    if any(v in behavior for v in BEHAVIOR_DISPATCH_VERBS):
        return False
    return True


def amended_behavior(obj: dict) -> str:
    """Return the original behavior text with one forwarding sentence
    appended. Uses the first declared peer."""
    behavior = (obj.get("behavior") or "").rstrip()
    if not behavior:
        return behavior
    peers = [p.get("object_id") for p in obj.get("peers", [])]
    target = peers[0]
    # Pick a natural object noun based on the role wording. Default "event".
    role = (obj.get("role") or "").lower()
    if "submission" in role:
        thing = "submission"
    elif "request" in role:
        thing = "request"
    elif "review" in role:
        thing = "review"
    elif "ticket" in role:
        thing = "ticket"
    elif "message" in role:
        thing = "message"
    elif "post" in role:
        thing = "post"
    elif "lead" in role:
        thing = "lead"
    elif "deal" in role or "quote" in role:
        thing = "deal"
    else:
        thing = "event"
    # Make sure final char ends with period before our append.
    if not behavior.endswith("."):
        behavior = behavior + "."
    return f"{behavior} Then forward the {thing} to {target}."


def main():
    target = os.path.realpath("data/zapier/workflows-mods.jsonl")
    backup = "data/zapier/workflows-mods.jsonl.bak_pre_forward_fix"
    print(f"Loading {target}")
    if not os.path.exists(backup):
        shutil.copy2(target, backup)
        print(f"  Backup created: {backup}")

    lines_in = []
    with open(target) as f:
        for line in f:
            lines_in.append(line.rstrip("\n"))
    print(f"  {len(lines_in)} TCs loaded")

    obj_amended = 0
    tc_amended = 0
    lines_out = []
    for raw in lines_in:
        d = json.loads(raw)
        any_amended = False
        for obj in d.get("objects", []):
            if needs_forward_instruction(obj):
                obj["behavior"] = amended_behavior(obj)
                obj_amended += 1
                any_amended = True
        if any_amended:
            tc_amended += 1
        lines_out.append(json.dumps(d, ensure_ascii=False))

    with open(target, "w") as f:
        for line in lines_out:
            f.write(line + "\n")

    print(f"Amended {obj_amended} object behavior texts across {tc_amended} TCs.")


if __name__ == "__main__":
    main()
