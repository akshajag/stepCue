"""Export a privacy-safe StepCue data snapshot for a NemoClaw / OpenShell sandbox.

The sandbox gets episodes, cue settings, agent decisions and phone check-ins, and
nothing else: no names, no emails, no location trail, no pairing key, no accounts.
Free text (briefs, reasoning, spoken check-in transcripts) is scrubbed with
nvidia_nim.scrub first.

Run from the project folder:  python nemoclaw/make_snapshot.py   -> nemoclaw/snapshot/
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import nvidia_nim  # noqa: E402

KEEP_STATE = ("status", "freeze_index", "bpm", "haptic", "alert_level", "caretaker_brief", "episodes", "motion")


def snapshot(out_dir=os.path.join(HERE, "snapshot"), root=ROOT):
    names = nvidia_nim.known_names(os.path.join(root, "users.json"))
    os.makedirs(out_dir, exist_ok=True)

    def load(name, default):
        try:
            with open(os.path.join(root, name), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return default

    state = load("state.json", {})
    state = nvidia_nim.scrub_obj({k: state[k] for k in KEEP_STATE if k in state}, names)
    comp = load("companion.json", {})
    comp = nvidia_nim.scrub_obj({"checkins": comp.get("checkins", [])[-50:],
                                 "confirmations": comp.get("confirmations", {})}, names)
    decisions = []
    try:
        with open(os.path.join(root, "advisor_log.jsonl"), encoding="utf-8") as f:
            for line in f.readlines()[-100:]:
                try:
                    decisions.append(nvidia_nim.scrub_obj(json.loads(line), names))
                except ValueError:
                    pass
    except OSError:
        pass

    for name, data in (("state.json", state), ("companion.json", comp)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
    with open(os.path.join(out_dir, "advisor_log.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(d) + "\n" for d in decisions)
    return {"episodes": len(state.get("episodes", [])), "decisions": len(decisions),
            "checkins": len(comp["checkins"]), "dir": out_dir}


if __name__ == "__main__":
    print(snapshot())
