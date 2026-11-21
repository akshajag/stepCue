"""Atomic, race-safe access to the shared state.json.

Contract:
  * writers write state.json.tmp then os.replace() it over state.json
  * readers catch JSONDecodeError / FileNotFoundError and retry

Three processes write this file (classifier, advisor, dashboard reset), so a
plain read-modify-write would lose updates. update_state() serialises writers
with a small lock file; the on-disk format and the tmp+replace rule are
unchanged.
"""
import copy
import json
import os
import time
from contextlib import contextmanager

from config import DEFAULT_STATE, STATE_FILE, STATE_TMP

_LOCK = STATE_FILE + ".lock"
_STALE_LOCK_SEC = 2.0


def read_state(retries=20, delay=0.01):
    """Return the current state dict. Never raises on a missing/torn file."""
    for _ in range(retries):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                merged = copy.deepcopy(DEFAULT_STATE)
                merged.update(data)
                return merged
        except FileNotFoundError:
            return copy.deepcopy(DEFAULT_STATE)
        except (json.JSONDecodeError, PermissionError, OSError):
            pass
        time.sleep(delay)
    return copy.deepcopy(DEFAULT_STATE)


def write_state(state):
    """Atomically replace state.json (tmp file + os.replace)."""
    with open(STATE_TMP, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
    # On Windows os.replace can fail briefly while a reader holds the file.
    for attempt in range(50):
        try:
            os.replace(STATE_TMP, STATE_FILE)
            return
        except PermissionError:
            time.sleep(0.004)
    os.replace(STATE_TMP, STATE_FILE)


@contextmanager
def _locked(timeout=1.0):
    deadline = time.monotonic() + timeout
    fd = None
    while True:
        try:
            fd = os.open(_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except (FileExistsError, PermissionError):
            try:
                if time.time() - os.path.getmtime(_LOCK) > _STALE_LOCK_SEC:
                    os.remove(_LOCK)       # holder crashed
                    continue
            except OSError:
                pass
            if time.monotonic() > deadline:
                break                      # proceed unlocked rather than stall
            time.sleep(0.002)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
            try:
                os.remove(_LOCK)
            except OSError:
                pass


def update_state(mutator):
    """Read-modify-write under the lock. `mutator(state)` edits in place.

    Only touch the keys your process owns inside the mutator:
      classifier -> status, freeze_index, episodes (except `analyzed`)
      advisor    -> bpm, haptic, alert_level, caretaker_brief, episodes[].analyzed
    """
    with _locked():
        state = read_state()
        mutator(state)
        write_state(state)
        return state


def reset_state():
    with _locked():
        write_state(copy.deepcopy(DEFAULT_STATE))


def state_age_sec():
    """Seconds since state.json was last written (inf if missing)."""
    try:
        return max(0.0, time.time() - os.path.getmtime(STATE_FILE))
    except OSError:
        return float("inf")
