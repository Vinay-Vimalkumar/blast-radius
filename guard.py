"""Supervise the selector so that every commit gets exactly one valid line before the batch limit.

The selector runs as a child process. Its stdout lines are checked against the output rules (one JSON object per
commit, known commit, known test numbers, no repeats, at most eight occupied workers) and passed through unchanged;
anything else is dropped. A commit left unanswered gets the full catalog, which can never omit a failing test:

- the child stops early (crash, kill, exit): the commit it was working on gets the full catalog and a fresh child
  continues with the rest, a limited number of times;
- the child prints nothing for a long time: the same, as for a stopped child;
- the batch deadline arrives: every unanswered commit gets the full catalog.

Only the standard library is used, so a fault in the selector's own modules or maps cannot reach this file.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROLE = "BR_SELECTOR_ROLE"
SKIP = "BR_SELECTOR_SKIP"
BUDGET = "BR_SELECTOR_BUDGET"

SMALL_BATCH = 30
SMALL_DEADLINE = 150.0
LARGE_DEADLINE = 225.0

STALL = 100.0
ATTEMPTS = 3
STOCK_BUDGET = 85.0
RESERVE = 20.0


def _reject(_):
    raise ValueError("not standard JSON")


def checked_commit(raw, pending, numbers):
    """The commit a stdout line answers, or None when the line breaks an output rule."""
    try:
        row = json.loads(raw.decode("utf-8"), parse_constant=_reject)
    except (ValueError, RecursionError):
        return None
    if type(row) is not dict or len(row) != 2 or "commit" not in row or "workers" not in row:
        return None
    commit, workers = row["commit"], row["workers"]
    if type(commit) is not str or commit not in pending or type(workers) is not list:
        return None
    seen = set()
    occupied = 0
    for group in workers:
        if type(group) is not list:
            return None
        occupied += bool(group)
        for number in group:
            if type(number) is not int or number not in numbers or number in seen:
                return None
            seen.add(number)
    return commit if occupied <= 8 else None


def full_catalog(millis):
    """A worker plan holding every test: least-loaded packing, cheapest of one to eight workers."""
    ordered = sorted(millis, key=lambda n: (-millis[n], n))
    if not ordered:
        return []
    best, best_cost = [ordered], sum(millis.values()) + 2000
    for count in range(2, min(len(ordered), 8) + 1):
        workers = [[] for _ in range(count)]
        loads = [0] * count
        for number in ordered:
            slot = loads.index(min(loads))
            workers[slot].append(number)
            loads[slot] += millis[number]
        cost = max(loads) + 2000 * count
        if cost < best_cost:
            best, best_cost = workers, cost
    return best


class _Output:
    def __init__(self, commits, millis, stream):
        self.lock = threading.Lock()
        self.order = commits
        self.pending = set(commits)
        self.numbers = frozenset(millis)
        self.millis = millis
        self.stream = stream
        self.closed = False
        self.progress = time.monotonic()
        self._full = None

    def remaining(self):
        with self.lock:
            return [commit for commit in self.order if commit in self.pending]

    def _write(self, commit, line):
        self.stream.write(line + b"\n")
        self.stream.flush()
        self.pending.discard(commit)

    def forward(self, raw):
        with self.lock:
            if self.closed:
                return
            commit = checked_commit(raw, self.pending, self.numbers)
            if commit is None:
                return
            self._write(commit, raw)
            self.progress = time.monotonic()

    def _fallback(self, commit):
        if self._full is None:
            self._full = full_catalog(self.millis)
        row = {"commit": commit, "workers": self._full}
        self._write(commit, json.dumps(row, separators=(",", ":")).encode("utf-8"))

    def give_full(self, commit):
        with self.lock:
            if not self.closed and commit in self.pending:
                self._fallback(commit)

    def close(self):
        """Stop forwarding and answer everything still open with the full catalog."""
        with self.lock:
            self.closed = True
            for commit in self.order:
                if commit in self.pending:
                    self._fallback(commit)


def _pump(pipe, output):
    try:
        for raw in iter(pipe.readline, b""):
            if raw.endswith(b"\n"):
                output.forward(raw[:-1])
    except (OSError, ValueError):
        pass


def _stop(proc):
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _log(message):
    try:
        print(f"guard: {message}", file=sys.stderr, flush=True)
    except OSError:
        pass


def _inputs():
    data = Path(os.environ.get("INPUT_DIR", "/data"))
    tests = json.loads((data / "tests.json").read_bytes())["tests"]
    millis = {test["number"]: round(1000 * test["seconds"]) for test in tests}

    commits = [patch.stem for patch in sorted((data / "commits").glob("*.patch"))]
    return commits, millis


def supervise(script, command=None, stream=None):
    """Run the selector under supervision and exit the process; return only if supervision cannot start."""
    began = time.monotonic()
    try:
        commits, millis = _inputs()
        stream = stream or sys.stdout.buffer
        output = _Output(commits, millis, stream)
        base = list(command) if command else [sys.executable, "-u", script]
    except Exception as exc:
        _log(f"not supervising ({type(exc).__name__})")
        return
    try:
        _run(base, output, began + (SMALL_DEADLINE if len(commits) <= SMALL_BATCH else LARGE_DEADLINE))
    except BaseException as exc:
        _log(f"supervision failed ({type(exc).__name__})")
    try:
        output.close()
    finally:
        os._exit(0)


def _run(base, output, deadline):
    for attempt in range(ATTEMPTS):
        todo = output.remaining()
        now = time.monotonic()
        if not todo or now >= deadline - RESERVE / 4:
            return
        env = dict(os.environ)
        env[ROLE] = "worker"
        if attempt:
            answered = [commit for commit in output.order if commit not in set(todo)]
            env[SKIP] = json.dumps(answered)
            env[BUDGET] = repr(max(0.0, min(STOCK_BUDGET, deadline - now - RESERVE)))
        proc = subprocess.Popen(base, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, env=env)
        with output.lock:
            output.progress = time.monotonic()
        reader = threading.Thread(target=_pump, args=(proc.stdout, output), daemon=True)
        reader.start()
        outcome = None
        while outcome is None:
            now = time.monotonic()
            if proc.poll() is not None:
                outcome = "exit"
            elif now >= deadline:
                outcome = "deadline"
            elif now - output.progress > STALL:
                outcome = "stall"
            else:
                time.sleep(0.05)
        if outcome != "exit":
            _stop(proc)


        grace = time.monotonic() + (2.0 if outcome == "exit" else 1.0)
        while reader.is_alive() and output.remaining() and time.monotonic() < grace:
            reader.join(0.05)
        if outcome == "deadline":
            _log("deadline reached")
            return
        todo = output.remaining()
        if not todo:
            return
        _log(f"selector stopped ({outcome}, code {proc.returncode}) at {todo[0]}; using the full suite for it")
        output.give_full(todo[0])
