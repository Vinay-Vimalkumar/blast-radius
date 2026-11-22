import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import os
import subprocess
import sys
from functools import cache
from pathlib import Path
from unittest import TestResult

from map_store import read_map
from semantics import semantic_lines


@cache
def _profile():
    return read_map(Path(__file__).parent / "maps" / "replay.json.gz")


class PatchedImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, changes, root):
        self.source = {}
        for path, body in changes.items():
            name = path[:-3].replace("/", ".").removesuffix(".__init__")
            self.source[name] = (root / path, body)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.source:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or Path(spec.origin).resolve() != self.source[fullname][0]:
            raise ValueError("patched module does not match the pinned source")
        spec.loader = self
        return spec

    def exec_module(self, module):
        path, source = self.source[module.__name__]
        exec(compile(source.encode("utf-8"), str(path), "exec", dont_inherit=True), module.__dict__)


def _hits(index, path, numbers):
    return bool(numbers.intersection(index.get(path, ())))


def _plan_max(changes, data):
    affected = set()
    for change in changes:
        path = change.path
        if change.structural or not path.startswith(("sqlglot/", "tests/")) or not path.endswith(".py"):
            return None
        if path not in data["files"]:
            continue
        numbers = semantic_lines(path, change.original, change.patched, change.removed)
        if numbers is None or _hits(data["imports"], path, numbers) or _hits(data["outside"], path, numbers):
            return None
        for i, call in enumerate(data["calls"]):
            if _hits(call["lines"], path, numbers) or _hits(call["tokenization"], path, numbers):
                affected.add(i)
    if affected and any(c["size"] > 100_000 for c in data["calls"][min(affected):]):
        return None
    return affected


def _run_max(data):
    mod = importlib.import_module("tests.test_parser")
    index = 0
    original = mod.parse_one
    from sqlglot.errors import ParseError

    def parse(sql, **opts):
        nonlocal index
        case = data["calls"][index]
        index += 1
        if hashlib.sha256(sql.encode()).hexdigest() != case["sql"] or opts != case["args"]:
            raise RuntimeError("stress-test invocation changed")
        if case["size"] > 100_000:
            exception, message = case["exception"]
            if exception != "ParseError":
                raise ValueError("unexpected exception class")
            raise ParseError(message)
        return original(sql, **opts)

    mod.parse_one = parse
    result = TestResult()
    mod.TestParser("test_max_nodes").run(result)
    return result.wasSuccessful() and index == len(data["calls"])


def _child(request):
    root = Path(request["root"]).resolve()
    sys.path.insert(0, str(root))
    sys.meta_path.insert(0, PatchedImports(request["patch"], root))
    return _run_max(_profile())


def _verify(changes, root, timeout):
    req = {"root": str(root), "patch": {c.path: c.patched for c in changes}}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    try:
        proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker"],
                              input=json.dumps(req).encode(), stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 79


def refine_replay(changes, chosen, cov=None, catalog=None):
    data = _profile()
    node = data["node"]
    if node not in chosen:
        return chosen
    plan = _plan_max(changes, data)
    if plan is None:
        return chosen
    root = Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot"))
    if not plan or _verify(changes, root, 3):
        return chosen - {node}
    return chosen


if __name__ == "__main__":
    try:
        ok = _child(json.load(sys.stdin))
    except Exception:
        ok = False
    raise SystemExit(79 if ok else 1)
