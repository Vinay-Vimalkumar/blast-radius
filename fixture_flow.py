import io
import os
from functools import cache
from pathlib import Path
from types import FunctionType

from map_store import read_map


@cache
def _index():
    return read_map(Path(__file__).parent / "maps" / "fixture_flow.json.gz")


@cache
def _helpers(root):
    path = root / "tests" / "helpers.py"
    namespace = {"__file__": str(path)}
    exec(compile(path.read_bytes(), str(path), "exec"), namespace)
    return namespace


def _rows(root, path, text, loader):
    namespace = _helpers(root)
    expected = str(root / path)
    def read(name, encoding):
        if os.path.normcase(os.path.normpath(name)) != os.path.normcase(expected) or encoding != "utf-8":
            raise ValueError("unexpected fixture access")
        return io.StringIO(text)
    fn = FunctionType(namespace[loader].__code__, {**namespace, "open": read})
    rows = fn(path.removeprefix("tests/fixtures/"))
    if loader == "load_sql_fixture_pairs":
        return [(tuple(meta.items()), sql, expected) for meta, sql, expected in rows]
    return [(s,) for s in rows]


def refine_fixture(change, readers, cov):
    if change.structural or not readers:
        return readers
    entries = _index().get(change.path)
    if not entries:
        return readers
    root = Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot")).resolve()
    result = set(readers)
    for loader in {entry[0] for node, entry in entries.items() if node in readers}:
        try:
            old = _rows(root, change.path, change.original, loader)
            new = _rows(root, change.path, change.patched, loader)
        except (IndexError, KeyError, OSError, TypeError, ValueError, RecursionError):
            continue
        if len(old) != len(new):
            changed = {-1}
        else:
            changed = set()
            for a, b in zip(old, new):
                for i, (x, y) in enumerate(zip(a, b)):
                    if x == y:
                        continue
                    changed.add(i)
                    if i == 0 and loader == "load_sql_fixture_pairs":
                        previous, current = dict(x), dict(y)
                        for key in previous.keys() | current.keys():
                            if key not in previous or key not in current or previous[key] != current[key]:
                                changed.add("meta:" + key)
                                if not previous.get(key) or not current.get(key):
                                    changed.add("nonempty-label:" + key)
        for node, (kind, used) in entries.items():
            if kind == loader and changed.isdisjoint(used):
                result.discard(node)
    return result
