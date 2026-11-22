"""Remove heavy tests when a patch cannot change anything they read on their baseline path.

Four heavy tests are covered: test_execute_tpcds and test_execute_tpch, which run in a parent (every collected
module imported: setUpClass tables, parse and pickle of the queries, DuckDB transpile, unpickled worker results)
and in spawned pool workers (unpickle, find_tables, execute); test_optimized_tpch (DuckDB transpile of the TPC-H
queries and their optimized forms); and test_generator_memory (every dialect's generator, constructed repeatedly).
A process can leave its baseline path only by running changed code or by reading a state entry whose value
changed. ds_reads.json.gz (build_ds_reads.py) records for each process role the code objects it ran, the module
and class attributes it read (with receiver and defining class), the dictionary keys it looked up, misses
included, the dictionaries it used whole, its LOAD_GLOBAL names and the names given to reflective builtins.

For a patch made only of sqlglot source edits (plus documentation, or fixtures a test never opens):
  1. a changed code object that ran in one of a test's roles keeps that test (no subprocess), and so does any
     added, changed or removed statement that runs at import with an effect outside modeled state (ds_imports.py:
     a reducer registered with a pickler, an assignment into another module, a call of an unlisted function);
  2. otherwise one child imports the patched sources as a pool worker and then as the full session, and compares,
     against the image's recorded baseline (--record at build time): the loaded modules, process settings and the
     pickling reducers (copyreg's and the spawned pool's ForkingPickler's), the digest of every read entry
     (dictionaries by looked-up key, or whole when used whole) and which read entries share one mutable object,
     class and module attribute names, class MROs, enum members and the special methods of classes in use; any
     function the patch adds or changes that runs during those imports keeps every test;
  3. a changed read entry keeps the test; an added or removed attribute keeps it when its class is in use and the
     name can be reached (read through a subclass, used by executed code, reflected, or a special method).
Anything else (other files, errors, timeouts, missing records, a budget the schedule gain cannot pay) keeps it.
"""
import gzip
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import types
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
EH = "tests/test_executor.py::TestExecutor::test_execute_tpch"
OH = "tests/test_executor.py::TestExecutor::test_optimized_tpch"
GM = "tests/test_generator.py::TestGenerator::test_generator_memory"

HEAVY = {DS: ("worker", "parent"), EH: ("worker_h", "parent_h"), OH: ("oh",), GM: ("gm",)}

CONTEXTS = (("worker", ("worker", "worker_h")), ("session", ("parent", "parent_h", "oh", "gm")))
PROFILE = HERE / "ds_reads.json.gz"
BASELINE = HERE / "ds_unread_baseline.json.gz"
FORMAT = 6

EXTERNAL_WATCH = ("_pickle", "asyncio", "atexit", "concurrent", "copyreg", "ctypes", "faulthandler",
                  "multiprocessing", "pickle", "signal", "socket", "subprocess", "threading", "tracemalloc")
EXPECTED_CPU = 2.5
WALL_NEEDED = 8.0
WALL_LIMIT = 20.0


def _module_name(path):
    name = path[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


class Profile:
    def __init__(self, body):
        names = body["names"]
        records = set()
        for ids in body["stages"].values():
            records.update(names[i] for i in ids)
        self.attr_reads = {}
        self.keys = {}
        self.whole = set()
        self.module_reads = set()
        self.dynamic = set()
        self.reflected = set()
        self.code = set()
        self.receivers = set()
        self.enums_whole = set()
        for record in records:
            kind, *rest = record.split("\t")
            if kind == "a":
                receiver, definer, attr = rest
                self.attr_reads.setdefault(attr, set()).add((receiver, definer))
                self.receivers.add(receiver)
            elif kind == "k":
                self.keys.setdefault(rest[0], set()).add(rest[1])
            elif kind == "w":
                self.whole.add(rest[0])
            elif kind == "m":
                self.module_reads.add((rest[0], rest[1]))
            elif kind == "n":
                self.dynamic.add(rest[0])
            elif kind == "r":
                self.reflected.add(rest[0])
            elif kind == "f":
                self.code.add((rest[0], rest[1], int(rest[2])))
            elif kind == "e":
                self.enums_whole.add(rest[0])
        self.globals = {(path, name) for path, name in body["globals"]}
        self.co_names = set(body["co_names"])
        self.co_strings = set(body["co_strings"])
        self.code_classes = set()
        for path, qualname, _ in self.code:
            outer = qualname.split(".<locals>", 1)[0]
            if "." in outer:
                self.code_classes.add(_module_name(path) + "." + outer.rsplit(".", 1)[0])

    def rows(self):
        """Entries whose values the patched process must reproduce, and dictionary rows checked by key."""
        plain = set()
        for attr, pairs in self.attr_reads.items():
            plain.update(f"c {definer} {attr}" for _, definer in pairs)
        plain.update(f"m {module} {attr}" for module, attr in self.module_reads)
        plain.update(f"m {_module_name(path)} {name}" for path, name in self.globals if path.endswith(".py"))
        keyed = {row: sorted(keys) for row, keys in self.keys.items() if row not in self.whole}
        whole = set(self.whole)
        plain -= set(keyed) | whole
        return sorted(plain), keyed, sorted(whole)

    def in_use(self, family):
        return bool(family & self.receivers or family & self.code_classes or family & self.reflected)


@lru_cache(maxsize=1)
def profiles():
    with gzip.open(PROFILE, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    return data, {role: Profile(body) for role, body in data["roles"].items()}


def changed_functions(path, original, patched):
    """(path, qualname, firstlineno) of baseline code objects whose code differs, positions ignored."""
    from semantics import _without_positions

    def walk(code, out):
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                out.setdefault(const.co_qualname, []).append(const)
                walk(const, out)
        return out

    from ds_imports import code_of

    old = walk(code_of(path, original), {})
    new = walk(code_of(path, patched), {})
    changed = set()
    for qualname, codes in old.items():
        others = new.get(qualname, [])
        if not (len(codes) == len(others)
                and all(_without_positions(a) == _without_positions(b) for a, b in zip(codes, others))):
            changed.update((path, qualname, code.co_firstlineno) for code in codes)
    return changed





def _process_before():
    import copyreg
    return dict(os.environ), list(sys.path), dict(copyreg.dispatch_table), list(sys.meta_path)


def _content(item):
    """Content digest of a reducer or other pickling hook: code, bound arguments and closures, not just a name, so
    two functools.partial objects over different values differ."""
    import ds_rows

    try:
        return ds_rows.digest(item, frozenset())
    except Exception as exc:
        return f"undigestable {type(item).__module__}.{type(item).__qualname__} {type(exc).__name__}"


def _pickling():
    """What the pickling a test process does depends on: copyreg's table and extension registry, and for the
    spawned pool the dispatch table a fresh ForkingPickler actually uses (its class's copy of copyreg's table plus
    its extra reducers), with the ForkingPickler class and the reduction module themselves."""
    import copyreg
    import io

    state = {
        "copyreg": sorted([repr(k), _content(v)] for k, v in copyreg.dispatch_table.items()),
        "extension": sorted(map(repr, copyreg._extension_registry.items())),
        "copyreg_module": sorted([k, _content(v)] for k, v in vars(copyreg).items()
                                 if callable(v) and not k.startswith("__")),
    }
    reduction = sys.modules.get("multiprocessing.reduction")
    if reduction is not None:
        pickler = reduction.ForkingPickler(io.BytesIO())
        state["pool_table"] = sorted([repr(k), _content(v)] for k, v in pickler.dispatch_table.items())
        state["pool_override"] = _content(getattr(pickler, "reducer_override", None))
        state["pool_class"] = sorted([k, _content(v)] for k, v in vars(reduction.ForkingPickler).items()
                                     if k not in ("__dict__", "__weakref__", "__doc__"))
        state["pool_module"] = sorted([k, _content(v)] for k, v in vars(reduction).items()
                                      if callable(v) and not k.startswith("__"))
    return state


def _process_digest(before):
    """Process settings, and what importing sqlglot changed in the environment, sys.path and pickling."""
    import gc
    import warnings
    environ, path, _, meta = before
    changed_environ = sorted((k, v) for k, v in os.environ.items() if environ.get(k) != v)
    removed_environ = sorted(set(environ) - set(os.environ))
    external = sorted(name for name in sys.modules if name.split(".")[0] in EXTERNAL_WATCH)
    return hashlib.sha256(json.dumps([
        sys.getrecursionlimit(), sys.getswitchinterval(), gc.isenabled(), len(warnings.filters),
        repr(sys.modules["decimal"].getcontext()) if "decimal" in sys.modules else None,
        changed_environ, removed_environ, list(sys.path) == path, _pickling(), external,
        [type(m).__qualname__ for m in sys.meta_path if m not in meta],
    ], default=repr, sort_keys=True).encode()).hexdigest()[:24]


class _ImportWatch:
    """Records which of the given (path, qualname) functions start running while enabled (sys.monitoring)."""

    def __init__(self, watch):
        self.wanted = {}
        for path, qualname in watch:
            self.wanted.setdefault("/" + path, set()).add(qualname)
        self.seen = set()
        monitoring = sys.monitoring
        self.tool = next(i for i in range(6) if monitoring.get_tool(i) is None)
        monitoring.use_tool_id(self.tool, "ds_unread imports")
        monitoring.register_callback(self.tool, monitoring.events.PY_START, self._start)

    def _start(self, code, offset):
        filename = code.co_filename.replace("\\", "/")
        for suffix, names in self.wanted.items():
            if filename.endswith(suffix):
                if code.co_qualname in names:
                    self.seen.add(f"{suffix[1:]}:{code.co_qualname}")
                break
        return sys.monitoring.DISABLE

    def enable(self, on):
        sys.monitoring.set_events(self.tool, sys.monitoring.events.PY_START if on else 0)


def _shared(before):
    """Names, MROs, enum members and special methods of every loaded sqlglot class, and process settings."""
    import enum

    import ds_rows
    import fe_replay as fr

    arbitrary = fr._arbitrary()
    classes, modules, mros, objects, enums, dunders = {}, {}, {}, {}, {}, {}
    for name in ds_rows.sqlglot_modules():
        module = sys.modules[name]
        modules[name] = sorted(k for k in vars(module) if k not in fr.MODULE_SKIP)
        for cls in ds_rows.classes_of(module):
            qual = ds_rows.class_name(cls)
            for attr, item in vars(cls).items():
                if (attr.startswith("__") and attr.endswith("__") and attr not in fr.CLASS_SKIP
                        and attr not in ("__slots__", "__match_args__", "__abstractmethods__", "__parameters__",
                                         "__orig_bases__", "_abc_impl")
                        and not isinstance(item, (types.MemberDescriptorType, types.GetSetDescriptorType))):
                    dunders[f"c {qual} {attr}"] = ds_rows.digest(item, arbitrary)
            if issubclass(cls, enum.Enum):
                enums[qual] = [[alias, m.name, repr(m.value)] for alias, m in cls.__members__.items()]
            classes[qual] = sorted(k for k, v in vars(cls).items() if k not in ds_rows.SKIP_CLASS
                                   and not isinstance(v, (types.MemberDescriptorType, types.GetSetDescriptorType)))
            mros[qual] = [ds_rows.class_name(b) for b in cls.__mro__[1:]]
            objects[qual] = cls
    hierarchy = {}
    for qual, cls in objects.items():
        for base in cls.__mro__:
            hierarchy.setdefault(ds_rows.class_name(base), []).append(qual)
    shared = {"loaded": ds_rows.sqlglot_modules(), "process": _process_digest(before), "classes": classes,
              "modules": modules, "mros": mros, "enums": enums, "dunders": dunders}
    return shared, hierarchy, objects


def _entries(objects, profile_rows, memo):
    """Digests of one profile's read entries; dictionaries keyed by the looked-up keys only."""
    import ds_rows
    import fe_replay as fr

    arbitrary = fr._arbitrary()
    plain, keyed, whole = profile_rows

    def value(row):
        kind, _, rest = row.partition(" ")
        if kind == "d":
            if ("d", rest) not in memo:
                cls = objects.get(rest)
                if cls is None:
                    raise KeyError(row)
                memo["d", rest] = sys.modules["sqlglot.generator"]._build_dispatch(cls)
            return memo["d", rest]
        owner, _, attr = rest.rpartition(" ")
        if kind == "m":
            return vars(sys.modules[owner])[attr]
        return vars(objects[owner])[attr]

    def digest(row):
        if ("r", row) not in memo:
            try:
                memo["r", row] = ds_rows.digest(value(row), arbitrary)
            except KeyError:
                memo["r", row] = None
        return memo["r", row]

    digests = {row: digest(row) for row in plain + whole}
    keys = {}
    for row, wanted in keyed.items():
        try:
            table = value(row)
        except KeyError:
            keys[row] = None
            continue
        if type(table) is not dict:
            keys[row] = {"*": digest(row)}
            continue
        if ("k", row) not in memo:
            memo["k", row] = {ds_rows.digest(key, frozenset()): key for key in table}
        found = {}
        for kd in wanted:
            if kd in memo["k", row]:
                found[kd] = ds_rows.digest(table[memo["k", row][kd]], arbitrary)
        keys[row] = found


    roots = []
    for row in sorted({*plain, *whole, *keyed}):
        if row.startswith("d "):
            continue
        try:
            item = value(row)
        except KeyError:
            continue
        only = None
        if row in keyed and type(item) is dict:
            only = [memo["k", row][kd] for kd in keyed[row] if kd in memo.get(("k", row), {})]
        roots.append((row, item, only))
    return {"digests": digests, "keys": keys, "aliases": _topology(roots, arbitrary)}


def _topology(roots, arbitrary=frozenset(), depth_limit=8):
    """{location: first location} for every mutable container reached a second time from the read entries.

    A location is the row, then /key or /index steps; dictionary rows read by key are walked under those keys only.
    """
    import enum

    first, shared = {}, {}

    def label(key):
        if key is None or isinstance(key, (str, int, float, bool)):
            return repr(key)
        if isinstance(key, enum.Enum):
            return f"{type(key).__qualname__}.{key.name}"
        if isinstance(key, type):
            return f"{key.__module__}.{key.__qualname__}"
        if isinstance(key, tuple):
            return "(" + ",".join(label(k) for k in key) + ")"
        return f"<{type(key).__qualname__}>"

    def walk(item, where, depth, only=None):
        if isinstance(item, (dict, list, set, bytearray)):
            if id(item) in first:
                shared[where] = first[id(item)]
                return
            first[id(item)] = where
        if depth >= depth_limit or id(item) in arbitrary:
            return
        if isinstance(item, dict):
            keys = only if only is not None else sorted(item, key=label)
            for key in keys:
                if key in item:
                    walk(item[key], f"{where}/{label(key)}", depth + 1)
        elif isinstance(item, (list, tuple)):
            for index, element in enumerate(item):
                walk(element, f"{where}/{index}", depth + 1)

    for row, item, only in roots:
        walk(item, row, 0, only)
    return shared


def _observe(request):
    """Child: import the patched sources as each process kind does; return states (record) or reasons."""
    import importlib
    import unittest

    import fe_replay as fr

    fr.child_setup(request.get("sources") or {}, cpu_limit=int(request.get("cpu_limit", 30)))
    _, roles = profiles()
    wanted = set(request.get("roles") or roles)
    base = None
    if not request.get("record"):
        with gzip.open(BASELINE, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        if data.get("format") != FORMAT:
            raise ValueError("baseline format")
        base = data
    out = {"contexts": {}, "roles": {}}


    import concurrent.futures.process
    import multiprocessing.spawn
    before = _process_before()
    watch = _ImportWatch(request["watch"]) if request.get("watch") else None
    for context, members in CONTEXTS:
        if watch is not None:
            watch.enable(True)
        for name in request["worker_modules" if context == "worker" else "session_modules"]:
            importlib.import_module(name)
        if watch is not None:
            watch.enable(False)
        if sys.modules["sqlglot.expressions.core"].UNITTEST is not True:
            raise RuntimeError("sqlglot was imported without its test-mode checks")
        members = [role for role in members if role in wanted]
        if not members:
            continue
        shared, hierarchy, objects = _shared(before)
        memo = {}
        if base is None:
            out["contexts"][context] = shared
        for role in members:
            entries = _entries(objects, roles[role].rows(), memo)
            if base is None:
                out["roles"][role] = entries
            else:
                now = {**shared, **entries}
                then = {**base["contexts"][context], **base["roles"][role]}
                found = conflicts(_delta(now, then), roles[role], hierarchy)
                if watch is not None and watch.seen:
                    found.insert(0, f"new or changed function ran during import: {sorted(watch.seen)[:3]}")
                out["roles"][role] = found
    out["cpu"] = round(time.process_time(), 4)
    return out





def _delta(now, base):
    """Changed entries among the profiled ones, plus added and removed attribute names and changed MROs."""
    delta = {}
    if now["loaded"] != base["loaded"]:
        delta["loaded"] = {"kind": "value"}
    if now["process"] != base["process"]:
        delta["process"] = {"kind": "value"}
    for row, digest in base["digests"].items():
        if now["digests"].get(row, "missing") != digest:
            delta[row] = {"kind": "value"}
    for row, keys in base["keys"].items():
        mine = now["keys"].get(row, "missing")
        if keys is None or mine is None or mine == "missing" or "*" in keys or "*" in mine:
            if mine != keys:
                delta[row] = {"kind": "value"}
            continue
        changed = sorted(k for k in set(keys) | set(mine) if keys.get(k) != mine.get(k))
        if changed:
            delta[row] = {"kind": "keys", "changed": changed}
    for qual, names in now["classes"].items():
        old = base["classes"].get(qual)
        if old is None:
            delta[f"c {qual} __mro__"] = {"kind": "added"}
            continue
        if now["mros"][qual] != base["mros"][qual]:
            delta[f"c {qual} __mro__"] = {"kind": "value"}
        for attr in set(names) ^ set(old):
            delta.setdefault(f"c {qual} {attr}", {"kind": "added" if attr in names else "removed"})
    for qual in set(base["classes"]) - set(now["classes"]):
        delta[f"c {qual} __mro__"] = {"kind": "removed"}
    for row, digest in now["dunders"].items():
        if row in base["dunders"] and base["dunders"][row] != digest:
            delta.setdefault(row, {"kind": "special"})
    for qual, members in now["enums"].items():
        old = base["enums"].get(qual)
        if old is None or members == old:
            continue

        before = [tuple(m) for m in old]
        kept = [tuple(m) for m in members if tuple(m) in set(before)]
        additive = kept == before and len(members) > len(before)
        delta[f"c {qual} __members__"] = {"kind": "added" if additive else "value"}
    mine, theirs = now.get("aliases", {}), base.get("aliases", {})
    for row in set(mine) | set(theirs):
        if mine.get(row) != theirs.get(row):
            delta.setdefault(row, {"kind": "value"})
    for module, names in now["modules"].items():
        old = base["modules"].get(module)
        if old is None:
            continue
        for attr in set(names) ^ set(old):
            delta.setdefault(f"m {module} {attr}", {"kind": "added" if attr in names else "removed"})
    return delta


def conflicts(delta, profile, hierarchy):
    """Reasons the patched process can differ from its baseline path; empty when none."""
    reasons = []
    new_classes = {row[2:-len(" __mro__")] for row, info in delta.items()
                   if row.endswith(" __mro__") and info["kind"] == "added"}
    if new_classes and "__subclasses__" in profile.co_names:
        reasons.append("new classes while __subclasses__ is used")
    for row, info in sorted(delta.items()):
        if row in ("loaded", "process"):
            reasons.append(row)
            continue
        kind, _, rest = row.partition(" ")
        if kind == "d" or info["kind"] in ("value", "keys") and kind in ("c", "m") and not row.endswith(
                (" __mro__", " __members__")) and info["kind"] != "special":
            reasons.append(f"{row}: read entry differs")
            continue
        owner, _, attr = rest.rpartition(" ")
        if kind == "m":
            reflected = f"module:{owner}" in profile.reflected
            if attr in ("__getattr__", "__dir__"):
                reasons.append(f"{row}: module attribute hook")
            elif ((owner, attr) in profile.module_reads or attr in profile.dynamic or reflected
                    or any((p, attr) in profile.globals for p in (owner.replace(".", "/") + ".py",
                                                                    owner.replace(".", "/") + "/__init__.py"))):
                reasons.append(f"{row}: reachable module name")
            continue
        if owner in new_classes:
            continue
        if attr == "__members__":
            if info["kind"] != "added" or owner in profile.enums_whole or any(
                    name in profile.whole for name in (f"c {owner} _member_map_", f"c {owner} _value2member_map_")):
                reasons.append(f"{row}: enum members changed")
            continue
        family = set(hierarchy.get(owner, [owner])) | {owner}
        if not profile.in_use(family):
            continue
        if attr == "__mro__":
            reasons.append(f"{row}: class in use")
            continue
        if info["kind"] == "special":
            reasons.append(f"{row}: special method of a class in use changed")
            continue
        through = {pair for pair in profile.attr_reads.get(attr, ()) if pair[0] in family}
        if through:
            reasons.append(f"{row}: lookup through this class")
        elif attr in profile.dynamic:
            reasons.append(f"{row}: dynamic name")
        elif family & profile.reflected:
            reasons.append(f"{row}: reflected class")
        elif attr.startswith("__") and attr.endswith("__"):
            reasons.append(f"{row}: special method of a class in use")
        elif attr in profile.co_names or attr in profile.co_strings:
            reasons.append(f"{row}: name used by executed code")
    return reasons





@lru_cache(maxsize=1)
def _baseline():
    import fe_replay as fr

    with gzip.open(BASELINE, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    profile_hash = hashlib.sha256(PROFILE.read_bytes()).hexdigest()
    root = Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot")).resolve()
    import ds_rows

    if (data.get("format") != FORMAT or data["python"] != sys.version or data["profile"] != profile_hash
            or data["tree"] != fr._load("fe_baseline.json.gz")["tree"] or not fr._environment()
            or profiles()[0]["tree"] != data["tree"]
            or profiles()[0].get("digest_version") != ds_rows.DIGEST_VERSION
            or profiles()[0].get("test_mode") is not True):
        raise ValueError("ds_unread baseline does not match this environment")
    return data


def _profiled_modules():
    """sqlglot modules the profiled test processes load: the child's imports and the real pool processes' modules.
    None when unknown (then every edited module is checked)."""
    try:
        data = _baseline()
        with gzip.open(HERE / "fe_modules.json.gz", "rt", encoding="utf-8") as stream:
            modules = json.load(stream)
    except (OSError, ValueError, KeyError):
        return None
    loaded = set(modules.get("worker", ())) | set(modules.get("parent", ()))
    for context in data["contexts"].values():
        loaded |= set(context["loaded"])
    return loaded


def _request(sources):
    import fe_replay as fr

    data = _baseline()
    return {"sources": sources, "worker_modules": data["worker_modules"],
            "session_modules": fr._load("fe_baseline.json.gz")["session_modules"]}


def _run(request, limit):
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    with subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--child"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True, env=env, start_new_session=True) as proc:
        try:
            output = proc.communicate(json.dumps(request), timeout=limit)[0]
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
    lines = output.strip().splitlines()
    if proc.returncode != 0 or not lines:
        raise RuntimeError(f"child exit {proc.returncode}")
    return json.loads(lines[-1])


def _inert(change, cov, test):
    """Documentation, or a fixture file the coverage map shows other tests read and this test never opens."""
    path = change.path
    if path.endswith(".py"):
        return False
    if path.startswith(("docs/", ".github/")) or path.endswith(".md"):
        return True
    if cov is None or not path.startswith("tests/fixtures/") or "/tpc-" in path:
        return False
    readers = cov.fixture(path)
    return bool(readers) and test not in readers


def verdicts(changes, tests, limit=WALL_LIMIT, cov=None):
    """{test: (removable, reasons)} for the given heavy tests."""
    reasons = {test: [] for test in tests}
    if len({c.path for c in changes}) != len(changes):
        return {test: (False, ["repeated file"]) for test in tests}
    sources, code = {}, set()
    for change in changes:
        path = change.path
        if not change.structural and change.path.startswith("sqlglot/") and path.endswith(".py"):
            code |= changed_functions(path, change.original, change.patched)
            sources[path] = change.patched
            continue
        for test in tests:
            if change.structural or not _inert(change, cov, test):
                reasons[test].append(f"unsupported change {path}")
    effects, watch = [], set()
    if sources:
        import ds_imports

        loaded = _profiled_modules()
        for change in changes:


            if change.path in sources and (loaded is None or _module_name(change.path) in loaded):
                added = ds_imports.new_or_changed_functions(change.path, change.original, change.patched)
                watch |= {(change.path, qualname) for qualname in added}
                effects += [f"{change.path}: {reason}" for reason in ds_imports.import_effects(
                    change.path, change.original, change.patched, sources, added)]
    if effects:
        for test in tests:
            reasons[test].append(f"import-time effects: {effects[:3]}")
    _, roles = profiles()
    for test in tests:
        if not sources:
            reasons[test].append("no sqlglot source change")
        ran = sorted(f"{p}:{q}" for p, q, n in code if any((p, q, n) in roles[r].code for r in HEAVY[test]))
        if ran:
            reasons[test].append(f"changed code ran: {ran[:5]}")
    open_tests = [test for test in tests if not reasons[test]]
    if open_tests:
        request = _request(sources)
        request["roles"] = sorted({role for test in open_tests for role in HEAVY[test]})
        request["watch"] = sorted(watch)
        observed = _run(request, limit)["roles"]
        for test in open_tests:
            reasons[test] += [f"{role}: {text}" for role in HEAVY[test] for text in observed[role]]
    return {test: (not reasons[test], reasons[test]) for test in tests}


def verdict(changes, limit=WALL_LIMIT, cov=None):
    """(removable, reasons) for test_execute_tpcds alone."""
    return verdicts(changes, [DS], limit, cov)[DS]


def _cost(nodes):
    from fe_replay import _durations
    from schedule import distribute

    table = _durations()
    millis = {number: duration for number, duration in table.values()}
    workers = [w for w in distribute((table[node][0] for node in nodes if node in table), millis) if w]
    return max(sum(millis[n] for n in w) for w in workers) + 2000 * len(workers) if workers else 0


def _saving(chosen, removable, full_cost):
    """Schedule seconds saved by removing the given tests; nothing when padding already fills the catalog."""
    before = _cost(chosen)
    if full_cost is not None and before >= full_cost:
        return 0.0
    return (before - _cost(set(chosen) - set(removable))) / 1000


def refine_unread(changes, chosen, cov, catalog, verify_until, full_cost=None, log=None):
    heavy = [test for test in HEAVY if test in chosen and test in catalog and test not in cov.missing
             and test in cov.nodes]
    if not heavy:
        return chosen
    try:
        heavy = [test for test in heavy if all(role in profiles()[1] for role in HEAVY[test])]
        if not heavy:
            return chosen
        remaining = verify_until - time.monotonic()
        if remaining < WALL_NEEDED:
            return chosen
        if _saving(chosen & catalog, heavy, full_cost) < 1.5 * EXPECTED_CPU + 1.0:
            return chosen
        results = verdicts(changes, heavy, min(WALL_LIMIT, remaining - 1), cov)
        cleared = {test for test, (removable, _) in results.items() if removable}
        if log is not None:
            log(f"ds_unread cleared {sorted(cleared)}; kept {[(t, r[:2]) for t, (ok, r) in results.items() if not ok]}")
        return chosen - cleared
    except Exception as exc:
        if log is not None:
            log(f"ds_unread error {type(exc).__name__}")
        return chosen


def record(dest):
    """Image build step: observe the unpatched processes and keep the baseline next to the profile."""
    import fe_replay as fr

    import ds_rows

    data, _ = profiles()
    if data.get("digest_version") != ds_rows.DIGEST_VERSION:
        raise SystemExit("profile keys were digested by another ds_rows version")
    if data.get("test_mode") is not True:
        raise SystemExit("profile was not taken with sqlglot's test-mode checks")
    with gzip.open(HERE / "fe_modules.json.gz", "rt", encoding="utf-8") as stream:
        modules = json.load(stream)
    worker_modules = modules.get("worker_order") or modules["worker"]
    session = fr._load("fe_baseline.json.gz")["session_modules"]
    request = {"sources": {}, "worker_modules": worker_modules, "session_modules": session, "cpu_limit": 120,
               "record": True}
    observed = _run(request, 300)
    second = _run(request, 300)
    for context, members in CONTEXTS:
        for role in (r for r in members if r in observed["roles"]):
            now = {**second["contexts"][context], **second["roles"][role]}
            then = {**observed["contexts"][context], **observed["roles"][role]}
            if _delta(now, then):
                raise SystemExit(f"{role} observation is not reproducible")
    record = {"format": FORMAT, "python": sys.version, "tree": fr._load("fe_baseline.json.gz")["tree"],
              "profile": hashlib.sha256(PROFILE.read_bytes()).hexdigest(), "worker_modules": worker_modules,
              "contexts": observed["contexts"], "roles": observed["roles"]}
    if data["tree"] != record["tree"]:
        raise SystemExit("profile and baseline trees differ")
    with gzip.open(dest, "wt", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True)
    print(f"ds_unread baseline recorded, child cpu {observed['cpu']}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        with os.fdopen(os.dup(1), "w") as channel, open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            request = json.load(sys.stdin)
            sys.path.insert(0, str(HERE))
            channel.write(json.dumps(_observe(request)) + "\n")
    elif sys.argv[1:2] == ["--record"]:
        sys.exit(record(sys.argv[2]))
