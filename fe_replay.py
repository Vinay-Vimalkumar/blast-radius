import ast
import gzip
import hashlib
import json
import os
import re
import signal
import struct
import subprocess
import sys
import time
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
FIXTURE = "optimizer/tpc-ds/tpc-ds.sql"
PINNED = ("tests/__init__.py", "tests/helpers.py", "tests/test_executor.py", "tests/fixtures/" + FIXTURE)
WALL_NEEDED = 12.0
WALL_LIMIT = 30.0


def _digest(value):
    data = value if isinstance(value, bytes) else value.encode("utf-8", "surrogatepass")
    return hashlib.sha256(data).hexdigest()[:24]


@lru_cache(maxsize=None)
def _load(name):
    with gzip.open(HERE / name, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def tree_hash(root):
    total = hashlib.sha256()
    names = sorted(p.relative_to(root).as_posix() for p in (root / "sqlglot").rglob("*.py"))
    for name in [*names, *PINNED]:
        file = root / name
        total.update(name.encode())
        total.update(hashlib.sha256(file.read_bytes()).digest() if file.exists() else b"absent")
    return total.hexdigest()



@lru_cache(maxsize=1)
def _environment():
    try:
        baseline = _load("fe_baseline.json.gz")
        root = Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot")).resolve()
        return (baseline["python"] == sys.version and baseline["tree"] == tree_hash(root)
                and _load("fe_consumer.json.gz")["tree"] == baseline["tree"])
    except (OSError, EOFError, ValueError, KeyError):
        return False


@lru_cache(maxsize=1)
def _line_index():
    data = _load("fe_lines.json.gz")
    if data["tree"] != _load("fe_baseline.json.gz")["tree"]:
        raise ValueError("line index and baseline come from different trees")
    return ({path: frozenset(lines) for path, lines in data["parent"].items()}, data["worker"], data["queries"],
            data["plan"])


def _plan(changed):
    parent_lines, worker_lines, count, _ = _line_index()
    parent = any(not lines.isdisjoint(parent_lines.get(path, ())) for path, lines in changed.items())
    mask = 0
    for path, lines in changed.items():
        table = worker_lines.get(path, {})
        for line in lines:
            mask |= int(table.get(str(line), "0"), 16)
    worker = [index for index in range(count) if mask >> index & 1]
    if not parent and not worker:
        return True, list(range(count))
    return parent, worker


def _planned(changed, worker):
    """The replayed queries that must also build their plan: those whose rendering or plan construction runs a
    changed line. A change that no recorded step runs replays everything, plans included."""
    _, worker_lines, _, plan_lines = _line_index()
    mask = known = 0
    for path, lines in changed.items():
        table, seen = plan_lines.get(path, {}), worker_lines.get(path, {})
        for line in lines:
            mask |= int(table.get(str(line), "0"), 16)
            known |= int(seen.get(str(line), "0"), 16)
    if not known:
        return list(worker)
    return [index for index in worker if mask >> index & 1]


@lru_cache(maxsize=1)
def _durations():
    data = Path(os.environ.get("INPUT_DIR", "/data"))
    tests = json.loads((data / "tests.json").read_bytes())["tests"]
    return {test["id"]: (test["number"], round(1000 * test["seconds"])) for test in tests}


def _saving(chosen, catalog, full_cost):
    from schedule import distribute

    table = _durations()
    millis = {number: duration for number, duration in table.values()}

    def cost(nodes):
        workers = [worker for worker in distribute((table[node][0] for node in nodes), millis) if worker]
        return max(sum(millis[n] for n in worker) for worker in workers) + 2000 * len(workers) if workers else 0

    nodes = [node for node in chosen if node in catalog and node in table]
    before = cost(nodes)
    if full_cost is not None and before >= full_cost:
        return 0.0
    return (before - cost([node for node in nodes if node != DS])) / 1000


def _expected_cpu(parent, worker, planned=()):
    baseline = _load("fe_baseline.json.gz")
    per_query, per_plan = baseline["worker_cpu"], baseline["plan_cpu"]
    return (0.5 + (2.6 if parent else 0.0) + (0.4 if worker else 0.0)
            + sum(per_query.get(str(index), 0.01) + 0.01 for index in worker)
            + sum(per_plan.get(str(index), 0.05) + 0.01 for index in planned))


_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_MUTATORS = frozenset("""
    add append appendleft clear difference_update discard extend extendleft insert
    intersection_update pop popitem popleft remove reverse rotate setdefault sort
    symmetric_difference_update update __delattr__ __delitem__ __setattr__ __setitem__
""".split())


def _shared_write(tree, rows):
    if not rows:
        return False
    rows = sorted(rows)

    def self_args(node):
        return (isinstance(node, ast.Attribute) and node.attr == "args"
                and isinstance(node.value, ast.Name) and node.value.id == "self")

    @lru_cache(None)
    def bindings(scope):
        shared = set()
        args = scope.args.posonlyargs + scope.args.args
        instance = bool(args and args[0].arg == "self")
        pending = [scope.body] if isinstance(scope, ast.Lambda) else list(scope.body)
        while pending:
            node = pending.pop()
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                shared.update(node.names)
            if isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del)):
                if (isinstance(node, ast.Name) and node.id == "self") or self_args(node):
                    instance = False
            if not isinstance(node, (*_FUNCTION, ast.ClassDef)):
                pending.extend(ast.iter_child_nodes(node))
        return shared, instance

    def instance_args(node, scope):
        return self_args(node) and bindings(scope)[1]

    def shared_target(target, scope):
        if isinstance(target, (ast.List, ast.Tuple)):
            return any(shared_target(part, scope) for part in target.elts)
        if isinstance(target, ast.Starred):
            return shared_target(target.value, scope)
        if isinstance(target, ast.Name):
            return target.id.isupper() or target.id in bindings(scope)[0]

        owner = target.value if isinstance(target, ast.Subscript) else target
        return not instance_args(owner, scope)

    def visit(node, scope):
        if hasattr(node, "lineno"):
            index = bisect_left(rows, node.lineno)
            if index == len(rows) or rows[index] > node.end_lineno:
                return False
        if isinstance(node, _FUNCTION):
            body = [node.body] if isinstance(node, ast.Lambda) else node.body
            return any(visit(stmt, node) for stmt in body)
        if isinstance(node, ast.ClassDef):
            scope = None
        if scope is not None:
            if isinstance(node, (ast.Global, ast.Nonlocal, ast.AugAssign)):

                return True
            if isinstance(node, (ast.Assign, ast.Delete)):
                if any(shared_target(target, scope) for target in node.targets):
                    return True
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                if shared_target(node.target, scope):
                    return True
            elif isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del)):
                if shared_target(node, scope):
                    return True
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name in {"setattr", "delattr", "setitem", "delitem"}:
                    return True
                if isinstance(func, ast.Attribute) and func.attr in _MUTATORS:
                    if not instance_args(func.value, scope):
                        return True
        return any(visit(child, scope) for child in ast.iter_child_nodes(node))

    return visit(tree, None)


def _writes(change):
    from semantics import _baseline, patched_tree

    return (_shared_write(patched_tree(change.path, change.patched), change.added)
            or _shared_write(_baseline(change.path, change.original)[0], change.removed))


def _admissible(changes, cov, catalog, consumer):
    from import_effects import import_consumers
    from semantics import semantic_lines

    from fe_fence import trivia_only

    if len({change.path for change in changes}) != len(changes):
        return "repeated file"
    sources, changed = {}, {}
    for change in changes:
        path = change.path
        if change.structural:
            return "structural change"
        if not path.endswith(".py"):
            directory, _, name = path.rpartition("/")
            if directory != "tests/fixtures/optimizer" or not name.endswith(".sql"):
                return "unsupported fixture edit"
            readers = cov.fixture(path)
            if not readers or DS in readers:
                return "fixture reads overlap or are unknown"
            continue
        if not path.startswith("sqlglot/"):
            return "python edit outside sqlglot"
        lines = semantic_lines(path, change.original, change.patched, change.removed)
        if lines is None:
            return "definition-time edit"
        if not lines:
            if not trivia_only(change):
                return "nontrivial additional source"
            continue
        if _writes(change):
            return "edit writes shared state"
        if set(lines).intersection(consumer.get(path, ())):
            return "changed line runs on the execution side"
        if import_consumers(path, lines, cov, catalog):
            return "changed line runs during import"
        sources[path] = change.patched
        changed[path] = set(lines)
    return (sources, changed) if sources else "no executable edit"


def refine_frontend(changes, chosen, cov, catalog, verify_until, full_cost=None):
    if DS not in chosen or DS not in catalog or DS in cov.missing or DS not in cov.nodes:
        return chosen
    try:
        remaining = verify_until - time.monotonic()
        if remaining < WALL_NEEDED:
            return chosen
        admitted = _admissible(changes, cov, catalog, _load("fe_consumer.json.gz")["test_execute_tpcds"])
        if isinstance(admitted, str) or not _environment():
            return chosen
        parent, worker = _plan(admitted[1])
        planned = _planned(admitted[1], worker)
        if _saving(chosen, catalog, full_cost) < 1.5 * _expected_cpu(parent, worker, planned) + 1.0:
            return chosen
        verdict = _replay({"sources": admitted[0], "parent": parent, "worker": worker, "plan": planned},
                          min(WALL_LIMIT, remaining - 1))
        return chosen - {DS} if verdict == "ok" else chosen
    except Exception:
        return chosen


def _replay(request, limit):
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
            while True:
                try:
                    os.waitpid(-proc.pid, 0)
                except ChildProcessError:
                    break
            if sys.exc_info()[0] is subprocess.TimeoutExpired:
                return "timeout"
            raise
    lines = output.strip().splitlines()
    return lines[-1] if proc.returncode == 0 and lines else f"child exit {proc.returncode}"



class _Mismatch(Exception):
    pass


class Stages:

    def __init__(self):
        self.cpu = {}
        self.per = {}
        self.kind = None
        self.index = None
        self.began = 0.0

    def __call__(self, kind, index):
        self.kind, self.index = kind, index
        return self

    def __enter__(self):
        self.began = time.process_time()

    def __exit__(self, *exc):
        spent = time.process_time() - self.began
        self.cpu[self.kind] = self.cpu.get(self.kind, 0.0) + spent
        self.per[self.index] = self.per.get(self.index, 0.0) + spent
        return False

    def report(self):
        return {"cpu": {kind: round(value, 4) for kind, value in self.cpu.items()},
                "per": {str(index): round(value, 4) for index, value in self.per.items()}}


def _install(root, sources):
    import importlib.abc
    import importlib.util

    modules = {}
    for path, text in sources.items():
        name = path[:-3].replace("/", ".")
        package = name.endswith(".__init__")
        if package:
            name = name[: -len(".__init__")]
        modules[name] = (root / path, text, package)

    class Overlay(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in modules:
                return None
            file, _, package = modules[fullname]
            spec = importlib.util.spec_from_loader(fullname, self, origin=str(file), is_package=package)
            if package and spec is not None:
                spec.submodule_search_locations = [str(file.parent)]
            return spec

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            file, text, _ = modules[module.__name__]
            module.__file__ = str(file)
            exec(compile(text.encode("utf-8"), str(file), "exec", dont_inherit=True), module.__dict__)

    sys.meta_path.insert(0, Overlay())


def prepare():
    import unittest

    import sqlglot
    import sqlglot.expressions.core as core

    if not core.UNITTEST:
        raise RuntimeError("the project was imported without its test-mode validation")
    import sqlglot.errors
    import sqlglot.executor
    import sqlglot.executor.python
    import sqlglot.executor.table
    import sqlglot.optimizer
    import sqlglot.planner
    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

    return list(load_sql_fixture_pairs(FIXTURE)), TPCDS_SCHEMA


def _freezer(track, arbitrary=frozenset()):
    import collections
    import enum
    import functools
    import re
    import types

    seen, keep, objects, classes, queued, slot_names = {}, [], [], [], set(), {}
    none = type(None)
    native = (types.BuiltinFunctionType, types.MethodDescriptorType, types.WrapperDescriptorType,
              types.GetSetDescriptorType, types.MemberDescriptorType, types.ClassMethodDescriptorType,
              types.MethodWrapperType)

    def cref(cls):
        return f"{cls.__module__}.{cls.__qualname__}"

    def slots(kind):
        names = slot_names.get(kind)
        if names is None:
            names = []
            for klass in kind.__mro__:
                declared = klass.__dict__.get("__slots__", ())
                for name in (declared,) if isinstance(declared, str) else declared:
                    if name not in ("__dict__", "__weakref__"):
                        names.append(name)
            slot_names[kind] = names
        return names

    def stable(key):
        kind = type(key)
        if kind is str or kind is int or kind is bool or kind is none or kind is float or kind is bytes:
            return True
        if kind is tuple:
            return all(stable(part) for part in key)
        return isinstance(key, enum.Enum)

    def order_key(value):
        kind = type(value)
        if kind is str or kind is int or kind is bool or kind is none or kind is float or kind is bytes:
            return f"{kind.__name__}:{value!r}"
        if isinstance(value, enum.Enum):
            return f"E:{cref(kind)}:{value.name}"
        if isinstance(value, type):
            return f"C:{cref(value)}"
        local, local_objects, _ = _freezer(False, arbitrary)
        return "Z:" + json.dumps([local(value), local_objects], separators=(",", ":"))

    def freeze(x):
        kind = type(x)
        if kind is str or kind is int or kind is bool or kind is none:
            return x
        if kind is float or kind is bytes or kind is complex:
            return [kind.__name__, repr(x)]
        if isinstance(x, enum.Enum):
            return ["E", cref(kind), x.name]
        if isinstance(x, type):
            if track and str(x.__module__).startswith("sqlglot") and id(x) not in queued:
                queued.add(id(x))
                classes.append(x)
            return ["C", cref(x)]
        if kind is types.ModuleType:
            return ["M", x.__name__]
        if isinstance(x, native):
            return ["B", getattr(x, "__qualname__", kind.__name__)]
        if kind is re.Pattern:
            return ["P", x.pattern, x.flags]
        if isinstance(x, (str, int, float, bytes)):
            return ["V", cref(kind), repr(x)]
        number = seen.get(id(x))
        if number is not None:
            return ["R", number]
        number = seen[id(x)] = len(objects)
        keep.append(x)
        objects.append(None)
        if kind is types.FunctionType:
            cells = []
            for cell in x.__closure__ or ():
                try:
                    cells.append(freeze(cell.cell_contents))
                except ValueError:
                    cells.append(["empty"])
            state = ["F", x.__module__, x.__qualname__, freeze(x.__defaults__), freeze(x.__kwdefaults__), cells,
                     freeze(x.__dict__) if x.__dict__ else None]
        elif kind is types.MethodType:
            state = ["m", freeze(x.__func__), freeze(x.__self__)]
        elif kind is classmethod or kind is staticmethod:
            state = [kind.__name__, freeze(x.__func__)]
        elif kind is property:
            state = ["p", freeze(x.fget), freeze(x.fset), freeze(x.fdel)]
        elif kind is functools.partial:
            state = ["partial", freeze(x.func), freeze(x.args), freeze(x.keywords)]
        else:
            if isinstance(x, dict):
                entries = list(x.items())
                if id(x) in arbitrary:
                    entries.sort(key=lambda entry: order_key(entry[0]))
                body = ["D", [[freeze(k), freeze(v)] for k, v in entries]]
            elif isinstance(x, (list, tuple, collections.deque)):
                values = list(x)
                if id(x) in arbitrary:
                    values.sort(key=order_key)
                body = ["L" if isinstance(x, list) else "T" if isinstance(x, tuple) else "Q", [freeze(v) for v in values]]
            elif isinstance(x, (set, frozenset)):
                values = list(x)
                if not all(stable(value) for value in values):
                    values.sort(key=order_key)
                body = ["S", [freeze(v) for v in values]]
            else:
                body = None
            if kind in (dict, list, tuple, set, frozenset):
                state = body
            elif body is None and not str(kind.__module__).startswith(("sqlglot", "tests")):
                try:
                    text = repr(x)
                except Exception:
                    text = None
                state = ["O", cref(kind), text if text is not None and "0x" not in text else None]
            else:
                fields = []
                for name in slots(kind):
                    try:
                        fields.append([name, freeze(getattr(x, name))])
                    except AttributeError:
                        pass
                own = getattr(x, "__dict__", None)
                if own:
                    fields.append(["__dict__", [[freeze(k), freeze(v)] for k, v in own.items()]])
                state = ["I", cref(kind), body, fields]
        objects[number] = state
        return ["R", number]

    return freeze, objects, classes


def tree_digest(tree):
    freeze, objects, _ = _freezer(False)
    return _digest(json.dumps([freeze(tree), objects], separators=(",", ":")))


MODULE_SKIP = frozenset({"__builtins__", "__loader__", "__spec__", "__cached__", "__file__", "__path__", "__doc__",
                         "__annotations__", "__annotate__", "__conditional_annotations__"})
CLASS_SKIP = frozenset({"__dict__", "__weakref__", "__doc__", "__module__", "__qualname__", "__firstlineno__",
                        "__static_attributes__", "__annotations__", "__annotate__", "__annotate_func__",
                        "__annotations_cache__", "__parameters__", "__orig_bases__", "__classdictcell__"})


ARBITRARY_ORDER = ("EXPRESSION_METADATA", "_CANONICALIZE_TYPES")


def _arbitrary():
    found, seen = set(), set()

    def visit(owner):
        for name, value in list(vars(owner).items()):
            if name in ARBITRARY_ORDER and isinstance(value, (dict, tuple, list)):
                found.add(id(value))
            elif isinstance(value, type) and id(value) not in seen and str(value.__module__).startswith("sqlglot"):
                seen.add(id(value))
                visit(value)

    for name in [name for name in sys.modules if name == "sqlglot" or name.startswith("sqlglot.")]:
        module = sys.modules[name]
        if module is not None:
            visit(module)
    return frozenset(found)


def state_parts():
    freeze, objects, classes = _freezer(True, _arbitrary())
    rows = []
    names = sorted(name for name in sys.modules if name == "sqlglot" or name.startswith("sqlglot."))
    rows.append(("loaded", names))
    for name in names:
        module = sys.modules[name]
        if module is not None:
            rows.append((f"module {name}",
                         [[key, freeze(value)] for key, value in list(vars(module).items()) if key not in MODULE_SKIP]))
    position = 0
    while position < len(classes):
        cls = classes[position]
        position += 1
        bases = [f"{base.__module__}.{base.__qualname__}" for base in cls.__mro__[1:]]
        rows.append((f"class {cls.__module__}.{cls.__qualname__}",
                     [bases, [[key, freeze(value)] for key, value in list(vars(cls).items()) if key not in CLASS_SKIP]]))
    rows.append(("objects", objects))
    import gc
    import warnings

    rows.append(("process", [sys.getrecursionlimit(), sys.getswitchinterval(), gc.isenabled(), len(warnings.filters),
                             repr(sys.modules["decimal"].getcontext()) if "decimal" in sys.modules else None]))
    return rows


def state_digest():
    return _digest(json.dumps(state_parts(), separators=(",", ":")))


def _loaded():
    import sqlglot.generator as generator

    modules = sorted(name for name in sys.modules if name.startswith("sqlglot"))
    dispatch = sorted(f"{cls.__module__}.{cls.__qualname__}" for cls in getattr(generator, "_DISPATCH_CACHE", ()))
    return json.dumps([modules, dispatch])


def plan_digest(plan):
    """Digest of what the executor can read from a plan, free of object addresses and of set order.

    An expression held by the plan or by a step is named by the tree it hangs in and its place in that tree, and
    every such tree is digested whole, parents and annotations included. A step is digested from its own fields
    and, bottom-up, from its dependencies; each step is then paired with its dependents, and the plan carries its
    root together with every step reachable through either kind of edge.
    """
    from sqlglot import exp
    from sqlglot.planner import Step

    trees, keep, merkle, active = {}, [], {}, set()

    def text(value):
        return json.dumps(value, separators=(",", ":"))

    def cref(cls):
        return f"{cls.__module__}.{cls.__qualname__}"

    def place(node):
        path, top = [], node
        while top.parent is not None:
            path.append([top.arg_key, top.index])
            top = top.parent
        path.reverse()
        current = top
        for key, index in path:
            current = current.args.get(key) if isinstance(current, exp.Expr) else None
            if index is not None and isinstance(current, list) and -len(current) <= index < len(current):
                current = current[index]
        if current is not node:
            return ["loose", tree_digest(node)]
        digest = trees.get(id(top))
        if digest is None:
            keep.append(top)
            digest = trees[id(top)] = tree_digest(top)
        return ["node", digest, path]

    def show(value):
        kind = type(value)
        if value is None or kind is bool or kind is int or kind is str:
            return value
        if kind is float:
            return ["float", repr(value)]
        if isinstance(value, exp.Expr):
            return place(value)
        if isinstance(value, Step):
            return ["step", step_digest(value)]
        if kind is list or kind is tuple:
            return ["L" if kind is list else "T", [show(item) for item in value]]
        if kind is dict:
            return ["D", [[show(key), show(item)] for key, item in value.items()]]
        if kind is set or kind is frozenset:
            return ["S", sorted(text(show(item)) for item in value)]
        if isinstance(value, type):
            return ["C", cref(value)]
        return ["O", tree_digest(value)]

    def step_digest(step):
        digest = merkle.get(id(step))
        if digest is None:
            if id(step) in active:
                raise ValueError("a plan step depends on itself")
            active.add(id(step))
            fields = [[name, show(value)] for name, value in sorted(vars(step).items())
                      if name not in ("dependencies", "dependents")]
            below = sorted(step_digest(dependency) for dependency in step.dependencies)
            active.discard(id(step))
            keep.append(step)
            digest = merkle[id(step)] = _digest(text([cref(type(step)), fields, below]))
        return digest

    def linked(step):
        return _digest(text([step_digest(step), sorted(step_digest(dependent) for dependent in step.dependents)]))

    reached, pending = {}, [plan.root]
    while pending:
        step = pending.pop()
        if id(step) not in reached:
            reached[id(step)] = step
            pending.extend(step.dependencies)
            pending.extend(step.dependents)
    fields = [[name, show(value)] for name, value in sorted(vars(plan).items()) if name not in ("root", "_dag")]
    graph = sorted([linked(step), sorted(linked(dependency) for dependency in dependencies)]
                   for step, dependencies in plan._dag.items())
    return _digest(text([cref(type(plan)), fields, linked(plan.root),
                         sorted(linked(step) for step in reached.values()), graph]))


def worker_role(pairs, blobs, emit, stages, schema, final_state=True, planned=frozenset()):
    import pickle

    from sqlglot import find_tables
    from sqlglot.optimizer import optimize
    from sqlglot.planner import Plan
    from sqlglot.schema import ensure_schema

    for index, blob in blobs:
        with stages("load", index):
            expression = pickle.loads(blob)
        if not pairs[index][0].get("execute"):
            continue
        with stages("tables", index):
            found = find_tables(expression)
        emit(f"tables{index}", json.dumps(sorted(table.name for table in found)))
        mapping = ensure_schema(schema, dialect=None)
        with stages("optimize", index):
            optimized = optimize(expression, mapping, leave_tables_isolated=True, dialect=None)
        with stages("digest", index):
            state = tree_digest(optimized)
        emit(f"optimized{index}", state)
        with stages("render", index):
            text = optimized.sql(pretty=True)
        emit(f"render{index}", text)
        if index in planned:
            with stages("plan", index):
                plan = Plan(optimized)
            with stages("plandigest", index):
                state = plan_digest(plan)
            emit(f"plan{index}", state)
        emit(f"schema{index}", json.dumps(schema, sort_keys=True, default=repr))
        emit(f"loaded{index}", _loaded())
    if final_state:
        with stages("state", -1):
            state = state_digest()
        emit("state", state)


class _Feeder:

    def __init__(self, fd):
        os.set_blocking(fd, False)
        self.fd = fd
        self.pending = bytearray()
        self.broken = False

    def add(self, index, blob):
        self.pending += struct.pack("<iI", index, len(blob))
        self.pending += blob
        self.pump()

    def pump(self):
        try:
            while self.pending and not self.broken:
                del self.pending[: os.write(self.fd, bytes(self.pending[: 1 << 16]))]
        except BlockingIOError:
            pass
        except OSError:
            self.broken = True

    def finish(self):
        try:
            os.set_blocking(self.fd, True)
            while self.pending and not self.broken:
                del self.pending[: os.write(self.fd, bytes(self.pending[: 1 << 16]))]
        except OSError:
            self.broken = True
        os.close(self.fd)


def _blobs(stream):
    while True:
        header = stream.read(8)
        if len(header) < 8:
            return
        index, size = struct.unpack("<iI", header)
        yield index, stream.read(size)


def _emitter(artifacts, expected):
    if expected is None:
        def emit(name, text):
            artifacts[name] = _digest(text)
    else:
        def emit(name, text):
            value = _digest(text)
            if expected.get(name) != value:
                raise _Mismatch(name)
            artifacts[name] = value
    return emit


def parent_role(pairs, order, duck, emit, stages, ship=None, halted=None):
    import pickle

    from sqlglot import parse_one, transpile

    for index in order:
        if halted is not None and halted():
            return
        with stages("parse", index):
            expression = parse_one(pairs[index][1])
        with stages("dump", index):
            blob = pickle.dumps(expression)
        emit(f"blob{index}", blob)
        if ship is not None:
            ship(index, blob)
    done = set()
    for index in duck:
        if halted is not None and halted():
            return
        sql = pairs[index][1]
        if sql in done:
            continue
        done.add(sql)
        with stages("duck", index):
            text = transpile(sql, write="duckdb")[0]
        emit(f"duck{index}", text)


def session_start(modules):
    import importlib

    for name in modules:
        importlib.import_module(name)
    from sqlglot.dialects.dialect import Dialect

    Dialect._classes.clear()


def _forked(function):
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(read)
            try:
                reply = function()
            except _Mismatch as exc:
                reply = {"verdict": f"differs at {exc}"}
            except BaseException as exc:
                reply = {"verdict": f"error {type(exc).__name__}"}
            with os.fdopen(write, "wb") as stream:
                stream.write(json.dumps(reply).encode())
            if reply["verdict"] != "ok":
                os.kill(os.getppid(), signal.SIGUSR1)
        finally:
            os._exit(0)
    os.close(write)
    return pid, read


def _collect(pid, read):
    with os.fdopen(read, "rb") as stream:
        raw = stream.read()
    os.waitpid(pid, 0)
    try:
        return json.loads(raw)
    except ValueError:
        return {"verdict": "error child lost"}


def run_roles(pairs, schema, order, duck, expected=None, stages_factory=Stages, final_state=True, work=None,
              session=None, planned=frozenset()):
    feed = set(order if work is None else work)
    stop = []
    previous = signal.signal(signal.SIGUSR1, lambda *_: stop.append(1))
    children = {}

    def finished(stages, artifacts):
        return {"verdict": "ok", "artifacts": artifacts, "report": stages.report(),
                "cpu": round(time.process_time(), 4)}

    if session is not None:
        def late_parent():
            session_start(session)
            stages, artifacts = stages_factory(), {}
            parent_role(pairs, order, duck, _emitter(artifacts, expected and expected["parent"]), stages)
            return finished(stages, artifacts)

        children["session"] = _forked(late_parent)

    feeder = None
    if feed:
        blob_r, blob_w = os.pipe()

        def worker():
            os.close(blob_w)
            stages, artifacts = stages_factory(), {}
            with os.fdopen(blob_r, "rb") as stream:
                worker_role(pairs, _blobs(stream), _emitter(artifacts, expected and expected["worker"]),
                            stages, schema, final_state, planned)
            return finished(stages, artifacts)

        children["worker"] = _forked(worker)
        os.close(blob_r)
        feeder = _Feeder(blob_w)

    def halted():
        if feeder is not None:
            feeder.pump()
        return bool(stop) or (feeder is not None and feeder.broken)

    def ship(index, blob):
        if feeder is not None and index in feed:
            feeder.add(index, blob)

    stages, artifacts, verdict = stages_factory(), {}, "ok"
    try:
        parent_role(pairs, order, duck, _emitter(artifacts, expected and expected["parent"]), stages, ship, halted)
    except _Mismatch as exc:
        verdict = f"differs at {exc}"
    except BaseException as exc:
        verdict = f"error {type(exc).__name__}"
    if verdict != "ok" or stop:
        for pid, _ in children.values():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if feeder is not None:
        feeder.finish()
    replies = {name: _collect(pid, read) for name, (pid, read) in children.items()}
    signal.signal(signal.SIGUSR1, previous)
    if verdict == "ok":
        failed = [reply["verdict"] for reply in replies.values() if reply["verdict"] != "ok"]
        if failed:
            verdict = next((text for text in failed if text != "error child lost"), failed[0])
    worker_reply = replies.get("worker", {})
    session_reply = replies.get("session", {})
    reports = {"parent": stages.report(), "worker": worker_reply.get("report", {}),
               "session": session_reply.get("report", {}), "session_artifacts": session_reply.get("artifacts", {}),
               "parent_cpu": round(time.process_time(), 4), "worker_cpu": worker_reply.get("cpu", 0),
               "session_cpu": session_reply.get("cpu", 0)}
    return verdict, artifacts, worker_reply.get("artifacts", {}), reports


def child_setup(sources, cpu_limit=25):
    root = Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot")).resolve()
    try:
        import resource
        if cpu_limit:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    except (ImportError, ValueError, OSError):
        pass
    sys.setrecursionlimit(20000)
    sys.path.insert(0, str(root))
    os.chdir(root)
    _install(root, sources)
    return root


def _child_main(protocol):
    request = json.load(sys.stdin)
    child_setup(request.get("sources") or {})
    try:
        baseline = _load("fe_baseline.json.gz")
        if baseline["python"] != sys.version:
            verdict = "python differs"
        else:
            pairs, schema = prepare()
            everything = list(range(len(pairs)))
            executed = [index for index in everything if pairs[index][0].get("execute")]
            work = sorted(set(request.get("worker", everything)))
            if work and not set(work).intersection(executed):
                work = sorted([*work, baseline["anchor"]])
            if request.get("parent", True):
                order, duck = everything, executed
            else:
                order, duck = work, []
            late = baseline["session_modules"] if request.get("parent", True) else None
            planned = frozenset(request.get("plan", ())).intersection(work, executed)
            verdict, parent, worker, reports = run_roles(pairs, schema, order, duck, expected=baseline, work=work,
                                                         session=late, planned=planned)
            if verdict == "ok":
                wanted = {f"blob{index}" for index in order}
                if duck:
                    wanted.update(name for name in baseline["parent"] if name.startswith("duck"))
                needed = {f"{kind}{index}" for index in set(work).intersection(executed)
                          for kind in ("tables", "optimized", "render", "schema", "loaded")}
                needed.update(f"plan{index}" for index in planned)
                if work:
                    needed.add("state")
                if (parent.keys() != wanted or worker.keys() != needed
                        or (late is not None and reports["session_artifacts"].keys() != wanted)):
                    verdict = "differs at missing artifacts"
    except BaseException as exc:
        verdict = f"error {type(exc).__name__}"
    protocol.write(verdict + "\n")


if __name__ == "__main__" and sys.argv[1:2] == ["--child"]:
    with os.fdopen(os.dup(1), "w") as channel, open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        _child_main(channel)
