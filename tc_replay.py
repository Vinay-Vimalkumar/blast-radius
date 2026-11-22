import ast
import hashlib
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

from map_store import read_map
from patches import Change


HERE = Path(__file__).resolve().parent
MAP = HERE / "maps" / "tc_replay.json.gz"
PARSE_MAP = HERE / "maps" / "tc_parse.json.gz"
ARG_MAP = HERE / "maps" / "tc_arguments.json.gz"
EX = "tests/test_executor.py::TestExecutor::test_"
OP = "tests/test_optimizer.py::TestOptimizer::test_"
TARGETS = {EX + "execute_tpcds", EX + "execute_tpch", EX + "optimized_tpch",
           OP + "tpch", OP + "tpcds"}
_wall_left = 20.0
DUCK = "sqlglot/generators/duckdb.py"
RULES = {"sqlglot/optimizer/" + name + ".py" for name in (
    "eliminate_ctes", "merge_subqueries", "optimize_joins", "pushdown_predicates",
    "unnest_subqueries",
)}
FILES = RULES | {DUCK, "sqlglot/parser.py", "sqlglot/generator.py"}
BOUNDARY_FILES = (
    "tests/helpers.py", "tests/test_executor.py", "tests/test_optimizer.py",
    "tests/fixtures/optimizer/tpc-h/tpc-h.sql", "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql",
    "sqlglot/serde.py", "sqlglot/executor/__init__.py", "sqlglot/optimizer/optimizer.py",
)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@lru_cache(maxsize=2)
def boundary_hashes(root):
    return {p: digest((root / p).read_text(encoding="utf-8")) for p in BOUNDARY_FILES}


@lru_cache(maxsize=1)
def profile():
    data = read_map(MAP)
    if data["version"] != 2:
        raise ValueError("unsupported TPC map")
    return data


@lru_cache(maxsize=1)
def parse_profile():
    return read_map(PARSE_MAP)


@lru_cache(maxsize=1)
def argument_profile():
    try:
        data = read_map(ARG_MAP)
        pars = parse_profile()
        if (data["version"] != 1 or data["complete"] is not True or data["source"] != pars["source"]
                or data["profile"] != hashlib.sha256(PARSE_MAP.read_bytes()).hexdigest()
                or not isinstance(data["reads"], list) or len(data["reads"]) != len(pars["rows"])):
            return {}
        sites = set(data["sites"])
        unread = {}
        for site in sites:
            line, start, end = map(int, site.split(":"))
            if line < 1 or start < 0 or end <= start:
                return {}
            unread.setdefault(line, []).append((start, end, site))
        if any(row is not None and (not isinstance(row, list) or not sites.issuperset(row)) for row in data["reads"]):
            return {}
        return {line: [(start, end, {i for i, row in enumerate(data["reads"]) if row is not None and site not in row})
                       for start, end, site in spans] for line, spans in unread.items()}
    except (OSError, EOFError, ValueError, TypeError, KeyError, AttributeError):
        return {}


def unread_rows(change, line):
    old = change.original.splitlines()[line - 1].encode()
    new = change.patched.splitlines()[line - 1].encode()
    for start, end, unused in argument_profile().get(line, ()):
        replacement = {b"True": b"False", b"False": b"True"}.get(old[start:end])
        if replacement is not None and old[:start] + replacement + old[end:] == new:
            return unused
    return set()


def replay_parse(call):
    from sqlglot import exp
    from sqlglot.dialects.dialect import Dialect
    method, sql, into, options = call
    dialect = Dialect()
    parsed = (dialect.parse(sql, **options) if method == "parse"
              else dialect.parse_into(getattr(exp, into), sql, **options))
    return tree_state(parsed)


def observe(task, fixtures):
    from sqlglot import exp, parse_one, transpile

    kind, suite, index = task
    schema, cases = fixtures[suite]
    meta, sql, expected = cases[index]
    if kind in ("duck", "duck_expected"):
        return transpile(sql if kind == "duck" else expected, write="duckdb")[0]
    expression = parse_one(sql)
    if kind == "ship":
        from sqlglot.serde import dump
        return json.dumps(dump(expression), ensure_ascii=False, separators=(",", ":"))
    import pickle
    from sqlglot.optimizer import optimize
    if kind == "assert":
        if meta.keys() - {"execute"} or (suite == "h" and meta.get("execute")):
            raise ValueError("unmodeled optimizer test options")
        tree = optimize(expression, schema=schema)
        for node in tree.walk():
            for key, arg in node.args.items():
                for child in arg if isinstance(arg, list) else (arg,):
                    if isinstance(child, exp.Expr) and (child.arg_key != key or child.parent is not node):
                        raise ValueError("broken optimizer parent links")
        actual = pickle.loads(pickle.dumps(tree)).sql(pretty=True, dialect=None)
        if actual != expected:
            raise ValueError("optimizer fixture assertion differs")
        return actual
    if kind == "plan":
        from sqlglot.schema import ensure_schema
        tree = optimize(pickle.loads(pickle.dumps(expression)), ensure_schema(schema, dialect=None),
                        leave_tables_isolated=True, dialect=None)
        return tree_state(tree)
    raise ValueError("unknown boundary")


def tree_state(tree):
    from enum import Enum
    from sqlglot import exp

    objects = []
    seen = {}

    def freeze(x):
        if x is None or type(x) in (str, bool, int, float):
            return x
        if isinstance(x, Enum):
            return ["enum", type(x).__module__, type(x).__qualname__, x.name]
        ident = id(x)
        if ident in seen:
            return ["ref", seen[ident]]
        i = len(objects)
        seen[ident] = i
        objects.append(None)
        if type(x) in (list, tuple):
            state = [type(x).__name__, [freeze(v) for v in x]]
        elif type(x) is dict:
            state = ["dict", [[freeze(k), freeze(v)] for k, v in x.items()]]
        elif isinstance(x, exp.Expression):
            slots = []
            for cls in type(x).__mro__:
                for k in cls.__dict__.get("__slots__", ()):
                    slots.append([k, freeze(getattr(x, k))])
            state = ["expr", type(x).__module__, type(x).__qualname__, slots,
                     freeze(vars(x)) if hasattr(x, "__dict__") else None]
        else:
            raise ValueError("unmodeled expression state")
        objects[i] = state
        return ["ref", i]

    freeze(tree)
    return json.dumps(objects, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def get_fixtures():
    from tests.helpers import TPCH_SCHEMA, TPCDS_SCHEMA, load_sql_fixture_pairs
    return {s: (schema, list(load_sql_fixture_pairs(f"optimizer/tpc-{s}/tpc-{s}.sql")))
            for s, schema in (("h", TPCH_SCHEMA), ("ds", TPCDS_SCHEMA))}


def _supported(change, data):
    if (change.structural or change.path not in FILES
            or digest(change.original) != data["sources"][change.path]):
        return None
    old, new = change.original.splitlines(keepends=True), change.patched.splitlines(keepends=True)
    if len(old) != len(new):
        return None
    edits = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
    if len(edits) != 1:
        return None
    i = edits[0]
    site = data["owners"][change.path].get(str(i + 1))
    if not site:
        return None
    import io
    import token
    import tokenize
    def tokens(line):
        result = []
        try:
            for t in tokenize.generate_tokens(io.StringIO(line.lstrip()).readline):
                result.append((t.type, t.string))
        except (tokenize.TokenError, IndentationError):
            pass
        return result
    a, b = tokens(old[i]), tokens(new[i])
    if len(a) != len(b) or old[i][:len(old[i])-len(old[i].lstrip())] != new[i][:len(new[i])-len(new[i].lstrip())]:
        return None
    changed = [j for j, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(changed) != 1:
        return None
    j = changed[0]
    (t, x), (u, y) = a[j], b[j]
    if t != u:
        return None
    if t in (token.NUMBER, token.STRING):
        xv, yv = ast.literal_eval(x), ast.literal_eval(y)
        if type(xv) != type(yv) or type(yv) not in (str, int, float) or "__" in str(yv):
            return None
        if t == token.STRING:
            if (not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9 ]{0,63}", yv)
                    or not any(s in old[i] for s in (".args.get(", ".sql(expression, ",
                                                       "._match_text_seq(", "._match_texts("))):
                return None
    elif {x, y} <= {"True", "False", "None"}:
        pass
    elif j >= 2 and a[j-2:j] == [(token.NAME, "TokenType"), (token.OP, ".")] and y in data["tokens"]:
        pass
    elif not (t == token.OP and {x, y} <= {"==", "!=", "<", "<=", ">", ">="}):
        return None
    if old[i].replace(x, y, 1) != new[i] and not any(
        old[i][:p] + y + old[i][p+len(x):] == new[i]
        for p in range(len(old[i])) if old[i].startswith(x, p)
    ):
        return None
    return site


class _PatchedImport:
    def __init__(self, root, changes):
        self.root = root
        self.sources = {p[:-3].replace("/", "."): (p, s) for p, s in changes.items()}

    def find_spec(self, fullname, path, target=None):
        if fullname in self.sources:
            return importlib.util.spec_from_loader(fullname, self, origin=str(self.root / self.sources[fullname][0]))

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path, source = self.sources[module.__name__]
        module.__file__ = str(self.root / path)
        exec(compile(source, module.__file__, "exec"), module.__dict__)


def _child(request):
    import resource
    import signal
    import time
    import unittest
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    root = Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot")).resolve()
    sys.path.insert(0, str(root))
    hook = _PatchedImport(root, request["changes"])
    sys.meta_path.insert(0, hook)
    for name in hook.sources:
        importlib.import_module(name)
    fixtures = get_fixtures()
    proved = []
    observed = {}
    class Budget(BaseException):
        pass
    def expired(*args):
        raise Budget()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 4)
    try:
        for node in sorted(request["nodes"], key=lambda n: (
                n != EX + "execute_tpcds", n.startswith(OP), n != EX + "execute_tpch")):
            pending = [task for i, (task, _, users) in enumerate(request["tasks"])
                       if node in users and i not in observed]
            if (sum(len(task[1][1]) for task in pending if task[0] == "parse_call") > 90_000
                    or sum(task[0] in ("plan", "assert") for task in pending) > 12):
                continue
            for i, (task, expected, users) in enumerate(request["tasks"]):
                if node not in users:
                    continue
                if i not in observed:
                    if time.process_time() > 2.65:
                        return proved
                    try:
                        output = replay_parse(task[1]) if task[0] == "parse_call" else observe(task, fixtures)
                        observed[i] = digest(output) == expected
                    except Exception:
                        observed[i] = False
                if not observed[i]:
                    break
            else:
                proved.append(node)
    except Budget:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return proved


def refine_tc(changes: list[Change], chosen: set[str], cov, catalog: set[str]) -> set[str]:
    global _wall_left
    targets = chosen & catalog & TARGETS
    if not targets or _wall_left < 0.2 or len(changes) != 1 or changes[0].path not in FILES:
        return chosen
    start = time.monotonic()
    try:
        data = profile()
        change = changes[0]
        site = _supported(change, data)
        if not site:
            return chosen
        root = Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot")).resolve()
        if data["pins"] != boundary_hashes(root):
            return chosen
        candidates = targets - cov.missing
        line = next(i for i, (a, b) in enumerate(zip(change.original.splitlines(),
                                                    change.patched.splitlines()), 1) if a != b)
        if site != "*":
            from import_effects import import_consumers
            candidates -= import_consumers(change.path, {line}, cov, catalog)
        elif change.path != DUCK:
            return chosen
        group = "duck" if change.path == DUCK else "rule" if change.path in RULES else "general"
        nodes = candidates.intersection(data["tests"][group])
        tasks = {}
        extra = []
        discharged = set()
        if change.path == "sqlglot/parser.py" and site != "*":
            try:
                pars = parse_profile()
                if pars["source"] == data["sources"][change.path] and pars["version"] == 1:
                    key = change.path + "#" + str(line) if line in pars.get("lines", ()) else site
                    nodes = set()
                    extra_nodes = {n for n in candidates & pars["tests"].keys()
                                   if key not in pars["outside"][n]}
                    unused = unread_rows(change, line)
                    for i, (call, result, hits) in enumerate(pars["rows"]):
                        if key in hits:
                            users = [n for n in extra_nodes if i in pars["tests"][n]]
                            if i in unused:
                                discharged.update(users)
                            elif users:
                                extra.append([["parse_call", call], result, users])
                else:
                    return chosen
            except FileNotFoundError:
                return chosen
        else:
            extra_nodes = set()
        for node in sorted(nodes):
            for row in data["tests"][group][node]:
                task, expected, hits = data["rows"][row]
                if group == "duck" and site == "*" and task[0] == "ship":
                    continue
                if site == "*" or site in hits:
                    tasks.setdefault(row, []).append(node)
        nodes = nodes | extra_nodes
        if not nodes:
            return chosen
        payload = {"changes": {change.path: change.patched}, "nodes": sorted(nodes),
                   "tasks": extra + [[*data["rows"][i][:2], nodeids] for i, nodeids in tasks.items()]}
        complete = discharged - {node for _, _, users in payload["tasks"] for node in users}
        if complete:
            chosen = chosen - complete
            nodes -= complete
            if not nodes:
                return chosen
            payload["nodes"] = sorted(nodes)
        proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--child"],
                              input=json.dumps(payload), capture_output=True, text=True, timeout=min(6, _wall_left),
                              env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"})
        if proc.returncode == 0:
            proved = json.loads(proc.stdout)
            if isinstance(proved, list) and set(proved) <= nodes:
                return chosen.difference(proved)
    except Exception:
        pass
    finally:
        _wall_left -= time.monotonic() - start
    return chosen


if __name__ == "__main__" and sys.argv[1:] == ["--child"]:
    request = json.load(sys.stdin)
    with os.fdopen(os.dup(1), "w") as protocol, open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        result = _child(request)
        protocol.write(json.dumps(result))
