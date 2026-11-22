import hashlib
import json
import marshal
import os
import subprocess
import sys
import time
from functools import cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAP = HERE / "packed_maps" / "mn_reduced.marshal"
NODE = "tests/test_parser.py::TestParser::test_max_nodes"
WIDTHS = (1, 2, 3, 5, 8, 16)
COLUMN = "x" + "=x" * 100
VERSION = 3
SLOWER = 6.0
BUDGET = 8.0


def _giant(width):
    return "SELECT " + ",".join(COLUMN for _ in range(width))


def _calibrate():
    best = None
    for _ in range(3):
        start = time.process_time()
        total = 0
        for i in range(200_000):
            total += i & 3
        spent = time.process_time() - start
        best = spent if best is None else min(best, spent)
    return best


def _observe():
    from sqlglot import exp, parse_one
    from sqlglot.dialects.dialect import Dialect
    from sqlglot.errors import ParseError

    def outcome(sql, **options):
        try:
            tree = parse_one(sql, **options)
        except ParseError as error:
            text = str(error)
            return ["ParseError", "Maximum number of AST nodes" in text, text.split(". Line")[0][:80]]
        except Exception as error:
            return ["raises", type(error).__name__]
        return ["tree", type(tree).__name__, len(tree.expressions) if isinstance(tree, exp.Select) else -1]

    out = {
        "call1": outcome("SELECT 1", max_nodes=1_000_000),
        "call2": outcome("SELECT " + ",".join(f"col{i}" for i in range(500)) + " FROM t", max_nodes=1_000_000),
        "call5": outcome("SELECT " + ",".join(f"x={i}" for i in range(100)), max_nodes=5),
        "call6": outcome("SELECT 1, 2, 3"),
    }
    dialect = Dialect.get_or_raise(None)
    mon = sys.monitoring
    tool = mon.DEBUGGER_ID
    mon.use_tool_id(tool, "max_nodes reduced run")
    count = [0, 0]

    def on_start(code, offset):
        count[0] += 1

    def on_jump(code, offset, destination):
        count[1] += 1

    mon.register_callback(tool, mon.events.PY_START, on_start)
    mon.register_callback(tool, mon.events.JUMP, on_jump)
    watched = mon.events.PY_START | mon.events.JUMP
    rows = []
    calibration = _calibrate()
    began = time.process_time()
    for width in WIDTHS:
        sql = _giant(width)
        count[0] = count[1] = 0
        mon.set_events(tool, watched)
        try:
            tokens = dialect.tokenize(sql)
            scanned = list(count)
            kinds = hashlib.blake2b(",".join(str(t.token_type) for t in tokens[:405]).encode(), digest_size=8).hexdigest()
            texts = hashlib.blake2b("\0".join(t.text for t in tokens).encode("utf-8", "surrogatepass"),
                                    digest_size=8).hexdigest()
            parser = dialect.parser(max_nodes=10**12)
            count[0] = count[1] = 0
            trees = parser.parse(tokens, sql)
            parsed = list(count)
            shape = [type(trees[0]).__name__, len(trees), len(trees[0].expressions)]
        except Exception as error:
            mon.set_events(tool, 0)
            rows.append({"width": width, "raises": type(error).__name__})
            continue
        finally:
            mon.set_events(tool, 0)
        nodes = parser._node_count
        row = {"width": width, "tokens": len(tokens), "kinds": kinds, "texts": texts, "count": nodes, "shape": shape,
               "scan_entries": scanned[0], "scan_jumps": scanned[1], "parse_entries": parsed[0],
               "parse_jumps": parsed[1], "index": parser._index}
        for name, limit in (("margin_one", nodes - 1), ("exact", nodes), ("inside", nodes // 2)):
            probe = dialect.parser(max_nodes=limit)
            try:
                probe.parse(dialect.tokenize(sql), sql)
                row[name] = ["no error"]
            except ParseError as error:
                row[name] = ["ParseError", f"Maximum number of AST nodes ({limit}) exceeded" in str(error),
                             probe._index, probe._node_count - limit]
            except Exception as error:
                row[name] = ["raises", type(error).__name__]
        rows.append(row)
    worked = time.process_time() - began
    mon.free_tool_id(tool)
    out["widths"] = rows
    sample = _giant(3)
    out["public_margin"] = outcome(sample, max_nodes=300)
    out["public_inside"] = outcome(sample, max_nodes=150)
    out["public_exact"] = outcome(sample, max_nodes=301)
    return {"observed": out, "cpu": [calibration, worked]}


def _pinned_shape(obs):
    rows = obs["widths"]
    for row in rows:
        w = row["width"]
        if (row.get("count") != 100 * w + 1 or row.get("tokens") != 202 * w or row.get("shape") != ["Select", 1, w]
                or row.get("margin_one", [None])[:2] != ["ParseError", True] or row.get("exact") != ["no error"]
                or row.get("inside", [None])[:2] != ["ParseError", True]):
            return False
    for key in ("scan_entries", "scan_jumps", "parse_entries", "parse_jumps"):
        steps = [(b[key] - a[key]) / (b["width"] - a["width"]) for a, b in zip(rows, rows[1:])]
        if max(steps) - min(steps) > (1 if key.endswith("jumps") else 0) or len(set(steps[1:])) != 1:
            return False
    return True


def _child_main(request):
    try:
        import resource
        seconds = int(request.get("cpu", 12))
        resource.setrlimit(resource.RLIMIT_AS, (1536 << 20, 1536 << 20))
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
    except Exception:
        pass
    root = os.path.realpath(request["root"])
    sources = request.get("sources") or {}
    if sources:
        from import_state import _install
        _install(root, sources, {})
    sys.path.insert(0, root)
    os.chdir(root)
    import unittest
    return _observe()


def _run_child(request, timeout):
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--child"],
                              input=json.dumps(request), capture_output=True, text=True, timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 79:
        return None
    lines = proc.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines else None


def record(root, output=MAP):
    runs = [_run_child({"root": str(Path(root).resolve()), "cpu": 600}, 300) for _ in range(5)]
    if (any(run is None for run in runs) or any(run["observed"] != runs[0]["observed"] for run in runs)
            or not _pinned_shape(runs[0]["observed"])):
        raise SystemExit("pinned reduced run is not reproducible or not of the expected shape")
    ratio = sorted(run["cpu"][1] / run["cpu"][0] for run in runs)[len(runs) // 2]
    Path(output).parent.mkdir(exist_ok=True)
    Path(output).write_bytes(marshal.dumps({"version": VERSION, "python": sys.version, "observed": runs[0]["observed"],
                                            "ratio": ratio}))


@cache
def _pinned():
    data = marshal.loads(MAP.read_bytes(), allow_code=False)
    if data["version"] != VERSION or data["python"] != sys.version:
        raise ValueError("reduced-run recording does not match this interpreter")
    return data


def verdict(sources, root, budget=BUDGET):
    pinned = _pinned()
    got = _run_child({"root": str(root), "sources": sources}, budget)
    if got is None:
        return "no result"
    if got["observed"] != pinned["observed"]:
        return "differs"
    calibration, worked = got["cpu"]
    if calibration <= 0 or worked / calibration > SLOWER * pinned["ratio"]:
        return "slower"
    return "same"


def _adds_threshold(change):
    import io
    import tokenize

    try:
        for tok in tokenize.generate_tokens(io.StringIO(change.patched).readline):
            if tok.type == tokenize.NUMBER and tok.start[0] in change.added:
                try:
                    if int(tok.string.replace("_", ""), 0) > 3:
                        return True
                except ValueError:
                    return True
    except (tokenize.TokenError, SyntaxError):
        return True
    return False


def refine_max_nodes(changes, chosen, cov, catalog, deadline=None):
    if NODE not in chosen or NODE not in catalog or NODE in cov.missing or NODE not in cov.nodes:
        return chosen
    try:
        sources = {}
        for change in changes:
            path = change.path
            if change.structural:
                return chosen
            if path.endswith(".py"):
                if not path.startswith("sqlglot/") or path.endswith("__main__.py"):
                    return chosen
                if NODE in cov.file_execution(path):
                    from semantics import semantic_lines



                    sites = semantic_lines(path, change.original, change.patched, change.removed)
                    if sites is None:
                        sites = change.removed
                        if not sites:
                            return chosen
                    if NODE in cov.lines(path, sites) or _adds_threshold(change):
                        return chosen
                sources[path] = change.patched
            elif NODE in cov.fixture(path):
                return chosen
        if not sources:
            return chosen
        budget = BUDGET if deadline is None else min(BUDGET, deadline - time.monotonic())
        if budget < 4:
            return chosen
        result = verdict(sources, os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot"), budget)
        if result == "same":
            return set(chosen) - {NODE}
    except Exception:
        pass
    return chosen


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        request = json.load(sys.stdin)
        with os.fdopen(os.dup(1), "w") as protocol, open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            try:
                payload = _child_main(request)
            except BaseException:
                raise SystemExit(1)
            protocol.write(json.dumps(payload))

        os._exit(79)
    if sys.argv[1:2] == ["--record"]:
        record(sys.argv[2])
