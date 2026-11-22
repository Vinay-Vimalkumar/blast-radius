import marshal
import os
import select
import struct
import sys
import time

ROOT = os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot")
HEAVY_FILES = frozenset({"tests/test_executor.py", "tests/test_optimizer.py", "tests/test_sqlglotc.py"})
HEAVY_COLLECTED = frozenset({
    "sqlglot.executor", "sqlglot.executor.context", "sqlglot.executor.env", "sqlglot.executor.python",
    "sqlglot.executor.table", "sqlglot.generators.python", "sqlglot.planner",
    "sqlglot.optimizer.canonicalize_internal_names",
})
SKIP_IDS = frozenset({"tests/test_parser.py::TestParser::test_max_nodes"})
IGNORED_PREFIXES = ("docs/", ".github/")


_REGISTRY_TESTS = (
    "tests/dialects/test_dialect.py::TestDialect::test_custom_dialect",
    "tests/dialects/test_dialect.py::TestDialect::test_get_or_raise",
    "tests/test_dialect_entry_points.py::TestDialectEntryPoints::test_entry_point_plugin_discovery_legacy_api",
    "tests/test_dialect_entry_points.py::TestDialectEntryPoints::test_entry_point_plugin_discovery_modern_api",
    "tests/test_dialect_entry_points.py::TestDialectEntryPoints::test_entry_point_plugin_not_found",
)

MAX_MS = int(os.environ.get("CHECK_RUN_MAX_MS", "1000"))
MAX_TESTS = int(os.environ.get("CHECK_RUN_MAX_TESTS", "120"))
CPU_CAP = float(os.environ.get("CHECK_RUN_CPU_CAP", "40"))
WALL_CAP = float(os.environ.get("CHECK_RUN_WALL_CAP", "25"))

_SPAWN_EVENTS = frozenset({
    "os.fork", "os.forkpty", "os.posix_spawn", "os.spawn", "os.exec", "os.system",
    "subprocess.Popen", "_posixsubprocess.fork_exec", "pty.spawn",
})


def _send(fd, obj):
    data = marshal.dumps(obj)
    view = memoryview(struct.pack("<I", len(data)) + data)
    while view:
        view = view[os.write(fd, view):]


def _recv(fd, deadline=None):
    buf = bytearray()
    size = None
    while True:
        if deadline is not None:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                raise TimeoutError("no reply")
        chunk = os.read(fd, 1 << 16)
        if not chunk:
            return None
        buf += chunk
        if size is None and len(buf) >= 4:
            size = struct.unpack_from("<I", buf)[0]
        if size is not None and len(buf) >= 4 + size:
            return marshal.loads(bytes(buf[4:4 + size]), allow_code=False)


REGISTRY_NAMES = ("_classes", ".classes", "_try_load", "DIALECT_MODULE_NAMES", "entry_points")


def _module_name(path):
    name = path[:-3].replace("/", ".")
    if name.endswith(".__init__"):
        return name[:-9], True
    return name, False


def _new_lines(change, lines):
    removed, added = change.removed, change.added
    out = set()
    old = new = 1
    top = max(lines)
    while old <= top:
        if old in removed or new in added:
            olds, news = [], []
            while old in removed:
                olds.append(old)
                old += 1
            while new in added:
                news.append(new)
                new += 1
            if any(n in lines for n in olds):
                out.update(news)
            continue
        if old in lines:
            out.add(new)
        old += 1
        new += 1
    return out


TIERS = 4


def _closure():
    import json
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fork_closure.json"), "rb") as stream:
            return {name: frozenset(mods) for name, mods in json.load(stream)["dialects"].items()}
    except (OSError, ValueError, KeyError):
        return {}


def _scan_patches():
    from pathlib import Path
    order, last = {}, {}
    try:
        patches = sorted((Path(os.environ.get("INPUT_DIR", "/data")) / "commits").glob("*.patch"))
        for index, patch in enumerate(patches):
            order[patch.name] = index
            for line in patch.read_text(encoding="utf-8", errors="replace").split("\n"):
                if line.startswith(("+++ b/", "--- a/")):
                    path = line[6:].split("\t", 1)[0].strip()
                    if path.endswith(".py"):
                        last[_module_name(path)[0]] = index
    except OSError:
        return {}, {}
    return order, last


class _Client:
    def __init__(self):
        self.proc = None
        self.dead = False
        self.req = []
        self.resp = []
        self.loaded = []
        self.first = {}
        self.spent = 0.0
        self.jobs = 0
        self.closure = None
        self.order = {}
        self.last = {}
        self.dialects = set()
        self.wall = 0.0
        self.skipped = None

    def start(self, light_files):
        import atexit
        import subprocess
        pairs = [os.pipe() + os.pipe() for _ in range(TIERS)]
        child = [fd for p in pairs for fd in (p[0], p[3])]
        here = os.path.dirname(os.path.abspath(__file__))
        code = f"import sys; sys.path.insert(0, {here!r}); import fork_check; fork_check._zygote_main()"
        self.proc = subprocess.Popen(
            [sys.executable, "-c", code, *map(str, child)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=child, start_new_session=True, close_fds=True)
        atexit.register(self.close)
        for p in pairs:
            os.close(p[0])
            os.close(p[3])
        self.req = [p[1] for p in pairs]
        self.resp = [p[2] for p in pairs]
        deadline = time.monotonic() + 30
        for level in range(3):
            if level:
                _send(self.req[level - 1], {"op": "next", "files": light_files if level == 2 else []})
            hello = _recv(self.resp[level], deadline)
            if not hello or hello.get("level") != level or hello.get("errors"):
                raise RuntimeError(f"tier {level} did not start: {hello and hello.get('errors')}")
            self.loaded.append(frozenset(hello["modules"]))
            self.spent += hello["cpu"]
            if level == 2:
                self.first = hello["first"]

    def widen(self, patch):
        if self.closure is None:
            self.closure = _closure()
            self.order, self.last = _scan_patches()
        if not self.closure:
            return
        index = self.order.get(getattr(patch, "name", None), 0)
        base = self.loaded[2]
        allowed = sorted(name for name, mods in self.closure.items()
                         if name not in self.dialects and all(self.last.get(m, -1) < index for m in mods - base))
        deadline = time.monotonic() + 30
        if len(self.loaded) == 3:
            _send(self.req[2], {"op": "next", "files": ["sqlglot.dialects." + name for name in allowed]})
            reply = _recv(self.resp[3], deadline)
            if not reply or reply.get("level") != 3 or reply.get("errors"):
                raise RuntimeError(f"tier 3 did not start: {reply and reply.get('errors')}")
            self.loaded.append(frozenset(reply["modules"]))
        elif allowed:
            _send(self.req[3], {"op": "preload", "files": ["sqlglot.dialects." + name for name in allowed]})
            reply = _recv(self.resp[3], deadline)
            if not reply or reply.get("errors"):
                raise RuntimeError(f"tier 3 preload failed: {reply and reply.get('errors')}")
            self.loaded[3] = frozenset(reply["modules"])
        else:
            return
        self.dialects.update(allowed)
        self.spent += reply["cpu"]

    def run(self, level, job):
        _send(self.req[level], job)
        reply = _recv(self.resp[level], time.monotonic() + job["wall"] + 10)
        if reply is None:
            raise RuntimeError("tier closed its pipe")
        self.spent += reply.get("cpu", 0.0)
        return reply

    def close(self):
        proc, self.proc = self.proc, None
        self.dead = True
        if proc is None:
            return
        import signal
        import subprocess
        for level in reversed(range(TIERS)):
            try:
                _send(self.req[level], {"op": "quit"})
            except OSError:
                pass
        for fd in self.req + self.resp:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()
        while True:
            try:
                os.waitpid(-proc.pid, 0)
            except ChildProcessError:
                break


_client = _Client()
_order = {}


def check_run(changes, chosen, cov, numbers, millis, verify_until, patch=None):
    began = time.monotonic()
    try:
        return _check_run(changes, chosen, cov, numbers, millis, verify_until, patch)
    except _Unprepared:
        return _skip(chosen, "unprepared")
    except Exception:
        if _client.proc is not None:
            _client.close()
        _client.dead = True
        return chosen
    finally:
        _client.wall += time.monotonic() - began


def _skip(chosen, reason):
    _client.skipped = reason
    return chosen


def _check_run(changes, chosen, cov, numbers, millis, verify_until, patch):
    if not chosen or _client.dead or sys.platform != "linux":
        return chosen
    if len(chosen) > MAX_TESTS:
        return _skip(chosen, "many")
    if chosen.issuperset(numbers):
        return _skip(chosen, "full-catalog")
    tests = []
    total = 0
    heavy = 0
    for node in chosen:
        number = numbers.get(node)
        if number is None:
            continue
        file, _, rest = node.partition("::")
        cls, _, method = rest.partition("::")
        if file in HEAVY_FILES or node in SKIP_IDS or not method or "[" in method:
            heavy += 1
        total += millis[number]
        tests.append((number, file[:-3].replace("/", "."), cls, method))
    if heavy:
        return _skip(chosen, "heavy")
    if not tests or total > MAX_MS:
        return _skip(chosen, "slow")

    try:
        modules, need_line = _prepare(changes, chosen, cov, numbers)
    except Exception as exc:
        raise _Unprepared() from exc
    if isinstance(modules, str):
        return _skip(chosen, modules)

    cpu = 0.6 + 6.0 * total / 1000
    wall = 4 + 4 * cpu
    if time.monotonic() + wall > verify_until or _client.spent + cpu > CPU_CAP or _client.wall + cpu > WALL_CAP:
        return _skip(chosen, "budget")
    tests.sort()
    partial = _partial_context(changes, chosen, numbers, millis, tests)
    return _dispatch(changes, chosen, numbers, patch, tests, total, modules, need_line, cpu, wall, partial)


def _partial_context(changes, chosen, numbers, millis, tests):
    """Return additional ordered cases, or None to keep whole-job verification."""


    if (len(tests) != len(chosen) or len(tests) < 2 or MAX_MS > 1000
            or any(c.path.startswith("tests/") for c in changes)):
        return None
    if not all(node in numbers for node in _REGISTRY_TESTS):
        return None
    gross = sum(millis[t[0]] for t in tests[:-1])
    context = []
    for node in _REGISTRY_TESTS:
        n = numbers[node]
        if n < tests[-1][0] and node not in chosen:
            file, cls, method = node.split("::")
            context.append((n, file[:-3].replace("/", "."), cls, method))


    extra = sum(millis[t[0]] for t in context)
    if gross <= extra + 50 or len(tests) + len(context) > MAX_TESTS:
        return None
    return context


def _prepare(changes, chosen, cov, numbers):
    from import_effects import _index
    from semantics import patched_code, semantic_lines
    modules = []
    need_line = set()
    for change in changes:
        path = change.path
        if path.startswith(IGNORED_PREFIXES) or path.endswith(".md"):
            continue
        if change.structural or not path.endswith(".py") or not path.startswith(("sqlglot/", "tests/")):
            return "file", None
        name, is_pkg = _module_name(path)
        if name in HEAVY_COLLECTED or path in HEAVY_FILES:
            return "heavy-module", None
        text = change.patched.split("\n")
        if any(word in text[n - 1] for n in change.added if 0 < n <= len(text) for word in REGISTRY_NAMES):
            return "registry", None
        lines = semantic_lines(path, change.original, change.patched, change.removed)
        targets = []
        must = None
        if lines is not None:
            must = sorted(numbers[node] for node in cov.fixture(path) & chosen if node in numbers)
        if lines:
            targets = sorted(_new_lines(change, lines))
            if not targets:
                return "no-target", None
            need_line.update(numbers[node] for node in cov.lines(path, lines) & chosen if node in numbers)
            import_lines = _index()[0].get(path, {})
            if any(import_lines.get(str(n)) for n in lines):
                must = None
        code = patched_code(path, change.patched)
        modules.append((name, os.path.join(ROOT, path), is_pkg, marshal.dumps(code),
                        change.patched.encode("utf-8"), targets, bool(lines != set()), must))
    if not modules or not any(m[6] for m in modules):
        return "no-change", None
    return modules, need_line


def _dispatch(changes, chosen, numbers, patch, tests, total, modules, need_line, cpu, wall, partial=None):
    if not _order:
        for node, number in numbers.items():
            file = node.partition("::")[0]
            if number < _order.get(file, 1 << 30):
                _order[file] = number
    if _client.proc is None:
        light = sorted((f for f in _order if f not in HEAVY_FILES), key=_order.get)
        _client.start([f[:-3].replace("/", ".") for f in light])

    names = {m[0] for m in modules}
    level = 2
    if names & _client.loaded[2]:
        level = 1
    if names & _client.loaded[1]:
        level = 0
    if level == 2:
        _client.widen(patch)
        if len(_client.loaded) == TIERS and not names & _client.loaded[3]:
            level = 3
    context = {t[0] for t in partial or ()}
    if partial is not None:
        tests = sorted([*tests, *partial])
    imports = []
    if level < 2:
        files = {t[1] for t in tests}
        if partial is not None:

            files.add("tests.test_dialect_entry_points")
        files.update(_client.first[name] for name in names if name in _client.first)
        keys = {f[:-3].replace("/", "."): n for f, n in _order.items()}
        imports = sorted(files, key=lambda f: keys.get(f, 1 << 30))
    _client.jobs += 1
    job = {"op": "job", "id": _client.jobs, "modules": modules, "tests": tests, "need_line": sorted(need_line),
           "imports": imports, "cpu": cpu, "wall": wall, "context": context}
    reply = _client.run(level, job)
    return _retained(reply, chosen, numbers, tests, partial)


def _retained(reply, chosen, numbers, tests, partial):
    if (reply.get("status") != "ok" or reply.get("exit") != 0 or reply.get("moved") != []
            or reply.get("taint") != [] or not reply.get("fixtures_closed")):
        return chosen
    status = reply.get("tests")
    if not isinstance(status, dict) or status.keys() != {t[0] for t in tests}:
        return chosen
    if all(status.get(t[0]) == "pass" for t in tests):
        return {node for node in chosen if node not in numbers}
    if partial is None:
        return chosen


    context = {t[0] for t in partial}
    passed = set()
    for n, module, _, _ in tests:
        verdict = status.get(n)
        if verdict != "pass":
            if verdict not in ("fail", "error") or n in context:
                return chosen
            if any(t[1] == module and t[0] > n for t in tests):
                passed.difference_update(t[0] for t in tests if t[1] == module)
            break
        passed.add(n)
    if not passed.difference(context):
        return chosen
    return {node for node in chosen if numbers.get(node) not in passed}


def _zygote_main():
    fds = [int(arg) for arg in sys.argv[1:1 + 2 * TIERS]]
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != here]
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    sys.argv[:] = ["pytest"]
    os.chdir(ROOT)
    _serve(0, fds, [])


def _self_cpu():
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _serve(level, fds, files):
    import gc
    import importlib
    start = _self_cpu() if level == 0 else time.process_time()
    errors = []
    first = {}
    try:
        if level == 0:
            import collections, contextlib, copy, datetime, decimal, difflib, enum, functools, inspect, itertools
            import json, linecache, locale, logging, math, numbers, pprint, re, resource, shutil, signal
            import string, tempfile, textwrap, threading, traceback, types, typing, warnings
            import importlib.machinery, importlib.util
            import unittest, unittest.mock
        elif level == 1:
            import sqlglot
            if not os.path.abspath(sqlglot.__file__).startswith(ROOT + os.sep):
                errors.append("sqlglot is not the pinned checkout")
        else:
            for name in files:
                before = set(sys.modules)
                importlib.import_module(name)
                if level == 2:
                    for module in set(sys.modules) - before:
                        if module.startswith("sqlglot."):
                            first[module] = name
    except BaseException as exc:
        errors.append(repr(exc)[:300])
    gc.collect()
    gc.freeze()
    req, resp = fds[2 * level], fds[2 * level + 1]
    higher = fds[2 * level + 2:]
    def loaded():
        return sorted(m for m in sys.modules if m == "sqlglot" or m.startswith(("sqlglot.", "tests.")) or m == "tests")

    spent = (_self_cpu() if level == 0 else time.process_time()) - (start if level else 0.0)
    _send(resp, {"level": level, "modules": loaded(), "first": first, "errors": errors, "cpu": spent})
    child = 0
    while True:
        try:
            msg = _recv(req)
        except Exception:
            msg = None
        if not msg or msg.get("op") == "quit":
            break
        if msg["op"] == "next" and not child:
            pid = os.fork()
            if pid == 0:
                os.close(req)
                os.close(resp)
                _serve(level + 1, fds, msg.get("files") or [])
                os._exit(0)
            child = pid
            for fd in higher:
                os.close(fd)
            higher = []
        elif msg["op"] == "job":
            try:
                reply = _run_job(level, msg, [req, resp, *higher])
            except BaseException as exc:
                reply = {"status": "tier-error", "note": repr(exc)[:300]}
            _send(resp, reply)
        elif msg["op"] == "preload":
            began = time.process_time()
            problems = []
            try:
                for name in msg.get("files") or []:
                    importlib.import_module(name)
            except BaseException as exc:
                problems.append(repr(exc)[:300])
            gc.collect()
            gc.freeze()
            _send(resp, {"modules": loaded(), "errors": problems, "cpu": time.process_time() - began})
    if child:
        _reap(child, 3.0)
    os._exit(0)


def _reap(pid, patience):
    import signal
    end = time.monotonic() + patience
    while time.monotonic() < end:
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if done:
            return
        time.sleep(0.01)
    try:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    except (ProcessLookupError, ChildProcessError):
        pass


def _run_job(level, job, inherited):
    import resource
    import shutil
    import signal
    tmp = f"/tmp/check-run-{os.getpid()}-{job['id']}"
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        code = 70
        try:
            os.close(r)
            for fd in inherited:
                try:
                    os.close(fd)
                except OSError:
                    pass
            _worker(level, job, w, tmp)
            code = 0
        except BaseException:
            code = 71
        finally:
            os._exit(code)
    os.close(w)
    deadline = start + job["wall"]
    chunks = []
    status = "ok"
    while True:
        left = deadline - time.monotonic()
        if left <= 0 or not select.select([r], [], [], left)[0]:
            status = "timeout"
            break
        chunk = os.read(r, 1 << 16)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(r)
    if status == "timeout":
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _, exit_status = os.waitpid(pid, 0)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    shutil.rmtree(tmp, ignore_errors=True)
    reply = None
    if status == "ok" and exit_status == 0:
        data = b"".join(chunks)
        try:
            if len(data) - 4 != struct.unpack_from("<I", data)[0]:
                raise ValueError("incomplete worker reply")
            reply = marshal.loads(data[4:], allow_code=False)
        except Exception:
            status = "crash"
    elif exit_status and status == "ok":
        status = "crash"
    if not isinstance(reply, dict):
        reply = {"status": status}
    reply["exit"] = exit_status
    reply["cpu"] = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    return reply


def _state():
    import builtins
    import decimal
    import gc
    import locale
    import logging
    import resource
    import signal
    import threading
    ctx = decimal.getcontext()
    mask = os.umask(0)
    os.umask(mask)

    def handler(number):
        value = signal.getsignal(number)
        return id(value) if callable(value) else repr(value)

    sqlglot = sys.modules.get("sqlglot")
    return {
        "recursionlimit": sys.getrecursionlimit(),
        "switchinterval": sys.getswitchinterval(),
        "int_max_str_digits": sys.get_int_max_str_digits(),
        "trace": sys.gettrace() is None and sys.getprofile() is None,
        "hooks": (id(sys.excepthook), id(sys.unraisablehook), id(sys.displayhook), id(threading.excepthook)),
        "streams": (id(sys.stdin), id(sys.stdout), id(sys.stderr)),
        "cwd": os.getcwd(),
        "umask": mask,
        "environ": tuple(sorted(os.environ.items())),
        "sys.path": tuple(sys.path),
        "import system": (tuple(map(id, sys.meta_path)), tuple(map(id, sys.path_hooks)), sys.dont_write_bytecode),
        "argv": tuple(sys.argv),
        "gc": (gc.isenabled(), gc.get_threshold()),
        "logging": (logging.root.level, logging.root.manager.disable, logging.raiseExceptions,
                    tuple(map(id, logging.root.handlers))),
        "decimal": (ctx.prec, ctx.rounding, ctx.Emin, ctx.Emax, ctx.capitals, ctx.clamp,
                    tuple(sorted((str(k), v) for k, v in ctx.traps.items()))),
        "locale": locale.setlocale(locale.LC_ALL),
        "signals": tuple(handler(n) for n in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM, signal.SIGPROF,
                                              signal.SIGXCPU, signal.SIGPIPE)),
        "threads": threading.active_count(),
        "builtins": tuple(sorted((k, id(v)) for k, v in vars(builtins).items())),
        "rlimits": tuple(resource.getrlimit(n) for n in (resource.RLIMIT_STACK, resource.RLIMIT_NOFILE,
                                                         resource.RLIMIT_AS, resource.RLIMIT_CPU)),
        "monitoring": tuple(sys.monitoring.get_tool(i) for i in range(6)),
        "sqlglot.pretty": getattr(sqlglot, "pretty", None),
    }


class _Abort(KeyboardInterrupt):
    pass


class _Unprepared(Exception):
    pass


def _worker(level, job, out_fd, tmp):
    import _imp
    import importlib
    import importlib.util
    import logging
    import resource
    import signal
    import tempfile
    import types
    import unittest
    import warnings
    from importlib.machinery import SourceFileLoader

    budget = job["cpu"]
    hard = int(budget) + 2
    resource.setrlimit(resource.RLIMIT_CPU, (hard, hard + 1))
    for limit, value in ((resource.RLIMIT_AS, 2 << 30), (resource.RLIMIT_CORE, 0), (resource.RLIMIT_FSIZE, 32 << 20)):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)
    os.close(null)
    os.mkdir(tmp, 0o700)
    tempfile.tempdir = tmp

    reply = {"status": "ok", "tests": {}, "moved": [], "taint": []}
    status = reply["tests"]
    current = [None]
    phase = [None]
    blocked = set()
    line_hits = set()
    class_hits = set()

    def alarm(signum, frame):
        reply["status"] = "cpu-timeout"
        raise _Abort()

    signal.signal(signal.SIGPROF, alarm)
    signal.setitimer(signal.ITIMER_PROF, budget, 0.25)

    mods = {}
    sources = {}
    watched = set()
    targets = {}
    must_reach = {}
    reached = set()
    for name, path, is_pkg, blob, source, lines, changed, must in job["modules"]:
        must_reach[path] = None if must is None else frozenset(must)
        if name in sys.modules:
            reply["status"] = "preloaded"
            _send(out_fd, reply)
            return
        code = marshal.loads(blob)
        _imp._fix_co_filename(code, path)
        mods[name] = (path, is_pkg, code)
        sources[path] = source
        watched.add(path)
        watched.add(importlib.util.cache_from_source(path))
        if lines:
            targets[name] = (code, frozenset(lines))
    names = {os.path.basename(p) for p in watched}

    class Loader(SourceFileLoader):
        def get_code(self, fullname):
            reached.add(mods[fullname][0])
            return mods[fullname][2]

        def get_data(self, path):
            data = sources.get(path)
            if data is None:
                return super().get_data(path)
            reached.add(path)
            return data

    import builtins
    import io
    copies = {}
    real_open = builtins.open

    def open_(file, *args, **kwargs):
        try:
            path = os.fspath(file)
        except TypeError:
            path = None
        if type(path) is str and path.rpartition("/")[2] in names:
            full = os.path.abspath(path)
            data = sources.get(full)
            if data is not None:
                copy = copies.get(full)
                if copy is None:
                    copy = copies[full] = os.path.join(tmp, f"src{len(copies)}-" + os.path.basename(full))
                    with real_open(copy, "wb") as stream:
                        stream.write(data)
                reached.add(full)
                file = copy
        return real_open(file, *args, **kwargs)

    builtins.open = io.open = open_

    class Finder:
        @staticmethod
        def find_spec(fullname, path=None, target=None):
            entry = mods.get(fullname)
            if entry is None:
                return None
            where = [os.path.dirname(entry[0])] if entry[1] else None
            return importlib.util.spec_from_file_location(
                fullname, entry[0], loader=Loader(fullname, entry[0]), submodule_search_locations=where)

    sys.meta_path.insert(0, Finder)

    monitor = sys.monitoring
    owned = {}
    tool = None
    if targets:
        for candidate in (4, 3):
            try:
                monitor.use_tool_id(candidate, "fork_check")
                tool = candidate
                break
            except ValueError:
                continue
        if tool is None:
            reply["status"] = "no-monitor"
            _send(out_fd, reply)
            return
        events = monitor.events.LINE | monitor.events.PY_START
        for code, lines in targets.values():
            stack = [code]
            while stack:
                block = stack.pop()
                own = {n for _, _, n in block.co_lines() if n in lines}
                if own and block.co_flags & 1:
                    owned[block] = own
                    monitor.set_local_events(tool, block, events)
                stack.extend(c for c in block.co_consts if type(c) is types.CodeType)

        importing = [0]
        exec_module = SourceFileLoader.exec_module

        def exec_counted(loader, module):
            importing[0] += 1
            try:
                return exec_module(loader, module)
            finally:
                importing[0] -= 1

        SourceFileLoader.exec_module = exec_counted

        def hit():
            if importing[0]:
                return
            if current[0] is not None:
                line_hits.add(current[0])
            elif phase[0] is not None:
                class_hits.add(phase[0])

        def on_line(code, line):
            if line in owned.get(code, ()):
                hit()
            return monitor.DISABLE

        def on_start(code, offset):
            if code.co_firstlineno in owned.get(code, ()):
                hit()

        monitor.register_callback(tool, monitor.events.LINE, on_line)
        monitor.register_callback(tool, monitor.events.PY_START, on_start)

    def audit(event, args):
        if event == "open":
            path = args[0]
            if type(path) is bytes:
                path = os.fsdecode(path)
            if type(path) is str and path.rpartition("/")[2] in names and os.path.abspath(path) in watched:
                reply["taint"].append(path)
        elif event in _SPAWN_EVENTS:
            blocked.add(current[0])
            raise RuntimeError("fork_check: process creation is not verified")

    sys.addaudithook(audit)

    class Handler(logging.Handler):
        def emit(self, record):
            self.format(record)

        def handleError(self, record):
            raise

    before = _state()
    try:
        broken = {}
        if level == 0:
            importlib.import_module("sqlglot")
        for name in job["imports"]:
            try:
                importlib.import_module(name)
            except _Abort:
                raise
            except BaseException as exc:
                broken[name] = repr(exc)[:200]
        cases = []
        index = {}
        for number, module_name, class_name, method in job["tests"]:
            status[number] = "notrun"
            if module_name in broken:
                status[number] = "import"
                continue
            module = sys.modules.get(module_name)
            if module is None:
                try:
                    module = importlib.import_module(module_name)
                except _Abort:
                    raise
                except BaseException as exc:
                    broken[module_name] = repr(exc)[:200]
                    status[number] = "import"
                    continue
            cls = getattr(module, class_name, None)
            if not (isinstance(cls, type) and issubclass(cls, unittest.TestCase)):
                status[number] = "noclass"
                continue
            try:
                case = cls(method)
            except Exception:
                status[number] = "nomethod"
                continue
            cases.append((number, case))
            index[id(case)] = number
        need_line = set(job["need_line"])
        context = job.get("context", ())
        bad_classes = set()
        seen = {}

        class Result(unittest.TestResult):
            def startTest(self, test):
                super().startTest(test)
                number = index.get(id(test))
                current[0] = number
                seen[number] = []
                if tool is not None:
                    monitor.restart_events()

            def stopTest(self, test):
                super().stopTest(test)
                number = index.get(id(test))
                current[0] = None
                events = seen.get(number)
                if number is None or events is None:
                    return
                verdict = "pass" if events == ["pass"] else next((e for e in events if e != "pass"), "noresult")
                if verdict == "pass":
                    if number in blocked:
                        verdict = "blocked"
                    elif number not in context and any(
                        path not in reached and (must is None or number in must) for path, must in must_reach.items()
                    ):
                        verdict = "nomodule"
                    elif number in need_line and number not in line_hits and type(test) not in class_hits:
                        verdict = "noline"
                status[number] = verdict
                if verdict != "pass":
                    self.stop()

            def note(self, test, what):
                number = index.get(id(test))
                if number is None:
                    number = index.get(id(getattr(test, "test_case", None)))
                    if number is not None and what == "pass":
                        return
                if number is None:
                    bad_classes.add(phase[0])
                    self.stop()
                else:
                    seen.setdefault(number, []).append(what)

            def addSuccess(self, test):
                self.note(test, "pass")

            def addFailure(self, test, err):
                self.note(test, "fail")

            def addError(self, test, err):
                self.note(test, "error")

            def addSkip(self, test, reason):
                self.note(test, "skip")

            def addExpectedFailure(self, test, err):
                self.note(test, "xfail")

            def addUnexpectedSuccess(self, test):
                self.note(test, "upass")

            def addSubTest(self, test, subtest, err):
                if err is not None:
                    self.note(test, "fail" if issubclass(err[0], test.failureException) else "error")

        result = Result()
        suite = unittest.TestSuite()
        handler = Handler()
        handler.setFormatter(logging.Formatter("%(levelname)-8s %(name)s:%(filename)s:%(lineno)d %(message)s", "%H:%M:%S"))
        logging.root.addHandler(handler)

        def fixtures(test):
            if tool is not None:
                monitor.restart_events()
            with warnings.catch_warnings(record=True):
                if not sys.warnoptions:
                    warnings.filterwarnings("always", category=DeprecationWarning)
                    warnings.filterwarnings("always", category=PendingDeprecationWarning)
                phase[0] = getattr(result, "_previousTestClass", None)
                suite._tearDownPreviousClass(test, result)
                if test is None:
                    suite._handleModuleTearDown(result)
                    return False
                phase[0] = type(test)
                suite._handleModuleFixture(test, result)
                suite._handleClassSetUp(test, result)
                result._previousTestClass = type(test)
                phase[0] = None
                if getattr(type(test), "_classSetupFailed", False) or getattr(result, "_moduleSetUpFailed", False):
                    bad_classes.add(type(test))
                    result.stop()
                    return False
                test(result)
                return True

        for number, case in cases:
            if result.shouldStop:
                break
            fixtures(case)
        fixtures(None)
        reply["fixtures_closed"] = not bad_classes and None not in blocked and not broken
        logging.root.removeHandler(handler)
        for number, case in cases:
            if type(case) in bad_classes:
                status[number] = "fixture"
    except _Abort:
        signal.setitimer(signal.ITIMER_PROF, 0)
        reply["status"] = "cpu-timeout"
        if current[0] is not None:
            status[current[0]] = "timeout"
    signal.setitimer(signal.ITIMER_PROF, 0)
    signal.signal(signal.SIGPROF, alarm)
    after = _state()
    if before["sqlglot.pretty"] is None:
        after["sqlglot.pretty"] = None
    reply["moved"] = sorted(key for key in before if before[key] != after[key])
    served = True
    for name, (path, _, code) in mods.items():
        module = sys.modules.get(name)
        if module is not None and (type(getattr(module, "__loader__", None)) is not Loader
                                   or getattr(module, "__file__", None) != path):
            served = False
    if not served:
        reply["taint"].append("patched module not served by the check run")
    _send(out_fd, reply)


def _build_closure(dest):
    import importlib
    import json
    import unittest
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != os.path.dirname(os.path.abspath(__file__))]
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    importlib.import_module("sqlglot")
    from sqlglot.dialects import DIALECT_MODULE_NAMES
    result = {}
    for name in sorted(set(DIALECT_MODULE_NAMES)):
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            before = set(sys.modules)
            importlib.import_module("sqlglot.dialects." + name)
            _send(w, sorted(m for m in set(sys.modules) - before if m.startswith("sqlglot.")))
            os._exit(0)
        os.close(w)
        result[name] = _recv(r)
        os.close(r)
        os.waitpid(pid, 0)
    with open(dest, "w", encoding="utf-8") as stream:
        json.dump({"dialects": result}, stream, sort_keys=True)


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--closure":
    _build_closure(sys.argv[2])
