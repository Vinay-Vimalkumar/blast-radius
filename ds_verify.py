from contextlib import contextmanager
from functools import lru_cache
import dis
import hashlib
import io
import tokenize
import json
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

from ds_operations import BASE, TEST, candidates, key, input_digest

HERE = Path(__file__).resolve().parent
FIELDS = ("co_name", "co_qualname", "co_filename", "co_firstlineno", "co_argcount",
          "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals", "co_stacksize", "co_flags",
          "co_names", "co_varnames", "co_freevars", "co_cellvars", "co_linetable", "co_exceptiontable")


def fingerprint():
    return [sys.implementation.cache_tag, platform.python_version(), sys.flags.optimize,
            sys._xoptions.get("no_debug_ranges") is not None, struct.calcsize("P")]


def compare(old, new, changes):
    if any(getattr(old, f) != getattr(new, f) for f in FIELDS):
        raise ValueError("metadata")
    if len(old.co_consts) != len(new.co_consts):
        raise ValueError("constants")
    changed_indices = set()
    for i, (a, b) in enumerate(zip(old.co_consts, new.co_consts)):
        if type(a) is not type(b):
            raise ValueError("constant type")
        if type(a) is types.CodeType:
            compare(a, b, changes)
        elif type(a) in (str, int) and a != b:
            changed_indices.add(i)
        elif a is not b and repr(a) != repr(b):
            raise ValueError("constant")
    if old.co_code == new.co_code and not changed_indices:
        return
    a = list(dis.get_instructions(old))
    b = list(dis.get_instructions(new))
    if len(a) != len(b):
        raise ValueError("instructions")
    targets = candidates(old)
    used = set()
    for x, y in zip(a, b):
        if (x.offset,x.opname,x.positions) != (y.offset,y.opname,y.positions):
            raise ValueError("layout")
        if x.opcode in dis.hasconst and x.arg in changed_indices:
            if x.opname not in {"LOAD_CONST","RETURN_CONST"} or x.arg != y.arg:
                raise ValueError("indirect constants")
            used.add(x.arg)
        elif x.arg == y.arg:
            continue
        elif x.opname not in {"BINARY_OP","COMPARE_OP","IS_OP","CONTAINS_OP","LOAD_SMALL_INT"}:
            raise ValueError("operand")
        if x.offset not in targets:
            raise ValueError("scope")
        changes.append((key(old),x.offset,targets[x.offset]))
    if used != changed_indices:
        raise ValueError("unreferenced constant")


def eligible(change, table):
    if change.structural or len(change.removed) != 1 or change.removed != change.added:
        return
    path = change.path
    item = table["files"].get(path)
    if not item or str(next(iter(change.removed))) not in item["lines"]:
        return
    if hashlib.sha256(change.original.encode()).hexdigest() != item["sha256"]:
        return
    a, b = change.original.splitlines(keepends=True), change.patched.splitlines(keepends=True)
    line, = change.removed
    if len(a) != len(b) or a[:line-1] != b[:line-1] or a[line:] != b[line:]:
        return
    if len(a[line-1].encode()) != len(b[line-1].encode()):
        return
    def tokens(source):
        return list(tokenize.generate_tokens(io.StringIO(source).readline))
    before, after = tokens(change.original), tokens(change.patched)
    if len(before) != len(after):
        return
    altered = [(u, v) for u, v in zip(before, after) if u[:4] != v[:4]]
    if len(altered) != 1:
        return
    u, v = altered[0]
    if u.type != v.type or u.type not in (tokenize.OP, tokenize.NUMBER, tokenize.STRING):
        return
    if u.start != v.start or u.end != v.end or u.start[0] != line or u.end[0] != line:
        return
    col, end = u.start[1], u.end[1]
    if a[line-1][:col] != b[line-1][:col] or a[line-1][end:] != b[line-1][end:]:
        return
    changes = []
    compare(compile(change.original,path,"exec",dont_inherit=True),
            compile(change.patched,path,"exec",dont_inherit=True),changes)
    if len(changes) != 1:
        return
    fn, offset, loc = changes[0]
    if loc != line:
        return
    for owner, off, queries in item["lines"][str(line)]:
        if (fn,offset) == (owner,off):
            return queries


@contextmanager
def overlay(change):
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        rel = Path(change.path)
        for depth in range(len(rel.parts)):
            parent = Path(*rel.parts[:depth])
            source = BASE / parent
            here = stage / parent
            here.mkdir(exist_ok=True)
            for child in source.iterdir():
                if child.name == rel.parts[depth] or child.name == "__pycache__":
                    continue
                if depth or child.name in {"sqlglot","tests"}:
                    (here / child.name).symlink_to(child, target_is_directory=child.is_dir())
        (stage / "sitecustomize.py").write_bytes((HERE / "ds_origin.py").read_bytes())
        out = stage / rel
        out.write_text(change.patched, encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(stage), PYTHON_CPU_COUNT="1", OPENBLAS_NUM_THREADS="1",
                   OMP_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONHASHSEED="0",
                   PYTHONDONTWRITEBYTECODE="1", TPC_ORIGIN_STAGE=str(stage),
                   TPC_ORIGIN_FILE=change.path, TPC_ORIGIN_HASH=hashlib.sha256(out.read_bytes()).hexdigest())
        yield stage, env

def run_witness(change, queries, deadline):
    with overlay(change) as (stage, env):
        run = [sys.executable, "-B", str(HERE / "ds_worker.py"), str(stage), change.path,
               env["TPC_ORIGIN_HASH"], *(str(i) for i in queries)]
        import signal
        with subprocess.Popen(run, cwd=stage, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              start_new_session=True, text=True) as proc:
            try:
                stdout = proc.communicate(timeout=max(0.01, deadline-time.monotonic()))[0]
            except BaseException:
                try:
                    os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                raise
        return proc.returncode == 0 and stdout.strip() == "verified"


@lru_cache(maxsize=1)
def _profile():
    import gzip
    with gzip.open(HERE / "ds_sites.json.gz","rt") as f:
        return json.load(f)

def refine_ds(changes, chosen, cov, catalog, deadline):
    if (TEST not in chosen or TEST not in catalog or TEST in cov.missing or TEST not in cov.nodes
            or len(changes) != 1 or time.monotonic() >= deadline):
        return chosen
    try:
        table = _profile()
        if (fingerprint() != table["python"] or os.getenv("SQLGLOT_ROOT",str(BASE)) != str(BASE)
                or os.getenv("SKIP_INTEGRATION") != "0"):
            return chosen
        qs = eligible(changes[0], table)
        if qs and input_digest() == table["inputs"] and run_witness(changes[0], qs,min(deadline,time.monotonic()+7)):
            return chosen - {TEST}
    except (OSError, ValueError, KeyError, IndexError, SyntaxError, RuntimeError, subprocess.SubprocessError, tokenize.TokenError):
        pass
    return chosen
