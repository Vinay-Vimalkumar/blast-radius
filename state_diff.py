import ast
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import types
from enum import Enum
from functools import cache
from pathlib import Path

from map_store import read_map

MAP = Path(__file__).resolve().parent / "maps" / "state.json.gz"
VALID_PATH = re.compile(r"sqlglot/(?:[A-Za-z_]\w*/)*[A-Za-z_]\w*\.py\Z", re.ASCII)
VERSION = 3


def _json(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("ascii")).hexdigest()


def _typename(cls):
    return type.__getattribute__(cls, "__module__") + "." + type.__getattribute__(cls, "__qualname__")


class Unsupported(ValueError):
    pass


class Fingerprints:

    def __init__(self):
        self.memo = {}
        self.active = set()

    def value(self, obj):
        kind = type(obj)
        if obj is None or kind is bool or kind is str or kind is int:
            return obj
        if kind is bytes:
            return ["bytes", obj.hex()]
        if kind is float:
            return ["float", struct.pack(">d", obj).hex()]
        if obj is Ellipsis:
            return ["ellipsis"]
        if isinstance(obj, type):
            return ["class", _typename(obj)]
        if isinstance(obj, Enum):
            return ["enum", _typename(kind), object.__getattribute__(obj, "_name_")]
        key = id(obj)
        if key in self.memo:
            return self.memo[key][1]
        if key in self.active or len(self.active) >= 64:
            raise Unsupported("cyclic or deep value")
        self.active.add(key)
        try:
            if kind is types.CodeType:
                parts = [obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, obj.co_freevars,
                         obj.co_cellvars, obj.co_argcount, obj.co_kwonlyargcount, obj.co_posonlyargcount,
                         obj.co_flags, obj.co_exceptiontable, obj.co_name, obj.co_qualname]
                data = ["code", *map(self.value, parts)]
            elif kind is types.FunctionType:
                cells = tuple(cell.cell_contents for cell in obj.__closure__ or ())
                data = ["fn", obj.__module__, obj.__qualname__, self.value(obj.__code__),
                        self.value(obj.__defaults__), self.value(obj.__kwdefaults__),
                        self.value(cells), self.value(obj.__dict__)]
            elif kind is types.MethodType:
                data = ["method", self.value(obj.__func__), self.value(obj.__self__)]
            elif kind in (dict, set, frozenset, tuple, list):
                if len(obj) > 20000:
                    raise Unsupported("large container")
                if kind is dict:
                    items = [[self.value(k), self.value(v)] for k, v in obj.items()]
                else:
                    items = [self.value(v) for v in obj]
                    if kind in (set, frozenset):
                        items.sort(key=_json)
                data = [kind.__name__, items]
            else:
                raise Unsupported("unmodeled type")
            result = ["digest", _digest(data)]
            self.memo[key] = (obj, result)
            return result
        finally:
            self.active.remove(key)

    def binding(self, value):
        try:
            return _digest(self.value(value))
        except (ValueError, TypeError, RecursionError):
            return None


def _module(path):
    return path[:-3].replace("/", ".").removesuffix(".__init__")


def snapshot(modules):
    fp = Fingerprints()
    rows, parents = {}, {}
    visited = set()

    def collect(owner, module, namespace):
        if owner in visited:
            return
        visited.add(owner)
        attrs = rows.setdefault(module, {}).setdefault(owner, {})
        for name, obj in namespace.items():
            if name.startswith("__"):
                continue
            if isinstance(obj, type) and _typename(obj) == owner + "." + name:
                clsname = _typename(obj)
                parents[clsname] = [_typename(base) for base in type.__getattribute__(obj, "__mro__")[1:]]
                collect(clsname, module, type.__getattribute__(obj, "__dict__"))
            if name.isupper() or name in {"arg_types", "required_args"} or (
                not name.startswith("_") and type(obj) in (dict, set, frozenset)
            ):
                attrs[name] = fp.binding(obj)

    for module in modules:
        collect(module, module, vars(sys.modules[module]))
    return rows, parents


def _runtime():
    return [sys.implementation.name, *sys.version_info[:3], importlib.util.MAGIC_NUMBER.hex()]


def _guard():
    def audit(event, args):
        if event == "open":
            if args[2] & (os.O_CREAT | os.O_WRONLY | os.O_RDWR | os.O_TRUNC | os.O_APPEND):
                raise PermissionError("write during import")
        elif event.startswith(("socket.", "subprocess.", "ctypes.", "shutil.", "os.spawn", "os.exec",
                               "os.fork", "os.posix_spawn", "os.startfile")) or event in {
            "os.system", "os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.link", "os.symlink",
            "os.chmod", "os.truncate", "os.utime", "os.chdir", "os.putenv", "os.unsetenv", "os.kill",
        }:
            raise PermissionError("external effect during import")
    sys.addaudithook(audit)


def _worker(request):
    import unittest

    root = Path(request["root"]).resolve(strict=True)
    overlay = request["overlay"]
    loaded = set()

    class Overlay(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            loaded.add(fullname)
            return compile(overlay[fullname].encode(), self.path, "exec", dont_inherit=True)

    class Finder:
        def find_spec(self, fullname, path, target=None):
            if fullname not in overlay:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or Path(spec.origin).resolve() != root / request["paths"][fullname]:
                raise ImportError("unexpected overlay origin")
            spec.loader = Overlay(fullname, spec.origin)
            return spec

    sys.path.insert(0, str(root))
    sys.meta_path.insert(0, Finder())
    _guard()
    modules = []
    for module in request["modules"]:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            if request.get("building"):
                continue
            raise
        if not Path(mod.__file__).resolve().is_relative_to(root):
            raise ImportError("wrong checkout")
        modules.append(module)
    if set(overlay) != loaded:
        raise ValueError("overlay not fully imported")
    return snapshot(modules)


def _child(root, modules, changes, building=False):
    paths = {_module(c.path): c.path for c in changes}
    request = {"root": str(root.resolve()), "modules": sorted(modules), "paths": paths,
               "overlay": {_module(c.path): c.patched for c in changes}, "building": building}
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.run(
            [sys.executable, "-s", "-S", "-B", str(Path(__file__).resolve()), "--child"],
            input=_json(request).encode("ascii"), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=directory, env={**os.environ, "PYTHONPATH": "", "PYTHONHASHSEED": "0",
                                "PYTHONOPTIMIZE": "0", "PYTHONNOUSERSITE": "1"},
            timeout=90 if building else 8, check=True,
        )
    return json.loads(result.stdout)


def _statements(body, owner):
    for stmt in body:
        if isinstance(stmt, ast.ClassDef):
            yield from _statements(stmt.body, owner + "." + stmt.name)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if len(targets) == 1 and isinstance(targets[0], ast.Name) and stmt.value is not None:
                yield owner, targets[0].id, stmt


def _same(a, b):
    return ast.dump(a) == ast.dump(b)


def _literal_map(node):
    if isinstance(node, ast.Dict) and all(isinstance(k, ast.Constant) and type(k.value) is str
                                          for k in node.keys) and all(
        isinstance(v, ast.Constant) and type(v.value) is str for v in node.values
    ):
        return [(k.value, v.value) for k, v in zip(node.keys, node.values)]
    return None


def _edits(change):
    module = _module(change.path)
    left, right = ast.parse(change.original.encode()), ast.parse(change.patched.encode())
    old, new = list(_statements(left.body, module)), list(_statements(right.body, module))
    if len(old) != len(new):
        return None
    edits = []
    for (owner, name, a), (owner2, name2, b) in zip(old, new):
        if (owner, name) != (owner2, name2):
            return None
        if _same(a.value, b.value):
            continue
        mapping, replacement = _literal_map(a.value), _literal_map(b.value)
        matches = mapping is not None and replacement is not None and (
            list(dict(mapping).items()) == list(dict(replacement).items())
            and (len(mapping) == len(replacement) or max(len(mapping), len(replacement)) <= 5)
        )
        if not (name.isupper() or name in {"arg_types", "required_args"}) or not (
            matches or (mapping is not None and replacement is not None and len(mapping) == len(replacement))
        ):
            return None
        edits.append((module, owner, name, matches, mapping is not None and replacement is not None))
        a.value = b.value = ast.Constant(None)
    if any(not e[3] for e in edits) and any(
        isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in TIME_FIELDS
        for n in ast.walk(left)
    ):
        return None
    return edits if edits and _same(left, right) else None


@cache
def _map():
    return read_map(MAP)


TIME_INPUTS = {"TIME_MAPPING", "FORMAT_MAPPING", "INVERSE_TIME_MAPPING"}
TIME_FIELDS = TIME_INPUTS | {"TIME_TRIE", "FORMAT_TRIE", "INVERSE_FORMAT_MAPPING",
                            "INVERSE_FORMAT_TRIE", "INVERSE_TIME_TRIE"}


def _time_readers(differences, catalog, cov, proof):
    from declarations import _lookup_index
    from import_effects import _index
    tests, records, _, access = _lookup_index()
    keep = catalog - (set(tests) & set(proof["collected"]))
    sites = {path: set(lines) for path, lines in proof["reflection"].items()}
    for owner, attr in differences:
        if (owner, attr, "*") not in records or attr not in access["reads"]:
            return None
        keep.update(tests[i] for i in records.get((owner, attr, "@"), ()))
        for path, lines in access["reads"][attr].items():
            sites.setdefault(path, set()).update(lines)
    for path, lines in sites.items():
        keep.update(cov.lines(path, lines))
    startup, footprints, _ = _index()
    affected = {mod for _, attr in differences for path, lines in access["reads"][attr].items()
                for line in lines for mod in startup.get(path, {}).get(str(line), ())}
    if affected:
        certified = set()
        for modules, nodes in footprints.values():
            if affected.isdisjoint(modules):
                certified.update(nodes)
        keep.update(catalog - certified)
    return keep


def _refine(changes, chosen, cov, catalog):
    if not changes or len(changes) > 4 or len(chosen) * 2 < len(catalog):
        return chosen, "already narrow or outside budget"
    if any(c.structural or not VALID_PATH.fullmatch(c.path) for c in changes):
        return chosen, "structural or non-package edit"
    if sum(len(c.original) + len(c.patched) for c in changes) > 1_500_000:
        return chosen, "source budget"
    info = _map()
    for c in changes:
        if "ranges" in info and not any(not c.removed or any(a <= n <= b for n in c.removed)
                                        for a, b in info["ranges"].get(c.path, ())):
            return chosen, "outside indexed declarations"
    if info["version"] != VERSION or info["runtime"] != _runtime():
        return chosen, "incompatible baseline"
    root = Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot"))
    edits = []
    paths = set()
    for c in changes:
        if c.path in paths or info["sources"].get(c.path) != hashlib.sha256(c.original.encode()).hexdigest():
            return chosen, "unverified baseline"
        paths.add(c.path)
        found = _edits(c)
        if not found:
            return chosen, "unsupported declaration edit"
        edits.extend(found)
    harmless = all(e[3] for e in edits)
    proof = info.get("time", {})
    owners = {e[1] for e in edits}
    if not harmless and not all(strings and name in TIME_INPUTS and owner in proof.get("owners", ())
                                for _, owner, name, _, strings in edits):
        return chosen, "no certified reader model"
    modules = {e[0] for e in edits}
    for _, owner, _, _, _ in edits:
        modules.update(info["descendants"].get(owner, ()))
    if len(modules) > 8:
        return chosen, "descendant budget"

    baseline = info["rows"]
    for module, owner, name, _, _ in edits:
        if not baseline.get(module, {}).get(owner, {}).get(name):
            return chosen, "missing baseline binding"
    rows, _ = _child(root, modules, changes)
    different = set()
    for module, owner, name, match, _ in edits:
        fields = {name} if match else TIME_FIELDS
        for field in fields:
            before = baseline[module][owner].get(field)
            after = rows[module][owner].get(field)
            if before is None or after is None:
                if field in baseline[module][owner] or field in rows[module][owner] or field == name:
                    return chosen, "unmodeled binding"
                continue
            if before != after:
                different.add((owner, field))
    for module, owner, name, match, _ in edits:
        if not rows[module][owner][name] or ((owner, name) in different) == match:
            return chosen, "unsupported or inconclusive comparison"
    keep = set(cov.missing) | (catalog - set(cov.nodes))
    for c in changes:
        keep.update(cov.fixture(c.path))
    if harmless and not different:
        return chosen & keep, "shadowed literal matches at runtime"
    if not different or any(owner not in owners or name not in TIME_FIELDS for owner, name in different):
        return chosen, "unexpected derived state"
    readers = _time_readers(different, catalog, cov, proof)
    if readers is None:
        return chosen, "incomplete reader support"
    return chosen & (keep | readers), f"verified {len(different)} time-format bindings"


def refine_state_diagnostics(changes, chosen, cov, catalog):
    try:
        return _refine(changes, chosen, cov, catalog)
    except (OSError, ValueError, KeyError, TypeError, IndexError, SyntaxError, RecursionError,
            EOFError, subprocess.SubprocessError) as exc:
        return chosen, "fallback: " + type(exc).__name__


def refine_state(changes, chosen, cov, catalog):
    return refine_state_diagnostics(changes, chosen, cov, catalog)[0]


if __name__ == "__main__" and sys.argv[1:] == ["--child"]:
    request = json.load(sys.stdin.buffer)
    if sys.platform.startswith("linux") and not request.get("building"):
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
        resource.setrlimit(resource.RLIMIT_AS, (768 << 20, 768 << 20))
    print(_json(_worker(request)))
