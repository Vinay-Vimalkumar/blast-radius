import os
import sys
import time
from functools import cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAP = HERE / "packed_maps" / "import_state.marshal"
VERSION = 5
SKIP_MODULE = frozenset({"__builtins__", "__loader__", "__spec__", "__cached__", "__file__", "__path__"})
SKIP_CLASS = frozenset({"__dict__", "__weakref__", "__module__", "__qualname__", "__firstlineno__",
                        "__static_attributes__", "__abstractmethods__", "_abc_impl", "__parameters__",
                        "__orig_bases__", "__type_params__"})
INERT = frozenset({"__annotations__", "__total__", "__required_keys__", "__optional_keys__", "__readonly_keys__",
                   "__mutable_keys__", "__annotate__"})
ENUM_STATE = frozenset({"_value2member_map_", "_hashable_values_", "_unhashable_values_", "_unhashable_values_map_",
                        "_member_map_", "_member_names_", "_value_repr_", "_member_type_", "_new_member_", "_use_args_"})
NAME_LOADS = frozenset({"LOAD_ATTR", "LOAD_GLOBAL", "LOAD_NAME", "LOAD_METHOD", "LOAD_SUPER_ATTR", "LOAD_DEREF",
                        "LOAD_CLOSURE", "LOAD_FROM_DICT_OR_GLOBALS", "LOAD_FROM_DICT_OR_DEREF", "IMPORT_FROM",
                        "LOAD_FAST_CHECK"})
PROLOGUE = frozenset({"RESUME", "COPY_FREE_VARS", "MAKE_CELL", "RETURN_GENERATOR"})
INF = 1 << 30
MIN_GAIN = 3.0
CHILD_SECONDS = [16.0]


def _line_map(original, patched, removed, added):
    mapping, old, new = {}, 1, 1
    total = patched.count("\n") + 1
    while new <= total:
        if new in added and old in removed:
            mapping[new] = old
            old += 1
            new += 1
        elif new in added:
            new += 1
        elif old in removed:
            old += 1
        else:
            mapping[new] = old
            old += 1
            new += 1
    return mapping


def _touched(original, patched, removed, added):
    points = set(removed)
    old, new = 1, 1
    total = patched.count("\n") + 1
    while new <= total:
        if new in added and old in removed:
            old += 1
            new += 1
        elif new in added:
            points.add(old - 0.5)
            new += 1
        elif old in removed:
            old += 1
        else:
            old += 1
            new += 1
    return points


def _install(root, sources, trees):
    import importlib.machinery as machinery

    patched = {}
    for path, text in sources.items():
        name = path[:-3].replace("/", ".")
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        patched[name] = (path, text)
    prefix = os.path.join(root, "")

    class Capture(machinery.SourceFileLoader):
        def get_code(self, fullname):
            code = super().get_code(fullname)
            trees[fullname] = (os.path.realpath(self.path)[len(prefix):].replace(os.sep, "/"), code)
            return code

    class Overlay(machinery.SourceFileLoader):
        def get_code(self, fullname):
            import marshal

            path, text = patched[fullname]
            code = marshal.loads(marshal.dumps(compile(text.encode("utf-8"), self.path, "exec", dont_inherit=True)))
            trees[fullname] = (path, code)
            return code

    class Finder:
        @staticmethod
        def find_spec(fullname, path=None, target=None):
            if fullname != "sqlglot" and not fullname.startswith("sqlglot."):
                return None
            spec = machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.origin is None:
                return spec
            if fullname in patched:
                if os.path.realpath(spec.origin) != os.path.join(root, *patched[fullname][0].split("/")):
                    raise ImportError("patched module is not where the pinned tree has it")
                spec.loader = Overlay(fullname, spec.origin)
            elif type(spec.loader) is machinery.SourceFileLoader:
                spec.loader = Capture(fullname, spec.origin)
            return spec

    sys.meta_path.insert(0, Finder)
    return set(patched)


def _module_names(root):
    base = os.path.join(str(root), "sqlglot")
    names = []
    for folder, folders, files in os.walk(base):
        if "__init__.py" not in files:
            folders[:] = []
            continue
        package = "sqlglot" + folder[len(base):].replace(os.sep, ".")
        if package != "sqlglot":
            names.append(package)
        names.extend(f"{package}.{file[:-3]}" for file in files
                     if file.endswith(".py") and file not in ("__init__.py", "__main__.py", "_version.py"))
    return ["sqlglot", *sorted(names)]


def _import_everything(names):
    import importlib
    import unittest

    failed = {}
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed[name] = type(exc).__name__
    return failed


def _index_code(trees, remap):
    from types import CodeType

    ident_of, code_of, file_of = {}, {}, {}
    for module, (file, top) in trees.items():
        table = remap.get(file)
        groups = {}
        stack = [top]
        while stack:
            block = stack.pop()
            nested = [c for c in block.co_consts if type(c) is CodeType]
            for const in nested:
                if const.co_flags & 1:
                    line = const.co_firstlineno
                    if table is not None:
                        line = table.get(line, f"+{line}")
                    groups.setdefault((const.co_qualname, line), []).append(const)
            stack.extend(reversed(nested))
        for (qualname, line), members in groups.items():
            for position, code in enumerate(members):
                ident = f"{module}.{qualname}@{line}" if len(members) == 1 else f"{module}.{qualname}@{line}#{position}"
                ident_of[id(code)] = ident
                code_of[ident] = code
                file_of[ident] = file
    return ident_of, code_of, file_of


def _has_unmapped_code(changes, sources, remap, base):
    for change in changes:
        path = change.path
        if path not in sources or change.patched != sources[path]:
            continue
        lines = change.patched.split("\n")
        added = (lines[n - 1] for n in change.added if 0 < n <= len(lines))
        if not any("def " in line or "lambda" in line or " for " in line or line.lstrip().startswith("@")
                   for line in added):
            continue
        module = path[:-3].replace("/", ".").removesuffix(".__init__")
        from semantics import patched_code

        top = patched_code(path, sources[path])
        _, codes, _ = _index_code({module: (path, top)}, remap)
        for ident in codes.keys() - base["code"].get(path, {}).keys():
            if ident not in base["extents"] and not any(ident in other for other in base["code"].values()):
                return True
    return False


def _code_extent(code):
    import dis
    from types import CodeType

    low, high = INF, 0
    stack = [code]
    while stack:
        block = stack.pop()
        stack.extend(c for c in block.co_consts if type(c) is CodeType)
        previous = ""
        for ins in dis.get_instructions(block):
            name = ins.opname
            if name in PROLOGUE or (name == "POP_TOP" and previous == "RETURN_GENERATOR"):
                previous = name
                continue
            previous = name
            where = ins.positions
            if where is None or where.lineno is None:
                continue
            low = min(low, where.lineno)
            high = max(high, where.end_lineno or where.lineno)
    return [low, high] if high else None


class _Snapshot:

    def __init__(self, loose, ident_of):
        import enum
        import functools
        import logging
        import re
        import types

        self.loose_paths = frozenset(loose)
        self.ident_of = ident_of
        self.memo = {}
        self.memo_loose = {}
        self.keep = []
        self.out = {}
        self.unmodeled = {}
        self.classes = []
        self.class_ids = set()
        self.pending = []
        self.first = {}
        self.current = ""
        self.loose = False
        self.path = {}
        self.low = INF
        self.top = {}
        self.Enum = enum.Enum
        self.partial = functools.partial
        self.Logger = logging.Logger
        self.Pattern = re.Pattern
        self.Function = types.FunctionType
        self.Method = types.MethodType
        self.Module = types.ModuleType
        self.Code = types.CodeType
        self.Proxy = types.MappingProxyType
        self.views = (type({}.keys()), type({}.values()), type({}.items()))
        self.builtin = (types.BuiltinFunctionType, types.MethodDescriptorType, types.WrapperDescriptorType,
                        types.GetSetDescriptorType, types.MemberDescriptorType, types.ClassMethodDescriptorType,
                        types.MethodWrapperType)
        self.alias = (types.GenericAlias, types.UnionType)
        self.address = re.compile(r" at 0x[0-9a-fA-F]+")

    @staticmethod
    def cref(cls):
        return f"{cls.__module__}.{cls.__qualname__}"

    def put(self, path, value, top=False):
        self.current = path
        self.loose = path in self.loose_paths
        token = self.tok(value, 0)
        self.out[path] = token if type(token) is int else hash(token)
        if top and isinstance(value, (dict, list, set, bytearray)):
            self.top.setdefault(id(value), []).append(path)

    def code_hash(self, code):
        consts = []
        for const in code.co_consts:
            kind = type(const)
            if kind is self.Code:
                ident = self.ident_of.get(id(const))
                consts.append(ident if ident is not None else ("code", self.code_hash(const)))
            elif kind is str:
                consts.append(const)
            elif kind is frozenset:
                consts.append(("fs", frozenset((type(e).__name__, repr(e)) for e in const)))
            else:
                consts.append((kind.__name__, repr(const)))
        return hash((code.co_code, tuple(consts), code.co_names, code.co_varnames, code.co_freevars,
                     code.co_cellvars, code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
                     code.co_flags, code.co_exceptiontable, code.co_qualname))

    def many(self, items, depth):
        get = (self.memo_loose if self.loose else self.memo).get
        tok = self.tok
        out = []
        for value in items:
            if type(value) is str:
                out.append(value)
            else:
                known = get(id(value))
                out.append(known if known is not None else tok(value, depth))
        return out

    def tok(self, x, depth):
        kind = type(x)
        if kind is str:
            return x
        if kind is int:
            return ("i", x)
        if x is None:
            return "\0None"
        if kind is bool:
            return "\0True" if x else "\0False"
        if kind is float or kind is complex:
            return ("f", repr(x))
        if kind is bytes:
            return ("b", x)
        memo = self.memo_loose if self.loose else self.memo
        ident = id(x)
        known = memo.get(ident)
        if known is not None:
            return known
        if isinstance(x, type):
            if (x.__module__ or "").startswith("sqlglot") and ident not in self.class_ids:
                self.class_ids.add(ident)
                self.classes.append(x)
            token = ("K", self.cref(x))
        elif isinstance(x, self.Enum):
            token = ("e", self.cref(kind), x.name)
        elif kind is self.Module:
            token = ("M", x.__name__)
        elif isinstance(x, self.builtin):
            token = ("B", getattr(x, "__qualname__", None) or repr(kind))
        elif kind is self.Pattern:
            token = ("re", x.pattern, x.flags)
        elif isinstance(x, self.Logger):
            token = ("L", x.name)
        elif kind is self.Code:
            token = ("code", self.code_hash(x))
        elif kind is self.Function and self.shared(x):
            code = x.__code__
            token = ("F", self.ident_of[id(code)]) if x.__name__ == code.co_name else ("F", self.ident_of[id(code)], x.__name__)
        else:
            token = None
        if token is not None:
            self.memo[ident] = self.memo_loose[ident] = token
            self.keep.append(x)
            return token
        at = self.path.get(ident)
        if at is not None:
            if at < self.low:
                self.low = at
            return ("^", depth - at)
        if depth > 60:
            self.unmodeled[self.current] = "depth:" + self.cref(kind)
            return ("deep", self.cref(kind))
        self.path[ident] = depth
        saved, self.low = self.low, INF
        try:
            token = self.build(x, kind, depth + 1)
        finally:
            del self.path[ident]
            low = self.low
            self.low = saved if low >= depth else min(saved, low)
        if low >= depth:
            memo[ident] = token
            self.keep.append(x)
        return token

    def shared(self, fn):
        code = fn.__code__
        ident = self.ident_of.get(id(code))
        if ident is None:
            return False
        qualname = code.co_qualname
        cells = code.co_freevars
        if "<locals>" in qualname or "<lambda>" in qualname or (cells and cells != ("__class__",)):
            return False
        first = self.first.get(ident)
        if first is None:
            self.first[ident] = fn
            self.pending.append((ident, fn))
            return True
        return first is fn

    def build(self, x, kind, depth):
        tok = self.tok
        loose = self.loose
        if kind is dict or isinstance(x, (dict, self.Proxy)):
            flat = self.many([part for pair in x.items() for part in pair], depth)
            extra = tok(x.default_factory, depth) if hasattr(x, "default_factory") else None
            if loose:
                return hash(("d~", self.cref(kind), frozenset(zip(flat[::2], flat[1::2])), extra))
            return hash(("d", self.cref(kind), tuple(flat), extra))
        if kind is list or kind is tuple or isinstance(x, (list, tuple)):
            items = self.many(x, depth)
            name = kind.__name__ if kind is list or kind is tuple else self.cref(kind)
            if loose:
                return hash((name + "~", tuple(sorted(hash(item) for item in items))))
            return hash((name, tuple(items)))
        if isinstance(x, (set, frozenset)):
            return hash((kind.__name__, frozenset(self.many(x, depth))))
        if kind is self.Function:
            code = x.__code__
            ident = self.ident_of.get(id(code))
            head = ("C", ident) if ident is not None else ("X", x.__module__, x.__qualname__, self.code_hash(code))
            cells = []
            for name, cell in zip(code.co_freevars, x.__closure__ or ()):
                if name == "__class__":
                    continue
                try:
                    cells.append(tok(cell.cell_contents, depth))
                except ValueError:
                    cells.append("\0empty")
            return hash((head, x.__name__, x.__doc__, tok(x.__defaults__, depth), tok(x.__kwdefaults__, depth),
                         tuple(cells), tok(x.__dict__, depth) if x.__dict__ else None))
        if kind is self.Method:
            return hash(("m", tok(x.__func__, depth), tok(x.__self__, depth)))
        if kind is classmethod or kind is staticmethod:
            return hash((kind.__name__, tok(x.__func__, depth)))
        if kind is property:
            return hash(("p", tok(x.fget, depth), tok(x.fset, depth), tok(x.fdel, depth), x.__doc__))
        if isinstance(x, self.partial):
            return hash(("P", tok(x.func, depth), tok(x.args, depth), tok(x.keywords, depth)))
        if isinstance(x, self.views):
            return hash((kind.__name__, tuple(self.many(list(x), depth))))
        if kind is bytearray:
            return hash(("ba", bytes(x)))
        module = kind.__module__ or ""
        if module == "__future__":
            return hash(("future", getattr(x, "compiler_flag", None)))
        if module.startswith("typing") or module in ("types", "collections.abc") or isinstance(x, self.alias):
            return hash(("T", repr(x)))
        if module.startswith("sqlglot") or hasattr(x, "__dict__") or hasattr(kind, "__slots__"):
            state = []
            for klass in kind.__mro__:
                slots = klass.__dict__.get("__slots__", ())
                for slot in (slots,) if isinstance(slots, str) else slots:
                    if slot in ("__dict__", "__weakref__"):
                        continue
                    try:
                        state.append((slot, tok(getattr(x, slot), depth)))
                    except AttributeError:
                        state.append((slot, "\0unset"))
            if hasattr(x, "__dict__"):
                state.append(("__dict__", tok(vars(x), depth)))
            if not module.startswith("sqlglot"):
                self.unmodeled.setdefault(self.current, "object:" + self.cref(kind))
            return hash(("O", self.cref(kind), tuple(state)))
        self.unmodeled[self.current] = "opaque:" + self.cref(kind)
        try:
            shown = self.address.sub("", repr(x))
        except Exception:
            shown = "?"
        return hash(("Q", self.cref(kind), shown))

    def run(self):
        names = sorted(n for n in sys.modules if n == "sqlglot" or n.startswith("sqlglot."))
        self.put("@modules", names)
        for name in names:
            module = sys.modules[name]
            if module is None:
                continue
            for key, value in list(vars(module).items()):
                if key not in SKIP_MODULE:
                    self.put(f"{name}.{key}", value, True)
        position = done = 0
        while position < len(self.classes) or done < len(self.pending):
            while position < len(self.classes):
                cls = self.classes[position]
                position += 1
                base = self.cref(cls)
                self.put(f"{base}.@bases", [self.cref(b) for b in cls.__mro__[1:]])
                for key, value in list(vars(cls).items()):
                    if key not in SKIP_CLASS:
                        self.put(f"{base}.{key}", value, True)
            while done < len(self.pending):
                ident, fn = self.pending[done]
                done += 1
                if fn.__defaults__ is not None or fn.__kwdefaults__ is not None:
                    self.put(f"fn:{ident}.__defaults__", (fn.__defaults__, fn.__kwdefaults__))
                if fn.__dict__:
                    self.put(f"fn:{ident}.__dict__", fn.__dict__)
                if fn.__doc__ is not None:
                    self.out[f"fn:{ident}.__doc__"] = hash(fn.__doc__)
        groups = sorted(tuple(sorted(paths)) for paths in self.top.values() if len(paths) > 1)
        self.out["@aliases"] = hash(tuple(groups))
        kinds = {}
        for reason in self.unmodeled.values():
            kinds[reason] = kinds.get(reason, 0) + 1
        self.out["@unmodeled"] = hash(tuple(sorted(kinds.items())))
        return self


def _process_probe():
    import decimal
    import locale
    import logging
    import warnings

    root = logging.getLogger()
    return {
        "recursion": sys.getrecursionlimit(),
        "switch": sys.getswitchinterval(),
        "environ": sorted(os.environ.items()),
        "cwd": os.getcwd(),
        "path": list(sys.path),
        "meta_path": [getattr(f, "__name__", type(f).__name__) for f in sys.meta_path],
        "hooks": [repr(sys.excepthook), repr(sys.displayhook), repr(sys.unraisablehook)],
        "root_logger": [root.level, len(root.handlers), logging.Logger.manager.disable],
        "warnings": [repr(f[:4]) for f in warnings.filters],
        "decimal": repr(decimal.getcontext()),
        "locale": list(locale.getlocale()),
        "builtins": sorted(vars(__import__("builtins"))),
        "others": sorted(n for n in sys.modules if n != "sqlglot" and not n.startswith("sqlglot.")),
    }


def _process_change(before, after, warned):
    import logging
    import typing

    out = {}
    for key, value in after.items():
        old = before[key]
        if value == old:
            out[key] = "same"
        elif isinstance(value, list) and all(isinstance(v, (str, tuple, list)) for v in value):
            out[key] = sorted(repr(v) for v in value if v not in old) + ["-" + repr(v) for v in old if v not in value]
        else:
            out[key] = repr(value)
    overloads = []
    for module, by_name in getattr(typing, "_overload_registry", {}).items():
        if str(module).startswith("sqlglot"):
            for qualname, by_line in by_name.items():
                overloads.append([module, qualname, [[repr(f.__defaults__), repr(f.__kwdefaults__)] for f in by_line.values()]])
    out["overloads"] = sorted(overloads)
    out["loggers"] = sorted((n, getattr(l, "level", None), len(getattr(l, "handlers", ())), getattr(l, "propagate", None))
                            for n, l in logging.Logger.manager.loggerDict.items() if n.startswith("sqlglot"))
    out["warned"] = sorted(set(warned))
    return out


def _snapshot_main(request):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (1536 << 20, 1536 << 20))
        seconds = int(request.get("cpu", 40))
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
    except Exception:
        pass
    ballast = [object() for _ in range(int(request.get("perturb", 0)))]
    root = os.path.realpath(request["root"])
    sources = request.get("sources", {})
    remap = {path: {int(k): v for k, v in table.items()} for path, table in request.get("remap", {}).items()}
    trees = {}
    _install(root, sources, trees)
    sys.path.insert(0, root)
    os.chdir(root)
    import unittest
    import warnings

    warned = []

    def note(message, category, filename, lineno, file=None, line=None):
        warned.append((category.__name__, str(message)))

    warnings.showwarning = note
    warnings.simplefilter("always")
    before = _process_probe()
    names = request["modules"]
    failed = _import_everything(names)
    ident_of, code_of, file_of = _index_code(trees, remap)
    snap = _Snapshot(request.get("loose", ()), ident_of).run()
    snap.put("@process", _process_change(before, _process_probe(), warned))
    wanted = None if request.get("all_code") else set(sources)
    code = {}
    for ident, block in code_of.items():
        if wanted is None or file_of[ident] in wanted:
            code.setdefault(file_of[ident], {})[ident] = snap.code_hash(block)
    result = {"bindings": snap.out, "code": code, "unmodeled": snap.unmodeled, "failed": failed,
              "loaded": sorted(file for file, _ in trees.values()), "modules": names}
    if request.get("extents"):
        result["extents"] = {ident: [file_of[ident], *extent] for ident, block in code_of.items()
                             if (extent := _code_extent(block)) is not None}
    del ballast
    return result


def _run_child(request, timeout):
    import json
    import marshal
    import subprocess

    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--snapshot"],
                              input=json.dumps(request).encode(), capture_output=True, timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 79 or not proc.stdout:
        return None
    return marshal.loads(proc.stdout, allow_code=False)


def _static_names(root, bindings):
    import ast
    import dis
    import types

    wanted = {path.rsplit(".", 1)[-1] for path in bindings if not path.startswith(("fn:", "@"))}
    names = {}
    for base in ("sqlglot", "tests"):
        for file in sorted((root / base).rglob("*.py")):
            rel = file.relative_to(root).as_posix()
            try:
                code = compile(ast.parse(file.read_bytes(), rel), rel, "exec", dont_inherit=True)
            except SyntaxError:
                continue
            stack = [code]
            while stack:
                block = stack.pop()
                stack.extend(k for k in block.co_consts if isinstance(k, types.CodeType))
                runtime = bool(block.co_flags & 0x1)
                for ins in dis.get_instructions(block):
                    line = ins.positions.lineno if ins.positions else None
                    if line is None:
                        continue
                    found = ()
                    if ins.opname in NAME_LOADS and isinstance(ins.argval, str):
                        found = (ins.argval,)
                    elif ins.opname == "LOAD_CONST":
                        values = ins.argval if isinstance(ins.argval, tuple) else (ins.argval,)
                        found = tuple(v for v in values if isinstance(v, str) and v.isidentifier())
                    for name in found:
                        if name in wanted:
                            entry = names.setdefault(name, {}).setdefault(rel, [[], []])
                            bucket = entry[0] if runtime else entry[1]
                            if line not in bucket:
                                bucket.append(line)
    return names


def record(root, output=MAP):
    import hashlib
    import marshal

    root = Path(root).resolve()
    names = _module_names(root)
    first = [_run_child({"root": str(root), "modules": names, "perturb": n, "cpu": 900}, 600) for n in (0, 1009, 5003, 20011, 70001, 150001)]
    if any(run is None or run["failed"] for run in first):
        raise SystemExit("pinned package did not import cleanly")
    keys = set().union(*(run["bindings"] for run in first))
    loose = sorted(k for k in keys if len({run["bindings"].get(k) for run in first}) > 1)
    second = [_run_child({"root": str(root), "modules": names, "loose": loose, "perturb": n, "all_code": True, "extents": n == 0, "cpu": 900}, 900)
              for n in (0, 2503, 40009, 100003)]
    if any(run is None or run["failed"] for run in second):
        raise SystemExit("pinned package did not import cleanly")
    unstable = sorted(k for k in keys if len({run["bindings"].get(k) for run in second}) > 1)
    base = second[0]
    if any(run["code"] != base["code"] for run in second):
        raise SystemExit("code digests are not reproducible")
    sources = hashlib.sha256()
    for file in sorted((root / "sqlglot").rglob("*.py")):
        sources.update(file.relative_to(root).as_posix().encode() + b"\0" + file.read_bytes())
    data = {"version": VERSION, "python": sys.version, "source": sources.hexdigest(), "bindings": base["bindings"],
            "code": base["code"], "extents": base["extents"], "loose": loose, "unstable": unstable,
            "loaded": base["loaded"], "modules": base["modules"], "names": _static_names(root, base["bindings"])}
    Path(output).parent.mkdir(exist_ok=True)
    Path(output).write_bytes(marshal.dumps(data))


_MILLIS = {}


def _millis(seconds):
    table = _MILLIS.get(id(seconds))
    if table is None:
        table = _MILLIS[id(seconds)] = {test: round(1000 * value) for test, value in (seconds or {}).items()}
    return table


@cache
def _baseline():
    import marshal

    data = marshal.loads(MAP.read_bytes(), allow_code=False)
    if data["version"] != VERSION or data["python"] != sys.version:
        raise ValueError("import-state map does not match this interpreter")
    return data


class PatchState:

    def __init__(self, changes, root, deadline=None, seconds=None):
        self.changes = changes
        self.root = Path(root)
        self.deadline = deadline
        self.seconds = seconds
        self._result = False
        self._touched = None

    def cost(self, tests):
        from schedule import distribute

        millis = _millis(self.seconds)
        workers = distribute([test for test in tests if test in millis], millis)
        if not workers:
            return 0.0
        return (max(sum(millis[test] for test in group) for group in workers) + 2000 * len(workers)) / 1000

    def changed(self):
        if self._result is False:
            try:
                self._result = self._compute()
            except Exception:
                self._result = None
        return self._result

    def touched(self, file):
        if self._touched is None:
            self._touched = {c.path: _touched(c.original, c.patched, c.removed, c.added)
                             for c in self.changes if not c.structural}
        return self._touched.get(file, ())

    def explained(self, key):
        extent = _baseline()["extents"].get(key[5:])
        if extent is None:
            return False
        file, low, high = extent
        return any(low <= point <= high for point in self.touched(file))

    def _compute(self):
        base = _baseline()
        sources, remap = {}, {}
        for change in self.changes:
            path = change.path
            if change.structural:
                return None
            if path.startswith("sqlglot/") and path.endswith(".py"):
                if path not in base["loaded"] or path.endswith("__main__.py"):
                    return None
                module = path[:-3].replace("/", ".").removesuffix(".__init__") + "."
                if any(name.startswith(module) for name in base["unstable"]):
                    return None
                sources[path] = change.patched
                remap[path] = _line_map(change.original, change.patched, change.removed, change.added)
        if not sources:
            return None
        if CHILD_SECONDS[0] <= 0:
            return None
        budget = 8.0
        if self.deadline is not None:
            budget = min(budget, self.deadline - time.monotonic())
        if budget < 3:
            return None
        if _has_unmapped_code(self.changes, sources, remap, base):
            return None
        start = time.monotonic()
        got = _run_child({"root": str(self.root), "sources": sources, "remap": remap, "loose": base["loose"],
                          "modules": base["modules"]}, budget)
        CHILD_SECONDS[0] -= time.monotonic() - start

        if got is None or got["failed"] or got["loaded"] != base["loaded"]:
            return None
        skip = set(base["unstable"])
        old, new = base["bindings"], got["bindings"]
        differs = [k for k in old.keys() | new.keys() if k not in skip and old.get(k) != new.get(k)]
        for file in sources:
            before, after = base["code"].get(file, {}), got["code"].get(file, {})
            differs.extend("code:" + k for k in before.keys() | after.keys() if before.get(k) != after.get(k))

        return sorted(differs)


@cache
def _collection():
    from map_store import MAPS, read_map
    data = read_map(MAPS / "collection.json.gz")
    if data["schema"] != 1:
        raise ValueError("unsupported collection map")
    return data


def collection_tests(path, lines, tests_by_file):
    table = _collection()["lines"].get(path, {})
    found = set()
    for line in lines:
        for test_file in table.get(str(line), ()):
            found |= tests_by_file.get(test_file, set())
    return found


def _lazy(path, lines, cov):
    from import_effects import _index

    readers = {read for start, end, read in _index()[2].get(path, ()) if any(start <= n <= end for n in lines)}
    return (cov.lines(path, readers) if readers else set()), readers


def _after_import(path, lines, cov, catalog, tests_by_file):
    cached, reads = _lazy(path, lines, cov)
    if set(tests_by_file) - set(_collection()["files"]):
        from import_effects import import_consumers
        return cached | import_consumers(path, lines, cov, catalog)
    return cached | collection_tests(path, set(lines) | reads, tests_by_file)


def settled_consumers(change, lines, consumers, state, tests_by_file, cov, catalog, already=()):
    try:
        if state is None or len(consumers) < 40 or state.seconds is None:
            return consumers
        kept = _after_import(change.path, lines, cov, catalog, tests_by_file) & set(consumers)
        already = set(already)
        gain = state.cost(already | set(consumers)) - state.cost(already | kept)
        if gain < MIN_GAIN:
            return consumers
        changed = state.changed()
        if changed is None:
            return consumers
        for key in changed:
            if not key.startswith("code:") or not state.explained(key):

                return consumers

        return kept
    except Exception:
        return consumers


def _function_readers(ident, cov, catalog, tests_by_file):
    extent = _baseline()["extents"].get(ident)
    if extent is None:
        return None
    file, low, high = extent
    if not cov.present(file):
        return None
    lines = set(range(low, high + 1))
    return cov.lines(file, lines) | _after_import(file, lines, cov, catalog, tests_by_file)


def binding_readers(path, cov, catalog, tests_by_file):
    base = _baseline()
    if path.startswith("@"):
        return None
    if path.startswith("code:"):
        return _function_readers(path[5:], cov, catalog, tests_by_file)
    if path.startswith("fn:"):
        ref, attr = path.rsplit(".", 1)
        if attr in INERT:
            return set()
        if attr != "__defaults__":
            return None
        return _function_readers(ref[3:], cov, catalog, tests_by_file)
    name = path.rsplit(".", 1)[-1]
    if name in INERT:
        return set()
    if name in ENUM_STATE:
        return None
    if name == "@bases" or (name.startswith("__") and name.endswith("__")):
        return None
    if name.endswith("_sql") or ".executor." in path:
        return None
    found = set()
    for file, (runtime, definition) in base["names"].get(name, {}).items():
        if file.startswith("tests/"):
            if definition or not cov.present(file):
                found |= tests_by_file.get(file, set())
        if runtime:
            found |= cov.lines(file, runtime)
    return found


def _default_body(change):
    try:
        from default_profile import _body
        return _body(change)
    except Exception:
        return None


def narrow(change, fallback, state, cov, catalog, tests_by_file):
    try:
        if state is None or len(fallback) < 40 or state.seconds is None or state.cost(fallback) < 2 * MIN_GAIN:
            return fallback
        keep = cov.fixture(change.path) | cov.missing | (catalog - set(cov.nodes))
        body = _default_body(change)
        if body is not None:
            from import_effects import import_consumers
            if not cov.present(change.path):
                return fallback
            readers = cov.lines(change.path, body) | import_consumers(change.path, body | change.removed, cov, catalog)
            return (readers | keep) & set(fallback)
        changed = state.changed()
        if changed is None:
            return fallback
        narrowed = set()
        for path in changed:
            readers = binding_readers(path, cov, catalog, tests_by_file)
            if readers is None:

                return fallback
            narrowed |= readers
        result = (narrowed | keep) & set(fallback)

        return result
    except Exception:
        return fallback


if __name__ == "__main__":
    if sys.argv[1:2] == ["--snapshot"]:
        import json
        import marshal

        request = json.loads(sys.stdin.buffer.read())
        with os.fdopen(os.dup(1), "wb") as protocol, open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            try:
                payload = _snapshot_main(request)
            except BaseException:
                raise SystemExit(1)
            protocol.write(marshal.dumps(payload))

        os._exit(79)
    if sys.argv[1:2] == ["--record"]:
        record(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else MAP)
