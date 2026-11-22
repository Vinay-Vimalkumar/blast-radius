"""What a sqlglot source edit changes in the code that runs while its module is imported.

ds_unread compares the state that importing leaves in sqlglot's own modules and classes, plus a few process
settings. It cannot see what an import does elsewhere: a reducer registered with a pickler that a spawned pool uses,
a monkeypatched standard-library function, a handler added to a logger. This module lists every statement that runs
at import (module and class bodies, decorators, defaults, base classes) that the edit adds, changes or removes, and
accepts it only when each of its effects lands in state the comparison models:
  - definitions; assignments and deletions whose targets are names, or attributes and items of sqlglot objects;
  - calls of sqlglot code other than functions the edit adds or changes, of methods of sqlglot objects and of fresh
    values, of a short list of pure builtins and standard-library functions;
  - imports of sqlglot modules and of standard-library modules without import-time effects.
Anything else is reported. new_or_changed_functions() gives the functions whose code the edit adds or changes; the
check's child keeps the heavy tests when any of them runs while the patched modules are imported.
"""
import ast
import os
import types
from functools import lru_cache
from pathlib import Path


SAFE_IMPORTS = frozenset({
    "__future__", "abc", "array", "base64", "binascii", "bisect", "builtins", "calendar", "codecs", "collections",
    "contextlib", "copy", "dataclasses", "datetime", "decimal", "difflib", "enum", "fractions", "functools",
    "hashlib", "heapq", "inspect", "io", "itertools", "json", "keyword", "locale", "logging", "math", "numbers",
    "operator", "os", "pathlib", "pprint", "random", "re", "shlex", "statistics", "string", "struct", "sys",
    "textwrap", "time", "types", "typing", "typing_extensions", "unicodedata", "uuid", "warnings", "weakref",
    "zoneinfo",
})

SAFE_CALL_MODULES = frozenset({
    "array", "base64", "binascii", "bisect", "calendar", "collections", "contextlib", "copy", "dataclasses",
    "datetime", "difflib", "enum", "fractions", "functools", "hashlib", "heapq", "itertools", "json", "keyword",
    "math", "numbers", "operator", "pprint", "re", "shlex", "statistics", "string", "struct", "textwrap", "types",
    "typing", "typing_extensions", "unicodedata", "uuid", "zoneinfo",
})
SAFE_CALLS = frozenset({
    "abc.abstractmethod", "decimal.Decimal", "logging.getLogger", "os.environ.get", "os.getenv", "os.path.abspath",
    "os.path.basename", "os.path.dirname", "os.path.join", "random.Random", "sys.intern",
    "weakref.WeakKeyDictionary", "weakref.WeakValueDictionary", "weakref.WeakSet", "weakref.ref",
})
SAFE_BUILTINS = frozenset({
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr", "classmethod", "complex",
    "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset", "getattr", "hasattr", "hash", "hex",
    "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next", "object", "oct", "ord",
    "pow", "property", "range", "repr", "reversed", "round", "set", "slice", "sorted", "staticmethod", "str", "sum",
    "super", "tuple", "type", "zip",
})


SAFE_BUILTIN_ATTRS = frozenset({"bytes.fromhex", "dict.fromkeys", "float.fromhex", "int.from_bytes", "str.maketrans"})

SAFE_NAMES = frozenset({"__annotations__", "__doc__", "__file__", "__module__", "__name__", "__package__",
                        "__qualname__", "__spec__", "Ellipsis", "NotImplemented"})

SAFE_REFERENCES = frozenset({
    "os.curdir", "os.linesep", "os.name", "os.pardir", "os.pathsep", "os.sep", "sys.byteorder", "sys.float_info",
    "sys.hexversion", "sys.implementation", "sys.maxsize", "sys.platform", "sys.version_info",
})


SAFE_DUNDERS = frozenset({"__doc__", "__members__", "__module__", "__name__", "__qualname__"})
_BAD = {"ext": 3, "changed": 2, "builtin": 1, "sqlglot": 1, "literal": 0}


def _builtin_exception(name):
    import builtins

    item = getattr(builtins, name, None)
    return isinstance(item, type) and issubclass(item, BaseException)


def _module_name(path):
    name = path[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def _type_checking(test):
    return ast.unparse(test) in ("TYPE_CHECKING", "t.TYPE_CHECKING", "typing.TYPE_CHECKING")


@lru_cache(maxsize=16)
def code_of(path, text):
    """Module code of a source text, compiled once: the selector reuses unpatched originals across patches."""
    return compile(text, path, "exec", dont_inherit=True)


@lru_cache(maxsize=4)
def _tree_of(path, text):
    return ast.parse(text, path)


def _import_time(code):
    """What a module or class body runs at import, positions ignored: functions stand in by qualname (their bodies
    run when called), class bodies are compared in full (they run at import)."""
    return code.replace(co_firstlineno=1, co_linetable=b"", co_consts=tuple(
        ("function", item.co_qualname) if isinstance(item, types.CodeType) and item.co_flags & 0x1
        else _import_time(item) if isinstance(item, types.CodeType) else item
        for item in code.co_consts))


def new_or_changed_functions(path, original, patched):
    """Qualnames of functions (lambdas and nested ones included) whose code the edit adds or changes."""
    from semantics import _without_positions

    def walk(code, out):
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                if const.co_flags & 0x1:
                    out.setdefault(const.co_qualname, []).append(const)
                walk(const, out)
        return out

    old = walk(code_of(path, original), {})
    new = walk(code_of(path, patched), {})
    changed = set()
    for qualname, codes in new.items():
        others = old.get(qualname, [])
        if not (len(codes) == len(others)
                and all(_without_positions(a) == _without_positions(b) for a, b in zip(codes, others))):
            changed.add(qualname)
    return changed


class _Scope:
    """Where names bound in a module or class body come from: sqlglot, a changed function, elsewhere, a literal."""

    def __init__(self, parent=None):
        self.parent, self.names, self.stars = parent, {}, []

    def bind(self, name, origin):
        old = self.names.get(name)
        if old is None or _BAD[origin.split(":", 1)[0]] > _BAD[old.split(":", 1)[0]]:
            self.names[name] = origin

    def lookup(self, name):
        import builtins

        scope = self
        while scope is not None:
            if name in scope.names:
                return scope.names[name]
            if scope.stars and not hasattr(builtins, name):
                kind, _, module = scope.stars[0].partition(":")
                return f"{kind}:{module}.{name}"
            scope = scope.parent
        return f"builtin:{name}"


_SCOPES = {}


def _module_scope(path, text):
    """Top-level bindings of a sqlglot module (cached)."""
    key = (path, hash(text))
    if key not in _SCOPES:
        try:
            _SCOPES[key] = _Module(path, ast.parse(text), set()).scopes[""]
        except (SyntaxError, ValueError, RecursionError):
            _SCOPES[key] = None
    return _SCOPES[key]


class _Module:
    def __init__(self, path, tree, changed, sources=None):
        self.path, self.tree, self.changed = path, tree, changed
        self.sources = sources or {}
        self.module = _module_name(path)
        self.package = self.module if path.endswith("/__init__.py") else self.module.rpartition(".")[0]
        self.future = any(isinstance(s, ast.ImportFrom) and s.module == "__future__"
                          and any(a.name == "annotations" for a in s.names) for s in tree.body)
        self.units = []
        self.scopes = {}
        self._visit(tree.body, "", _Scope())



    def _imported(self, node, alias):
        if isinstance(node, ast.Import):
            dotted = alias.name if alias.asname else alias.name.split(".")[0]
        else:
            base = node.module or ""
            if node.level:
                parts = self.package.split(".")
                parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                base = ".".join(p for p in [*parts, base] if p)
            dotted = f"{base}.{alias.name}" if base else alias.name
        return ("sqlglot:" if dotted.split(".")[0] == "sqlglot" else "ext:") + dotted

    def origin(self, expr, scope):
        if isinstance(expr, ast.Name):
            return scope.lookup(expr.id)
        if isinstance(expr, ast.Attribute):
            base = self.origin(expr.value, scope)
            kind, _, dotted = base.partition(":")
            if kind == "literal":
                return base
            if kind == "changed":
                return f"sqlglot:{dotted}.{expr.attr}"
            return f"{kind}:{dotted}.{expr.attr}"
        if isinstance(expr, ast.Subscript):
            base = self.origin(expr.value, scope)
            return base if base == "literal" else base + "[]"
        if isinstance(expr, ast.Call):
            base = self.origin(expr.func, scope)
            kind, _, dotted = base.partition(":")
            if kind == "literal" or kind == "builtin" and dotted.split(".")[0] in SAFE_BUILTINS:
                return "literal"
            if kind == "changed":
                return f"sqlglot:{dotted}()"
            return f"{kind}:{dotted}()"
        if isinstance(expr, (ast.Starred, ast.NamedExpr)):
            return self.origin(expr.value, scope)
        return "literal"

    def _module_text(self, dotted):
        root = Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot"))
        for path in (dotted.replace(".", "/") + ".py", dotted.replace(".", "/") + "/__init__.py"):
            if path in self.sources:
                return path, self.sources[path]
            try:
                return path, (root / path).read_text(encoding="utf-8")
            except OSError:
                continue
        return None, None

    def resolve(self, origin, depth=0):
        """Follow a sqlglot origin through module attributes that other sqlglot modules import from elsewhere."""
        kind, _, dotted = origin.partition(":")
        if kind != "sqlglot" or depth > 5:
            return origin
        parts = dotted.split(".")
        for i in range(len(parts) - 1, 0, -1):
            path, text = self._module_text(".".join(parts[:i]))
            if text is None:
                continue
            name = parts[i].split("(", 1)[0].split("[", 1)[0]
            scope = _module_scope(path, text)
            if scope is None or name not in scope.names:
                return origin
            target = scope.names[name]
            if target == "literal":
                return "literal"
            if target.startswith("changed:"):
                return origin
            rest = parts[i][len(name):] + "".join("." + p for p in parts[i + 1:])
            resolved = target + rest
            if resolved == origin:
                return origin
            return resolved if resolved.startswith("ext:") else self.resolve(resolved, depth + 1)
        return origin

    def _bind_target(self, target, origin, scope):
        if isinstance(target, ast.Name):
            scope.bind(target.id, origin)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._bind_target(item, origin, scope)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value, origin, scope)

    def _visit(self, body, qual, scope):
        self.scopes[qual] = scope
        for s in body:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{qual}.{s.name}" if qual else s.name
                scope.bind(s.name, f"changed:{name}" if name in self.changed else f"sqlglot:{self.module}.{name}")
                self.units.append(("def", qual, s))
            elif isinstance(s, ast.ClassDef):
                name = f"{qual}.{s.name}" if qual else s.name
                scope.bind(s.name, f"sqlglot:{self.module}.{name}")
                self.units.append(("class", qual, s))
                self._visit(s.body, name, _Scope(self.scopes[""]))
            elif isinstance(s, ast.If) and _type_checking(s.test):
                self._visit(s.orelse, qual, scope)
            elif isinstance(s, (ast.If, ast.While)):
                self.units.append(("test", qual, s.test))
                self._visit(s.body, qual, scope)
                self._visit(s.orelse, qual, scope)
            elif isinstance(s, (ast.For, ast.AsyncFor)):
                self._bind_target(s.target, self.origin(s.iter, scope) + "[]", scope)
                self.units.append(("for", qual, s))
                self._visit(s.body, qual, scope)
                self._visit(s.orelse, qual, scope)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                for item in s.items:
                    if item.optional_vars is not None:
                        self._bind_target(item.optional_vars, self.origin(item.context_expr, scope), scope)
                self.units.append(("with", qual, s))
                self._visit(s.body, qual, scope)
            elif isinstance(s, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                self._visit(s.body, qual, scope)
                for handler in s.handlers:
                    if handler.name:
                        scope.bind(handler.name, "literal")
                    self.units.append(("except", qual, handler))
                    self._visit(handler.body, qual, scope)
                self._visit(s.orelse, qual, scope)
                self._visit(s.finalbody, qual, scope)
            else:
                if isinstance(s, (ast.Import, ast.ImportFrom)):
                    for alias in s.names:
                        if alias.name != "*":
                            scope.bind(alias.asname or alias.name.split(".")[0], self._imported(s, alias))
                        else:
                            origin = self._imported(s, alias)
                            scope.stars.append(origin[: -len(".*")] if origin.endswith(".*") else origin)
                elif isinstance(s, ast.Assign):
                    for target in s.targets:
                        self._bind_target(target, self.origin(s.value, scope), scope)
                elif isinstance(s, ast.AnnAssign) and s.value is not None:
                    self._bind_target(s.target, self.origin(s.value, scope), scope)
                self.units.append(("stmt", qual, s))



    def key(self, unit):
        """What the unit runs at import, positions ignored; function and class bodies are separate units."""
        kind, qual, node = unit
        if kind == "def":
            parts = [node.name, node.decorator_list, node.args.defaults,
                     [d for d in node.args.kw_defaults if d is not None]]
            if not self.future:
                parts += [node.returns, [a.annotation for a in (*node.args.posonlyargs, *node.args.args,
                                                                 *node.args.kwonlyargs)]]
        elif kind == "class":
            parts = [node.name, node.decorator_list, node.bases, node.keywords]
        elif kind == "for":
            parts = [node.target, node.iter]
        elif kind == "with":
            parts = [node.items]
        elif kind == "except":
            parts = [node.type, node.name]
        else:
            parts = [node]

        from semantics import _dump_key

        def dump(value):
            if isinstance(value, list):
                return [dump(v) for v in value]
            if isinstance(value, ast.AST):
                return _dump_key(value)
            return repr(value)

        return repr((kind, qual, dump(parts)))

    def check(self, unit):
        kind, qual, node = unit
        scope = self.scopes[qual]
        if kind == "def":
            exprs = [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]
            if not self.future:
                exprs += [a.annotation for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                          if a.annotation is not None]
                exprs += [node.returns] if node.returns is not None else []
            return self.check_decorators(node, scope) + [r for e in exprs for r in self.check_expr(e, scope)]
        if kind == "class":
            return self.check_decorators(node, scope) + [
                r for e in (*node.bases, *(k.value for k in node.keywords)) for r in self.check_expr(e, scope)]
        if kind == "test":
            return self.check_expr(node, scope)
        if kind == "for":
            return self.check_target(node.target, scope) + self.check_expr(node.iter, scope)
        if kind == "with":
            return [r for item in node.items for r in self.check_expr(item.context_expr, scope)
                    + (self.check_target(item.optional_vars, scope) if item.optional_vars is not None else [])]
        if kind == "except":
            return self.check_expr(node.type, scope) if node.type is not None else []
        return self.check_stmt(node, scope)

    def check_stmt(self, s, scope):
        if isinstance(s, ast.Expr):
            if isinstance(s.value, ast.Constant):
                return []
            return self.check_expr(s.value, scope)
        if isinstance(s, ast.Assign):
            return [r for t in s.targets for r in self.check_target(t, scope)] + self.check_expr(s.value, scope)
        if isinstance(s, ast.AnnAssign):
            reasons = self.check_target(s.target, scope)
            if s.value is not None:
                reasons += self.check_expr(s.value, scope)
            if not self.future and isinstance(s.target, ast.Name):
                reasons += self.check_expr(s.annotation, scope)
            return reasons
        if isinstance(s, ast.AugAssign):
            return self.check_target(s.target, scope) + self.check_expr(s.value, scope)
        if isinstance(s, ast.Delete):
            return [r for t in s.targets for r in self.check_target(t, scope)]
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            reasons = []
            for alias in s.names:
                origin = self._imported(s, alias)
                kind, _, dotted = origin.partition(":")
                top = dotted.split(".")[0]
                if kind != "sqlglot" and top not in SAFE_IMPORTS:
                    reasons.append(f"imports {dotted}")
            return reasons
        if isinstance(s, (ast.Pass, ast.Global, ast.Nonlocal, ast.Break, ast.Continue)):
            return []
        if isinstance(s, ast.Raise):
            return [r for e in (s.exc, s.cause) if e is not None for r in self.check_expr(e, scope)]
        if isinstance(s, ast.Assert):
            return [r for e in (s.test, s.msg) if e is not None for r in self.check_expr(e, scope)]
        return [f"unsupported import-time statement {type(s).__name__}"]

    def check_target(self, target, scope):
        if isinstance(target, ast.Name):
            return []
        if isinstance(target, (ast.Tuple, ast.List)):
            return [r for item in target.elts for r in self.check_target(item, scope)]
        if isinstance(target, ast.Starred):
            return self.check_target(target.value, scope)
        if isinstance(target, (ast.Attribute, ast.Subscript)):
            reasons = self.check_expr(target.value, scope)
            if isinstance(target, ast.Subscript):
                reasons += self.check_expr(target.slice, scope)
            origin = self.resolve(self.origin(target.value, scope))
            if origin.split(":", 1)[0] in ("ext", "builtin"):
                reasons.append(f"assigns into {origin.partition(':')[2]}")
            return reasons
        return [f"unsupported target {type(target).__name__}"]

    def check_decorators(self, node, scope):
        """A decorator is called with the definition at import, whether or not it is spelled as a call."""
        reasons = []
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                reasons += self.check_call(ast.Call(func=decorator, args=[], keywords=[]), scope)
            reasons += self.check_expr(decorator, scope)
        return reasons

    def check_expr(self, expr, scope):
        """Calls, and every name or attribute chain the expression reads: changed import-time code may only reach
        sqlglot objects, fresh values, listed builtins and listed library modules, so it cannot hand an outside
        object to a call that looks harmless (functools.partial, map, a builtin type's method)."""
        reasons = []
        local = {n.id for c in ast.walk(expr) if isinstance(c, (ast.comprehension, ast.NamedExpr))
                 for n in ast.walk(c.target) if isinstance(n, ast.Name)}
        stack = [(expr, None)]
        while stack:
            node, parent = stack.pop()
            if isinstance(node, (ast.Await, ast.Yield, ast.YieldFrom)):
                reasons.append(f"unsupported expression {type(node).__name__}")
            if isinstance(node, ast.Call):
                reasons += self.check_call(node, scope)
            if (isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__")
                    and node.attr not in SAFE_DUNDERS):
                reasons.append(f"uses {node.attr}")
            if (isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)) and isinstance(node.ctx, ast.Load)
                    and not (isinstance(parent, (ast.Attribute, ast.Subscript)) and parent.value is node)
                    and not (isinstance(parent, ast.Call) and parent.func is node)
                    and not self._root_name(node) in local):
                reasons += self.check_reference(node, scope)
            for child in ast.iter_child_nodes(node):
                if isinstance(node, ast.Lambda) and child is node.body:
                    continue
                stack.append((child, node))
        return reasons

    @staticmethod
    def _root_name(node):
        while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
            node = node.func if isinstance(node, ast.Call) else node.value
        return node.id if isinstance(node, ast.Name) else None

    def reference_ok(self, origin):
        kind, _, dotted = origin.partition(":")
        if kind in ("literal", "sqlglot", "changed"):
            return True
        if kind == "builtin":
            bare = dotted
            while bare.endswith("[]"):
                bare = bare[:-2]
            if not any(c in bare for c in ".[("):
                return bare in SAFE_BUILTINS or bare in SAFE_NAMES or _builtin_exception(bare)
            return dotted in SAFE_BUILTIN_ATTRS
        top = dotted.split(".")[0].split("[")[0].split("(")[0]
        constant = any(dotted == ref or dotted.startswith((ref + "[", ref + ".")) for ref in SAFE_REFERENCES)
        return (top in SAFE_CALL_MODULES or dotted in SAFE_CALLS or constant
                or (top == "logging" and dotted.rsplit(".", 1)[-1].isupper()))

    def check_reference(self, node, scope):
        origin = self.resolve(self.origin(node, scope))
        return [] if self.reference_ok(origin) else [f"references {origin.partition(':')[2]}"]

    def check_call(self, call, scope):
        origin = self.resolve(self.origin(call.func, scope))
        kind, _, dotted = origin.partition(":")
        if kind in ("literal", "sqlglot"):
            return []
        if kind == "changed":
            return [f"calls new or changed function {dotted} at import"]
        if kind == "builtin":
            if any(c in dotted for c in ".[("):
                return [] if dotted in SAFE_BUILTIN_ATTRS else [f"calls builtin {dotted}"]
            return [] if dotted in SAFE_BUILTINS or _builtin_exception(dotted) else [f"calls builtin {dotted}"]
        top = dotted.split(".")[0].split("[")[0].split("(")[0]
        if top in SAFE_CALL_MODULES or dotted in SAFE_CALLS:
            return []
        return [f"calls {dotted}"]


def import_effects(path, original, patched, sources=None, changed=None):
    """Reasons the edit may change what importing the module does outside modeled state; empty when none.

    sources maps the patch's edited paths to their patched text, for resolving names imported from them; changed
    is new_or_changed_functions() of this edit when the caller already has it.
    """
    if _import_time(code_of(path, original)) == _import_time(code_of(path, patched)):
        return []
    if changed is None:
        changed = new_or_changed_functions(path, original, patched)
    old = _Module(path, _tree_of(path, original), set())
    new = _Module(path, _tree_of(path, patched), changed, sources)
    before = {}
    for unit in old.units:
        before.setdefault(old.key(unit), []).append(unit)
    after = {}
    for unit in new.units:
        after.setdefault(new.key(unit), []).append(unit)
    reasons = []
    for key, units in after.items():
        for unit in units[len(before.get(key, ())):]:
            reasons += new.check(unit)
    for key, units in before.items():
        for unit in units[len(after.get(key, ())):]:
            reasons += [f"removed: {r}" for r in old.check(unit)]
    return sorted(set(reasons))
