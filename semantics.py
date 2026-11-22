import ast
import difflib
import marshal
import os
import time
from functools import lru_cache
from importlib.util import MAGIC_NUMBER, cache_from_source
from types import CodeType

try:
    import _ast_native
except ImportError:
    _ast_native = None


def _dump_key(node):
    if _ast_native is not None:
        try:
            return _ast_native.dump(node, ast.AST, "_fields")
        except (AttributeError, TypeError, ValueError, RecursionError):
            pass
    return ast.dump(node)


_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_DEFINITION_SCOPES = (*_DEFINITIONS, ast.If, ast.For, ast.AsyncFor, ast.While, ast.With,
                      ast.AsyncWith, ast.Try, ast.TryStar, ast.ExceptHandler, ast.Match, ast.match_case)


def _without_positions(code):
    return code.replace(
        co_firstlineno=1,
        co_linetable=b"",
        co_consts=tuple(
            _without_positions(value) if isinstance(value, CodeType) else value
            for value in code.co_consts
        ),
    )


def _pinned_code(path, data):
    """Bytecode the image compiled from exactly these source bytes, or None.

    compileall used the same flags as compile(..., "exec", dont_inherit=True); a timestamp .pyc whose
    recorded mtime and size match the unchanged file, holding these exact bytes, is that compilation.
    """
    try:
        file = os.path.join(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot"), *path.split("/"))
        with open(file, "rb") as stream:
            if stream.read() != data:
                return None
            stat = os.fstat(stream.fileno())
        with open(cache_from_source(file), "rb") as stream:
            pyc = stream.read()
        if (pyc[:4] != MAGIC_NUMBER or pyc[4:8] != b"\0\0\0\0"
                or int.from_bytes(pyc[8:12], "little") != int(stat.st_mtime) & 0xFFFFFFFF
                or int.from_bytes(pyc[12:16], "little") != stat.st_size & 0xFFFFFFFF):
            return None
        code = marshal.loads(pyc[16:])
    except (OSError, ValueError, EOFError, TypeError, NotImplementedError):
        return None
    return code if type(code) is CodeType else None


@lru_cache(maxsize=6)
def _baseline(path, source):
    data = source.encode("utf-8")
    tree = ast.parse(data, path)
    code = _pinned_code(path, data)
    if code is None:
        code = compile(tree, path, "exec", dont_inherit=True)
    lines = set()

    def collect(block):
        lines.update(n for start, end, n in block.co_lines() if end > start and n)
        for value in block.co_consts:
            if isinstance(value, CodeType):
                collect(value)

    collect(code)
    return tree, _without_positions(code), lines


@lru_cache(maxsize=4)
def _patched(path, source):
    tree = ast.parse(source.encode("utf-8"), path)

    try:
        code = compile(tree, path, "exec", dont_inherit=True)
    except Exception as exc:

        code = exc.with_traceback(None)
    return tree, code


def patched_tree(path, source):
    return _patched(path, source)[0]


_MUTATING_CALLS = frozenset({
    "setattr", "delattr", "exec", "eval", "__setattr__", "__delattr__", "__setitem__",
    "__delitem__", "setdefault", "update", "add", "append", "extend", "insert", "remove",
    "clear", "discard", "pop", "popitem", "register", "unregister", "reverse", "sort",
})

_REFLECTIVE_WRITES = {"setattr", "delattr", "__setattr__", "__delattr__", "__setitem__",
                      "__delitem__", "exec", "eval"}


def _shared_receiver(node, names):
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if isinstance(node, ast.Attribute) and (node.attr.isupper() or node.attr in {"__dict__", "arg_types"}):
            return True
        node = node.value
    return isinstance(node, ast.Name) and node.id in names


def _may_write_shared_state(scope, shared, writers, immutable=None, *, base_renderer=False):
    nodes = list(ast.walk(scope))
    aliases = set()
    copies = {}
    for node in nodes:
        if isinstance(node, ast.Global):
            return True
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if name in writers:
                return True
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if _shared_receiver(node.value, shared):
                if immutable is None or not immutable(node.value):
                    aliases.update(t.id for t in targets if isinstance(t, ast.Name))
            elif base_renderer and isinstance(node.value, ast.Name):
                copies.setdefault(node.value.id, set()).update(t.id for t in targets if isinstance(t, ast.Name))


    if copies and aliases:
        pending = list(aliases)
        while pending:
            source = pending.pop()
            for target in copies.get(source, ()):
                if target not in aliases:
                    aliases.add(target)
                    pending.append(target)
    roots = shared | aliases
    for node in nodes:
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if _shared_receiver(node.value, roots):
                return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATING_CALLS:
                if _shared_receiver(node.func.value, roots):
                    return True
            if not base_renderer and any(isinstance(arg, ast.Name) and arg.id in aliases for arg in node.args):
                return True
    return False


def _changed_shared_scopes(tree, lines, universe=None, path=None):
    shared = {alias.asname or alias.name.split(".")[0]
              for stmt in tree.body if isinstance(stmt, (ast.Import, ast.ImportFrom)) for alias in stmt.names}
    shared.update(stmt.name for stmt in tree.body if isinstance(stmt, ast.ClassDef))

    def visit(scope, owner, writers):
        writers = set(writers)
        for stmt in scope.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, (ast.Name, ast.Attribute)):
                name = stmt.value.id if isinstance(stmt.value, ast.Name) else stmt.value.attr
                if name in writers:
                    writers.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
        for stmt in scope.body:
            first = min((d.lineno for d in getattr(stmt, "decorator_list", ())), default=stmt.lineno)
            if lines and not any(first <= n <= stmt.end_lineno for n in lines):
                continue
            if isinstance(stmt, ast.ClassDef):
                yield from visit(stmt, (*owner, stmt.name), writers)
            else:
                immutable = None
                if universe is not None and len(owner) == 1 and isinstance(stmt, ast.FunctionDef):
                    immutable = _ConstantRead(universe, path, owner[0], stmt)
                if _may_write_shared_state(stmt, shared, writers, immutable, base_renderer=path == "sqlglot/generator.py"):
                    yield owner, _dump_key(stmt)

    return tuple(visit(tree, (), _REFLECTIVE_WRITES))






















_IMMUTABLE_CONSTANTS = (str, bytes, int, float, complex, bool, type(None))
_LOOKUP_HOOKS = frozenset({"__getattr__", "__getattribute__", "__setattr__", "__delattr__", "__set_name__",
                           "__init_subclass__"})
_DYNAMIC_CALLS = frozenset({"setattr", "delattr", "vars", "__setattr__", "__delattr__", "new_class"})
_UNIVERSE_DIRS = ("sqlglot", "tests")
_PROOF_STATS = {}


def _proof_note(reason):
    _PROOF_STATS[reason] = _PROOF_STATS.get(reason, 0) + 1


class _Unproven(Exception):
    pass


def _immutable_literal(node):
    if isinstance(node, ast.Constant):
        return type(node.value) in _IMMUTABLE_CONSTANTS
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float, complex)
    if isinstance(node, ast.Tuple):
        return all(_immutable_literal(item) for item in node.elts)
    return False


_PINNED_TEXT = {}


def _class_topology(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield _dump_key(node)
        elif isinstance(node, ast.ClassDef):
            yield node.name, tuple(_dump_key(base) for base in (*node.bases, *node.keywords, *node.decorator_list))
_PINNED_TREE = {}


class _Universe:
    """sqlglot/ and tests/ of the pinned checkout, with the patch's own text in place of every patched file."""

    def __init__(self, changes, root=None):
        self.root = root or os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot")
        self.complete = not any(c.structural or not c.original or not c.patched for c in changes)
        self.deadline = time.monotonic() + 2.5
        if self.complete:
            self.complete = all(tuple(_class_topology(_baseline(c.path, c.original)[0]))
                                == tuple(_class_topology(patched_tree(c.path, c.patched)))
                                for c in changes if c.path.endswith(".py"))
        self.overrides = {c.path: c.patched for c in changes if c.path.endswith(".py")}
        self.trees = {}
        self.files_read = 0
        self.files_parsed = 0

    def paths(self):
        key = ("paths", self.root)
        if key not in _PINNED_TEXT:
            found = []
            for top in _UNIVERSE_DIRS:
                for folder, dirs, files in os.walk(os.path.join(self.root, top)):
                    dirs[:] = sorted(d for d in dirs if d != "__pycache__")
                    rel = os.path.relpath(folder, self.root).replace(os.sep, "/")
                    found.extend(f"{rel}/{name}" for name in sorted(files) if name.endswith(".py"))
            _PINNED_TEXT[key] = tuple(found)
        return sorted(set(_PINNED_TEXT[key]) | {p for p in self.overrides if p.startswith(_UNIVERSE_DIRS)})

    def text(self, path):
        if path in self.overrides:
            return self.overrides[path]
        key = (self.root, path)
        if key not in _PINNED_TEXT:
            try:
                with open(os.path.join(self.root, *path.split("/")), "rb") as stream:
                    _PINNED_TEXT[key] = stream.read().decode("utf-8")
            except (OSError, UnicodeDecodeError):
                _PINNED_TEXT[key] = None
            self.files_read += 1
        return _PINNED_TEXT[key]

    def tree(self, path):
        if path not in self.trees:
            text = self.text(path)
            if text is None:
                raise _Unproven("unreadable module")
            if path in self.overrides:
                tree = patched_tree(path, text)
            else:
                key = (self.root, path)
                if key not in _PINNED_TREE:
                    _PINNED_TREE[key] = ast.parse(text.encode("utf-8"), path)
                    self.files_parsed += 1
                tree = _PINNED_TREE[key]
            self.trees[path] = tree
        return self.trees[path]

    def module_path(self, module):
        base = module.replace(".", "/")
        for candidate in (f"{base}.py", f"{base}/__init__.py"):
            if candidate in self.overrides or os.path.isfile(os.path.join(self.root, *candidate.split("/"))):
                return candidate
        return None

    def mentioning(self, word):
        return [p for p in self.paths() if word in (self.text(p) or "")]


def _package(path):
    parts = path[:-3].split("/")
    return parts[:-1] if parts[-1] == "__init__" else parts[:-1]


class _ConstantRead:
    """Called with the value of `local = <value>`; True only when <value> is a proven immutable class constant."""

    def __init__(self, universe, path, owner, method):
        self.universe, self.path, self.owner, self.method = universe, path, owner, method

    def __call__(self, value):
        if not (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.attr.isupper()):
            return False
        _proof_note("attempt")
        try:
            self._prove(value.value.id, value.attr)
        except _Unproven as exc:
            _proof_note(f"fallback: {exc}")
            return False
        except (SyntaxError, ValueError, RecursionError, OverflowError):
            _proof_note("fallback: unparsable module")
            return False
        _proof_note("proven")
        return True

    def _prove(self, receiver, attr):
        universe, method = self.universe, self.method
        if not universe.complete:
            raise _Unproven("patch adds, deletes or renames a file")
        args = [*method.args.posonlyargs, *method.args.args]
        decorators = [d.id if isinstance(d, ast.Name) else None for d in method.decorator_list]
        if not args or args[0].arg != receiver or any(d != "classmethod" for d in decorators):
            raise _Unproven("receiver is not the method's own self or cls")
        for node in ast.walk(method):
            if isinstance(node, ast.Name) and node.id == receiver and isinstance(node.ctx, (ast.Store, ast.Del)):
                raise _Unproven("receiver rebound")
            if isinstance(node, ast.arg) and node.arg == receiver and node is not args[0]:
                raise _Unproven("receiver shadowed")
        start = (self.path, self.owner)
        self._class(start)
        reachable = self._reachable(start)
        bodies = set()
        for key in reachable:
            mro = self._mro(key)
            bodies.update(mro)
            binding = None
            for cls in mro:
                node = self._class(cls)
                for stmt in node.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name in _LOOKUP_HOOKS:
                        raise _Unproven(f"{stmt.name} on {cls[1]}")
                    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                        if any(isinstance(t, ast.Name) and t.id in _LOOKUP_HOOKS for t in targets):
                            raise _Unproven(f"lookup hook bound on {cls[1]}")
                        if any(isinstance(t, ast.Name) and t.id == "__slots__" for t in targets):
                            slots = stmt.value
                            if not (isinstance(slots, (ast.Tuple, ast.List)) and _immutable_literal(ast.Tuple(slots.elts))):
                                raise _Unproven(f"__slots__ of {cls[1]} not literal")
                            if any(isinstance(e, ast.Constant) and e.value == attr for e in slots.elts):
                                raise _Unproven(f"__slots__ of {cls[1]} names {attr}")
                if binding is None:
                    found = self._binding(node, attr)
                    if found is not None:
                        binding = (cls, found)
            if binding is None:
                raise _Unproven(f"{key[1]} inherits no class-body binding of {attr}")
            if not _immutable_literal(binding[1]):
                raise _Unproven(f"{key[1]} gets {attr} from {binding[0][1]} as a non-literal value")
        self._no_outside_writes(attr, {path for path, _ in bodies})

    def _binding(self, node, attr):
        """The one plain assignment of `attr` in this class body, None if there is none."""
        found = []
        todo = list(node.body)
        while todo:
            stmt = todo.pop()
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if stmt.name == attr:
                    raise _Unproven(f"{attr} defined as a function or class")
                continue
            for child in ast.walk(stmt):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    continue
                if isinstance(child, ast.Name) and child.id == attr and isinstance(child.ctx, (ast.Store, ast.Del)):
                    found.append(stmt)
                if isinstance(child, ast.alias) and (child.asname or child.name) == attr:
                    raise _Unproven(f"{attr} bound by an import in a class body")
        if not found:
            return None
        if len(found) > 1:
            raise _Unproven(f"{attr} bound more than once in one class body")
        stmt = found[0]
        if stmt not in node.body:
            raise _Unproven(f"{attr} bound inside a nested statement")
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            return stmt.value
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            return stmt.value
        raise _Unproven(f"{attr} bound by {type(stmt).__name__}")

    def _class(self, key):
        path, name = key
        matches = [s for s in self.universe.tree(path).body if isinstance(s, ast.ClassDef) and s.name == name]
        others = [s for s in self.universe.tree(path).body if not isinstance(s, ast.ClassDef)
                  and any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)
                          for n in ast.walk(s))]
        if len(matches) != 1 or others:
            raise _Unproven(f"{name} is not a single module-level class in {path}")
        node = matches[0]
        if node.decorator_list or node.keywords:
            raise _Unproven(f"{name} has a decorator or class keyword")
        return node

    def _resolve(self, path, expr):
        """The (path, class) a base expression names, or None for `object`."""
        tree = self.universe.tree(path)
        if isinstance(expr, ast.Name):
            if expr.id == "object":
                return None
            if any(isinstance(s, ast.ClassDef) and s.name == expr.id for s in tree.body):
                return path, expr.id
            for stmt in tree.body:
                if isinstance(stmt, ast.ImportFrom):
                    for alias in stmt.names:
                        if (alias.asname or alias.name) == expr.id:
                            return self._imported(path, stmt, alias.name)
            raise _Unproven(f"base {expr.id} of a class in {path} not resolved")
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
            for stmt in tree.body:
                if isinstance(stmt, ast.Import):
                    for alias in stmt.names:
                        if alias.asname == expr.value.id or (alias.asname is None and alias.name == expr.value.id):
                            target = self.universe.module_path(alias.name)
                            if target is not None:
                                return self._defined(target, expr.attr)
                if isinstance(stmt, ast.ImportFrom) and not stmt.level and stmt.module:
                    for alias in stmt.names:
                        if (alias.asname or alias.name) == expr.value.id:
                            target = self.universe.module_path(f"{stmt.module}.{alias.name}")
                            if target is not None:
                                return self._defined(target, expr.attr)
        raise _Unproven(f"base {ast.unparse(expr)} of a class in {path} not resolved")

    def _imported(self, path, stmt, name, depth=0):
        if depth > 8:
            raise _Unproven("import chain too long")
        if stmt.level:
            package = _package(path)[: len(_package(path)) - stmt.level + 1]
            module = ".".join([*package, *(stmt.module.split(".") if stmt.module else [])])
        else:
            module = stmt.module or ""
        target = self.universe.module_path(module)
        if target is None:
            raise _Unproven(f"module {module} outside the universe")
        tree = self.universe.tree(target)
        if any(isinstance(s, ast.ClassDef) and s.name == name for s in tree.body):
            return self._defined(target, name)
        for inner in tree.body:
            if isinstance(inner, ast.ImportFrom):
                for alias in inner.names:
                    if (alias.asname or alias.name) == name:
                        return self._imported(target, inner, alias.name, depth + 1)
        raise _Unproven(f"{name} not found in {module}")

    def _defined(self, path, name):
        self._class((path, name))
        return path, name

    def _bases(self, key):
        return [b for b in (self._resolve(key[0], e) for e in self._class(key).bases) if b is not None]

    def _mro(self, key, stack=()):
        if key in stack:
            raise _Unproven("class hierarchy cycle")
        sequences = [self._mro(base, (*stack, key)) for base in self._bases(key)]
        sequences.append(self._bases(key))
        result = [key]
        sequences = [list(s) for s in sequences if s]
        while sequences:
            for sequence in sequences:
                head = sequence[0]
                if not any(head in other[1:] for other in sequences):
                    break
            else:
                raise _Unproven("no consistent MRO")
            result.append(head)
            sequences = [[c for c in s if c != head] for s in sequences]
            sequences = [s for s in sequences if s]
        return result

    def _reachable(self, start):
        reachable = {start}
        frontier = [start]
        while frontier:
            key = frontier.pop()
            for path in self.universe.mentioning(key[1]):
                if time.monotonic() > self.universe.deadline:
                    raise _Unproven("class scan budget reached")
                tree = self.universe.tree(path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.alias) and node.name == key[1] and node.asname:
                        raise _Unproven(f"aliased class import in {path}")
                    if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
                        callee = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                        if callee == "new_class" or (callee == "type" and len(node.args) == 3):
                            raise _Unproven(f"class built dynamically in {path}")
                    if not isinstance(node, ast.ClassDef) or (path, node.name) in reachable:
                        continue
                    texts = [ast.unparse(b) for b in node.bases]
                    if not any(key[1] in text for text in texts):
                        continue
                    if node not in tree.body:
                        raise _Unproven(f"subclass {node.name} of {key[1]} is not module-level in {path}")
                    if any(isinstance(b, ast.Starred) for b in node.bases):
                        raise _Unproven(f"starred bases in {path}")
                    if key in self._bases((path, node.name)):
                        reachable.add((path, node.name))
                        frontier.append((path, node.name))
        return reachable

    def _no_outside_writes(self, attr, defining):
        universe = self.universe
        for path in sorted(set(universe.mentioning(attr)) | defining | set(universe.overrides)):
            if time.monotonic() > universe.deadline:
                raise _Unproven("binding scan budget reached")
            if not path.startswith(_UNIVERSE_DIRS):
                continue
            tree = universe.tree(path)
            strict = path in defining or path in universe.overrides
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for stmt in node.body:
                        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == attr:
                            raise _Unproven(f"{attr} has a method or descriptor binding in {path}")
                        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                            if any(isinstance(t, ast.Name) and t.id == attr for t in targets) and not _immutable_literal(stmt.value):
                                raise _Unproven(f"{attr} has a nonliteral class binding in {path}")
                if isinstance(node, ast.Attribute) and node.attr == attr and isinstance(node.ctx, (ast.Store, ast.Del)):
                    raise _Unproven(f"{attr} assigned outside its class body in {path}")
                if isinstance(node, ast.Attribute) and node.attr == "__dict__" and strict:
                    raise _Unproven(f"__dict__ used in {path}")
                if isinstance(node, ast.Call):
                    callee = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
                    if callee in _DYNAMIC_CALLS:
                        literal = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant)]
                        if strict or not literal or literal[0] == attr:
                            raise _Unproven(f"{callee}() in {path}")


def _executed_effects(node):
    if isinstance(node, ast.Lambda):
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                yield from _executed_effects(default)
        return
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in _MUTATING_CALLS:
            yield _dump_key(node)
            return
    if isinstance(node, (ast.AugAssign, ast.Delete)):
        yield _dump_key(node)
        return
    if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, ast.Store):
        yield _dump_key(node)
    for child in ast.iter_child_nodes(node):
        yield from _executed_effects(child)


def _statement_targets(statement):
    """What an expression statement binds or writes, so that retargeting an unchanged call still differs."""
    if isinstance(statement, ast.Assign):
        return tuple(_dump_key(target) for target in statement.targets)
    if isinstance(statement, ast.AnnAssign):
        return (_dump_key(statement.target),)
    return ()


def _definition_effects(node, owner=(), skip=None):
    if isinstance(node, _DEFINITIONS):
        owner = (*owner, node.name)
        for position, decorator in enumerate(node.decorator_list):
            if skip is None or not skip(node, decorator):
                yield owner, "decorator", position, _dump_key(decorator)
        if isinstance(node, ast.ClassDef):
            expressions = (*node.bases, *(keyword.value for keyword in node.keywords))
        else:
            args = node.args
            parameters = (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
            expressions = (*args.defaults, *args.kw_defaults, node.returns,
                           *(arg.annotation for arg in parameters if arg is not None))
        for position, value in enumerate(expressions):
            if value is not None and any(isinstance(n, ast.Call) for n in ast.walk(value)):
                yield owner, "call", position, _dump_key(value)
        if not isinstance(node, ast.ClassDef):
            return
    elif not isinstance(node, ast.Module):

        owner = (*owner, _dump_key(node))
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.expr, ast.withitem, ast.comprehension)):
                if isinstance(child, ast.Call):
                    yield owner, "control", _dump_key(child)
                else:
                    for effect in _executed_effects(child):
                        yield owner, "control", effect
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _DEFINITION_SCOPES):
            yield from _definition_effects(child, owner, skip)
        elif isinstance(child, ast.stmt):
            if isinstance(child, (ast.Expr, ast.Assign, ast.AnnAssign)) and isinstance(child.value, ast.Call):
                if skip is None or not skip(node, child):
                    yield owner, "expression", _statement_targets(child), _dump_key(child.value)
            else:
                for effect in _executed_effects(child):
                    yield owner, "statement", effect




_INERT_CONSTRUCTORS = frozenset({"frozenset", "set", "tuple"})
_INERT_DECORATORS = frozenset({"unsupported_args"})


def _literal(node):
    if isinstance(node, ast.Constant):
        return True
    return isinstance(node, (ast.Tuple, ast.List, ast.Set)) and all(_literal(element) for element in node.elts)


def _rebound(tree):
    """Inert callee names the module binds itself; only `from sqlglot.generator import unsupported_args` is allowed."""
    names = _INERT_CONSTRUCTORS | _INERT_DECORATORS
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            found = {node.id}
        elif isinstance(node, (*_DEFINITIONS, ast.ExceptHandler, ast.MatchAs, ast.MatchStar,
                               ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple)):
            found = {node.name}
        elif isinstance(node, ast.arg):
            found = {node.arg}
        elif isinstance(node, ast.MatchMapping):
            found = {node.rest}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            found = set()
            for alias in node.names:
                if alias.name == "*":
                    found |= names
                elif not (isinstance(node, ast.ImportFrom) and node.module == "sqlglot.generator"
                          and alias.name == "unsupported_args" and alias.asname is None):
                    found.add(alias.asname or alias.name.split(".")[0])
        else:
            continue
        bound |= found & names
    return bound


def _inert_call(node, bound, callees):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in callees
            and node.func.id not in bound and all(_literal(argument) for argument in node.args)
            and all(keyword.arg is not None and _literal(keyword.value) for keyword in node.keywords))


def _naming(text, token, application=None):
    return sorted(line.strip() for line in text.split("\n")
                  if token in line and not (application and line.strip().startswith(application)))


def _inert_kinds(changes):
    """Which inert calls the whole patch leaves trustworthy: none if it moves files, constants unless it edits a line
    naming builtins, decorators unless it edits sqlglot/generator.py or a non-decorator line naming unsupported_args."""
    if not changes or any(change.structural for change in changes):
        return frozenset()
    kinds = {"constant", "decorator"}
    if any(_naming(change.original, "builtins") != _naming(change.patched, "builtins") for change in changes):
        kinds.discard("constant")
    if any(change.path == "sqlglot/generator.py"
           or _naming(change.original, "unsupported_args", "@unsupported_args(")
           != _naming(change.patched, "unsupported_args", "@unsupported_args(") for change in changes):
        kinds.discard("decorator")
    return frozenset(kinds)


def _inert_skip(tree, kinds):
    bound = _rebound(tree)

    def skip(parent, node):
        if isinstance(node, ast.stmt):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            annotation = getattr(node, "annotation", None)
            return ("constant" in kinds and _inert_call(node.value, bound, _INERT_CONSTRUCTORS)
                    and all(isinstance(target, ast.Name) for target in targets)
                    and (annotation is None or not any(isinstance(n, ast.Call) for n in ast.walk(annotation))))

        return ("decorator" in kinds and isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                and _inert_call(node, bound, _INERT_DECORATORS))

    return skip


def order_sensitive_change(change, changes=None):
    path = change.path
    if path in {"sqlglot/parser.py", "sqlglot/generator.py"} or (
        path.startswith("sqlglot/parsers/") and path.endswith(".py")
    ):


        original = _baseline(path, change.original)[0]


        return _shared_scopes_differ(change, original, patched_tree(path, change.patched))
    return generator_initialization_changed(change, changes)


def generator_initialization_changed(change, changes=None):
    """True when a generator module's definition-time effects differ, or a changed scope may write shared state.
    Given the whole patch (`changes`), inert calls (see _inert_skip and _inert_kinds) are left out of the effect
    comparison on both sides; the deferred and reflective write guard below still decides every such patch."""
    if not change.path.startswith("sqlglot/generators/") or not change.path.endswith(".py"):
        return False
    original = _baseline(change.path, change.original)[0]
    patched = patched_tree(change.path, change.patched)
    if tuple(_definition_effects(original)) != tuple(_definition_effects(patched)):
        kinds = _inert_kinds(changes)
        if not kinds or (tuple(_definition_effects(original, skip=_inert_skip(original, kinds)))
                         != tuple(_definition_effects(patched, skip=_inert_skip(patched, kinds)))):
            return True
    if not _shared_scopes_differ(change, original, patched):
        return False
    if changes is None:
        return True


    before = _Universe([], root=None)
    after = _Universe(changes)
    return _shared_scopes_differ(change, original, patched, before, after)


def _shared_scopes_differ(change, original, patched, before=None, after=None):
    def affected(tree, lines, universe):
        if not lines and (change.removed or change.added):
            return ()
        return _changed_shared_scopes(tree, lines, universe, change.path)

    return affected(original, change.removed, before) != affected(patched, change.added, after)


def patched_code(path, source):
    """compile(source.encode("utf-8"), path, "exec", dont_inherit=True), shared by every stage of a patch."""
    code = _patched(path, source)[1]
    if isinstance(code, BaseException):

        raise type(code)(*code.args)
    return code


def _entry(node):
    while isinstance(node, (ast.IfExp, ast.BoolOp, ast.Compare)):
        if isinstance(node, ast.IfExp):
            node = node.test
        elif isinstance(node, ast.BoolOp):
            node = node.values[0]
        else:
            node = node.left
    return node


def _scope_changed(old, new):
    if (
        old.co_flags != new.co_flags
        or old.co_cellvars != new.co_cellvars
        or old.co_freevars != new.co_freevars
        or (set(old.co_varnames) ^ set(new.co_varnames)) & (set(old.co_names) | set(new.co_names))
    ):
        return True
    children = [v for v in old.co_consts if isinstance(v, CodeType)]
    others = [v for v in new.co_consts if isinstance(v, CodeType)]
    if [child.co_qualname for child in children] != [child.co_qualname for child in others]:
        return True
    return any(_scope_changed(a, b) for a, b in zip(children, others))


_PASS = [ast.Pass()]


def _docstring(statement):
    return (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str))


def _same_bodies(old, new):
    """The two bodies are equal statement for statement once a leading docstring is dropped from each, and
    likewise inside every function and class they define directly."""
    old = (old[1:] or _PASS) if old and _docstring(old[0]) else old
    new = (new[1:] or _PASS) if new and _docstring(new[0]) else new
    return len(old) == len(new) and all(_same_statement(a, b) for a, b in zip(old, new))


def _same_statement(a, b):
    if type(a) is not type(b):
        return False
    if not isinstance(a, _DEFINITIONS):
        return _dump_key(a) == _dump_key(b)
    for field, value in ast.iter_fields(a):
        other = getattr(b, field)
        if field == "body":
            if not _same_bodies(value, other):
                return False
        elif isinstance(value, list):
            if len(value) != len(other) or any(_dump_key(x) != _dump_key(y) for x, y in zip(value, other)):
                return False
        elif isinstance(value, ast.AST) or isinstance(other, ast.AST):
            if type(value) is not type(other) or _dump_key(value) != _dump_key(other):
                return False
        elif value != other:
            return False
    return True


def _within_docstrings(tree, source, lines):
    """Every given line that holds code lies in the docstring of the module or of a class or function."""
    rows = source.split("\n")
    pending = {line for line in lines if 0 < line <= len(rows) and rows[line - 1].strip()
               and not rows[line - 1].lstrip().startswith("#")}
    scopes = [tree]
    while scopes and pending:
        body = scopes.pop().body
        if body and _docstring(body[0]):
            pending.difference_update(range(body[0].lineno, body[0].end_lineno + 1))
        scopes.extend(statement for statement in body if isinstance(statement, _DEFINITIONS)
                      and any(statement.lineno <= line <= statement.end_lineno for line in pending))
    return not pending


def docstrings_only(path: str, original: str, patched: str, removed=None, added=None) -> bool:
    """True if the change is confined to docstrings: the compiled programs differ, yet the two sources are the
    same program once the docstring of the module and of every class and function is taken out. They can then
    differ only in __doc__ values, comments and layout.

    Neither the pinned project nor its tests read a docstring: the one use of __doc__ (the expression base
    class copies each subclass's own docstring onto it) feeds nothing, and the doctest loader in
    tests/test_docs.py is not a collected test. Such a change therefore acts like a comment change.
    """
    if _changed_lines(path, original, patched) is not None:
        return False
    try:

        old = _baseline(path, original)[0]
        if removed is not None and not _within_docstrings(old, original, removed):
            return False
        new = patched_tree(path, patched)
        if added is not None and not _within_docstrings(new, patched, added):
            return False
        return _same_bodies(old.body, new.body)
    except (SyntaxError, ValueError, RecursionError):
        return False


def semantic_lines(path: str, original: str, patched: str, old_lines: set[int]) -> set[int] | None:
    lines = _changed_lines(path, original, patched)
    return None if lines is None else lines.copy()


@lru_cache(maxsize=16)
def _changed_lines(path, original, patched):
    try:
        old, compiled_old, executable = _baseline(path, original)
        new = patched_tree(path, patched)
        compiled_new = patched_code(path, patched)
    except (SyntaxError, ValueError, RecursionError):
        return None
    if compiled_old == _without_positions(compiled_new):
        return set()
    if _scope_changed(compiled_old, compiled_new):
        return None

    lines = set()
    ambiguous = False

    def record(node, ancestors, runtime):
        nonlocal ambiguous
        if not runtime:
            ambiguous = True
            return
        for target in (node, *reversed(ancestors)):
            if isinstance(target, (*_DEFINITIONS, ast.Module, ast.Lambda)):
                break
            if not hasattr(target, "lineno"):
                continue
            if isinstance(target, (ast.If, ast.While, ast.IfExp)):
                target = target.test
            if isinstance(target, (ast.BoolOp, ast.Compare)):
                target = _entry(target)
            reached = executable.intersection(range(target.lineno, target.end_lineno + 1))
            if reached:
                lines.update(reached)
                return
        ambiguous = True

    def compare(a, b, ancestors, runtime):
        if type(a) is not type(b):
            if isinstance(a, _DEFINITIONS) or isinstance(b, _DEFINITIONS):
                record(a, ancestors, False)
            else:
                record(a, ancestors, runtime)
            return
        if not isinstance(a, ast.AST):
            if a != b:
                record(a, ancestors, runtime)
            return
        if _ast_native is not None and isinstance(a, ast.stmt):
            try:
                if _ast_native.equal(a, b, ast.AST, "_fields"):
                    return
            except (AttributeError, TypeError, ValueError, RecursionError):
                pass
        parents = ancestors + (a,)
        for field, av in ast.iter_fields(a):
            bv = getattr(b, field)
            body_runtime = runtime
            if isinstance(a, (*_DEFINITIONS, ast.Lambda)):
                body_runtime = field == "body" and not isinstance(a, ast.ClassDef)
            if not isinstance(av, list):
                compare(av, bv, parents, body_runtime)
                continue
            if len(av) == len(bv):
                for index, (left, right) in enumerate(zip(av, bv)):
                    doc = (
                        index == 0 and field == "body" and isinstance(a, _DEFINITIONS)
                        and any(isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant)
                                and isinstance(x.value.value, str) for x in (left, right))
                    )
                    compare(left, right, parents, body_runtime and not doc)
                continue
            if any(not isinstance(x, ast.stmt) for x in (*av, *bv)):
                record(a, ancestors, body_runtime)
                continue
            align = difflib.SequenceMatcher(
                None, [_dump_key(x) for x in av], [_dump_key(x) for x in bv], autojunk=False
            )
            for tag, i, j, p, q in align.get_opcodes():
                if tag == "equal":
                    continue
                edits = av[i:j] + bv[p:q]
                if (
                    any(isinstance(x, _DEFINITIONS) for x in edits)
                    or (isinstance(a, _DEFINITIONS) and field == "body" and i == 0
                        and any(isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant)
                                and isinstance(x.value.value, str) for x in edits))
                ):
                    record(a, ancestors, False)
                elif i < j:
                    for x in av[i:j]:
                        record(x, parents, body_runtime)
                elif av:
                    record(av[min(i, len(av) - 1)], parents, body_runtime)
                else:
                    record(a, ancestors, body_runtime)

    compare(old, new, (), False)
    return None if ambiguous else lines
