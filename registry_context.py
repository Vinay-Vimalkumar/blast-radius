import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path

DIALECT = "sqlglot/dialects/dialect.py"
INDEX = "sqlglot/dialects/__init__.py"
RESET = "tests/test_dialect_entry_points.py"
PINS = {
    DIALECT: "76e8553773db08405642d8ea65cd680322d94a7f1a94b2efdba4015728116bda",
    INDEX: "0769ac3b1bec8cacf5a24c63b3049271d0907e20bba94a2475bbcfdeac329ae9",
    "sqlglot/optimizer/__init__.py": "ad220f0033d3761d67d8048c8b4a730fb1ce47923e88c6bbe236b41067b83a82",
    RESET: "d16d3f02bc635f57d284ef9e3cec350b5e7b1ca46f4122c921fc57eb239bbe3e",
}
METHODS = {"get", "get_or_raise", "_try_load"}
ATTRIBUTES = METHODS | {"classes", "_classes"}
SCOPES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _mentions_registry(node):
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr == "_classes":
            return True
        if isinstance(child, ast.Name) and child.id == "_classes":
            return True
    return isinstance(node, ast.FunctionDef) and node.name == "get_or_raise"


def _signature(source, path):
    tree = ast.parse(source)
    if path == DIALECT:
        projected = []
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef) and stmt.name == "Dialect":
                stmt.body = [child for child in stmt.body if _mentions_registry(child)]
            elif isinstance(stmt, SCOPES) and stmt.name not in ("Dialects", "_Dialect"):
                continue
            projected.append(stmt)
        tree.body = projected
    return ast.dump(tree)


def dialect_provider(path):
    directory, _, filename = path.rpartition("/")
    return directory == "sqlglot/dialects" and filename.endswith(".py") and path not in (INDEX, DIALECT)


def lookup_rows(source, path):
    tree = ast.parse(source)
    aliases = {"Dialect"}
    packages = set()
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Import):
            packages.update(a.asname or a.name for a in stmt.names if a.name == "sqlglot")
        if isinstance(stmt, ast.ImportFrom) and (stmt.module or "").startswith("sqlglot"):
            aliases.update(a.asname or a.name for a in stmt.names if a.name == "Dialect")

    def bound(node):
        if isinstance(node, ast.Name):
            return node.id in aliases
        return (
            isinstance(node, ast.Attribute) and node.attr == "Dialect"
            and isinstance(node.value, ast.Name) and node.value.id in packages
        )

    def visit(node, owner):
        if isinstance(node, SCOPES):
            owner = (*owner, node.name)
        text = None
        if isinstance(node, ast.Subscript) and bound(node.value):
            text = ast.dump(node)
        elif isinstance(node, ast.Attribute) and bound(node.value) and node.attr in ATTRIBUTES:
            text = ast.dump(node)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if bound(node.func.value) and node.func.attr in METHODS:
                text = ast.dump(node)
        elif isinstance(node, ast.Import):
            imports = [(a.name, a.asname) for a in node.names if a.name == "sqlglot"]
            if imports:
                text = repr(imports)
        elif isinstance(node, ast.ImportFrom):
            bindings = [(a.name, a.asname) for a in node.names if a.name == "Dialect" or a.asname in aliases]
            if bindings:
                text = repr((node.module, node.level, bindings))
        if text is not None:
            yield owner, text, node.lineno, node.end_lineno
        for child in ast.iter_child_nodes(node):
            yield from visit(child, owner)

    rows = list(visit(tree, ()))
    if dialect_provider(path):
        for stmt in tree.body:
            if isinstance(stmt, SCOPES):
                stmt.body = []
            rows.append(((), ast.dump(stmt), stmt.lineno, stmt.lineno))
    return rows


def lookup_digest(rows):
    data = json.dumps([(owner, text) for owner, text, _, _ in rows], separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


@lru_cache(maxsize=3)
def _baseline(source, path):
    if hashlib.sha256(source.encode()).hexdigest() != PINS[path]:
        raise ValueError("unexpected registry source")
    return _signature(source, path)


@lru_cache(maxsize=1)
def _readers():
    data = json.loads(Path(__file__).with_name("registry_readers.json").read_bytes())
    return {path: (sha, digest, frozenset(lines)) for path, (sha, digest, lines) in data.items()}


def _lookup_changed(change):
    record = _readers().get(change.path)
    if record is None:
        return dialect_provider(change.path)
    sha, digest, lookup_lines = record
    if change.structural or hashlib.sha256(change.original.encode()).hexdigest() != sha:
        return True
    if not dialect_provider(change.path) and change.removed.isdisjoint(lookup_lines):
        lines = change.patched.split("\n")
        names = ("Dialect", "_classes", "get_or_raise", "_try_load")
        if not any(any(name in lines[i - 1] for name in names) for i in change.added):
            return False
    return lookup_digest(lookup_rows(change.patched, change.path)) != digest


def registry_context_required(changes):
    for change in changes:
        if change.structural and change.original:
            sha = hashlib.sha256(change.original.encode()).hexdigest()
            if sha in PINS.values() or any(sha == entry[0] for entry in _readers().values()):
                return True
        if change.path in PINS:
            if change.structural or _baseline(change.original, change.path) != _signature(change.patched, change.path):
                return True
        if change.path.startswith("sqlglot/") and _lookup_changed(change):
            return True
    return False
