import ast
import hashlib
import re
from functools import cache

from map_store import MAPS, read_map
from inert_files import safe_headers


SOURCE = "sqlglot/errors.py"
RENDERER = "highlight_sql"
COMPARISONS = (ast.Eq, ast.NotEq, ast.Lt, ast.Gt, ast.LtE, ast.GtE)
GAP = re.compile(rb"([ \t]*)(<=|>=|!=|==|<|>|\+|-)([ \t]*)")


@cache
def _observations():
    return read_map(MAPS / "slice_replay.json.gz")


def catalog_key(nodes):
    return hashlib.sha256("\0".join(sorted(nodes)).encode()).hexdigest()


def _outside(text, node):
    lines = text.encode().splitlines(keepends=True)
    left = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
    right = sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset
    data = b"".join(lines)
    return data[:left], data[right:]


def _operator(text, left, right):
    tail = _outside(text, left)[1]
    head = _outside(text, right)[0]
    width = len(tail) + len(head) - len(text.encode())
    match = GAP.fullmatch(tail[:width])
    if match is not None:
        return head[:-width] + match[1], match[3] + tail[width:]
    return None


def _tree(text):
    tree = ast.parse(text)
    function = next(stmt for stmt in tree.body if isinstance(stmt, ast.FunctionDef) and stmt.name == RENDERER)
    constants = {
        stmt.targets[0].id: stmt.value for stmt in tree.body
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Constant)
    }
    return function, constants


def _operations(text, function, touched):
    for node in ast.walk(function):
        if not isinstance(node, ast.expr) or not all(node.lineno <= line <= node.end_lineno for line in touched):
            continue
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], COMPARISONS):
            location = _operator(text, node.left, node.comparators[0])
            if location is not None:
                yield "compare", location, type(node.ops[0])
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            location = _operator(text, node.left, node.right)
            if location is not None:
                yield "offset", location, type(node.op)
        elif isinstance(node, ast.Constant) and type(node.value) is int and -5 <= node.value <= 256:
            yield "integer", _outside(text, node), node.value


def _edit(change):
    before, constants = _tree(change.original)
    after, revised = _tree(change.patched)
    parameter = before.args.defaults[-1].id
    original = constants[parameter]
    replacement = revised.get(parameter)
    if replacement is not None and _outside(change.original, original) == _outside(change.patched, replacement):
        if type(replacement.value) is int and -5 <= replacement.value <= 256 and replacement.value != original.value:
            return before, after, constants, original.value, replacement.value
    previous = {(family, site): value for family, site, value in _operations(change.original, before, change.removed)}
    for family, site, value in _operations(change.patched, after, change.added):
        if (family, site) in previous and previous[family, site] != value:
            return before, after, constants, original.value, original.value
    return None


def _make(function, constants):
    env = {name: node.value for name, node in constants.items()}
    tree = ast.Module(body=[function], type_ignores=[])
    exec(compile(tree, "<slice-replay>", "exec", dont_inherit=True), env)
    return env[function.name]


class _Input:
    def __init__(self, size, cached_characters=False):
        self.size = size
        self.full = f"\x000,{size}\x00" if size else ""
        self.characters = {} if cached_characters else None

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not isinstance(index, slice):
            raise ValueError("unsupported string access")
        start, stop, step = index.indices(self.size)
        if step != 1:
            raise ValueError("unsupported slice stride")
        if start >= stop:
            return ""
        if start == 0 and stop == self.size:
            return self.full
        atom = f"\x00{start},{stop}\x00"
        if self.characters is not None and stop - start == 1:
            return self.characters.setdefault(atom, atom)
        return atom


def _result(function, sql, positions, context):
    try:
        value = function(sql, positions, context)
    except ValueError:
        if not positions:
            return ("empty-positions",)
        raise
    return (value, tuple(x is sql.full for x in value),
            tuple(x is y for i, x in enumerate(value) for y in value[:i]),
            tuple(sql.characters.get(x) is x for x in value) if sql.characters is not None else ())


def slice_unchanged(changes, patch_text, cov, catalog):
    if len(changes) != 1:
        return False
    change = changes[0]
    if change.structural or change.path != SOURCE or cov.fixture(SOURCE):
        return False
    headers = [line for line in patch_text.splitlines() if line.startswith("diff --git ")]
    if headers != [f"diff --git a/{SOURCE} b/{SOURCE}"] or not safe_headers(patch_text):
        return False
    try:
        profile = _observations()
        if (profile["unmodeled"] or hashlib.sha256(change.original.encode()).hexdigest() != profile["source"]
                or catalog_key(catalog) != profile["catalog"]):
            return False
        edit = _edit(change)
        if edit is None or profile["metadata"]:
            return False
        original, patched, constants, old, new = edit
        before, after = _make(original, constants), _make(patched, constants)
        for size, coordinates, context, supplied in profile["calls"]:
            if old != new and (supplied or context != old):
                continue
            positions = [tuple(pair) for pair in coordinates]
            updated = new if old != new else context
            for cached_characters in (False, True):
                sql = _Input(size, cached_characters)
                if _result(before, sql, positions, context) != _result(after, sql, positions, updated):
                    return False
        return True
    except (OSError, ValueError, SyntaxError, KeyError, TypeError, AttributeError, IndexError, RecursionError, StopIteration):
        return False

