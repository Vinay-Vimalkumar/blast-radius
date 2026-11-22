import ast
from functools import lru_cache
from pathlib import Path

import value_profile as vp
from map_store import read_map
from semantics import _baseline, patched_tree

MAP = Path(__file__).parent / "maps" / "heavy.json.gz"
OPS = {**vp.OPS, ast.In: lambda v, c: v in c, ast.NotIn: lambda v, c: v not in c}
LITERAL_ERROR = object()


def literal(node):
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return LITERAL_ERROR
    if type(value) in (tuple, list, set):
        return value if all(type(x) in (str, int, bool, type(None)) for x in value) else LITERAL_ERROR
    return value if type(value) in (str, int, bool, type(None)) else LITERAL_ERROR


def compare(node):
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in OPS:
        if literal(node.comparators[0]) is not LITERAL_ERROR:
            return node.left, node.comparators[0], False
        if literal(node.left) is not LITERAL_ERROR and type(node.ops[0]) not in (ast.In, ast.NotIn):
            return node.comparators[0], node.left, True


def lookup(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and not node.keywords and len(node.args) in (1, 2) and isinstance(node.args[0], ast.Constant)
            and type(node.args[0].value) is str and (len(node.args) == 1 or isinstance(node.args[1], ast.Constant)))


def enum_compare(node):
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return
    op = type(node.ops[0])
    if op not in (ast.Is, ast.IsNot, ast.Eq, ast.NotEq, ast.In, ast.NotIn):
        return
    rhs = node.comparators[0]
    member = op in (ast.In, ast.NotIn)
    if member and not isinstance(rhs, (ast.Tuple, ast.List, ast.Set)):
        return
    attrs = rhs.elts if member else [rhs]
    if not attrs or not all(isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) for a in attrs):
        return
    qualifier = attrs[0].value.id
    if all(a.value.id == qualifier for a in attrs):
        return qualifier, [a.attr for a in attrs], op


@lru_cache(maxsize=1)
def load():
    return read_map(MAP)


def same(a, b):
    return ast.dump(a) == ast.dump(b)


def matching(old, new, line):
    if isinstance(old, ast.JoinedStr) or new is None:
        return
    if hasattr(old, "lineno") and not old.lineno <= line <= old.end_lineno:
        return
    if isinstance(old, ast.expr) and old.lineno <= line <= old.end_lineno:
        yield old, new
    if type(old) is not type(new):
        return
    for field, value in ast.iter_fields(old):
        other = getattr(new, field)
        if isinstance(value, ast.AST):
            yield from matching(value, other, line)
        elif isinstance(value, list) and len(value) == len(other):
            for a, b in zip(value, other):
                if isinstance(a, ast.AST):
                    yield from matching(a, b, line)


def observed_values(row, constants):
    span, strings, special = row
    points = list(strings)
    for index, val in enumerate((False, True, None)):
        if special & (1 << index):
            points.append(val)
    if span:
        marks = set(span)
        for value in constants:
            values = value if type(value) in (tuple, list, set) else (value,)
            for x in values:
                if type(x) in (int, bool):
                    marks.update(i for i in (x - 1, x, x + 1) if span[0] <= i <= span[1])
        points.extend(marks)
    return points


def evaluate(op, value, fixed, swapped):
    return OPS[op](fixed, value) if swapped else OPS[op](value, fixed)


def rule(old, new):
    e1, e2 = enum_compare(old), enum_compare(new)
    if e1 and e2 and e1[0] == e2[0] and same(old.left, new.left):
        def enum_stable(row):
            table, seen = row
            members = load()["enums"][table]
            def outcome(value, spec):
                names = [members[n] for n in spec[1]]
                result = value in names
                return not result if spec[2] in (ast.IsNot, ast.NotEq, ast.NotIn) else result
            return all(outcome(v, e1) == outcome(v, e2) for v in seen)
        return "e", enum_stable
    a, b = compare(old), compare(new)
    if a and b and a[2] == b[2] and same(a[0], b[0]):
        op1, op2 = type(old.ops[0]), type(new.ops[0])
        if op1 in vp.IDENTITY or op2 in vp.IDENTITY:
            if not (vp.singleton(a[1]) and vp.singleton(b[1])):
                return
        val1, val2 = literal(a[1]), literal(b[1])
        def unchanged(row):
            points = observed_values(row, (val1, val2))
            return bool(points) and all(evaluate(op1, x, val1, a[2]) == evaluate(op2, x, val2, b[2])
                                        for x in points)
        return "v", unchanged
    if isinstance(new, ast.BoolOp) and same(old, new.values[0]):
        stop_truth = isinstance(new.op, ast.Or)
        if a:
            val = literal(a[1])
            op = type(old.ops[0])
            if op not in vp.IDENTITY or vp.singleton(a[1]):
                def stops(row):
                    points = observed_values(row, (val,))
                    return bool(points) and all(evaluate(op, x, val, a[2]) == stop_truth for x in points)
                return "v", stops
        if lookup(old):
            default = old.args[1].value if len(old.args) == 2 else None
            if bool(default) == stop_truth:
                return "d", lambda row: old.args[0].value not in row[0]
    if lookup(old) and lookup(new) and same(old.func, new.func):
        x, y = old.args[0].value, new.args[0].value
        if x == y:
            return "d", lambda row: row[1]
        if len(old.args) == len(new.args) and all(same(a, b) for a, b in zip(old.args[1:], new.args[1:])):
            return "d", lambda row: x not in row[0] and y not in row[0]
    if isinstance(old, ast.BoolOp) and same(old.values[0], new):
        expected = 1 if isinstance(old.op, ast.And) else 2
        return "b", lambda mask: mask == expected
    if isinstance(old, ast.BoolOp) and isinstance(new, ast.BoolOp) and type(old.op) is type(new.op):
        if same(old.values[0], new.values[0]):
            expected = 1 if isinstance(old.op, ast.And) else 2
            return "b", lambda mask: mask == expected
    if isinstance(old, ast.IfExp) and isinstance(new, ast.IfExp) and same(old.test, new.test):
        if same(old.body, new.body):
            return "b", lambda mask: mask == 2
        if same(old.orelse, new.orelse):
            return "b", lambda mask: mask == 1


def refine_heavy(change, lines, candidates):
    if not candidates or not lines or not change.removed:
        return candidates
    try:
        data = load()
        paths = data["sites"].get(change.path)
        if not paths or not set(data["nodes"]).intersection(candidates):
            return candidates
        line = min(change.removed)
        if not any(int(site[0]) <= line <= int(site[2]) for site in (key.split(":") for key in paths)):
            return candidates
        if data["hashes"][change.path] != vp.digest(change.original):
            return candidates
        old_s, new_s = change.original.encode(), change.patched.encode()
        old_l, new_l = vp._offsets(old_s), vp._offsets(new_s)
        old_tree = _baseline(change.path, change.original)[0]
        new_tree = patched_tree(change.path, change.patched)
        for old, new in matching(old_tree, new_tree, line):
            key = vp.location(old)
            if key not in paths or not isinstance(new, ast.expr):
                continue
            start, end = vp._span(old, old_l)
            ns, ne = vp._span(new, new_l)
            if old_s[:start] != new_s[:ns] or old_s[end:] != new_s[ne:]:
                continue
            r = rule(old, new)
            if not r:
                continue
            kind, stable = r
            scope_changes = (ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await, ast.Lambda)
            if (kind == "b" or isinstance(new, ast.BoolOp)) and any(
                isinstance(n, scope_changes) for expr in (old, new) for n in ast.walk(expr)
            ):
                continue
            remove = set()
            for idx, rows in paths[key].get(kind, {}).items():
                node = data["nodes"][int(idx)]
                try:
                    if node in candidates and rows and all(stable(row) for row in rows):
                        remove.add(node)
                except (TypeError, KeyError):
                    continue
            if remove:
                candidates = set(candidates) - remove
        return candidates
    except (OSError, EOFError, IndexError, ValueError, TypeError, KeyError, AttributeError, RecursionError, SyntaxError):
        return candidates
