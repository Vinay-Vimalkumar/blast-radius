import ast
import hashlib
import operator
import re
import zlib
from functools import lru_cache
from pathlib import Path

from map_store import read_map
from semantics import _baseline, _dump_key, patched_tree


MAP = Path(__file__).resolve().parent / "maps" / "values.json.gz"
OPS = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Eq: operator.eq,
    ast.NotEq: operator.ne, ast.GtE: operator.ge, ast.Gt: operator.gt,
    ast.Is: operator.is_, ast.IsNot: operator.is_not,
}
IDENTITY = (ast.Is, ast.IsNot)
EQUALITY = (ast.Eq, ast.NotEq)
UNKNOWN = 1 << 7
UNORDERED = 1 << 6
SAMPLES = (None, True, False, 0, 1, "", object())
OPAQUE = 6


def singleton(node):
    return isinstance(node, ast.Constant) and (node.value is None or type(node.value) is bool)


def layout(node):
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    op = type(node.ops[0])
    if op not in OPS:
        return None
    if op in IDENTITY + EQUALITY:
        if singleton(node.comparators[0]):
            return "right"
        if singleton(node.left):
            return "left"
    if op in IDENTITY:
        return None
    return "relation"


def location(node):
    return f"{node.lineno}:{node.col_offset}:{node.end_lineno}:{node.end_col_offset}"


def digest(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _integer(node):
    return isinstance(node, ast.Constant) and type(node.value) is int


def _same(a, b):
    return _dump_key(a) == _dump_key(b)


def _difference(old, new):
    kind = layout(old)
    if not kind or not isinstance(new, ast.Compare) or len(new.ops) != 1:
        return None
    before, after = type(old.ops[0]), type(new.ops[0])
    if after not in OPS:
        return None
    left, right = old.left, old.comparators[0]
    new_left, new_right = new.left, new.comparators[0]
    old_fn, new_fn = OPS[before], OPS[after]
    if kind != "relation":
        if kind == "left":
            left, right, new_left, new_right = right, left, new_right, new_left
        if not _same(left, new_left) or not singleton(new_right) or after not in IDENTITY + EQUALITY:
            return None
        bits = 0
        for idx, value in enumerate(SAMPLES):
            if idx == OPAQUE and (before in EQUALITY or after in EQUALITY):
                bits |= 1 << idx
            elif old_fn(value, right.value) != new_fn(value, new_right.value):
                bits |= 1 << idx
        return bits
    if after in IDENTITY:
        return None
    if _same(left, new_left) and _same(right, new_right):
        bits = UNKNOWN
        for i, sign in enumerate((-1, 0, 1)):
            if old_fn(sign, 0) != new_fn(sign, 0):
                bits |= (1 << i) | (1 << (i + 3))
        if (before is ast.NotEq) != (after is ast.NotEq):
            bits |= UNORDERED
        return bits
    swapped = _integer(left) and _integer(new_left) and _same(right, new_right)
    if swapped:
        right, new_right = left, new_left
    elif not (_integer(right) and _integer(new_right) and _same(left, new_left)):
        return None
    a, b = right.value, new_right.value
    bits = UNKNOWN | UNORDERED | 0b111000
    for boundary in (a, b):
        for x in (boundary - 1, boundary, boundary + 1):
            l, r, nl, nr = (a, x, b, x) if swapped else (x, a, x, b)
            if old_fn(l, r) != new_fn(nl, nr):
                bits |= 1 << (1 + (l > r) - (l < r))
    return bits


@lru_cache(maxsize=1)
def _load():
    data = read_map(MAP)
    if data["version"] != 1:
        raise ValueError("unsupported operand profile")
    return data


def _offsets(data):
    return [0, *(m.end() for m in re.finditer(b"\n", data))]


def _span(node, lines):
    return lines[node.lineno - 1] + node.col_offset, lines[node.end_lineno - 1] + node.end_col_offset


def _comparisons_at(node, line):
    if hasattr(node, "lineno") and not node.lineno <= line <= node.end_lineno:
        return
    if isinstance(node, ast.JoinedStr) or type(node).__name__ == "TemplateStr":
        return
    if isinstance(node, ast.Compare):
        yield node
    for child in ast.iter_child_nodes(node):
        yield from _comparisons_at(child, line)


@lru_cache(maxsize=2)
def _baseline_tree(original, path=None):
    return ast.parse(original) if path is None else _baseline(path, original)[0]


def _edit(original, patched, old_lines, sites, path=None):
    before, after = original.encode(), patched.encode()
    old_offsets, new_offsets = _offsets(before), _offsets(after)
    if len(old_offsets) != len(new_offsets):
        return None
    line = next((i for i, (a, b) in enumerate(zip(before.split(b"\n"), after.split(b"\n")), 1)
                 if a != b), None)
    if line is None or (old_lines and line not in old_lines):
        return None
    new_sites = None
    for old in _comparisons_at(_baseline_tree(original, path), line):
        if location(old) not in sites:
            continue
        start, end = _span(old, old_offsets)
        if before[:start] != after[:start] or not after.endswith(before[end:]):
            continue
        if new_sites is None:
            new_tree = ast.parse(patched) if path is None else patched_tree(path, patched)
            new_sites = {(n.lineno, n.col_offset, n.end_lineno, n.end_col_offset): n
                         for n in _comparisons_at(new_tree, line)}
        pos = (old.lineno, old.col_offset, old.end_lineno,
               old.end_col_offset + len(after) - len(before))
        new = new_sites.get(pos)
        if new is None:
            continue
        ns, ne = _span(new, new_offsets)
        if before[:start] != after[:ns] or before[end:] != after[ne:]:
            continue
        bits = _difference(old, new)
        if bits is None or _same(old, new):
            continue
        if not _only_comparison_tokens(old, new, before, after, old_offsets, new_offsets):
            continue
        return location(old), bits
    return None


def _only_comparison_tokens(old, new, before, after, old_offsets, new_offsets):
    def parts(node, data, offsets):
        start, end = _span(node, offsets)
        a, b = _span(node.left, offsets)
        c, d = _span(node.comparators[0], offsets)
        return data[start:a], data[a:b], data[b:c], data[c:d], data[d:end]

    prefix, left, gap, right, suffix = parts(old, before, old_offsets)
    new_prefix, new_left, new_gap, new_right, new_suffix = parts(new, after, new_offsets)
    if prefix != new_prefix or suffix != new_suffix:
        return False
    if left != new_left and _same(old.left, new.left):
        return False
    if right != new_right and _same(old.comparators[0], new.comparators[0]):
        return False
    if gap == new_gap:
        return True
    pattern = rb"[ \t]*(?:is[ \t]+not|is|<=|>=|==|!=|<|>)[ \t]*"
    return bool(re.fullmatch(pattern, gap) and re.fullmatch(pattern, new_gap))


def refine(path, original, patched, old_lines, candidates):
    result = set(candidates)
    if not result:
        return result
    try:
        data = _load()
        path = str(path).replace("\\", "/")
        if not isinstance(original, str) or not isinstance(patched, str):
            return result
        original, patched = original.replace("\r\n", "\n"), patched.replace("\r\n", "\n")
        if data["hashes"].get(path) != digest(original) or path not in data["sites"]:
            return result
        sites = data["sites"][path]
        change = _edit(original, patched, set(old_lines or ()), sites, path)
        if change is None:
            return result
        site, differs = change
        removable = set()
        nodes = data["nodes"]
        for mask, indices in sites[site].items():
            if not 0 < int(mask) <= 255:
                return result
            if not int(mask) & differs:
                for i in indices:
                    if type(i) is not int or i < 0:
                        return result
                    removable.add(nodes[i])
        return result - removable
    except (OSError, EOFError, zlib.error, ValueError, KeyError, TypeError, IndexError,
            AttributeError, SyntaxError, RecursionError):
        return result
