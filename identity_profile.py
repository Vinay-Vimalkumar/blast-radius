import ast
import re
import zlib
from functools import lru_cache
from pathlib import Path

import value_profile as vp
from map_store import read_map


MAP = Path(__file__).resolve().parent / "maps" / "identity.json.gz"
UNKNOWN = 1 << 12
GAP = re.compile(rb"([)\s]*)(is[ \t]+not|is|==|!=)([\s(]*)")


@lru_cache(maxsize=1)
def _load():
    data = read_map(MAP)
    if data["version"] != 1:
        raise ValueError("unsupported identity profile")
    return data


def _edit(original, patched, old_lines, sites, path=None):
    before, after = original.encode(), patched.encode()
    offsets = vp._offsets(before)
    delta = len(after) - len(before)
    line = next((i for i, (a, b) in enumerate(zip(before.split(b"\n"), after.split(b"\n")), 1)
                 if a != b), None)
    if line is None or (old_lines and line not in old_lines):
        return None
    baseline = vp._baseline_tree(original, path)
    for node in vp._comparisons_at(baseline, line):
        key = vp.location(node)
        if key not in sites or len(node.ops) != 1:
            continue
        start = vp._span(node.left, offsets)[1]
        end = vp._span(node.comparators[0], offsets)[0]
        if before[:start] != after[:start] or before[end:] != after[end + delta:]:
            continue
        old = GAP.fullmatch(before[start:end])
        new = GAP.fullmatch(after[start:end + delta])
        if not old or not new:
            continue
        if (old[1].rstrip(b" \t") != new[1].rstrip(b" \t")
                or old[3].lstrip(b" \t") != new[3].lstrip(b" \t")):
            continue
        op1, op2 = old[2].split(), new[2].split()
        if (op1[0] == b"is") == (op2[0] == b"is"):
            continue
        new_tree = ast.parse(patched)
        for changed in vp._comparisons_at(new_tree, line):
            if (changed.lineno, changed.col_offset) != (node.lineno, node.col_offset):
                continue
            if (len(changed.ops) != 1 or not vp._same(node.left, changed.left)
                    or not vp._same(node.comparators[0], changed.comparators[0])):
                continue
            changed.ops = node.ops
            if not vp._same(baseline, new_tree):
                return None
            same_polarity = (op1 in ([b"is"], [b"=="])) == (op2 in ([b"is"], [b"=="]))
            return key, UNKNOWN | (0x666 if same_polarity else 0x999)
    return None


def refine_identity(path, original, patched, old_lines, candidates) -> set[str]:
    result = set(candidates)
    if not result or not isinstance(original, str) or not isinstance(patched, str):
        return result
    try:
        path = str(path).replace("\\", "/")
        data = _load()
        original, patched = original.replace("\r\n", "\n"), patched.replace("\r\n", "\n")
        sites = data["sites"].get(path)
        if not sites or data["hashes"].get(path) != vp.digest(original):
            return result
        change = _edit(original, patched, old_lines, sites, path)
        if change is None:
            return result
        site, differs = change
        retain, remove = set(), set()
        for mask, indices in sites[site].items():
            mask = int(mask)
            if not 0 < mask < UNKNOWN * 2:
                return result
            group = retain if mask & differs else remove
            for i in indices:
                if type(i) is not int or i < 0:
                    return result
                group.add(data["nodes"][i])
        return result - (remove - retain)
    except (OSError, EOFError, zlib.error, ValueError, KeyError, TypeError, IndexError,
            AttributeError, SyntaxError, RecursionError):
        return result
