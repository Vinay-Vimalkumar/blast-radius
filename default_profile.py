from functools import cache
import hashlib

from import_effects import import_consumers
from map_store import MAPS, read_map
from semantics import patched_code


@cache
def _sites():
    data = read_map(MAPS / "defaults.json.gz")
    if data["schema"] != 1:
        raise ValueError("unsupported index version")
    return data


def _literal(text):
    if text in (b"True", b"False"):
        return text == b"True"
    if not text or len(text) > 256 or not text.isascii() or not text[:1].isdigit() or text.strip() != text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        return None


def _body(change):
    if change.structural or len(change.removed) != 1 or change.removed != change.added:
        return None
    data = _sites()
    sites = data["sites"].get(change.path)
    if not sites:
        return None
    old = change.original.encode("utf-8")
    new = change.patched.encode("utf-8")
    if hashlib.sha256(old).hexdigest() != data["hashes"][change.path]:
        return None
    line, = change.removed
    line_start = sum(map(len, old.splitlines(keepends=True)[:line - 1]))
    for _, _, site_line, column, end_column, first, last in sites:
        if site_line != line:
            continue
        start = line_start + column
        end = line_start + end_column
        new_end = end + len(new) - len(old)
        if new_end < start or old[:start] != new[:start] or old[end:] != new[new_end:]:
            continue
        previous = _literal(old[start:end])
        edited = _literal(new[start:new_end])
        if previous is not None and type(previous) is type(edited) and previous != edited:
            patched_code(change.path, change.patched)
            return set(range(first, last + 1))
    return None


def refine_defaults(changes, chosen, cov, catalog):
    if len(changes) != 1:
        return chosen
    change = changes[0]
    try:
        body = _body(change)
        if not body or not cov.present(change.path):
            return chosen
        consumers = import_consumers(change.path, body | change.removed, cov, catalog)
        return cov.lines(change.path, body) | consumers | cov.fixture(change.path) | cov.missing | (catalog - set(cov.nodes))
    except (OSError, EOFError, IndexError, KeyError, ValueError, TypeError, SyntaxError, RecursionError):
        return chosen
