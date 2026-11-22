"""Consumers of new, class-keyed expressions in the pinned SQLGlot registry."""
import ast
import hashlib
import re
from pathlib import Path

from declarations import _lookup_index
from map_store import expression_consumer_profile
from semantics import _dump_key


PROFILE = Path(__file__).with_name("expression_consumers.json")
REGISTRIES = {"FUNCTIONS", "FUNCTION_BY_NAME", "ALL_FUNCTIONS", "EXPR_CLASSES",
              "EXPRESSION_METADATA", "TRANSFORMS"}


def _declaration(node):
    if (not isinstance(node, ast.ClassDef) or not re.fullmatch(r"[A-Z][A-Za-z0-9]*", node.name)
            or node.decorator_list or node.keywords or node.type_params
            or [ast.unparse(b) for b in node.bases] != ["Expression", "Func"]
            or len(node.body) != 1):
        return False
    stmt = node.body[0]
    if (not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1
            or not isinstance(stmt.targets[0], ast.Name) or stmt.targets[0].id != "arg_types"
            or not isinstance(stmt.value, ast.Dict)):
        return False
    keys = {k.value for k in stmt.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return ("this" in keys and len(keys) == len(stmt.value.keys)
            and all(isinstance(v, ast.Constant) and type(v.value) is bool for v in stmt.value.values))


def _new_classes(change):
    old, new = ast.parse(change.original.encode()), ast.parse(change.patched.encode())
    existing = {n.name for n in old.body if isinstance(n, ast.ClassDef)}
    added = [n for n in new.body if isinstance(n, ast.ClassDef) and n.name not in existing]
    if not added or not all(map(_declaration, added)):
        return None
    names = {n.name for n in added}
    if len(names) != len(added):
        return None

    available = set()
    for stmt in new.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "sqlglot.expressions.core":
            available.update(a.name for a in stmt.names if not a.asname)
        if stmt in added and not {"Expression", "Func"} <= available:
            return None
    new.body = [n for n in new.body if n not in added]
    return names if _dump_key(old) == _dump_key(new) else None


def _returns_set(comp):
    if not isinstance(comp, ast.DictComp) or len(comp.generators) != 1:
        return None
    gen = comp.generators[0]
    if (gen.ifs or gen.is_async or not isinstance(gen.iter, ast.Set)
            or not isinstance(gen.target, ast.Name) or not isinstance(comp.key, ast.Name)
            or comp.key.id != gen.target.id or not isinstance(comp.value, ast.Dict)
            or len(comp.value.keys) != 1 or not isinstance(comp.value.keys[0], ast.Constant)
            or comp.value.keys[0].value != "returns"
            or not isinstance(comp.value.values[0], ast.Attribute)
            or ast.unparse(comp.value.values[0].value) != "exp.DType"):
        return None
    return gen.iter


def _only_new_metadata(change, names):
    old, new = ast.parse(change.original.encode()), ast.parse(change.patched.encode())
    removed = set()
    for stmt in new.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
        elif isinstance(stmt, ast.AnnAssign):
            target = stmt.target
        else:
            continue
        if not isinstance(target, ast.Name) or target.id != "EXPRESSION_METADATA" or not isinstance(stmt.value, ast.Dict):
            continue
        for key, value in zip(stmt.value.keys, stmt.value.values):
            group = _returns_set(value) if key is None else None
            if group is None:
                continue
            keep = []
            for element in group.elts:
                if (isinstance(element, ast.Attribute) and isinstance(element.value, ast.Name)
                        and element.value.id == "exp" and element.attr in names):
                    if element.attr in removed:
                        return False
                    removed.add(element.attr)
                else:
                    keep.append(element)
            group.elts = keep
    return bool(removed) and _dump_key(old) == _dump_key(new)


def _eligible(changes, profile):
    classes = [c for c in changes if c.path in profile["modules"]]
    types = [c for c in changes if c.path in profile["typing"]]
    if len(classes) != 1 or len(changes) != len(classes) + len(types) or len(types) > 1:
        return None
    if any(c.structural or c.removed or hashlib.sha256(c.original.encode()).hexdigest() != profile["hashes"][c.path]
           for c in changes):
        return None
    names = _new_classes(classes[0])
    if not names or types and not _only_new_metadata(types[0], names):
        return None
    spellings = {re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper() for name in names}
    lower = {name.lower() for name in names}
    if (len(lower) != len(names) or len(spellings) != len(names)
            or lower.intersection(profile["reserved"]) or spellings.intersection(profile["functions"])):
        return None
    return names, spellings


def refine_expression_additions(changes, chosen, cov, catalog):
    if len(changes) not in {1, 2}:
        return chosen
    class_candidate = False
    for change in changes:
        if change.path.startswith("sqlglot/expressions/") and change.added:
            lines = change.patched.splitlines()
            class_candidate = any(lines[i - 1].startswith("class ") for i in change.added)
            if class_candidate:
                break
    if not class_candidate:
        return chosen
    try:
        profile = expression_consumer_profile()
        affected = _eligible(changes, profile)
        if not affected:
            return chosen
        names, spellings = affected
        tests, lookups, _, _ = _lookup_index()
        keep = set(catalog).difference(tests) | cov.missing | set(profile["opaque"])
        for c in changes:
            keep.update(cov.fixture(c.path))
        keys = {"str:" + spelling for spelling in spellings} | {"member:" + name for name in names}
        keys.update("str:" + name.lower() for name in names)
        keys.add("*")
        for owner, attr in profile["owners"]:
            if (owner, attr, "*") not in lookups:
                return chosen
            for key in keys:
                keep.update(tests[i] for i in lookups.get((owner, attr, key), ()))
        return chosen & keep
    except (OSError, ValueError, KeyError, IndexError, AttributeError, TypeError, SyntaxError):
        return chosen
