"""Narrow pure edits to a dialect's return-type registry using actual test lookups."""
import ast
import hashlib
from functools import cache

from declarations import _outside_dict
from map_store import MAPS, expression_consumer_profile, read_map
from schedule import distribute


@cache
def _readers():
    return read_map(MAPS / "typing_keys.json.gz")


def _return_groups(source):
    tree = ast.parse(source)
    for stmt in tree.body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "EXPRESSION_METADATA"
                and isinstance(stmt.value, ast.Dict)):
            continue
        for key, comp in zip(stmt.value.keys, stmt.value.values):
            if key is not None or not isinstance(comp, ast.DictComp) or len(comp.generators) != 1:
                continue
            gen = comp.generators[0]
            if (gen.ifs or gen.is_async or not isinstance(gen.target, ast.Name)
                    or not isinstance(comp.key, ast.Name) or comp.key.id != gen.target.id
                    or not isinstance(gen.iter, ast.Set) or not isinstance(comp.value, ast.Dict)
                    or len(comp.value.keys) != 1 or not isinstance(comp.value.keys[0], ast.Constant)
                    or comp.value.keys[0].value not in {"returns", "annotator"}):
                continue
            dtype = comp.value.values[0]
            if comp.value.keys[0].value == "returns":
                if not (isinstance(dtype, ast.Attribute) and isinstance(dtype.value, ast.Attribute)
                        and dtype.value.attr == "DType" and isinstance(dtype.value.value, ast.Name)
                        and dtype.value.value.id == "exp"):
                    continue
            elif not isinstance(dtype, ast.Name) or dtype.id == gen.target.id:
                continue
            members = set()
            for elt in gen.iter.elts:
                if not (isinstance(elt, ast.Attribute) and isinstance(elt.value, ast.Name)
                        and elt.value.id == "exp"):
                    break
                members.add(elt.attr)
            else:
                yield gen.iter, members


def _changed_members(change):


    added = change.added
    new_groups = {_outside_dict(change.patched, node): members for node, members in _return_groups(change.patched)
                  if not added or all(node.lineno <= n <= node.end_lineno for n in added)}
    if not new_groups:
        return set()
    for old, left in _return_groups(change.original):
        if change.removed and not all(old.lineno <= n <= old.end_lineno for n in change.removed):
            continue
        right = new_groups.get(_outside_dict(change.original, old))
        if right is not None and left != right:
            return left ^ right
    return set()


def _affected(node, groups):

    while node:
        if node in groups:
            return True
        if "::" in node:
            node = node.rpartition("::")[0]
        else:
            node = node.rpartition("/")[0]
    return False


def refine_typing_metadata(changes, chosen, cov):
    if len(changes) != 1 or not chosen:
        return chosen
    change = changes[0]
    if (change.structural or not change.path.startswith("sqlglot/typing/")
            or change.path not in expression_consumer_profile()["typing"]):
        return chosen
    if hashlib.sha256(change.original.encode()).hexdigest() != expression_consumer_profile()["hashes"][change.path]:
        return chosen
    try:
        names = _changed_members(change)
        if not names:
            return chosen
        profile = _readers()
        if not names.issubset(profile["exports"]):
            return chosen
        groups = set()
        owners = ["*"] + [owner for owner, paths in profile["origins"].items() if change.path in paths]
        for name in names:
            for owner in owners:
                groups.update(profile["groups"].get(owner + ":" + profile["exports"][name], ()))
        if "" in groups:
            return chosen

        keep = set(cov.fixture(change.path)) | cov.missing
        for path, inspected in profile["direct_files"].items():
            if "*" in inspected or names.intersection(inspected):
                groups.add(path)
        groups.update(cov.fixture(change.path))
        complete = set(profile["complete"])



        for i, node in enumerate(profile["order"]):
            if _affected(node, groups):
                keep.update(profile["order"][i:])
                break
        keep.update(chosen - complete)
        return chosen & keep
    except (OSError, ValueError, SyntaxError, TypeError, KeyError, AttributeError, RecursionError):
        return chosen


def refine_typing_schedule(changes, workers, cov, tests, millis, padding):
    if not workers or len(changes) != 1 or not changes[0].path.startswith("sqlglot/typing/"):
        return workers
    numbered = [n for worker in workers for n in worker]
    chosen = {tests[n]["id"] for n in numbered}
    kept = refine_typing_metadata(changes, chosen, cov)
    if kept == chosen:
        return workers
    proposal = distribute([n for n in numbered if tests[n]["id"] in kept], millis)
    if padding is not None:
        proposal = padding.fill(proposal)

    def cost(groups):
        load = max((sum(millis[n] for n in group) for group in groups), default=0)
        return load + 2000 * len(groups)



    return proposal if cost(proposal) < cost(workers) else workers
