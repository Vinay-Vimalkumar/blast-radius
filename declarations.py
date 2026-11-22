import ast
import copy
import hashlib
import os
import re
from collections import defaultdict
from functools import cache
from pathlib import Path

from map_store import expression_consumer_profile, read_grouped, read_map
from semantics import _baseline, patched_tree


@cache
def _index():
    path = Path(__file__).resolve().parent / "maps" / "declarations.json.gz"
    grouped = read_grouped(path)
    if grouped is not None:
        data, records = grouped
    else:
        data = read_map(path)
        records = {tuple(row[:-1]): row[-1] for row in data["records"]}
    descendants = defaultdict(set)
    for child, bases in data["parents"].items():
        for base in bases:
            descendants[base].add(child)
    copies = defaultdict(set)
    for src, attr, dst in data.get("copies", ()):
        copies[src, attr].add(dst)
    return descendants, records, data["tests"], set(data["cached"]), copies


def _declarations(node, owner):
    for stmt in node.body:
        if isinstance(stmt, ast.ClassDef):
            yield from _declarations(stmt, f"{owner}.{stmt.name}")
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            target = stmt.target if isinstance(stmt, ast.AnnAssign) else stmt.targets[0]
            if isinstance(target, ast.Name) and stmt.value is not None:
                yield owner, target.id, stmt


def _key(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return "str:" + node.value
    if isinstance(node, ast.Attribute):
        return "member:" + node.attr
    return None


def _outside_dict(source, node):
    lines = source.encode("utf-8").splitlines(keepends=True)
    return (b"".join(lines[:node.lineno - 1]) + lines[node.lineno - 1][:node.col_offset],
            lines[node.end_lineno - 1][node.end_col_offset:] + b"".join(lines[node.end_lineno:]))


def expression_contract_narrowed(change, state=None):
    if not change.path.startswith("sqlglot/expressions/") or not change.path.endswith(".py"):
        return False
    before, after = _declaration_trees(change.path, change.original, change.patched)
    schemas = []
    touched = set()
    for tree, lines in ((before, change.removed), (after, change.added)):
        entries = {}
        for owner, name, statement in _declarations(tree, ""):
            if name == "arg_types":
                entries[owner] = statement.value
                if any(statement.lineno <= line <= statement.end_lineno for line in lines):
                    touched.add(owner)
        schemas.append(entries)
    old, new = schemas
    for owner in touched.intersection(old):
        try:
            original = ast.literal_eval(old[owner])
            updated = ast.literal_eval(new[owner])
        except (KeyError, ValueError, TypeError):
            return True
        if not isinstance(original, dict) or not isinstance(updated, dict):
            return True
        if original.keys() - updated.keys() or any(required and not original.get(k) for k, required in updated.items()):
            if (state is not None and len(state.changes) == 1 and not change.structural
                    and isinstance(old[owner], ast.Dict) and isinstance(new[owner], ast.Dict)
                    and all(type(k) is str and type(v) is bool for d in (original, updated) for k, v in d.items())
                    and _outside_dict(change.original, old[owner]) == _outside_dict(change.patched, new[owner])):
                prefix = change.path[:-3].replace("/", ".") + owner + "."
                bindings = state.changed()

                if bindings and set(bindings) <= {prefix + "arg_types", prefix + "required_args"}:
                    continue
            return True
    return False


@cache
def _lookup_index():
    folder = Path(__file__).resolve().parent / "maps"
    grouped = read_grouped(folder / "declarations-lookups.json.gz")
    if grouped is not None:
        data, records = grouped
    else:
        data = read_map(folder / "declarations-lookups.json.gz")
        records = {tuple(r[:-1]): r[-1] for r in data["records"]}
    access = read_map(folder / "declarations-access.json.gz")
    aliases = {(a, b): (c, d) for a, b, c, d in data.get("aliases", ())}
    return data["tests"], records, aliases, access


def _safe_reference(node, known, symbols):
    if isinstance(node, (ast.Constant, ast.Lambda)):
        return isinstance(node, ast.Constant) or not (node.args.defaults or any(node.args.kw_defaults))
    if isinstance(node, (ast.Name, ast.Attribute)):
        name = ast.unparse(node)
        if name in known:
            return True
        owner, _, member = name.rpartition(".")
        if owner.split(".")[0] not in known:
            return False
        if owner == "exp.DType":
            owner = "DType"
        return member in symbols.get(owner, ())
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_safe_reference(e, known, symbols) for e in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float)
    return False


def _entry_edits(a, b, symbols):
    if isinstance(a, ast.BinOp) and isinstance(b, ast.BinOp) and isinstance(a.op, (ast.BitOr, ast.Sub)):
        if type(a.op) != type(b.op) or ast.dump(a.left) != ast.dump(b.left):
            return None
        a, b = a.right, b.right
    if not isinstance(a, (ast.Dict, ast.Set)) or type(a) != type(b):
        return None
    known = set()
    pending = [a]
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Lambda, ast.SetComp, ast.ListComp, ast.DictComp, ast.GeneratorExp)):
            continue
        if isinstance(node, (ast.Name, ast.Attribute)):
            known.add(ast.unparse(node))
        pending.extend(ast.iter_child_nodes(node))
    def entries(node):
        items = zip(node.keys, node.values) if isinstance(node, ast.Dict) else ((k, None) for k in node.elts)
        result = {}
        for k, v in items:
            if k is None or isinstance(k, ast.Starred):
                key = "spread:" + ast.dump(v if k is None else k)
            else:
                key = _key(k)
                if key is None or not _safe_reference(k, known, symbols):
                    return None
            if key in result:
                return None
            result[key] = v
        return result
    x, y = entries(a), entries(b)
    if x is None or y is None:
        return None
    shared = x.keys() & y.keys()
    if [k for k in x if k in shared] != [k for k in y if k in shared]:
        return None
    changes = x.keys() ^ y.keys()
    shape = bool(changes)
    for key in shared:
        if x[key] is not None and ast.dump(x[key]) != ast.dump(y[key]):
            changes.add(key)
    for key in changes:
        if key.startswith("spread:"):
            return None
        v = y.get(key)
        if v is not None and not _safe_reference(v, known, symbols):
            return None
    return changes, shape


def _alias_group(value):
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr == "fromkeys" and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "dict" and len(value.args) == 2 and not value.keywords):
        return ()
    keys, handler = value.args
    if not (isinstance(keys, ast.Tuple) and isinstance(handler, ast.Lambda)
            and not handler.args.defaults and not any(handler.args.kw_defaults)):
        return ()
    names = tuple(k.value for k in keys.elts if isinstance(k, ast.Constant)
                  and type(k.value) is str and re.fullmatch(r"[A-Z][A-Z0-9_]*", k.value))
    return names if len(names) == len(keys.elts) == len(set(names)) else ()


def _dispatch_handler(value):
    bound = isinstance(value, ast.Attribute) and value.attr == "from_arg_list"
    if bound:
        target = value.value
    elif isinstance(value, ast.Lambda):
        args = value.args
        if (len(args.args) != 1 or args.posonlyargs or args.vararg or args.kwonlyargs
                or args.kwarg or args.defaults or not isinstance(value.body, ast.Call)
                or value.body.args):
            return False
        target = value.body.func
    else:
        return False
    if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
            and target.value.id == "exp"):
        return False
    try:
        from sqlglot import exp
    except ImportError:
        return False
    cls = vars(exp).get(target.attr)
    if (type(cls) is not type or not issubclass(cls, exp.Func)
            or any(getattr(cls, attr, None) is not getattr(exp.Expression, attr)
                   for attr in ("__init__", "__new__", "__getattribute__", "__setattr__", "_set_parent"))):
        return False
    if bound:
        method = next((vars(t)["from_arg_list"] for t in cls.__mro__ if "from_arg_list" in vars(t)), None)
        return method is vars(exp.Func)["from_arg_list"] and type(method) is classmethod
    argument = args.args[0].arg
    if argument in {"dict", "exp", "seq_get"}:
        return False
    for kw in value.body.keywords:
        v = kw.value
        if kw.arg not in cls.arg_types:
            return False
        if isinstance(v, ast.Constant) and (v.value is None or type(v.value) is bool):
            continue
        if isinstance(v, ast.Name) and v.id == argument:
            continue
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "seq_get"
                and len(v.args) == 2 and not v.keywords and isinstance(v.args[0], ast.Name)
                and v.args[0].id == argument and isinstance(v.args[1], ast.Constant)
                and type(v.args[1].value) is int):
            continue
        return False
    return True


def _grouped_function_edits(before, after):
    if not isinstance(before, ast.Dict) or not isinstance(after, ast.Dict):
        return None
    def entries(node):
        mapping = {}
        for key, value in zip(node.keys, node.values):
            group = _alias_group(value) if key is None else ()
            if group:
                keys, handler = group, value.args[1]
            elif isinstance(key, ast.Constant) and type(key.value) is str:
                keys, handler = (key.value,), value
            elif key is None:
                keys, handler = (("spread", ast.dump(value)),), value
            else:
                return None
            for k in keys:
                if k in mapping:
                    return None
                mapping[k] = (group, ast.dump(handler), handler)
        return mapping
    old, new = entries(before), entries(after)
    if old is None or new is None:
        return None
    changed = {k for k in old.keys() | new.keys() if k not in old or k not in new or old[k][:2] != new[k][:2]}
    if (not changed or any(type(k) is not str for k in changed)
            or not any(index[k][0] for k in changed for index in (old, new) if k in index)
            or [k for k in old if k not in changed] != [k for k in new if k not in changed]):
        return None
    for k in changed:
        for index in (old, new):
            if k in index and not _dispatch_handler(index[k][2]):
                return None
    return {"str:" + k for k in changed}, True


_DUCKDB_PARSER = "43d5f72b9fc84a064e0f59855c33bc04ca113d606f2f5ad194c2b1e926889535"


@cache
def _opaque_dispatch_readers(source):
    readers = set(expression_consumer_profile()["opaque"])
    profiles = (
        ("tests/test_executor.py", "270a844a7b0b84045efde017135e6208ca07d7d93eba395be17c20c6e918923d",
         ("TestExecutor::test_execute_tpcds", "TestExecutor::test_execute_tpch")),
        ("tests/test_generator.py", "3b3781e8219d97c80a882d5285bdf9824f306abf85c4cafb7e5dce7631243215",
         ("TestGenerator::test_generator_memory",)),
        ("tests/test_optimizer.py", "35c02d71cacfa5d8e24319ef245963da3c1151b891281bac1864a00fa183cbf7",
         ("TestOptimizer::test_tpcds",)),
    )
    for file, digest, methods in profiles:
        path = source / file
        if path.is_file() and hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest() == digest:
            readers.difference_update(f"{file}::{method}" for method in methods)
    return readers


def _observed(rows, cov, relevant=None):
    found = set()
    for filename, lines in rows.items():
        found.update(cov.lines(filename, lines))
        if relevant is not None and relevant <= found:
            break
    return found


def _lookup_candidates(cls, attr, impacted, keys, shape, module, cov):
    tests, records, aliases, access = _lookup_index()
    origins = [aliases.get((owner, attr), (owner, attr)) for owner in impacted]
    if not any((*origin, "*") in records for origin in origins):
        return None
    eager = set(access["eager"].get(attr, ()))
    point_dispatch = attr in {"FUNCTIONS", "TRANSFORMS", "TYPE_MAPPING"}
    word_keyword = attr == "KEYWORDS" and (
        not shape or all(re.fullmatch("str:[A-Z][A-Z0-9]*", key) for key in keys)
    )
    derived = ()
    if cls != module and attr in {"TIME_MAPPING", "FORMAT_MAPPING", "INVERSE_TIME_MAPPING"}:
        derived = access.get("format_dependencies", {}).get(attr, ())
    if not point_dispatch and not word_keyword and not derived and ("*" in eager or eager & impacted):
        return None
    if cls == module and (attr in access["aliases"] or eager or attr in _index()[3]):
        return None
    exposed, hit = set(), set()
    for origin in origins:
        exposed.update(records.get((*origin, "@"), ()))
        for key in ("*", *keys):
            hit.update(records.get((*origin, key), ()))
    if derived:
        hit.update(exposed)
        for owner in impacted:
            for name in derived:
                hit.update(records.get((owner, name, "@"), ()))
    result = {tests[i] for i in hit}
    missing = set(cov.nodes) - set(tests)
    reads = access["reads"].get(attr, {})
    if not reads:
        return None
    result.update(missing & _observed(reads, cov, missing))
    for name in derived:
        result.update(missing & _observed(access["reads"].get(name, {}), cov, missing))
    if not point_dispatch and not word_keyword:
        opaque = _observed(access["opaque"].get(attr, {}), cov)
        result.update(opaque if cls == module else opaque & {tests[i] for i in exposed})
    return result


def _value_keys(before, after):
    if not isinstance(before, ast.Dict) or not isinstance(after, ast.Dict):
        return None
    if len(before.keys) != len(after.keys):
        return None
    changed = []
    for k, v, k2, v2 in zip(before.keys, before.values, after.keys, after.values):
        if (ast.dump(k) if k else None) != (ast.dump(k2) if k2 else None):
            return None
        if ast.dump(v) != ast.dump(v2):
            token = _key(k)
            if token is None:
                return None
            changed.append(token)
    return changed


def _literal_edits(before, after):
    if type(before) is not type(after):
        if isinstance(before, (ast.Constant, ast.UnaryOp)) and isinstance(after, (ast.Constant, ast.UnaryOp)):
            try:
                a, b = ast.literal_eval(before), ast.literal_eval(after)
                return type(a) is type(b) and isinstance(a, (int, float))
            except ValueError:
                pass
        return False
    if isinstance(before, ast.Constant):
        return type(before.value) is type(after.value)
    if isinstance(before, list):
        return len(before) == len(after) and all(_literal_edits(x, y) for x, y in zip(before, after))
    if isinstance(before, ast.AST):
        return all(_literal_edits(a, getattr(after, field)) for field, a in ast.iter_fields(before))
    return before == after


@cache
def _module_references(root, attr):
    refs = defaultdict(set)
    word = re.compile(rb"\b" + re.escape(attr.encode()) + rb"\b")
    for folder in ("sqlglot", "tests"):
        for path in (root / folder).rglob("*.py"):
            source = path.read_bytes()
            if not word.search(source):
                continue
            relative = path.relative_to(root).as_posix()
            tree = ast.parse(source, relative)
            todo = [(tree, False, tree)]
            while todo:
                node, runtime, stmt = todo.pop()
                if isinstance(node, ast.stmt):
                    stmt = node
                if isinstance(node, ast.alias) and node.name == attr and node.asname:
                    return None
                symbol = (
                    isinstance(node, ast.Name) and node.id == attr and isinstance(node.ctx, ast.Load)
                    or isinstance(node, ast.Attribute) and node.attr == attr and isinstance(node.ctx, ast.Load)
                    or isinstance(node, ast.Constant) and node.value == attr
                )
                if symbol:
                    if not runtime:
                        return None
                    refs[relative].update((node.lineno, stmt.lineno))
                for field, value in ast.iter_fields(node):
                    active = runtime
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                        active = field == "body"
                    if isinstance(value, list):
                        todo.extend((item, active, stmt) for item in value if isinstance(item, ast.AST))
                    elif isinstance(value, ast.AST):
                        todo.append((value, active, stmt))
    return refs or None


def _declaration_trees(path, original, patched):
    for text in (original, patched):
        if text.startswith("\ufeff") or any("coding" in line for line in text.split("\n", 2)[:2]):
            break
    else:
        try:
            return _baseline(path, original)[0], patched_tree(path, patched)
        except (SyntaxError, ValueError, RecursionError, OverflowError):
            pass
    return ast.parse(original), ast.parse(patched)


def _mask_values(scope, changed):
    result = copy.copy(scope)
    result.body = []
    for stmt in scope.body:
        if stmt in changed:
            stmt = copy.copy(stmt)
            stmt.value = ast.Constant(None)
        elif isinstance(stmt, ast.ClassDef):
            stmt = _mask_values(stmt, changed)
        result.body.append(stmt)
    return result


@cache
def _private_name_unused(root, path, name):
    if not (root / path).is_file() or not (root / "tests").is_dir():
        return False
    token = name.encode()
    for folder in ("sqlglot", "tests"):
        for file in (root / folder).rglob("*.py"):
            source = file.read_bytes()
            if token not in source:
                continue
            if file == root / path:
                return False
            tree = ast.parse(source)

            for cls in (stmt for stmt in tree.body if isinstance(stmt, ast.ClassDef)):
                if any(isinstance(stmt, ast.FunctionDef) and stmt.name == name for stmt in cls.body):
                    for node in ast.walk(cls):
                        if isinstance(node, ast.FunctionDef) and node.name == name:
                            node.name = "_unrelated"
                        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                                and node.value.id == "self" and node.attr == name):
                            node.attr = "_unrelated"
            if name in ast.unparse(tree):
                return False
    return True


def _private_transform_helper(path, old, new, root, cov):
    if not path.startswith("sqlglot/generators/"):
        return None
    existing = {node.name for node in old.body if isinstance(node, ast.FunctionDef)}
    helpers = [node for node in new.body if isinstance(node, ast.FunctionDef) and node.name not in existing]
    if len(helpers) != 1:
        return None
    helper, = helpers
    if (not re.fullmatch(r"_[a-z][a-z0-9_]*", helper.name) or helper.decorator_list or helper.args.defaults
            or any(helper.args.kw_defaults) or helper.type_params
            or not any(isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__"
                       and any(name.name == "annotations" for name in stmt.names) for stmt in old.body)):
        return None
    if ([v.arg for v in (*helper.args.posonlyargs, *helper.args.args)] != ["self", "expression"]
            or helper.args.kwonlyargs or helper.args.vararg or helper.args.kwarg):
        return None
    allowed_calls = {"self.func", "expression.sql_name", "expression.args.get", "exp.If", "exp.Literal.number"}
    body = ast.Module(helper.body, [])
    allowed = (ast.Module, ast.If, ast.Return, ast.Assign, ast.Name, ast.Load, ast.Store, ast.Attribute,
               ast.Constant, ast.Call, ast.keyword, ast.Compare, ast.Lt)
    locals_ = {n.id for n in ast.walk(body) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    if locals_ & {"self", "expression", "exp"}:
        return None
    for node in ast.walk(body):
        if not isinstance(node, allowed):
            return None
        if isinstance(node, ast.Name) and node.id not in locals_ | {"self", "expression", "exp"}:
            return None
        if isinstance(node, ast.Attribute) and (not isinstance(node.ctx, ast.Load) or node.attr.startswith("__")):
            return None
        if isinstance(node, ast.Call) and ast.unparse(node.func) not in allowed_calls:
            return None

    if not _private_name_unused(root, path, helper.name):
        return None
    replacements = []
    for cls in new.body:
        if not isinstance(cls, ast.ClassDef) or helper.lineno >= cls.lineno:
            continue
        for stmt in cls.body:
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "TRANSFORMS"
                    and isinstance(stmt.value, ast.Dict)):
                continue
            for i, (key, value) in enumerate(zip(stmt.value.keys, stmt.value.values)):
                if isinstance(value, ast.Name) and value.id == helper.name:
                    if not (isinstance(key, ast.Attribute) and isinstance(key.value, ast.Name) and key.value.id == "exp"):
                        return None
                    replacements.append((cls.name, stmt.value, i, key.attr))
    if not replacements:
        return None
    owners = {owner for owner, table, index, key in replacements}
    if len(owners) != 1:
        return None
    compile(new, path, "exec", dont_inherit=True)
    for _, table, i, _ in reversed(replacements):
        del table.keys[i], table.values[i]
    new.body.remove(helper)
    if ast.dump(old) != ast.dump(new):
        return None

    try:
        from sqlglot import exp
    except ImportError:
        return None
    for _, _, _, key in replacements:
        expression = vars(exp).get(key)
        if not isinstance(expression, type) or not issubclass(expression, exp.Expr) or expression.__name__ != key:
            return None
    module = path[:-3].replace("/", ".")
    cls = module + "." + owners.pop()
    descendants, _, _, _, copies = _index()
    impacted = descendants.get(cls)
    if not impacted or any(copies.get((owner, "TRANSFORMS")) for owner in impacted):
        return None
    _, records, aliases, _ = _lookup_index()
    if any((*aliases.get((owner, "TRANSFORMS"), (owner, "TRANSFORMS")), "*") not in records for owner in impacted):
        return None
    keys = {"member:" + key for _, _, _, key in replacements}
    return _lookup_candidates(cls, "TRANSFORMS", impacted, keys, True, module, cov)












_RENDERER_TRANSFORMS = {
    ("sqlglot/transforms.py", "eliminate_distinct_on"): ("c53660809a3fdfdd", {"Select"}),
    ("sqlglot/transforms.py", "eliminate_qualify"): ("76e632f05e358214", {"Select"}),
    ("sqlglot/transforms.py", "eliminate_semi_and_anti_joins"): ("5bceec839cfc035f", set()),
    ("sqlglot/transforms.py", "eliminate_window_clause"): ("9594ed1aa85dfdcc", set()),
    ("sqlglot/transforms.py", "unqualify_unnest"): ("61ba20fbf7b3ecbd", set()),
    ("sqlglot/transforms.py", "unnest_generate_date_array_using_recursive_cte"): ("5d98aac2db3942dd", set()),
    ("sqlglot/transforms.py", "add_within_group_for_percentiles"): ("fe1e12e1c2a3dd9e", {"WithinGroup"}),
    ("sqlglot/transforms.py", "remove_unique_constraints"): ("5c75bac191573ecb", set()),
    ("sqlglot/transforms.py", "ctas_with_tmp_tables_to_create_tmp_view"): ("27e7fa7d6c38f0ae", {"Create"}),
    ("sqlglot/generators/sqlite.py", "_transform_create"): ("99e630d6b1c37e16", set()),
    ("sqlglot/generators/sqlite.py", "_generated_to_auto_increment"): ("ff8e5eef66816037", set()),
    ("sqlglot/generators/postgres.py", "_auto_increment_to_serial"): ("fca81908700c2c5f", set()),
    ("sqlglot/generators/postgres.py", "_serial_to_generated"): ("dcfec37c3d7d5323", set()),
}
_EXPR_ROOT = "sqlglot.expressions.core.Expr"
_CONNECTOR = "sqlglot.expressions.core.Connector"


def _ast_digest(node):
    import hashlib
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()[:16]


def _future_annotations(tree):
    return any(isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__"
               and any(alias.name == "annotations" for alias in stmt.names) for stmt in tree.body)


def _exp_member(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "exp":
        return node.attr
    return None


def _inert_renderer(value):
    """A removed table value whose evaluation at class creation had no effect besides building it."""
    if isinstance(value, ast.Lambda):
        return not (value.args.defaults or any(value.args.kw_defaults))
    if isinstance(value, ast.Call):
        return (isinstance(value.func, ast.Name) and value.func.id == "rename_func" and len(value.args) == 1
                and not value.keywords and isinstance(value.args[0], ast.Constant)
                and type(value.args[0].value) is str)
    return False

















_PURE_ATTRS = frozenset({"this", "expression", "expressions", "args", "is_int"})
_PURE_HELPERS = frozenset({"func", "_jsonobject_sql"})
_PURE_BUILTINS = frozenset({"len", "isinstance", "any"})
_PURE_COMPARE = (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)


def _pure_value(node, ctx):
    kind = type(node)
    if kind is ast.Constant:
        return type(node.value) in (str, int, float, bool, type(None))
    if kind is ast.Name:
        return isinstance(node.ctx, ast.Load) and node.id in ctx["readable"]
    if kind is ast.Attribute:
        if not isinstance(node.ctx, ast.Load):
            return False
        if isinstance(node.value, ast.Name) and node.value.id == "exp":
            return ctx["exp"] and node.attr in ctx["exp_classes"]
        return node.attr in _PURE_ATTRS and _pure_value(node.value, ctx)
    if kind is ast.Subscript:

        return (isinstance(node.ctx, ast.Load) and isinstance(node.slice, ast.Constant)
                and type(node.slice.value) is int and _list_value(node.value, ctx))
    if kind is ast.BoolOp:
        return all(_pure_value(value, ctx) for value in node.values)
    if kind is ast.UnaryOp:
        return isinstance(node.op, ast.Not) and _pure_value(node.operand, ctx)
    if kind is ast.Compare:

        operands = [node.left, *node.comparators]
        for op, left, right in zip(node.ops, operands, operands[1:]):
            if not isinstance(op, _PURE_COMPARE):
                return False
            if isinstance(op, (ast.Eq, ast.NotEq)) and not (isinstance(left, ast.Constant)
                                                             or isinstance(right, ast.Constant)):
                return False
        return all(_pure_value(value, ctx) for value in operands)
    if kind is ast.IfExp:
        return all(_pure_value(value, ctx) for value in (node.test, node.body, node.orelse))
    if kind in (ast.List, ast.Tuple):
        return isinstance(node.ctx, ast.Load) and all(_pure_value(value, ctx) for value in node.elts)
    if kind is ast.JoinedStr:

        return all((isinstance(part, ast.Constant) and type(part.value) is str)
                   or (isinstance(part, ast.FormattedValue) and part.conversion == -1 and part.format_spec is None
                       and isinstance(part.value, ast.Name) and part.value.id in ctx["str_names"]
                       and _pure_value(part.value, ctx))
                   for part in node.values)
    if kind is ast.Call:
        return _pure_call(node, ctx)
    return False


def _list_value(node, ctx):
    """A Python list: `<value>.expressions` or a local only ever bound to such a value or a list display."""
    if isinstance(node, ast.Name):
        return node.id in ctx["list_names"] and _pure_value(node, ctx)
    return (isinstance(node, ast.Attribute) and node.attr == "expressions" and isinstance(node.ctx, ast.Load)
            and _pure_value(node, ctx))


def _typed_locals(body, generator):
    """(str_names, list_names): locals every one of whose bindings is a str result, or a list, respectively."""
    bindings = {}
    for node in ast.walk(ast.Module(body=list(body), type_ignores=[])):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            bindings.setdefault(node.targets[0].id, []).append(node.value)

    def is_str(value):
        if isinstance(value, ast.Constant):
            return type(value.value) is str
        if isinstance(value, ast.JoinedStr):
            return True
        if isinstance(value, ast.IfExp):
            return is_str(value.body) and is_str(value.orelse)
        return (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name) and value.func.value.id == generator
                and value.func.attr in _PURE_HELPERS)

    def is_list(value):
        return (isinstance(value, (ast.List, ast.Tuple))
                or (isinstance(value, ast.Attribute) and value.attr == "expressions"))

    str_names = {name for name, values in bindings.items() if all(is_str(v) for v in values)}
    list_names = {name for name, values in bindings.items() if all(is_list(v) for v in values)}
    return str_names, list_names


def _pure_call(node, ctx):
    if node.keywords:
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return (func.id in _PURE_BUILTINS and func.id not in ctx["shadowed"] and func.id not in ctx["readable"]
                and all(not isinstance(arg, ast.Starred) and _pure_value(arg, ctx) for arg in node.args))
    if not isinstance(func, ast.Attribute):
        return False
    owner = func.value
    if isinstance(owner, ast.Name) and owner.id == ctx["generator"]:
        if func.attr not in _PURE_HELPERS or not node.args:
            return False
        if func.attr == "func" and not (isinstance(node.args[0], ast.Constant) and type(node.args[0].value) is str):
            return False
        return all(_list_value(arg.value, ctx) if isinstance(arg, ast.Starred) else _pure_value(arg, ctx)
                   for arg in node.args)
    if (isinstance(owner, ast.Attribute) and owner.attr == "args" and isinstance(owner.ctx, ast.Load)
            and _pure_value(owner.value, ctx)):
        if func.attr == "values":
            return not node.args
        if func.attr == "get":
            return (len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
                    and type(node.args[0].value) is str)
    return False


def _pure_block(body, ctx, docstring=False):
    for index, stmt in enumerate(body):
        if isinstance(stmt, ast.Expr):
            if not (docstring and index == 0 and isinstance(stmt.value, ast.Constant)
                    and type(stmt.value.value) is str):
                return False
        elif isinstance(stmt, ast.Assign):
            if (len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name) or stmt.type_comment
                    or stmt.targets[0].id in ctx["fixed"] or not _pure_value(stmt.value, ctx)):
                return False
            ctx["readable"].add(stmt.targets[0].id)
        elif isinstance(stmt, ast.If):
            if not (_pure_value(stmt.test, ctx) and _pure_block(stmt.body, ctx) and _pure_block(stmt.orelse, ctx)):
                return False
        elif isinstance(stmt, ast.Return):
            if stmt.value is not None and not _pure_value(stmt.value, ctx):
                return False
        else:
            return False
    return True


def _renderer_scope(tree, parents):
    """Names a renderer body may rely on in this module, or None when the module rebinds them."""
    bound = {}
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.setdefault(stmt.name, []).append(stmt)
            continue

        for node in ast.walk(stmt):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name == "*":
                        return None
                    bound.setdefault(alias.asname or alias.name.split(".")[0], []).append(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.setdefault(node.name, []).append(node)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.setdefault(node.id, []).append(node)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                return None
    def imported_once(name, module):
        found = bound.get(name, [])
        return (len(found) == 1 and isinstance(found[0], ast.ImportFrom) and found[0].module == module
                and not found[0].level and any(a.name == name and a.asname is None for a in found[0].names))

    if not imported_once("exp", "sqlglot") or bound.keys() & _PURE_BUILTINS:
        return None
    classes = frozenset(owner.rsplit(".", 1)[-1] for owner, bases in parents.items() if _EXPR_ROOT in bases)

    return {"shadowed": frozenset(bound), "exp_classes": classes,
            "rename_func": imported_once("rename_func", "sqlglot.dialects.dialect"),
            "generator": imported_once("generator", "sqlglot")}


def _pure_removed(value, scope):
    if isinstance(value, ast.Lambda):
        return _pure_lambda(value, scope)
    return scope["rename_func"]


_BASE_TABLE = ast.dump(ast.parse("generator.Generator.TRANSFORMS.items()", mode="eval").body)


@cache
def _base_transforms(root):
    """Pinned `Generator.TRANSFORMS` entries written after its last `**` spread, with the module's scope."""
    tree = ast.parse((root / "sqlglot/generator.py").read_bytes())
    cls = next((s for s in tree.body if isinstance(s, ast.ClassDef) and s.name == "Generator"), None)
    tables = [s.value for s in (cls.body if cls else ())
              if (isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
                  and s.targets[0].id == "TRANSFORMS")
              or (isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and s.target.id == "TRANSFORMS")]
    scope = _renderer_scope(tree, _parents())
    if len(tables) != 1 or not isinstance(tables[0], ast.Dict) or scope is None:
        return None
    explicit = {}
    for key, value in zip(tables[0].keys, tables[0].values):
        if key is None:
            explicit.clear()
            continue
        member = _exp_member(key)
        if member is None:
            return None
        explicit[member] = value
    return explicit, scope


def _inherited_pure(table, member, root, module_scope):
    """A key newly excluded from the inherited base table: its pinned renderer stops running too.

    Only `generator.Generator.TRANSFORMS` is supported, and only for a key written explicitly after the base
    table's last spread, so the inherited value is known.
    """
    if table != _BASE_TABLE or not module_scope["generator"]:
        return False
    base = _base_transforms(root)
    if base is None:
        return False
    entries, scope = base
    value = entries.get(member)
    return value is not None and _inert_renderer(value) and _pure_removed(value, scope)


def _pure_callable(params, body, scope, docstring=False):
    """Whether a renderer with these parameter names (generator first) and body fits the read-only grammar."""
    if len(set(params)) != len(params) or "exp" in params or _PURE_BUILTINS & set(params):
        return False
    generator = params[0] if params else None
    fixed = set(params) | {"exp"} | _PURE_BUILTINS
    str_names, list_names = _typed_locals(body, generator) if isinstance(body, list) else (set(), set())
    ctx = {"generator": generator, "readable": set(params[1:]), "fixed": fixed, "exp": True,
           "shadowed": scope["shadowed"] - {"exp"}, "exp_classes": scope["exp_classes"],
           "str_names": str_names - fixed, "list_names": list_names - fixed}
    if isinstance(body, list):
        return _pure_block(body, ctx, docstring=docstring)
    return _pure_value(body, ctx)


def _pure_lambda(node, scope):
    args = node.args
    if args.posonlyargs or args.kwonlyargs or args.kwarg or args.defaults or any(args.kw_defaults):
        return False
    params = [arg.arg for arg in args.args]
    if args.vararg is not None:
        params.append(args.vararg.arg)
    return _pure_callable(params, node.body, scope)


def _expression_annotation(node, scope):
    if node is None:
        return True
    if isinstance(node, ast.Constant) and type(node.value) is str:
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return False
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "exp"
            and node.attr in scope["exp_classes"])


def _pure_method(method, scope):
    """An added `<key>_sql(self, expression)` whose annotations and body fit the read-only grammar."""
    first, second = method.args.args
    returns = method.returns
    if (first.annotation is not None or first.type_comment or second.type_comment or method.type_comment
            or not _expression_annotation(second.annotation, scope)
            or not (returns is None or (isinstance(returns, ast.Name) and returns.id == "str")
                    or (isinstance(returns, ast.Constant) and returns.value == "str"))):
        return False
    return _pure_callable([first.arg, second.arg], method.body, scope, docstring=True)


def _excluded_keys(comprehension):
    """`{k: v for k, v in X.TRANSFORMS.items() if k != exp.A}` (or `not in (exp.A, ...)`): X and {A, ...}."""
    if not (isinstance(comprehension, ast.DictComp) and len(comprehension.generators) == 1):
        return None
    gen = comprehension.generators[0]
    if (gen.is_async or len(gen.ifs) != 1 or not isinstance(gen.target, ast.Tuple) or len(gen.target.elts) != 2
            or not all(isinstance(e, ast.Name) for e in gen.target.elts)
            or not isinstance(comprehension.key, ast.Name) or not isinstance(comprehension.value, ast.Name)
            or comprehension.key.id != gen.target.elts[0].id or comprehension.value.id != gen.target.elts[1].id
            or not (isinstance(gen.iter, ast.Call) and not gen.iter.args and not gen.iter.keywords
                    and isinstance(gen.iter.func, ast.Attribute) and gen.iter.func.attr == "items")):
        return None
    cond = gen.ifs[0]
    if not (isinstance(cond, ast.Compare) and len(cond.ops) == 1 and isinstance(cond.left, ast.Name)
            and cond.left.id == comprehension.key.id):
        return None
    if isinstance(cond.ops[0], ast.NotEq):
        members = [_exp_member(cond.comparators[0])]
    elif isinstance(cond.ops[0], ast.NotIn) and isinstance(cond.comparators[0], (ast.Tuple, ast.Set, ast.List)):
        members = [_exp_member(e) for e in cond.comparators[0].elts]
    else:
        return None
    if None in members or len(set(members)) != len(members):
        return None
    return ast.dump(gen.iter), set(members)


def _removed_pure(value, pure):
    """A removed value (a default-free lambda or `rename_func("...")`) must also satisfy `pure` when given."""
    return pure is None or pure(value)


def _transform_table_edits(before, after, pure=None, inherited=None):
    """Keys a TRANSFORMS dict display loses between two versions, or None for any other kind of edit.

    With `pure`, every removed lambda must satisfy it; with `inherited`, every key newly excluded from an
    inherited table must satisfy `inherited(iterated_table, key)`.
    """
    if not isinstance(before, ast.Dict) or not isinstance(after, ast.Dict):
        return None
    old = list(zip(before.keys, before.values))
    new = list(zip(after.keys, after.values))
    removed = set()
    i = 0
    for key, value in new:
        while i < len(old):
            okey, ovalue = old[i]
            i += 1
            if (key is None) != (okey is None):
                if okey is None:
                    return None
            elif key is None:
                if ast.dump(value) == ast.dump(ovalue):
                    break
                before_excl, after_excl = _excluded_keys(ovalue), _excluded_keys(value)
                if before_excl is None or after_excl is None or before_excl[0] != after_excl[0]:
                    return None
                if not before_excl[1] <= after_excl[1]:
                    return None
                excluded = after_excl[1] - before_excl[1]
                if inherited is not None and not all(inherited(after_excl[0], name) for name in excluded):
                    return None
                removed |= excluded
                break
            elif ast.dump(key) == ast.dump(okey):
                if ast.dump(value) != ast.dump(ovalue):
                    return None
                break
            member = _exp_member(okey)
            if member is None or not _inert_renderer(ovalue) or not _removed_pure(ovalue, pure):
                return None
            removed.add(member)
        else:
            return None
    for okey, ovalue in old[i:]:
        member = _exp_member(okey)
        if okey is None or member is None or not _inert_renderer(ovalue) or not _removed_pure(ovalue, pure):
            return None
        removed.add(member)
    if any(_exp_member(k) is None for k in before.keys if k is not None):
        return None
    return removed


def _added_renderers(old_cls, new_cls):
    """New public `<key>_sql` methods and the class's TRANSFORMS pair, or None for any other class edit."""
    names = {stmt.name for stmt in old_cls.body if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))}
    added, table = [], None
    remaining = list(old_cls.body)
    for stmt in new_cls.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name not in names:
            args = stmt.args
            if (stmt.decorator_list or getattr(stmt, "type_params", None) or args.posonlyargs or args.vararg
                    or args.kwonlyargs or args.kwarg or args.defaults or len(args.args) != 2
                    or not re.fullmatch(r"[a-z][a-z0-9]*_sql", stmt.name)):
                return None
            added.append(stmt)
            continue
        if not remaining:
            return None
        previous = remaining.pop(0)
        if ast.dump(previous) == ast.dump(stmt):
            continue
        if (isinstance(stmt, ast.Assign) and isinstance(previous, ast.Assign) and len(stmt.targets) == 1
                and len(previous.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                and isinstance(previous.targets[0], ast.Name) and stmt.targets[0].id == "TRANSFORMS"
                and previous.targets[0].id == "TRANSFORMS" and table is None):
            table = previous.value, stmt.value
            continue
        return None
    if remaining or not added:
        return None
    return added, table


def _mentions(data, token, ignored_prefixes):
    """Whether `token` occurs in `data` other than right after one of the ignored prefixes."""
    start = data.find(token)
    while start != -1:
        if not any(data[max(0, start - len(p)):start] == p for p in ignored_prefixes):
            return True
        start = data.find(token, start + 1)
    return False


def _module_path(owner, root):
    parts = owner.split(".")
    for end in range(len(parts) - 1, 0, -1):
        candidate = "/".join(parts[:end]) + ".py"
        if (root / candidate).is_file():
            return candidate
    return None


def _preprocess_transforms(tree):
    """(name, reference) for every transform passed to `preprocess` in a module, None when unresolvable."""
    local = {stmt.name for stmt in tree.body if isinstance(stmt, ast.FunctionDef)}
    imported = {alias.asname or alias.name: alias.name for stmt in tree.body
                if isinstance(stmt, ast.ImportFrom) and stmt.module == "sqlglot.transforms" for alias in stmt.names}
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        named = (isinstance(func, ast.Name) and (func.id == "preprocess" or imported.get(func.id) == "preprocess"))
        if not (named or (isinstance(func, ast.Attribute) and func.attr == "preprocess")):
            continue
        if not node.args:
            return None
        if not isinstance(node.args[0], ast.List):
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
                continue
            return None
        for item in node.args[0].elts:
            if (isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name)
                    and item.value.id == "transforms"):
                found.append(("sqlglot/transforms.py", item.attr))
            elif isinstance(item, ast.Name) and item.id in local:
                found.append((None, item.id))
            elif isinstance(item, ast.Name) and item.id in imported:
                found.append(("sqlglot/transforms.py", imported[item.id]))
            else:
                return None
    return found


def _renderer_migration(path, old, new, root, cov):
    if not _future_annotations(old) or not _future_annotations(new) or len(old.body) != len(new.body):
        return None
    pair = None
    for before, after in zip(old.body, new.body):
        if ast.dump(before) == ast.dump(after):
            continue
        if (pair is not None or not isinstance(before, ast.ClassDef) or not isinstance(after, ast.ClassDef)
                or before.name != after.name
                or [ast.dump(b) for b in before.bases] != [ast.dump(b) for b in after.bases]
                or [ast.dump(k) for k in before.keywords] != [ast.dump(k) for k in after.keywords]
                or before.decorator_list or after.decorator_list
                or getattr(before, "type_params", None) or getattr(after, "type_params", None)):
            return None
        pair = before, after
    if pair is None:
        return None
    edit = _added_renderers(*pair)
    if edit is None:
        return None
    added, table = edit
    parents = _parents()
    scope = _renderer_scope(new, parents)
    if scope is None or not all(_pure_method(method, scope) for method in added):
        return None
    removed = set()
    if table is not None:
        removed = _transform_table_edits(
            *table, pure=lambda value: _pure_removed(value, scope),
            inherited=lambda iterated, member: _inherited_pure(iterated, member, root, scope))
        if removed is None:
            return None
    module = path[:-3].replace("/", ".")
    cls = f"{module}.{pair[1].name}"
    descendants, _, _, _, copies = _index()
    if cls not in descendants:
        return None
    classes = set(removed)
    for method in added:
        key = method.name[:-4]
        matches = [o for o, bases in parents.items() if _EXPR_ROOT in bases and o.rsplit(".", 1)[-1].lower() == key]
        if len(matches) != 1:
            return None
        classes.add(matches[0].rsplit(".", 1)[-1])
    for name in classes:
        owners = [o for o, bases in parents.items() if _EXPR_ROOT in bases and o.rsplit(".", 1)[-1] == name]
        if len(owners) != 1 or _CONNECTOR in parents[owners[0]]:
            return None
    impacted = set(descendants.get(cls, {cls}))
    pending = list(impacted)
    while pending:
        for target in copies.get((pending.pop(), "TRANSFORMS"), ()):
            extra = descendants[target] - impacted
            impacted.update(extra)
            pending.extend(extra)

    contributors = set()
    for owner in impacted:
        for base in parents.get(owner, (owner,)):
            if _EXPR_ROOT not in parents.get(base, ()):
                contributors.add(base)
    trees = {path: new}
    transforms_tree = None
    for owner in sorted(contributors):
        rel = _module_path(owner, root)
        if rel is None:
            return None
        tree = trees.get(rel)
        if tree is None:
            data = (root / rel).read_bytes()
            if not _mentions(data, b"preprocess(", (b"def ", b"self.")):
                continue
            tree = trees[rel] = ast.parse(data)
        refs = _preprocess_transforms(tree)
        if refs is None:
            return None
        for source, name in refs:
            source = source or rel
            if source == "sqlglot/transforms.py":
                if transforms_tree is None:
                    transforms_tree = ast.parse((root / source).read_bytes())
                defining = transforms_tree
            else:
                defining = trees.get(source)
                if defining is None:
                    defining = trees[source] = ast.parse((root / source).read_bytes())
            func = next((s for s in defining.body if isinstance(s, ast.FunctionDef) and s.name == name), None)
            audited = _RENDERER_TRANSFORMS.get((source, name))
            if func is None or audited is None or _ast_digest(func) != audited[0] or audited[1] & classes:
                return None

    names = {method.name for method in added}
    class_read = re.compile(rb"\b" + re.escape(pair[1].name.encode()) + rb"\s*\.\s*TRANSFORMS\b")
    callers = set()
    for folder in ("sqlglot", "tests"):
        for file in sorted((root / folder).rglob("*.py")):
            rel = file.relative_to(root).as_posix()
            data = file.read_bytes()
            if not (class_read.search(data) or any(_mentions(data, n.encode(), (b"def ",)) for n in names)):
                continue
            tree = new if rel == path else ast.parse(data)
            if rel != path:
                prefix = rel[:-3].replace("/", ".")
                allowed = {id(n) for stmt in tree.body if isinstance(stmt, ast.ClassDef)
                           and f"{prefix}.{stmt.name}" in impacted for n in ast.walk(stmt)}
                if any(isinstance(n, ast.Attribute) and n.attr == "TRANSFORMS" and isinstance(n.value, ast.Name)
                       and n.value.id == pair[1].name and id(n) not in allowed for n in ast.walk(tree)):
                    return None
            for node in ast.walk(tree):
                referenced = ((isinstance(node, ast.Attribute) and node.attr in names)
                              or (isinstance(node, ast.Name) and node.id in names)
                              or (isinstance(node, ast.Constant) and node.value in names))
                if referenced:
                    if not cov.present(rel):
                        return None
                    callers.update(cov.lines(rel, {node.lineno}))
    result = _lookup_candidates(cls, "TRANSFORMS", impacted, {"member:" + n for n in classes}, True, module, cov)
    if result is None:
        return None
    return result | callers


@cache
def _parents():
    path = Path(__file__).resolve().parent / "maps" / "declarations.json.gz"
    grouped = read_grouped(path)
    data = grouped[0] if grouped is not None else read_map(path)
    return {child: frozenset(bases) for child, bases in data["parents"].items()}


def declaration_candidates(path, original, patched, old_lines, cov, *, single_change=False):
    if not path.startswith("sqlglot/"):
        return None
    old, new = _declaration_trees(path, original, patched)
    if single_change and path.startswith("sqlglot/generators/") and len(new.body) == len(old.body) + 1:

        result = _private_transform_helper(
            path, old, ast.parse(patched), Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot")), cov
        )
        if isinstance(result, set):
            return result | cov.missing
    if single_change and path.startswith("sqlglot/generators/") and len(new.body) == len(old.body):
        result = _renderer_migration(path, old, new, Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot")), cov)
        if isinstance(result, set):
            return result | cov.missing
    module = path[:-3].replace("/", ".").removesuffix(".__init__")
    left = list(_declarations(old, module))
    right = list(_declarations(new, module))
    if len(left) != len(right):
        return None
    changes = []
    changed_nodes = set()
    grouped = False
    for (cls, attr, a), (cls2, attr2, b) in zip(left, right):
        if (cls, attr) != (cls2, attr2):
            return None
        if old_lines and not any(a.lineno <= line <= a.end_lineno for line in old_lines):
            continue
        before, after = a.value, b.value
        if ast.dump(before) == ast.dump(after):
            continue
        keys = _value_keys(before, after)
        extended = None
        container = before.right if isinstance(before, ast.BinOp) else before
        if not _literal_edits(before, after) or (isinstance(container, (ast.Dict, ast.Set)) and not keys):
            symbols = _lookup_index()[3]["symbols"]
            extended = _entry_edits(before, after, symbols)
            if (extended is None and single_change and attr == "FUNCTIONS"
                    and cls == "sqlglot.parsers.duckdb.DuckDBParser"
                    and hashlib.sha256(original.encode()).hexdigest() == _DUCKDB_PARSER):
                extended = _grouped_function_edits(before, after)
                if extended is not None:
                    grouped = True
            if extended is None and not (
                isinstance(before, ast.Attribute) and isinstance(after, ast.Attribute)
                and ast.dump(before.value) == ast.dump(after.value)
                and _safe_reference(after, {ast.unparse(before).split(".")[0]}, symbols)
                or isinstance(before, ast.Constant) and isinstance(after, ast.Constant)
            ):
                return None
        if attr in {"TIME_MAPPING", "FORMAT_MAPPING", "INVERSE_TIME_MAPPING"}:
            if not isinstance(after, ast.Dict) or any(
                not isinstance(k, ast.Constant) or type(k.value) is not str
                or not isinstance(v, ast.Constant) or type(v.value) is not str
                for k, v in zip(after.keys, after.values) if k is not None
            ):
                return None
        changes.append((cls, attr, keys, extended))
        changed_nodes.update((a, b))
    if not changes or ast.dump(_mask_values(old, changed_nodes)) != ast.dump(_mask_values(new, changed_nodes)):
        return None

    descendants, records, tests, cached, copies = _index()
    names = set(cov.nodes).difference(tests)
    if grouped:
        if len(changes) != 1:
            return None
        names.update(_opaque_dispatch_readers(Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot"))))
    hits = set()
    for cls, attr, keys, extended in changes:
        if cls == module and not extended:
            if not attr.isupper():
                return None
            if keys:
                result = _lookup_candidates(cls, attr, {cls}, keys, False, module, cov)
                if result is not None:
                    names.update(result)
                    continue
            root = Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot"))
            refs = _module_references(root, attr)
            if not refs:
                return None
            for filename, lines in refs.items():
                names.update(cov.lines(filename, lines))
            continue
        if cls != module and cls not in descendants:
            return None
        impacted = set(descendants.get(cls, {cls}))
        pending = list(impacted)
        while pending:
            for target in copies.get((pending.pop(), attr), ()):
                additions = descendants[target] - impacted
                impacted.update(additions)
                pending.extend(additions)
        if extended and attr != "arg_types":
            changed_keys, shape = extended
            if not shape and (cls, attr, "*") in records:
                keys = changed_keys
            else:
                result = _lookup_candidates(cls, attr, impacted, changed_keys, shape, module, cov)
                if result is None:
                    return None
                names.update(result)
                continue
        if keys and (cls, attr, "*") in records:
            for child in impacted:
                for key in ["*", *keys]:
                    hits.update(records.get((child, attr, key), ()))
        elif attr == "arg_types" and (cls, attr) in records:
            for child in impacted:
                hits.update(records.get((child,), ()))
                for name in ("arg_types", "required_args"):
                    hits.update(records.get((child, name), ()))
        elif (cls, attr) in records and attr not in cached:
            hits.update(records[cls, attr])
        elif keys:
            result = _lookup_candidates(cls, attr, impacted, keys, False, module, cov)
            if result is None:
                return None
            names.update(result)
        else:
            return None
    return names | {tests[i] for i in hits} | cov.missing
