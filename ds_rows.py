"""Per-entry state rows of loaded sqlglot modules and classes.

A row is one module attribute ("m <module> <name>") or one class attribute ("c <module>.<qualname> <name>").
Values are frozen with the front-end replay's freezer, one freezer per row. Sets and frozensets are compared as
sets, whatever their iteration order, because that order differs between source-compiled and cached modules.
Enum members carry their values. Dictionary rows also keep one digest per key, so an edit that only adds keys
can be told apart from one that changes an existing entry. Generator dispatch tables, which sqlglot builds on
first use and caches for the rest of the process, are reported as virtual rows ("d <generator class>").
"""
import enum
import hashlib
import json
import sys
import types

import fe_replay as fr

SKIP_CLASS = fr.CLASS_SKIP | {"__slots__", "__match_args__", "_abc_impl", "__abstractmethods__", "__hash__",
                              "__init_subclass__", "__class_getitem__", "__orig_bases__", "__parameters__"}


def sqlglot_modules():
    return sorted(name for name, module in list(sys.modules.items())
                  if module is not None and (name == "sqlglot" or name.startswith("sqlglot.")))


def classes_of(module):
    """Classes defined in a module, including nested classes reachable through class attributes."""
    found, queue, seen = [], [], set()
    for value in list(vars(module).values()):
        if isinstance(value, type) and value.__module__ == module.__name__:
            queue.append(value)
    while queue:
        cls = queue.pop()
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        found.append(cls)
        for value in list(vars(cls).values()):
            if isinstance(value, type) and value.__module__ == module.__name__ and id(value) not in seen:
                queue.append(value)
    return sorted(found, key=lambda c: c.__qualname__)


def class_name(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


DIGEST_VERSION = 2


def _canon(value, freeze, arbitrary):
    """Freeze with sets in a canonical order, whatever their element types, and enum members with values.

    Containers that sqlglot builds from sets of classes (fe_replay.ARBITRARY_ORDER) have a process-dependent order
    and are compared as sorted collections. Elements are ordered by their own digest, computed with a fresh freezer,
    before the shared freezer numbers any object they reference, so the result does not depend on iteration order."""
    def sub(v):
        return _canon(v, freeze, arbitrary)

    def ordered(values):
        return sorted(values, key=lambda v: digest(v, arbitrary))

    if isinstance(value, (set, frozenset)):
        return ["S*", type(value).__name__, [sub(v) for v in ordered(value)]]
    if isinstance(value, enum.Enum):
        return ["E*", class_name(type(value)), value.name, repr(value.value)]
    if isinstance(value, dict):
        items = list(value.items())
        if id(value) in arbitrary:
            items.sort(key=lambda item: digest(item[0], arbitrary))
        return ["D*", type(value).__name__, [[sub(k), sub(v)] for k, v in items]]
    if isinstance(value, (list, tuple)) and type(value) in (list, tuple):
        if id(value) in arbitrary:
            return [type(value).__name__ + "*", [sub(v) for v in ordered(value)]]
        return [type(value).__name__, [sub(v) for v in value]]
    return freeze(value)


def digest(value, arbitrary):
    freeze, objects, _ = fr._freezer(False, arbitrary)
    head = _canon(value, freeze, arbitrary)
    return hashlib.sha256(json.dumps([head, objects], sort_keys=True, separators=(",", ":"),
                                     default=repr).encode("utf-8", "surrogatepass")).hexdigest()[:24]


def key_digests(value, arbitrary):
    out = {}
    for k, v in value.items():
        out[digest(k, arbitrary)] = digest(v, arbitrary)
    return out


def dispatch_tables():
    gen = sys.modules.get("sqlglot.generator")
    if gen is None:
        return {}
    tables = {}
    for module in [sys.modules[n] for n in sqlglot_modules()]:
        for cls in classes_of(module):
            if isinstance(cls, type) and issubclass(cls, gen.Generator):
                tables[class_name(cls)] = gen._build_dispatch(cls)
    return tables


def snapshot():
    """Return {row: digest} and {row: {key digest: value digest}} for dictionary rows."""
    arbitrary = fr._arbitrary()
    rows, keys = {}, {}
    names = sqlglot_modules()
    rows["loaded"] = hashlib.sha256(json.dumps(names).encode()).hexdigest()[:24]
    for name in names:
        module = sys.modules[name]
        for attr, value in list(vars(module).items()):
            if attr in fr.MODULE_SKIP:
                continue
            row = f"m {name} {attr}"
            rows[row] = digest(value, arbitrary)
            if type(value) is dict:
                keys[row] = key_digests(value, arbitrary)
        for cls in classes_of(module):
            prefix = f"c {class_name(cls)}"
            rows[f"{prefix} __mro__"] = hashlib.sha256(json.dumps(
                [class_name(b) for b in cls.__mro__[1:]]).encode()).hexdigest()[:24]
            if issubclass(cls, enum.Enum):
                rows[f"{prefix} __members__"] = digest(
                    [[m.name, repr(m.value)] for m in cls.__members__.values()], arbitrary)
            for attr, value in list(vars(cls).items()):
                if attr in SKIP_CLASS or isinstance(value, (types.MemberDescriptorType, types.GetSetDescriptorType)):
                    continue
                row = f"{prefix} {attr}"
                rows[row] = digest(value, arbitrary)
                if type(value) is dict:
                    keys[row] = key_digests(value, arbitrary)
    for name, table in dispatch_tables().items():
        row = f"d {name}"
        rows[row] = digest(table, arbitrary)
        keys[row] = key_digests(table, arbitrary)
    import gc
    import os
    import warnings
    rows["process"] = hashlib.sha256(json.dumps([
        sys.getrecursionlimit(), sys.getswitchinterval(), gc.isenabled(), len(warnings.filters),
        repr(sys.modules["decimal"].getcontext()) if "decimal" in sys.modules else None,
        sorted(os.environ.items()), list(sys.path), sorted(map(repr, __import__("copyreg").dispatch_table)),
    ], default=repr).encode()).hexdigest()[:24]
    return rows, keys


def delta(rows, keys, base_rows, base_keys):
    """Changed rows, with key-level detail for dictionary rows present on both sides."""
    changed = {}
    for row in sorted(set(rows) | set(base_rows)):
        a, b = base_rows.get(row), rows.get(row)
        if a == b:
            continue
        if a is None:
            changed[row] = {"kind": "added"}
        elif b is None:
            changed[row] = {"kind": "removed"}
        elif row in keys and row in base_keys:
            old, new = base_keys[row], keys[row]
            if old == new:
                changed[row] = {"kind": "order"}
                continue
            changed[row] = {"kind": "keys",
                            "added": sorted(set(new) - set(old)),
                            "removed": sorted(set(old) - set(new)),
                            "changed": sorted(k for k in set(old) & set(new) if old[k] != new[k])}
        else:
            changed[row] = {"kind": "value"}
    return changed
