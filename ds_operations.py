import dis
from functools import lru_cache
from pathlib import Path

BASE = Path("/opt/sqlglot")
TEST = "tests/test_executor.py::TestExecutor::test_execute_tpcds"

def candidates(code):
    if not code.co_flags & 2 or code.co_name.startswith("<"):
        return {}
    result = {}
    for item in dis.get_instructions(code):
        pos = item.positions
        if not pos or pos.lineno != pos.end_lineno or pos.col_offset is None:
            continue
        if item.opname in {"LOAD_CONST", "RETURN_CONST", "LOAD_SMALL_INT"}:
            if type(item.argval) not in {int, str}:
                continue
        elif item.opname not in {"BINARY_OP", "COMPARE_OP", "IS_OP", "CONTAINS_OP"}:
            continue
        result[item.offset] = pos.lineno
    return result

def key(code):
    return f"{code.co_firstlineno}:{code.co_qualname}"

@lru_cache(maxsize=1)
def input_digest():
    import hashlib
    sources = sorted((BASE/"sqlglot").rglob("*.py"))
    fixtures = BASE/"tests/fixtures/optimizer"
    sources.extend([BASE/"tests/test_executor.py",BASE/"tests/helpers.py",BASE/"tests/__init__.py"])
    for suite in ("tpc-h","tpc-ds"):
        sources.extend(sorted(p for p in (fixtures/suite).rglob("*") if p.is_file()))
    h = hashlib.sha256()
    for p in sources:
        h.update(str(p.relative_to(BASE)).encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()
