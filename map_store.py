import gzip
import hashlib
import json
import marshal
import os
from functools import cache
from pathlib import Path


MAPS = Path(__file__).resolve().parent / "maps"
PACKED = MAPS.parent / "packed_maps"


@cache
def expression_consumer_profile():
    return json.loads((MAPS.parent / "expression_consumers.json").read_bytes())


def tc_fixture_bytes(root):
    directory = Path(root) / "tests" / "fixtures" / "optimizer"
    return [(directory / suite / (suite + ".sql")).read_text(encoding="utf-8").encode("utf-8")
            for suite in ("tpc-h", "tpc-ds")]


def decode_parse_profile(contents, root=None):
    data = json.loads(gzip.decompress(contents))
    if "fixture_sha256" not in data:
        return data
    fixtures = tc_fixture_bytes(root or os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot"))
    if [hashlib.sha256(f).hexdigest() for f in fixtures] != data.pop("fixture_sha256"):
        raise ValueError("TPC fixture content differs from the recorded parse profile")
    for call, _, _ in data["rows"]:
        if isinstance(call[1], list):
            index, offset, size = call[1]
            if not (0 <= index < len(fixtures) and 0 <= offset < offset + size <= len(fixtures[index])):
                raise ValueError("TPC fixture reference outside recorded file")
            call[1] = fixtures[index][offset:offset + size].decode("utf-8")
    return data


def read_json_map(path):
    if path.name == "tc_parse.json.gz":
        return decode_parse_profile(path.read_bytes())
    with gzip.open(path, "rb") as stream:
        return json.load(stream)


def _packed_path(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return PACKED / f"{path.stem}-{digest}.marshal"


def read_map(path):
    if path.parent == MAPS and PACKED.is_dir():
        try:
            return marshal.loads(_packed_path(path).read_bytes(), allow_code=False)
        except (OSError, EOFError, ValueError, TypeError):
            pass
    return read_json_map(path)



GROUPED = {"declarations.json.gz": 1, "declarations-lookups.json.gz": 2}
_MISSING = object()


class GroupedRecords:
    """Read-only mapping equal to {tuple(row[:-1]): row[-1] for row in records}, loaded per key prefix."""

    def __init__(self, spans, blob, width):
        self._spans = spans
        self._blob = blob
        self._width = width
        self._loaded = {}

    def _group(self, key):
        prefix = key[:self._width]
        table = self._loaded.get(prefix)
        if table is None:
            span = self._spans.get(prefix)
            if span is None:
                table = {}
            else:
                with open(self._blob, "rb") as stream:
                    stream.seek(span[0])
                    table = marshal.loads(stream.read(span[1]), allow_code=False)
            self._loaded[prefix] = table
        return table

    def get(self, key, default=None):
        if type(key) is not tuple or len(key) < self._width:
            return default
        return self._group(key).get(key, default)

    def __contains__(self, key):
        return self.get(key, _MISSING) is not _MISSING

    def __getitem__(self, key):
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value


def _grouped_paths(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return PACKED / f"{path.stem}-{digest}.head", PACKED / f"{path.stem}-{digest}.groups"


def write_grouped(path, width):
    data = read_json_map(path)
    groups = {}
    for row in data["records"]:
        key = tuple(row[:-1])
        groups.setdefault(key[:width], {})[key] = row[-1]
    blob = bytearray()
    spans = {}
    for prefix, table in groups.items():
        chunk = marshal.dumps(table)
        spans[prefix] = (len(blob), len(chunk))
        blob += chunk
    head = {key: value for key, value in data.items() if key != "records"}
    head["@spans"] = spans
    head["@width"] = width
    head_path, blob_path = _grouped_paths(path)
    blob_path.write_bytes(bytes(blob))
    head_path.write_bytes(marshal.dumps(head))


def read_grouped(path):
    """Return (head, records) for a grouped map, or None when no matching grouped pack exists."""
    if path.parent != MAPS or path.name not in GROUPED or not PACKED.is_dir():
        return None
    try:
        head_path, blob_path = _grouped_paths(path)
        head = marshal.loads(head_path.read_bytes(), allow_code=False)
        if head.get("@width") != GROUPED[path.name] or not blob_path.is_file():
            return None
    except (OSError, EOFError, ValueError, TypeError):
        return None
    return head, GroupedRecords(head["@spans"], blob_path, head["@width"])


if __name__ == "__main__":
    PACKED.mkdir(exist_ok=True)
    for path in MAPS.glob("*.json.gz"):
        _packed_path(path).write_bytes(marshal.dumps(read_json_map(path)))
        if path.name in GROUPED:
            write_grouped(path, GROUPED[path.name])
