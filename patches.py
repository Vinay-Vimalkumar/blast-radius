import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


@dataclass
class Change:
    path: str
    original: str
    patched: str
    removed: set[int] = field(default_factory=set)
    added: set[int] = field(default_factory=set)
    structural: bool = False


HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _path(header, prefix):
    name = header[4:].split("\t", 1)[0]
    if name == "/dev/null":
        return None
    if not name.startswith(prefix):
        raise ValueError("unsupported diff path")
    name = name[len(prefix):]
    parts = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or parts.is_absolute() or ".." in parts.parts:
        raise ValueError("invalid path")
    return name


def read_changes(diff: str, root: Path) -> list[Change]:
    lines = diff.replace("\r\n", "\n").split("\n")
    result = []
    i = 0


    bare = False
    while i < len(lines):
        if not lines[i].startswith("--- "):
            if lines[i].startswith(("GIT binary patch", "Binary files ", "rename from ", "copy from ")):
                raise ValueError("non-text change")
            if lines[i].startswith("diff --git "):
                if bare:
                    raise ValueError("file added or deleted without content")
            elif lines[i].startswith(("new file mode ", "deleted file mode ")):
                bare = True
            i += 1
            continue
        bare = False
        before = _path(lines[i], "a/")
        after = _path(lines[i + 1], "b/")
        if not lines[i + 1].startswith("+++ ") or (before is None and after is None):
            raise ValueError("missing file header")
        if before:
            baseline = (root / before).resolve()
            if not baseline.is_relative_to(root.resolve()):
                raise ValueError("path outside checkout")
            original = baseline.read_text(encoding="utf-8")
        else:
            original = ""
        source = original.split("\n")
        if source[-1] == "":
            source.pop()
        change = Change(after or before, original, "", structural=before != after)
        new = []
        cursor = 0
        ends_in_newline = original.endswith("\n") if before else True
        i += 2
        while i < len(lines) and (hunk := HUNK.match(lines[i])):
            old_start, old_count, _, new_count = hunk.groups()
            old_left = int(old_count if old_count is not None else 1)
            new_left = int(new_count if new_count is not None else 1)
            start = int(old_start) - int(old_left > 0)
            if not cursor <= start <= len(source):
                raise ValueError("hunk outside source")
            new.extend(source[cursor:start])
            cursor = start
            last = ""
            i += 1
            while old_left or new_left or (i < len(lines) and lines[i].startswith("\\ No newline")):
                row = lines[i]
                i += 1
                if row.startswith("\\ No newline"):
                    if last in {" ", "+"}:
                        ends_in_newline = False
                    elif last == "-":
                        ends_in_newline = True
                    continue
                last, value = row[:1], row[1:]
                if last not in {" ", "-", "+"}:
                    raise ValueError("bad hunk")
                if last in {" ", "-"}:
                    if old_left <= 0 or source[cursor] != value:
                        raise ValueError("base revision mismatch")
                    if last == "-":
                        change.removed.add(cursor + 1)
                    old_left -= 1
                    cursor += 1
                if last in {" ", "+"}:
                    if new_left <= 0:
                        raise ValueError("bad hunk size")
                    new_left -= 1
                    new.append(value)
                    if last == "+":
                        change.added.add(len(new))
        new.extend(source[cursor:])
        change.patched = "\n".join(new) + ("\n" if new and ends_in_newline else "")
        result.append(change)
    if bare:
        raise ValueError("file added or deleted without content")
    if not result:
        raise ValueError("no text hunks")
    return result
