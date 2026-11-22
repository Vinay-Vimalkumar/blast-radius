"""Recognize plain article and license text patches."""
import re

_ARTICLE = re.compile(r"posts/[A-Za-z0-9][A-Za-z0-9_.-]*\.md")
_HEADER_LINES = (
    re.compile(r"diff --git a/(\S+) b/\1"),
    re.compile(r"index [0-9a-f]{7,64}\.\.[0-9a-f]{7,64}( 100644)?"),
    re.compile(r"new file mode 100644"),
    re.compile(r"--- (a/\S+|/dev/null)"),
    re.compile(r"\+\+\+ b/\S+"),
)


_MODE = re.compile(r"(old|new) mode 100(644|755)")


def metadata_only(patch_text):
    """True for a patch that changes the content of no file: an empty diff, or only regular-file
    permission changes (git's three-line "diff --git a/X b/X", "old mode", "new mode" blocks, with no
    index line or hunk). Nothing a test imports, parses or opens depends on the execute bit."""
    lines = [line for line in patch_text.replace("\r\n", "\n").split("\n") if line.strip()]
    for start in range(0, len(lines), 3):
        block = lines[start:start + 3]
        if (len(block) != 3 or not _HEADER_LINES[0].fullmatch(block[0])
                or not block[1].startswith("old mode ") or not block[2].startswith("new mode ")
                or not _MODE.fullmatch(block[1]) or not _MODE.fullmatch(block[2])):
            return False
    return True


def safe_headers(patch_text):
    """True only if every file header in the diff is one of the plain forms above."""
    in_header = False
    for line in patch_text.replace("\r\n", "\n").split("\n"):
        if line.startswith("diff --git ") or (line.startswith("--- ") and not in_header):
            in_header = True
        if line.startswith("@@"):
            in_header = False
            continue
        if in_header and not any(p.fullmatch(line) for p in _HEADER_LINES):
            return False
    return True


def inert_change(change, cov):
    if change.path == "LICENSE":
        eligible = not change.structural and bool(change.original)
    elif change.path.startswith("posts/") and _ARTICLE.fullmatch(change.path):
        eligible = not change.structural or change.original == ""
    else:
        return False
    return eligible and bool(change.patched) and "\0" not in change.patched and not cov.fixture(change.path)
