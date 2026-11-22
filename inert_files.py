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
