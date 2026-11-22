import ast
import io
import token
import tokenize

COMPARE = {"<", "<=", "==", "!=", ">=", ">", "in", "not in", "is", "is not"}
BOOL = {"and", "or"}


def _integer(text):
    if text in ("True", "False"):
        return True
    if not text or not text[0].isdigit() or any(c.isspace() for c in text):
        return False
    try:
        int(text, 0)
    except (SyntaxError, ValueError):
        return False
    return True


def _sites(source, row):
    tokens = []
    for item in tokenize.generate_tokens(io.StringIO(source).readline):
        if item.start[0] > row:
            break
        if item.start[0] == item.end[0] == row and item.type in (token.NAME, token.NUMBER, token.OP):
            tokens.append(item)
    for k, item in enumerate(tokens):
        yield item.start[1], item.end[1], item.string
        if k+1 < len(tokens):
            nxt = tokens[k+1]
            phrase = f"{item.string} {nxt.string}"
            if phrase in ("not in", "is not"):
                yield item.start[1], nxt.end[1], phrase


def single_replacement(change):
    if len(change.removed) != 1 or change.removed != change.added:
        return False
    row, = change.removed
    old = change.original.split("\n")
    new = change.patched.split("\n")
    if old[:row-1] != new[:row-1] or old[row:] != new[row:]:
        return False
    before, after = old[row-1], new[row-1]
    for left, right, prior in _sites(change.original, row):
        prefix, suffix = before[:left], before[right:]
        if not after.startswith(prefix) or not after.endswith(suffix):
            continue
        stop = len(after) - len(suffix)
        if stop <= left:
            continue
        replacement = after[left:stop]
        if prior == replacement:
            continue
        if ((prior in BOOL and replacement in BOOL)
                or (prior in COMPARE and replacement in COMPARE)
                or (_integer(prior) and _integer(replacement))):
            return True
    return False


def trivia_only(change):
    old = change.original.split("\n")
    new = change.patched.split("\n")
    for source, numbers in ((old,change.removed),(new,change.added)):
        if any(source[i-1].strip() and not source[i-1].lstrip().startswith("#") for i in numbers):
            return False
    return ast.dump(ast.parse(change.original)) == ast.dump(ast.parse(change.patched))
