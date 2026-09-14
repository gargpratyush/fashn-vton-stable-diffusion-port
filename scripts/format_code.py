"""Format owned C/C++ only; --base checks changed lines without rewriting inherited debt."""
import argparse
import difflib
from pathlib import Path, PurePosixPath
import re
import subprocess
import xml.etree.ElementTree as ET


ROOTS = {"src", "include", "examples", "tests"}
EXCLUDED = ("src/tokenizers/vocab", "examples/server/frontend")


def owned(name):
    path = PurePosixPath(name.replace("\\", "/"))
    return (bool(path.parts) and not path.is_absolute() and ".." not in path.parts and path.parts[0] in ROOTS
            and path.suffix in {".c", ".cpp", ".h", ".hpp"}
            and not any(path == PurePosixPath(p) or PurePosixPath(p) in path.parents for p in EXCLUDED))


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def candidates(repo, base=None):
    names = git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard").decode().split("\0")
    if base:
        git(repo, "rev-parse", "--verify", base + "^{commit}")
        changed = git(repo, "diff", "--name-only", "-z", "--diff-filter=ACMR", base).decode().split("\0")
        untracked = git(repo, "ls-files", "-z", "--others", "--exclude-standard").decode().split("\0")
        names = set(changed + untracked)
    return sorted({name for name in names if name and owned(name) and (repo / name).is_file()})


def changed_lines(before, after):
    result = set()
    for tag, _, _, start, end in difflib.SequenceMatcher(
            None, before.splitlines(), after.splitlines(), autojunk=False).get_opcodes():
        if tag != "equal":
            result.update(range(max(1, start + 1), max(start + 1, end) + 1))
    return result


def replacements(source, document, lines=None):
    result = []
    for item in ET.fromstring(document).findall("replacement"):
        offset, length = int(item.attrib["offset"]), int(item.attrib["length"])
        first = source[:offset].count(b"\n") + 1
        last = source[:offset + length].count(b"\n") + 1
        if lines is None or any(line in lines for line in range(first, last + 1)):
            result.append((offset, length, (item.text or "").encode()))
    return result


def run(repo, formatter, base=None, write=False):
    version = subprocess.check_output([formatter, "--version"], text=True)
    if not re.search(r"clang-format version 19\.", version):
        raise ValueError("Use clang-format 19.x for reproducible repository formatting")
    failed = []
    for name in candidates(repo, base):
        path = repo / name
        if (not path.resolve().is_relative_to(repo.resolve()) or
                not owned(path.resolve().relative_to(repo.resolve()).as_posix()) or any(
                p.is_symlink() for p in [path, *path.parents] if p != repo.parent)):
            raise ValueError("Refusing linked source path: " + name)
        source = path.read_bytes()
        lines = None
        if base:
            previous = subprocess.run(["git", "-C", str(repo), "show", f"{base}:{name}"], capture_output=True)
            # A new path has no baseline; format it in full.
            if previous.returncode == 0:
                lines = changed_lines(previous.stdout, source)
        document = subprocess.check_output(
            [formatter, "--style=file", "--output-replacements-xml", "--assume-filename=" + str(path)],
            input=source)
        edits = replacements(source, document, lines)
        if edits:
            failed.append(name)
            if write:
                for offset, length, text in reversed(edits):
                    source = source[:offset] + text + source[offset + length:]
                path.write_bytes(source)
            print(("formatted " if write else "needs formatting: ") + name)
    return 0 if write or not failed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Git commit to compare; omit to check all owned files")
    parser.add_argument("--formatter", default="clang-format")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only (default)")
    mode.add_argument("--write", action="store_true", help="Explicitly apply selected replacements")
    args = parser.parse_args()
    return run(Path(__file__).resolve().parent.parent, args.formatter, args.base, args.write)


if __name__ == "__main__":
    raise SystemExit(main())
