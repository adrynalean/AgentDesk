"""Build a real, sizable corpus from the Python standard library's own docs.

Introspects a broad set of stdlib modules and writes one markdown file per module
containing each public object's qualified name + docstring. Fully offline and
reproducible — a legitimate technical corpus for RAG.

Usage:  python scripts/build_corpus.py [out_dir]
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

MODULES = """
json os re collections itertools functools math statistics datetime random
textwrap string pathlib typing dataclasses enum heapq bisect decimal fractions
hashlib base64 secrets uuid urllib.parse html csv sqlite3 argparse logging
socket threading asyncio subprocess shutil glob io struct array copy pprint
difflib unittest gzip zipfile tarfile pickle shelve configparser calendar time
zoneinfo contextlib abc numbers operator weakref types traceback warnings
inspect ast dis tokenize keyword tempfile fnmatch stat filecmp mimetypes
smtplib email.utils http.client ftplib xml.etree.ElementTree queue sched
selectors signal ipaddress ssl codecs unicodedata locale gettext platform
""".split()


def _docstring(obj) -> str:
    return inspect.cleandoc(inspect.getdoc(obj) or "").strip()


def build(out_dir: str = "corpus") -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    objects = 0

    for modname in MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:  # noqa: BLE001
            continue
        lines: list[str] = [f"# Module `{modname}`\n"]
        if _docstring(mod):
            lines.append(f"{modname}: {_docstring(mod).splitlines()[0]}\n")

        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
            if not (inspect.isfunction(obj) or inspect.isclass(obj) or inspect.isbuiltin(obj)):
                continue
            if getattr(obj, "__module__", modname) != modname:
                continue  # skip re-exports
            doc = _docstring(obj)
            if not doc:
                continue
            lines.append(f"## {modname}.{name}\n{doc}\n")
            objects += 1
            if inspect.isclass(obj):  # include public methods
                for mname, meth in inspect.getmembers(obj, inspect.isfunction):
                    if mname.startswith("_"):
                        continue
                    mdoc = _docstring(meth)
                    if mdoc:
                        lines.append(f"### {modname}.{name}.{mname}\n{mdoc}\n")
                        objects += 1

        (out / f"{modname.replace('.', '_')}.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[corpus] wrote {objects} documented objects from {len(MODULES)} modules to {out}/")
    return objects


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "corpus")
