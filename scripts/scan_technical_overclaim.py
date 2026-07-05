#!/usr/bin/env python3
"""Lightweight scanner for technical overclaim language.

This is intentionally conservative: it warns; it does not rewrite prose.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path.cwd()
TARGETS = [ROOT / "book", ROOT / "process"]

PATTERNS = [
    r"\balways faster\b",
    r"\bsolves the problem\b",
    r"\boptimal\b",
    r"\bbest\b",
    r"\bproves\b",
    r"\bnegligible\b",
    r"\bclearly\b",
    r"\bobviously\b",
]


def iter_markdown_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for target in TARGETS:
        if target.exists():
            files.extend(target.rglob("*.md"))
    return files


def main() -> int:
    findings: list[str] = []
    combined = re.compile("|".join(PATTERNS), re.IGNORECASE)
    for path in iter_markdown_files():
        rel_path = path.relative_to(ROOT)
        if str(rel_path).startswith("book/evidence/recommended-papers"):
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if path.match("*/book/evidence/source-ledger/*.md") and line.startswith("Exact quote:"):
                continue
            if combined.search(line):
                findings.append(f"{rel_path}:{lineno}: check technical overclaim: {line.strip()}")

    if findings:
        print("\n".join(findings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
