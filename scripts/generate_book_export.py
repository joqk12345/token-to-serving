#!/usr/bin/env python3
"""Assemble the ready chapters into a standalone HTML export."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "book/chapters"
EXPORT = ROOT / "book/export"
SPINE = ROOT / "book/spine.yml"
SOURCE = EXPORT / "book-source.md"
HTML = EXPORT / "book.html"
PDF = EXPORT / "book.pdf"
PANDOC = shutil.which("pandoc")
TECTONIC = shutil.which("tectonic")
XELATEX = shutil.which("xelatex")
PDF_ENGINE = XELATEX or TECTONIC

CHAPTER_RE = re.compile(r"^[0-9][0-9]-.*\.md$")
H1_RE = re.compile(r"^#\s+(.+?)\s*$")


def fail(message: str) -> str:
    return f"FAIL: {message}"


def chapter_paths() -> list[pathlib.Path]:
    return sorted(
        p for p in CHAPTERS.glob("[0-9][0-9]-*.md") if CHAPTER_RE.match(p.name) and not p.name.endswith("-brief.md")
    )


def load_part_by_slug() -> dict[str, str]:
    part_by_slug: dict[str, str] = {}
    current_part: str | None = None
    if not SPINE.exists():
        return part_by_slug
    for raw in SPINE.read_text(encoding="utf-8").splitlines():
        part_match = re.match(r'^\s{2}- title:\s+"(.+?)"\s*$', raw)
        if part_match:
            current_part = part_match.group(1)
            continue
        slug_match = re.match(r'^\s{8}slug:\s+"(.+?)"\s*$', raw)
        if slug_match and current_part:
            part_by_slug[slug_match.group(1)] = current_part
    return part_by_slug


def strip_front_matter(text: str) -> str:
    if not text.startswith("---\n"):
        return text.lstrip()
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return text.lstrip()
    return parts[2].lstrip()


def tag_chapter_title(text: str, chapter_index: int) -> str:
    lines: list[str] = []
    title_tagged = False
    for line in text.splitlines():
        if not title_tagged:
            match = H1_RE.match(line)
            if match:
                title = match.group(1)
                lines.append(f"## Chapter {chapter_index}: {title} {{.chapter-title}}")
                title_tagged = True
                continue
        if line.startswith("### "):
            lines.append("#" + line)
        elif line.startswith("## "):
            lines.append("#" + line)
        else:
            lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def rewrite_export_links(text: str) -> str:
    text = text.replace("](chapters/", "](../chapters/")
    text = text.replace("](downloads/", "](../../downloads/")
    text = text.replace("](book/", "](../book/")
    return text


def read_source(rel: pathlib.Path) -> str:
    return rel.read_text(encoding="utf-8")


def build_source() -> str:
    lines = [
        "---",
        'title: "Large Language Model Systems: From Tokens to Serving Infrastructure"',
        'author: "gizamo"',
        'subtitle: "The Complete Engineering Journey from Tokens to Serving Infrastructure"',
        'lang: en',
        "---",
        "",
    ]

    part_by_slug = load_part_by_slug()
    current_part: str | None = None
    for chapter_index, path in enumerate(chapter_paths(), start=1):
        part_title = part_by_slug.get(path.stem)
        if part_title and part_title != current_part:
            lines.append(f"# {part_title} {{.part-title}}")
            lines.append("")
            current_part = part_title
        content = strip_front_matter(read_source(path))
        lines.append(tag_chapter_title(content, chapter_index).rstrip())
        lines.append("")

    refs = ROOT / "book/REFERENCES.md"
    if refs.exists():
        lines.append(rewrite_export_links(read_source(refs)).rstrip())
        lines.append("")

    index = ROOT / "book/INDEX.md"
    if index.exists():
        lines.append(rewrite_export_links(read_source(index)).rstrip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_assets() -> tuple[str, str]:
    EXPORT.mkdir(parents=True, exist_ok=True)
    source = build_source()
    css = build_css()
    SOURCE.write_text(source, encoding="utf-8")
    (EXPORT / "book.css").write_text(css, encoding="utf-8")
    return source, css


def build_css() -> str:
    css = """
    :root {
      color-scheme: light;
      --ink: #18212f;
      --muted: #566173;
      --border: #d9e0ea;
      --paper: #ffffff;
      --canvas: #f4f6f9;
      --accent: #173e76;
      --accent-soft: #e8eef8;
      --code-bg: #f6f8fb;
    }
    html { background: var(--canvas); }
    body {
      margin: 0 auto;
      max-width: 980px;
      padding: 24px 18px 80px;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background: var(--paper);
      box-shadow: 0 20px 60px rgba(15, 23, 42, .08);
    }
    header#title-block-header {
      padding: 36px 36px 8px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 28px;
    }
    h1.title {
      margin: 0 0 10px;
      font-size: clamp(30px, 4vw, 48px);
      line-height: 1.05;
      letter-spacing: -.02em;
    }
    #TOC {
      background: var(--accent-soft);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px 22px;
      margin: 0 36px 34px;
    }
    #TOC > ul { columns: 2; column-gap: 28px; }
    #TOC a { color: var(--accent); text-decoration: none; }
    #TOC a:hover { text-decoration: underline; }
    h1, h2, h3, h4 { line-height: 1.2; scroll-margin-top: 28px; }
    h1:not(.title) {
      margin-top: 56px;
      padding-top: 18px;
      border-top: 2px solid var(--border);
      break-before: page;
      page-break-before: always;
    }
    h1.part-title {
      color: var(--accent);
      font-size: 30px;
    }
    h2.chapter-title {
      margin-top: 42px;
      padding-top: 16px;
      border-top: 1px solid var(--border);
      break-before: page;
      page-break-before: always;
    }
    h2 { margin-top: 34px; }
    h3 { margin-top: 24px; }
    h4 { margin-top: 20px; }
    p, li { line-height: 1.74; font-size: 16px; }
    blockquote {
      margin: 18px 0;
      padding: 0 18px;
      border-left: 4px solid var(--accent);
      color: #243246;
    }
    code, pre {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.94em;
    }
    code {
      background: var(--code-bg);
      padding: 0.1em 0.3em;
      border-radius: 4px;
    }
    pre {
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px 18px;
      overflow: auto;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 18px 0 24px;
      font-size: 15px;
    }
    th, td {
      border: 1px solid var(--border);
      padding: 8px 10px;
      vertical-align: top;
    }
    th { background: var(--accent-soft); }
    a { color: var(--accent); }
    .chapter-title { page-break-after: avoid; }
    figure {
      margin: 28px auto 24px;
      text-align: center;
      page-break-inside: avoid;
      break-inside: avoid;
    }
    figure img {
      max-width: 100%;
      height: auto;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fff;
    }
    figure figcaption {
      margin-top: 10px;
      font-size: 14px;
      line-height: 1.45;
      color: var(--muted);
      text-align: left;
    }
    @media print {
      @page { size: letter; margin: 0.65in; }
      body { box-shadow: none; max-width: none; padding: 0 18mm 16mm; }
      #TOC { margin-left: 0; margin-right: 0; }
      h1:not(.title), h2.chapter-title { break-before: page; page-break-before: always; }
      figure img { border: none; border-radius: 0; }
    }
    """

    return css.strip() + "\n"


def render_html(source_path: pathlib.Path, html_path: pathlib.Path, css_path: pathlib.Path) -> None:
    if not PANDOC:
        raise RuntimeError("pandoc is not available")

    workdir = source_path.parent

    proc = subprocess.run(
        [
            PANDOC,
            source_path.name,
            "--from",
            "markdown+yaml_metadata_block+fenced_code_blocks+pipe_tables+smart",
            "--to",
            "html5",
            "--standalone",
            "--toc",
            "--toc-depth",
            "3",
            "--resource-path",
            str(workdir) + ":" + str(ROOT / "book"),
            "--metadata",
            "author=gizamo",
            "--metadata",
            "lang=en",
            "--metadata",
            "title=Large Language Model Systems: From Tokens to Serving Infrastructure",
            "--css",
            css_path.name,
            "--output",
            html_path.name,
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "pandoc export failed")


def render_pdf(source_path: pathlib.Path, pdf_path: pathlib.Path) -> None:
    if not PDF_ENGINE:
        raise RuntimeError("neither xelatex nor tectonic is available")

    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = "0"
    pdf_source = source_path
    cleanup_source: pathlib.Path | None = None
    source_text = source_path.read_text(encoding="utf-8")
    if ".svg)" in source_text:
        cleanup_source = source_path.with_name(f".{source_path.stem}.pdf-source.md")
        cleanup_source.write_text(source_text.replace(".svg)", ".pdf)"), encoding="utf-8")
        pdf_source = cleanup_source

    try:
        proc = subprocess.run(
            [
                PANDOC,
                str(pdf_source),
                "--from",
                "markdown+yaml_metadata_block+fenced_code_blocks+pipe_tables+smart",
                "--to",
                "pdf",
                "--standalone",
                "--toc",
                "--toc-depth",
                "3",
                "--resource-path",
                str(source_path.parent) + ":" + str(ROOT / "book"),
                "--metadata",
                "author=gizamo",
                "--metadata",
                "lang=en",
                "--metadata",
                "title=Large Language Model Systems: From Tokens to Serving Infrastructure",
                "--pdf-engine",
                PDF_ENGINE,
                "--variable",
                "geometry:margin=0.8in",
                "--variable",
                "papersize=letter",
                "--variable",
                "fontsize=11pt",
                "--output",
                str(pdf_path),
            ],
            cwd=source_path.parent,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if cleanup_source and cleanup_source.exists():
            cleanup_source.unlink()
    if proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "PDF export failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not PANDOC:
        print(fail("pandoc is not available"), file=sys.stderr)
        return 1
    if not PDF_ENGINE:
        print(fail("neither xelatex nor tectonic is available"), file=sys.stderr)
        return 1

    if args.check:
        source = build_source()
        css = build_css()
        EXPORT.mkdir(parents=True, exist_ok=True)
        tmp_source = EXPORT / ".book-source.check.md"
        tmp_html = EXPORT / ".book.check.html"
        tmp_pdf = EXPORT / ".book.check.pdf"
        for tmp_path in (tmp_source, tmp_html, tmp_pdf):
            if tmp_path.exists():
                tmp_path.unlink()
        try:
            tmp_source.write_text(source, encoding="utf-8")
            try:
                render_html(tmp_source, tmp_html, EXPORT / "book.css")
            except Exception as exc:  # pragma: no cover - surfaced to user
                print(fail(str(exc)), file=sys.stderr)
                return 1

            current_source = SOURCE.read_text(encoding="utf-8") if SOURCE.exists() else ""
            current_css = (EXPORT / "book.css").read_text(encoding="utf-8") if (EXPORT / "book.css").exists() else ""
            current_html = HTML.read_text(encoding="utf-8") if HTML.exists() else ""

            if current_source != source:
                print(fail("book/export/book-source.md is missing or stale"), file=sys.stderr)
                return 1
            if current_css != css:
                print(fail("book/export/book.css is missing or stale"), file=sys.stderr)
                return 1
            if current_html != tmp_html.read_text(encoding="utf-8"):
                print(fail("book/export/book.html is missing or stale"), file=sys.stderr)
                return 1
            try:
                render_pdf(tmp_source, tmp_pdf)
            except Exception as exc:  # pragma: no cover - surfaced to user
                print(fail(str(exc)), file=sys.stderr)
                return 1
            if not PDF.exists():
                print(fail("book/export/book.pdf is missing"), file=sys.stderr)
                return 1

            newest_input = max(SOURCE.stat().st_mtime, (EXPORT / "book.css").stat().st_mtime, HTML.stat().st_mtime)
            if PDF.stat().st_mtime < newest_input:
                print(fail("book/export/book.pdf is missing or stale"), file=sys.stderr)
                return 1
        finally:
            for tmp_path in (tmp_source, tmp_html, tmp_pdf):
                if tmp_path.exists():
                    tmp_path.unlink()

        print("PASS: book export is current")
        return 0

    try:
        source, css = write_assets()
        render_html(SOURCE, HTML, EXPORT / "book.css")
        render_pdf(SOURCE, PDF)
    except Exception as exc:  # pragma: no cover - surfaced to user
        print(fail(str(exc)), file=sys.stderr)
        return 1

    print("WROTE: book/export/book-source.md, book/export/book.html, and book/export/book.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
