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
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "book/chapters"
EXPORT = ROOT / "book/export"
SPINE = ROOT / "book/spine.yml"
SOURCE = EXPORT / "book-source.md"
HTML = EXPORT / "book.html"
PDF = EXPORT / "book.pdf"
COVER = ROOT / "book/figures/artwork/book-cover.svg"
PANDOC = shutil.which("pandoc")
TECTONIC = shutil.which("tectonic")
XELATEX = shutil.which("xelatex")
PDF_ENGINE = XELATEX or TECTONIC
RSVG_CONVERT = shutil.which("rsvg-convert")

CHAPTER_RE = re.compile(r"^[0-9][0-9]-.*\.md$")
H1_RE = re.compile(r"^#\s+(.+?)\s*$")
SVG_REF_RE = re.compile(r"\(([^)]+\.svg)\)")
ID_WORD_RE = re.compile(r"[^a-z0-9]+")


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


def pandoc_identifier(text: str) -> str:
    ident = ID_WORD_RE.sub("-", text.lower()).strip("-")
    return ident or "section"


def chapter_title(text: str) -> str:
    for line in text.splitlines():
        match = H1_RE.match(line)
        if match:
            return match.group(1)
    return "Untitled"


def tag_chapter_title(text: str, chapter_index: int) -> str:
    lines: list[str] = []
    title_tagged = False
    for line in text.splitlines():
        if not title_tagged:
            match = H1_RE.match(line)
            if match:
                title = match.group(1)
                heading = f"Chapter {chapter_index}: {title}"
                lines.append(f"## {heading} {{#{pandoc_identifier(heading)} .chapter-title}}")
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


def normalize_asset_ref(ref: str) -> pathlib.PurePosixPath:
    parts = [part for part in pathlib.PurePosixPath(ref).parts if part not in {"..", "."}]
    if not parts:
        raise RuntimeError(f"invalid asset reference: {ref!r}")
    return pathlib.PurePosixPath(*parts)


def convert_svg_to_pdf(svg_path: pathlib.Path, pdf_path: pathlib.Path) -> None:
    if not RSVG_CONVERT:
        raise RuntimeError("rsvg-convert is not available")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [RSVG_CONVERT, "-f", "pdf", "-o", str(pdf_path), str(svg_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or f"failed to convert {svg_path}")


def rewrite_svg_links(text: str, mapping: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        ref = match.group(1)
        return f"({mapping.get(ref, ref)})"

    return SVG_REF_RE.sub(repl, text)


def read_source(rel: pathlib.Path) -> str:
    return rel.read_text(encoding="utf-8")


def build_source() -> str:
    lines = [
        "---",
        'pagetitle: "Large Language Model Systems: From Tokens to Serving Infrastructure"',
        'lang: en',
        "---",
        "",
    ]
    if COVER.exists():
        lines.extend(
            [
                "\\thispagestyle{empty}",
                "",
                "![](../figures/artwork/book-cover.svg){.book-cover width=85%}",
                "",
                "\\newpage",
                "",
            ]
        )

    part_by_slug = load_part_by_slug()
    chapters = chapter_paths()
    lines.append("# Contents {#contents .contents-title}")
    lines.append("")
    contents_current_part: str | None = None
    for chapter_index, path in enumerate(chapters, start=1):
        part_title = part_by_slug.get(path.stem)
        if part_title and part_title != contents_current_part:
            lines.append(f"- [{part_title}](#{pandoc_identifier(part_title)})")
            contents_current_part = part_title
        content = strip_front_matter(read_source(path))
        heading = f"Chapter {chapter_index}: {chapter_title(content)}"
        lines.append(f"  - [{heading}](#{pandoc_identifier(heading)})")
    lines.append("- [References](#references)")
    lines.append("- [Index](#index)")
    lines.append("")
    lines.append("\\newpage")
    lines.append("")

    current_part: str | None = None
    for chapter_index, path in enumerate(chapters, start=1):
        part_title = part_by_slug.get(path.stem)
        if part_title and part_title != current_part:
            lines.append(f"# {part_title} {{#{pandoc_identifier(part_title)} .part-title}}")
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
    h1.contents-title {
      margin: 40px 36px 16px;
      padding-top: 0;
      border-top: none;
      color: var(--accent);
    }
    h1.contents-title + ul {
      margin: 0 36px 34px;
      padding: 20px 22px 20px 38px;
      background: var(--accent-soft);
      border: 1px solid var(--border);
      border-radius: 14px;
      columns: 2;
      column-gap: 28px;
    }
    h1.contents-title + ul a { color: var(--accent); text-decoration: none; }
    h1.contents-title + ul a:hover { text-decoration: underline; }
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
    img.book-cover {
      display: block;
      width: min(100%, 760px);
      height: auto;
      margin: 0 auto 36px;
      border: 1px solid var(--border);
      border-radius: 8px;
      break-after: page;
      page-break-after: always;
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
      img.book-cover {
        width: 100%;
        max-height: 9.1in;
        object-fit: contain;
        border: none;
        border-radius: 0;
      }
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
            "--resource-path",
            str(workdir) + ":" + str(ROOT / "book"),
            "--metadata",
            "lang=en",
            "--metadata",
            "pagetitle=Large Language Model Systems: From Tokens to Serving Infrastructure",
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
    if not RSVG_CONVERT:
        raise RuntimeError("rsvg-convert is not available")

    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = "0"
    source_text = source_path.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix=f".{source_path.stem}.svg-pdf-assets-", dir=source_path.parent) as asset_dir:
        pdf_asset_root = pathlib.Path(asset_dir)
        pdf_asset_link_root = pathlib.PurePosixPath(pdf_asset_root.name)
        replacement_map: dict[str, str] = {}

        for ref in sorted(set(SVG_REF_RE.findall(source_text))):
            svg_path = (source_path.parent / ref).resolve()
            if not svg_path.exists():
                raise RuntimeError(f"missing SVG asset: {ref}")
            asset_rel = normalize_asset_ref(ref)
            pdf_asset = pdf_asset_root / asset_rel.with_suffix(".pdf")
            convert_svg_to_pdf(svg_path, pdf_asset)
            replacement_map[ref] = (pdf_asset_link_root / asset_rel.with_suffix(".pdf")).as_posix()

        pdf_source = source_path
        cleanup_source: pathlib.Path | None = None
        if replacement_map:
            cleanup_source = source_path.with_name(f".{source_path.stem}.pdf-source.md")
            cleanup_source.write_text(rewrite_svg_links(source_text, replacement_map), encoding="utf-8")
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
                    "--resource-path",
                    str(source_path.parent) + ":" + str(ROOT / "book"),
                    "--metadata",
                    "lang=en",
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
        tmp_source = pathlib.Path(
            tempfile.NamedTemporaryFile(prefix=".book-source.", suffix=".check.md", dir=EXPORT, delete=False).name
        )
        tmp_html = pathlib.Path(
            tempfile.NamedTemporaryFile(prefix=".book.", suffix=".check.html", dir=EXPORT, delete=False).name
        )
        tmp_pdf = pathlib.Path(
            tempfile.NamedTemporaryFile(prefix=".book.", suffix=".check.pdf", dir=EXPORT, delete=False).name
        )
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
