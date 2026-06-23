#!/usr/bin/env python3
"""Generate a pdfpc sidecar (main.pdfpc) with per-slide speaker notes.

pdfpc auto-loads a ".pdfpc" file sitting next to the PDF and shows its notes
(Markdown) in the presenter console only. This maps each page of the deck to
the matching section of the speaker script (talk.md).

The deck order is deterministic:
    page 1            -> Title
    page 2            -> Outline
    then for each Part: 1 section-transition page -> the Part opener line
                        followed by one page per "## slide" -> its spoken text
So notes are assigned strictly by order; we assert the running total equals the
PDF page count before writing (a mismatch means the deck structure changed).

Struck-through text (~~...~~ = slides skipped aloud) is removed from the notes,
so each page shows exactly what to say.

Usage:
    python3 build_pdfpc_notes.py            # talk.md -> ../slides/main.pdfpc
    python3 build_pdfpc_notes.py talk.md /path/to/main.pdf
Re-run after editing talk.md (or after the deck structure changes).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MD = os.path.join(HERE, "talk.md")
DEFAULT_PDF = os.path.normpath(os.path.join(HERE, "..", "slides", "main.pdf"))

PART_RE = re.compile(r"^#\s+Part\s+.*$")
SLIDE_RE = re.compile(r"^##\s+(.*?)\s*(?:\[[^\]]*\])?\s*$")


def clean(text: str) -> str:
    """Drop struck-through (skipped) spans; keep the rest of the markdown."""
    return re.sub(r"~~.*?~~", "", text).strip()


def airy(text: str) -> str:
    """One sentence per paragraph (blank line between) so notes read easily.

    Splits at sentence-ending punctuation followed by whitespace and a capital,
    leaving decimals (9.47), abbreviations and mid-sentence capitals untouched.
    """
    text = re.sub(r"([.!?][\"')\]]?)\s+(?=[A-Z\"(])", r"\1\n\n", text)
    # collapse any accidental triple+ blank lines
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse(md: str):
    """Return ordered note blocks: [Title, Outline, opener, slide, slide, ...]."""
    lines = md.splitlines()
    blocks: list[tuple[str, str]] = []  # (label, markdown_text)
    label = None
    paras: list[str] = []
    buf: list[str] = []

    def flush_para():
        if buf:
            paras.append(" ".join(buf).strip())
            buf.clear()

    def flush_block():
        flush_para()
        if label is not None:
            text = clean("\n\n".join(p for p in paras if p.strip()))
            # Part 0 (front matter: Title/Outline) has no transition slide and no
            # opener line; drop any empty opener so pages stay aligned.
            if not (label == "Opener" and not text):
                blocks.append((label, text))
        paras.clear()

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("|"):
            continue
        if line.strip() == "---":
            continue
        if PART_RE.match(line):
            # Part header: the following quote lines (before first ##) are the opener.
            flush_block()
            label = "Opener"
            continue
        m = SLIDE_RE.match(line)
        if m:
            flush_block()
            label = m.group(1)
            continue
        if line.startswith("#"):
            # Doc H1 title line -> ignore (not a note block)
            continue
        if line.startswith(">"):
            content = line[1:].lstrip()
            if content == "":
                flush_para()
            else:
                buf.append(content)
            continue
        s = line.strip()
        if s.startswith("*") and s.endswith("*") and len(s) > 2 and label is not None:
            # stage direction (e.g. Image credits) -> keep as parenthetical
            flush_para()
            paras.append("(" + s.strip("*").strip() + ")")
            continue
    flush_block()
    return blocks


def page_count(pdf: str) -> int:
    out = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
    for ln in out.splitlines():
        if ln.startswith("Pages:"):
            return int(ln.split()[1])
    raise SystemExit("could not read page count from " + pdf)


def main():
    md_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MD
    pdf_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PDF
    with open(md_path, encoding="utf-8") as f:
        blocks = parse(f.read())

    notes = [text for _label, text in blocks]
    npages = page_count(pdf_path)

    if len(notes) != npages:
        sys.stderr.write(
            f"WARNING: {len(notes)} note blocks but {npages} PDF pages.\n"
            "The deck structure and the script are out of sync; mapping may be "
            "shifted. Check the printed alignment below.\n\n"
        )

    pdf_dir = os.path.dirname(os.path.abspath(pdf_path))
    pdf_name = os.path.basename(pdf_path)
    out_path = os.path.join(pdf_dir, os.path.splitext(pdf_name)[0] + ".pdfpc")

    # Per-page duration from word count at the speaker's measured pace (~128 wpm,
    # from rehearsal), then a running cumulative "target" time appended to each note
    # so you can check the count-up timer against it live (ahead = fast, behind = slow).
    wpm = 128.5

    def fmt(sec: int) -> str:
        return f"{sec // 60}:{sec % 60:02d}"

    cum = 0
    note_texts = []
    for t in notes:
        words = len(re.sub(r"[*_~`]", "", t).split())
        cum += round(words / wpm * 60)
        body = airy(t) if t else ""
        note_texts.append((body + f"\n\n[{fmt(cum)}]").strip())

    # pdfpc 4.x stores notes as JSON (pdfpcFormat 2). If such a file already
    # exists, patch the `note` fields in place (keeping idx/label/overlay so the
    # overlay grouping stays exactly as pdfpc computed it). Otherwise fall back
    # to the legacy "### N" text format, which pdfpc imports on first load.
    import json

    data = None
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                d = json.load(f)
            # Only reuse the existing JSON if its page count matches the deck;
            # a stale file (deck changed since) would shift the notes, so rewrite.
            if (
                isinstance(d, dict)
                and d.get("pdfpcFormat")
                and "pages" in d
                and len(d["pages"]) == len(note_texts)
            ):
                data = d
        except (ValueError, OSError):
            data = None

    if data is not None:
        pages = sorted(data["pages"], key=lambda p: p["idx"])
        if len(pages) != len(note_texts):
            sys.stderr.write(
                f"WARNING: {len(pages)} JSON pages vs "
                f"{len(note_texts)} notes; patching the overlap.\n"
            )
        for p, text in zip(pages, note_texts, strict=False):
            p["note"] = (text + "\n\n") if text else ""
        data["duration"] = 0  # 0 = count-up stopwatch (starts at 0:00)
        data["lastMinutes"] = 0
        data["disableMarkdown"] = False
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
    else:
        # No existing pdfpc file: write the legacy "### N" text format, where N is the
        # 1-based PDF page number. pdfpc imports it and does its OWN page-to-slide
        # mapping, which correctly handles its internal grouping of unnumbered frames
        # (title/outline/section transitions). Writing a JSON from scratch with guessed
        # label/overlay values would make pdfpc attach notes to the wrong slides.
        lines = ["[file]", pdf_name, "[notes]"]
        for i, text in enumerate(note_texts[:npages], start=1):
            lines.append(f"### {i}")
            lines.append(text if text else "(no notes)")
            lines.append("")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")

    # sanity print: page -> label -> first words
    print(f"Wrote {out_path}  ({len(notes)} blocks, {npages} pages)\n")
    for i, (label, text) in enumerate(blocks[:npages], start=1):
        first = re.sub(r"\s+", " ", re.sub(r"[*~]", "", text))[:60]
        print(f"  p{i:2d}  {label[:34]:34s} | {first}")


if __name__ == "__main__":
    main()
