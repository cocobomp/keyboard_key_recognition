#!/usr/bin/env python3
"""fetch_lm_corpus.py — download public-domain English text for the n-gram LM.

The language model must be trained on text you did NOT type during the sessions,
otherwise the "decoding" stage is just retrieving memorised prompts and the
resulting accuracy is meaningless.  This grabs a few Project Gutenberg books
into corpus/lm_train.txt.

    python fetch_lm_corpus.py                 # default set, ~2 MB of text
    python fetch_lm_corpus.py --out corpus/lm_train.txt --books 11 84 1342
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request

import kkr_common as kc

DEFAULT_BOOKS = [11, 84, 1342, 2701, 1661, 98]  # Alice, Frankenstein, P&P, Moby Dick, Holmes, Two Cities
URL = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
URL_ALT = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)


def strip_boilerplate(text: str) -> str:
    m = START.search(text)
    if m:
        text = text[m.end() :]
    m = END.search(text)
    if m:
        text = text[: m.start()]
    return text


def fetch(book_id: int, timeout: float) -> str | None:
    for url in (URL.format(id=book_id), URL_ALT.format(id=book_id)):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                raw = r.read().decode("utf-8", errors="replace")
            print(f"  {book_id}: {len(raw) // 1024} KB from {url}")
            return strip_boilerplate(raw)
        except Exception as exc:
            print(f"  {book_id}: {url} failed ({exc})")
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="corpus/lm_train.txt")
    p.add_argument("--books", type=int, nargs="*", default=DEFAULT_BOOKS)
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args(argv)

    parts = [t for t in (fetch(b, args.timeout) for b in args.books) if t]
    if not parts:
        print(
            "\nno text could be downloaded. Point --lm-corpus at any large English\n"
            "text file you already have instead — it only has to be text you did not\n"
            "type during the capture sessions.",
            file=sys.stderr,
        )
        return 1
    text = "\n\n".join(parts)
    kc.ensure_dir("/".join(args.out.split("/")[:-1]) or ".")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"\nwrote {args.out} ({len(text) // 1024} KB, {len(text.split())} words)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
