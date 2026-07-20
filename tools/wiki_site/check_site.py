#!/usr/bin/env python3
"""Fail-fast checks for a generated Omniluxia site."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="site/preview")
    args = parser.parse_args()
    root = Path(args.path).resolve()
    pages = sorted(root.glob("*.html"))
    missing: list[str] = []
    anchors: list[str] = []
    assets: list[str] = []

    for page in pages:
        text = page.read_text(encoding="utf-8")
        ids = set(re.findall(r'id=["\']([^"\']+)', text))
        for href in re.findall(r'href=["\']([^"\']+)', text):
            parsed = urlparse(html.unescape(href))
            if parsed.scheme or parsed.netloc or href.startswith("#"):
                if href.startswith("#") and href[1:] not in ids:
                    anchors.append(f"{page.name}: {href}")
                continue
            target = (page.parent / unquote(parsed.path)).resolve()
            if not target.exists():
                missing.append(f"{page.name}: {href}")
            if parsed.fragment and target.exists() and target.suffix == ".html":
                target_ids = set(re.findall(r'id=["\']([^"\']+)', target.read_text(encoding="utf-8")))
                if parsed.fragment not in target_ids:
                    anchors.append(f"{page.name}: {href}")
        for src in re.findall(r'src=["\']([^"\']+)', text):
            if src.startswith(("http:", "https:", "data:", "#")):
                continue
            target = (page.parent / unquote(src.split("#", 1)[0])).resolve()
            if not target.exists():
                assets.append(f"{page.name}: {src}")

    css = root / "assets" / "style.css"
    if css.exists():
        for value in re.findall(r"url\(['\"]?([^)'\"]+)", css.read_text(encoding="utf-8")):
            if value.startswith(("http:", "https:", "data:", "#")):
                continue
            if not (css.parent / value).resolve().exists():
                assets.append(f"style.css: {value}")

    print(f"pages: {len(pages)}")
    print(f"missing pages: {len(missing)}")
    print(f"bad anchors: {len(anchors)}")
    print(f"missing assets: {len(assets)}")
    for label, values in (("MISSING", missing), ("ANCHOR", anchors), ("ASSET", assets)):
        for value in values[:20]:
            print(f"{label}: {value}")
    return 1 if missing or anchors or assets else 0


if __name__ == "__main__":
    raise SystemExit(main())
