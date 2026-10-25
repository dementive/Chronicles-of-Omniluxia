"""Reject a Pages artifact that has lost the reviewed search or geographic atlas."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


def check(root: Path) -> list[str]:
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    def read(name):
        path = root / name
        require(path.is_file(), f"Missing {name}")
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    archive = read("all.html")
    require('data-search-results' in archive, "All Pages lacks the full-text results container")
    require('data-index-category' in archive, "All Pages lacks subject filtering")
    match = re.search(r'<script type="application/json" id="full-search-data">(.*?)</script>', archive, re.S)
    require(match is not None, "All Pages lacks its embedded full-text search index")
    if match:
        try:
            rows = json.loads(match.group(1).replace(r"\/", "/"))
            require(bool(rows) and all(r.get("text") and r.get("url") for r in rows), "Search index lacks article text or destinations")
            require(any("Hellas" in r.get("text", "") for r in rows), "Search index lost article-body text")
        except (ValueError, TypeError, AttributeError):
            errors.append("Full-text search index is invalid")

    atlas = read("world-atlas.html")
    require('data-atlas-viewer' in atlas, "World Atlas lacks its interactive viewer")
    require('data-geographic-map="true"' in atlas, "World Atlas lacks source-derived geographic geometry")
    require('data-atlas-select' in atlas and 'data-atlas-zoom="in"' in atlas, "World Atlas lacks region or zoom controls")
    require('data-reader-map=' not in atlas, "A static map placeholder survived the production renderer")
    match = re.search(r'<script type="application/json" data-atlas-data>(.*?)</script>', atlas, re.S)
    require(match is not None, "World Atlas lacks its embedded region data")
    if match:
        try:
            data = json.loads(match.group(1).replace(r"\/", "/"))
            require(data.get("originalSize") == [8192, 4096], "Atlas does not identify the original source grid")
            require(len(data.get("regions", [])) == 16, "Atlas lost regional coverage")
            for region in data.get("regions", []):
                require((root / region["url"]).is_file(), f"Missing region article: {region['url']}")
        except (ValueError, TypeError, KeyError):
            errors.append("Atlas region data is invalid")

    for name in ("world-relief.webp", "world-realms.png", "world-atlas-data.json"):
        require((root / "assets/maps/reader" / name).is_file(), f"Missing atlas asset: {name}")

    javascript = read("assets/site.js")
    require("wireFullSearch" in javascript and "wireGeographicAtlas" in javascript,
            "Published JavaScript lacks search or atlas initialization")
    for name in ("style.css", "site.js"):
        source, output = root.parent / "assets" / name, root / "assets" / name
        if source.is_file() and output.is_file():
            require(hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(output.read_bytes()).digest(),
                    f"Published {name} differs from the canonical wiki asset")
    for name, text in (("all.html", archive), ("world-atlas.html", atlas)):
        scripts = re.findall(r'<script\b[^>]*src="([^"]+)"', text)
        require(any(urlsplit(src).path == "assets/site.js" for src in scripts), f"{name} does not load the shared script")
        require('name="q"' in text, f"{name} lacks global archive search")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    errors = check(args.path.resolve())
    for error in errors:
        print("FEATURE ERROR:", error)
    if errors:
        return 1
    print("Published feature checks passed: full-text search, geographic atlas, controls, data, and canonical assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
