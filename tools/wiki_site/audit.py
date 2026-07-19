#!/usr/bin/env python3
"""Audit the generated wiki as a directed article graph."""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from html.parser import HTMLParser


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def page_links(path):
    parser = LinkParser()
    with open(path, encoding="utf-8") as f:
        parser.feed(f.read())
    return parser.links


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="docs")
    ap.add_argument("--report", default="wiki-audit.json")
    ap.add_argument("--markdown-report", default="wiki-audit.md")
    args = ap.parse_args()

    html_files = sorted(f for f in os.listdir(args.site) if f.endswith(".html"))
    pages = set(html_files)
    utility = {"index.html", "all.html", "credits.html", "404.html"}
    articles = {f for f in pages if not f.startswith("c-") and f not in utility}
    graph = {f: set() for f in articles}
    broken = []

    for source in html_files:
        for href in page_links(os.path.join(args.site, source)):
            if href.startswith(("http://", "https://", "mailto:", "#", "//")):
                continue
            target = href.split("#", 1)[0].split("?", 1)[0]
            if not target or target.startswith("assets/") or target.endswith((".xml", ".json")):
                continue
            if target not in pages:
                broken.append({"source": source, "target": href})
            elif source in articles and target in articles and target != source:
                graph[source].add(target)

    incoming = Counter()
    for targets in graph.values():
        incoming.update(targets)
    orphans = sorted(f for f in articles if incoming[f] == 0)
    no_outgoing = sorted(f for f in articles if not graph[f])
    weak = sorted(f for f in articles if incoming[f] + len(graph[f]) < 3)
    reciprocal = sum(1 for source, targets in graph.items()
                     for target in targets if source in graph.get(target, set())) // 2

    report = {
        "pages": len(pages),
        "articles": len(articles),
        "article_links": sum(map(len, graph.values())),
        "reciprocal_relationships": reciprocal,
        "broken_links": broken,
        "orphan_articles": orphans,
        "articles_without_outgoing_links": no_outgoing,
        "weakly_connected_articles": weak,
        "incoming_links": dict(sorted(incoming.items())),
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md = [
        "# Wiki graph audit", "",
        f"- Generated pages: {report['pages']}",
        f"- Articles: {report['articles']}",
        f"- Contextual article links: {report['article_links']}",
        f"- Reciprocal relationships: {reciprocal}",
        f"- Broken links: {len(broken)}",
        f"- Orphan articles: {len(orphans)}",
        f"- Articles without outgoing links: {len(no_outgoing)}",
        f"- Weakly connected articles: {len(weak)}", "",
    ]
    for title, items in [
        ("Broken links", [f"{x['source']} -> {x['target']}" for x in broken]),
        ("Orphan articles", orphans),
        ("Articles without outgoing links", no_outgoing),
        ("Weakly connected articles", weak),
    ]:
        md.extend([f"## {title}", ""])
        md.extend([f"- {item}" for item in items] or ["None."])
        md.append("")
    with open(args.markdown_report, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"{len(pages)} pages; {sum(map(len, graph.values()))} article links; "
          f"{len(orphans)} orphans; {len(broken)} broken links")
    if broken:
        for item in broken[:20]:
            print(f"  {item['source']} -> {item['target']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
