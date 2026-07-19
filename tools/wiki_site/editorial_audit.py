#!/usr/bin/env python3
"""Create a prioritized, evidence-based wiki expansion backlog."""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import build


PLACEHOLDER_RE = re.compile(
    r"\b(?:placeholder|todo|tbd|stub|work in progress|coming soon|"
    r"does not yet provide|needs? expansion|unfinished)\b|\bCHANGE\?",
    re.I,
)

SECTION_IDEAS = {
    "countries": ["Origins and legitimacy", "Government and factions", "Economy and warfare", "Neighbors and campaign situation"],
    "races": ["Origins", "Society and lifecycle", "Cultures and homelands", "Relations with other peoples"],
    "religions": ["Cosmology and doctrine", "Worship and institutions", "Sacred places and figures", "Political divisions"],
    "characters": ["Origins and family", "Major deeds", "Relationships and rivals", "Legacy"],
    "regions": ["Geography", "Peoples and settlements", "History", "Strategic importance"],
    "cultures": ["Language and naming", "Social customs", "Political traditions", "Neighbors and diaspora"],
    "history": ["Causes", "Principal actors", "Course of events", "Consequences and memory"],
    "magic": ["Principles", "Practitioners and training", "Uses and limitations", "Historical examples"],
    "newworld": ["Homeland and environment", "Society", "History", "Old World contact"],
    "lore": ["Origins", "Organization or nature", "Historical role", "Connections to the wider setting"],
}


def audit_page(page):
    headings = re.findall(r"^#{2,4}\s+(.+)$", page.raw, re.M)
    markers = [m.group(0) for m in PLACEHOLDER_RE.finditer(page.raw)]
    score = 0
    evidence = []
    if markers:
        score += 50
        evidence.append("explicit placeholder language: " + ", ".join(sorted(set(markers))))
    if page.words < 60:
        score += 45
        evidence.append(f"only {page.words} words")
    elif page.words < 100:
        score += 35
        evidence.append(f"only {page.words} words")
    elif page.words < 180:
        score += 20
        evidence.append(f"only {page.words} words")
    elif page.words < 300:
        score += 8
        evidence.append(f"brief at {page.words} words")
    if not headings and page.words < 400:
        score += 15
        evidence.append("no article sections")
    elif len(headings) == 1 and page.words < 500:
        score += 6
        evidence.append("only one article section")
    if len(page.backlinks) < 2:
        score += 8
        evidence.append(f"only {len(page.backlinks)} incoming wiki links")

    ideas = SECTION_IDEAS.get(page.category or "lore", SECTION_IDEAS["lore"])
    missing = [idea for idea in ideas
               if not any(idea.split()[0].casefold() in h.casefold() for h in headings)]
    return {
        "title": page.title,
        "file": os.path.basename(page.path),
        "category": page.category or "lore",
        "words": page.words,
        "headings": headings,
        "incoming_links": len(page.backlinks),
        "priority_score": score,
        "evidence": evidence,
        "suggested_sections": missing[:4],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", required=True)
    ap.add_argument("--json", default="wiki-editorial-backlog.json")
    ap.add_argument("--markdown", default="wiki-editorial-backlog.md")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    site = build.Site(args.wiki, os.devnull)
    site.load()
    site.classify()
    items = [audit_page(page) for page in site.pages
             if not page.is_home and not getattr(page, "is_hub", False)]
    items.sort(key=lambda item: (-item["priority_score"], item["words"], item["title"].casefold()))
    items = items[:args.limit]

    summary = {
        "audited_articles": len(site.pages) - 1,
        "high_priority": sum(item["priority_score"] >= 50 for item in items),
        "medium_priority": sum(30 <= item["priority_score"] < 50 for item in items),
        "items": items,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "# Omniluxia wiki editorial backlog", "",
        "Generated from article length, explicit placeholder language, section structure, and incoming links.", "",
        f"Audited **{summary['audited_articles']}** articles. The table lists the top {len(items)} expansion candidates.", "",
        "| Priority | Article | Category | Words | Why it needs work | Suggested expansion |",
        "|---:|---|---|---:|---|---|",
    ]
    for item in items:
        why = "; ".join(item["evidence"]) or "comparatively thin coverage"
        suggestions = ", ".join(item["suggested_sections"])
        lines.append(
            f"| {item['priority_score']} | {item['title']} | {build.CATEGORY_LABEL.get(item['category'], 'Lore')} "
            f"| {item['words']} | {why} | {suggestions} |")
    lines.extend([
        "", "## Editorial guidance", "",
        "Treat this as a triage list, not permission to invent canon. Expansion should first draw from the mod's localization, events, missions, country setup, deities, cultures, and map data. Clearly flag genuine lore decisions that cannot be resolved from those sources.",
    ])
    with open(args.markdown, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Audited {summary['audited_articles']} articles; wrote {len(items)} prioritized candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
