#!/usr/bin/env python3
"""
Chronicles of Omniluxia - static wiki site generator.

Reads the project wiki (a flat pile of markdown files) and renders a designed
static site into docs/, which GitHub Pages serves.

The wiki remains the single source of truth. Nothing about authoring changes:
pages are written on github.com as normal. The workflow in
.github/workflows/wiki-site.yml re-runs this on every wiki edit, so the published
site follows the wiki automatically.

Usage:
    python3 tools/wiki_site/build.py [--wiki PATH] [--out PATH] [--base URL]

Locally, clone the wiki beside the mod repo and it will be found automatically:
    git clone https://github.com/dementive/Chronicles-of-Omniluxia.wiki.git
"""

import argparse
import datetime as _dt
import glob
import html
import json
import os
import re
import shutil
import sys
import unicodedata
from collections import defaultdict

STEAM_WORKSHOP_URL = "https://steamcommunity.com/workshop/filedetails/?id=3154169256"

try:
    import markdown
except ImportError:
    sys.exit("This script needs python-markdown:  pip install markdown")

HERE = os.path.dirname(os.path.abspath(__file__))
# This script sits in <mod repo>/tools/wiki_site/.
MOD_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
# By default look for the wiki cloned as a sibling of the mod repo.
WIKI_ROOT = os.path.join(os.path.dirname(MOD_ROOT), "Chronicles-of-Omniluxia.wiki")
REGISTRY_PATH = os.path.join(HERE, "article_registry.json")

WIKI_URL_RE = re.compile(
    r"https?://github\.com/[^/]+/Chronicles-of-Omniluxia/wiki/([^)\s\"'#]+)(#[^)\s\"']*)?"
)

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
# Hub pages act as category roots: a page belongs to a hub that links to it.
# CATEGORIES is display order (nav, footer, index). CLASSIFY_PRIORITY is the
# order in which hubs get to claim a page, which is a different question --
# the tightly curated list hubs (Races, Religions) must claim before the
# essay-style hubs (Magic, History), which link to half the wiki in passing.

CATEGORIES = [
    # key,          hub page,               label,              blurb
    ("history",     "History",              "History",
     "The chronicle of Omniluxia, from the first ages to the fracture of empires."),
    ("magic",       "Magic",                "Magic",
     "Mana, spellcraft, and the disciplines that bend reality to imagination."),
    ("races",       "Races",                "Races",
     "The peoples of the world, from the near-human to the utterly alien."),
    ("religions",   "Religions",            "Religions",
     "Gods, prophets, pantheons, and the faiths that move nations."),
    ("cultures",    "Cultures",             "Cultures",
     "Tongues, traditions, and the culture groups that carry them."),
    ("regions",     "Regions",              "Regions",
     "Continents, seas, forests, and the storied places between them."),
    ("countries",   "Countries",            "Countries",
     "The realms of the world, standing and fallen alike — from the Zani "
     "successor states to the ancient dwarven kingdoms."),
    ("newworld",    "New-World",            "The New World",
     "The Jade Island, the Mushroom Isles, and the ten ancient races beyond the horizon."),
    ("characters",  "Important-Characters", "Characters",
     "Gods, demi-gods, emperors, generals, prophets, and rebels who shaped the age."),
]

CLASSIFY_PRIORITY = [
    "newworld",    # claims its ten races before the general Races hub can
    "races",
    "religions",
    "cultures",
    "characters",
    "regions",     # places before polities: a country page can mention a sea,
    "countries",   # but the sea is a region first
    "history",     # essay hubs last: they link everywhere incidentally
    "magic",
]

CATEGORY_LABEL = {k: lbl for k, _h, lbl, _b in CATEGORIES}
CATEGORY_BLURB = {k: b for k, _h, _l, b in CATEGORIES}
HUB_PAGES = {h: k for k, h, _l, _b in CATEGORIES}

FALLBACK_CATEGORY = "lore"
CATEGORY_LABEL[FALLBACK_CATEGORY] = "Lore"
CATEGORY_BLURB[FALLBACK_CATEGORY] = (
    "Deeper cuts of the setting: institutions, artefacts, and the odd corners "
    "of the world that resist tidy filing.")

# Pages no hub links to, or that the hubs classify wrongly. Everything that
# would otherwise fall through to the Lore catch-all is placed here by hand.
CATEGORY_OVERRIDES = {
    # The chronicle proper
    "Timeline": "history",
    "Historical-Events": "history",
    "Recent-Events": "history",
    "Luxterran-Calendar": "history",
    "Great-Collapse": "history",
    "Great-Northern-War": "history",
    "High-Elven-Civil-War": "history",
    "Jarenam-Invasion": "history",
    "The-Emergence": "history",
    "Tales-from-the-Zanic-Age": "history",

    # Spellcraft: the discipline pages are bare ability lists off Magic-Styles
    "Spell": "magic",
    "Magic-Styles": "magic",
    "Folk-Magic": "magic",
    "Divine-Wells": "magic",
    "Development-Rankings": "magic",
    "Aldic": "magic",
    "Amten": "magic",
    "Melodian": "magic",
    "Omnic": "magic",

    # Polities, standing and fallen
    "Anempanso": "countries",
    "Dwarven-Grandlands": "countries",
    "Eagelian-Kingdom": "countries",
    "Jarenam-Empire": "countries",
    "Northern-Empire": "countries",
    "Gevanni": "countries",

    # The dwarven founding pairs and the primordial parents are mythic figures
    "Ava": "characters",
    "Dwadais": "characters",
    "Ferja": "characters",
    "Frei": "characters",
    "Htgarth": "characters",
    "Summsir": "characters",
    "Stone-Father": "characters",
    "Mother-Earth": "characters",
    "Sumun-the-Great": "characters",

    # Languages belong with the cultures that speak them
    "Imperial-Zanic": "cultures",
    "Srroskku": "cultures",
    "Waal": "cultures",
    "Jaam": "cultures",

    # Places. Luxterra and Morrigon are Old World continents; the New World hub
    # names them only to say the New World lies beyond them.
    "Luxterra": "regions",
    "Morrigon": "regions",
    "Silver-Caves": "regions",
    "Eldritch-Forest": "regions",

    # Likewise the Zani Empire is referenced everywhere but belongs to no hub
    "Zani-Empire": "countries",
    "Kingdom-of-Srrorum": "countries",
    "Zanis": "characters",
    "Sumun": "characters",

    # Peoples
    "Silver-Halfling": "races",
    "Vampire": "races",
    "Wendaghan": "races",

    # Genuinely miscellaneous
    "Order-of-Sennmoggen": "lore",
    "Sources": "lore",
}

# Hero art assignment. The loading screens vary by terrain; each category draws
# from plates whose landscape suits it, so a category reads as visually coherent
# without every page looking identical.
CATEGORY_PLATES = {
    "history":    [13, 14, 9, 3],
    "magic":      [8, 12, 7],
    "races":      [6, 0, 8, 7, 4],
    "religions":  [8, 7, 5, 12],
    "cultures":   [6, 10, 5, 2],
    "regions":    [0, 4, 12, 2, 11],
    "countries":  [10, 13, 3, 1, 14],
    "newworld":   [6, 0, 7, 4],
    "characters": [13, 2, 10, 9],
    "lore":       [5, 11, 1, 12, 0],
}

FEATURED = ["Timeline", "Magic", "Zani-Empire", "Races", "Religions", "New-World"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_key(name):
    """Fold a page name to a comparison key.

    The wiki contains filenames using the Unicode hyphen U+2010 (Blood-Stained,
    Ular-Pria, Jarenam-Invasion) while links to them use an ASCII hyphen. Both
    must resolve to the same page.
    """
    n = unicodedata.normalize("NFKD", name)
    n = n.replace("‐", "-").replace("‑", "-").replace("–", "-")
    n = n.replace("_", "-").replace(" ", "-")
    n = re.sub(r"-+", "-", n)
    return n.strip("-").lower()


def slugify(name):
    """Output filename stem for a page."""
    return normalize_key(name)


def titleize(name):
    return name.replace("‐", "-").replace("_", " ").replace("-", " ").strip()


def strip_tags(s):
    return re.sub(r"<[^>]+>", "", s)


def reading_time(text):
    words = len(re.findall(r"\w+", text))
    return max(1, round(words / 220))


# ---------------------------------------------------------------------------
# Page model
# ---------------------------------------------------------------------------

class Page:
    def __init__(self, path, wiki_root):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.key = normalize_key(self.name)
        self.slug = slugify(self.name)
        self.is_home = self.key == "home"
        self.out = "index.html" if self.is_home else f"{self.slug}.html"
        with open(path, encoding="utf-8") as f:
            self.raw = f.read()

        self.title = titleize(self.name)
        self.epigraph = None
        self.epigraph_by = None
        self.category = None
        self.links = set()          # keys of pages this page links to
        self.backlinks = set()
        self.html = ""
        self.toc = []
        self.summary = ""
        self.aliases = []
        self.auto_links = set()
        self.words = len(re.findall(r"\w+", self.raw))
        self.mtime = os.path.getmtime(path)

    # -- parsing ------------------------------------------------------------
    def parse_front(self):
        """Pull the H1 title and a leading blockquote epigraph out of the body."""
        body = self.raw

        m = re.match(r"\s*#\s+(.+?)\s*\n", body)
        if m:
            self.title = m.group(1).strip()
            # Strip trailing decorative emoji from headings.
            self.title = re.sub(r"[\s\U0001F300-\U0001FAFF☀-➿]+$", "",
                                self.title).strip()
            body = body[m.end():]

        m = re.match(r"\s*>\s*(.+?)\n(?:\s*\n|$)", body, re.S)
        if m:
            quote = " ".join(l.strip().lstrip(">").strip()
                             for l in m.group(1).splitlines())
            # "text" - Attribution
            am = re.match(r'^[""\'"]?(.*?)[""\'"]?\s*[-—–]\s*(.+)$', quote)
            if am and len(am.group(2)) < 80:
                self.epigraph, self.epigraph_by = am.group(1).strip(), am.group(2).strip()
            else:
                self.epigraph = quote.strip('"“”')
            body = body[m.end():]

        self.body_md = body

    def collect_links(self, resolver):
        for m in WIKI_URL_RE.finditer(self.raw):
            k = normalize_key(m.group(1))
            if resolver(k):
                self.links.add(k)
        for m in re.finditer(r"\[[^\]]*\]\(([^):\s]+?)(#[^)]*)?\)", self.raw):
            k = normalize_key(m.group(1))
            if resolver(k):
                self.links.add(k)
        for m in re.finditer(r'<a href="([^":]+?)(#[^"]*)?"', self.raw):
            k = normalize_key(m.group(1))
            if resolver(k):
                self.links.add(k)
        self.links.discard(self.key)


# ---------------------------------------------------------------------------
# Site build
# ---------------------------------------------------------------------------

class Site:
    def __init__(self, wiki, out, base=""):
        self.wiki = wiki
        self.out = out
        self.base = base.rstrip("/")
        self.pages = []
        self.by_key = {}
        self.by_category = defaultdict(list)
        self.alias_targets = {}
        self.ambiguous_aliases = set()
        self.excluded_aliases = set()

    # -- load ---------------------------------------------------------------
    def load(self):
        files = sorted(f for f in os.listdir(self.wiki)
                       if f.endswith(".md") and not f.startswith("_"))
        for f in files:
            p = Page(os.path.join(self.wiki, f), self.wiki)
            p.parse_front()
            self.pages.append(p)
            self.by_key[p.key] = p

        # Some pages share an H1 with another page (Magic-Styles.md is headed
        # "# Magic", same as Magic.md). Two identically named entries in an
        # index are useless, so collisions fall back to the filename.
        seen = defaultdict(list)
        for p in self.pages:
            seen[p.title.lower()].append(p)
        for title, group in seen.items():
            if len(group) > 1:
                for p in group:
                    p.title = titleize(p.name)

        print(f"  loaded {len(self.pages)} pages")
        self.load_registry()

    def load_registry(self):
        """Build the canonical title/alias -> article lookup.

        Titles and filenames work automatically. article_registry.json only
        needs to contain adjectival, plural, historical, or otherwise unusual
        forms, plus exclusions for names that are too ambiguous to auto-link.
        """
        config = {"aliases": {}, "exclude": []}
        if os.path.isfile(REGISTRY_PATH):
            with open(REGISTRY_PATH, encoding="utf-8") as f:
                config = json.load(f)
        self.excluded_aliases = {a.casefold() for a in config.get("exclude", [])}

        candidates = defaultdict(set)
        for p in self.pages:
            for alias in {p.title, titleize(p.name)}:
                if len(alias.strip()) >= 3:
                    candidates[alias.strip().casefold()].add(p.key)
        for target_name, aliases in config.get("aliases", {}).items():
            target = self.by_key.get(normalize_key(target_name))
            if not target:
                raise ValueError(f"Registry alias target does not exist: {target_name}")
            for alias in aliases:
                candidates[alias.strip().casefold()].add(target.key)

        self.ambiguous_aliases = {a for a, keys in candidates.items() if len(keys) > 1}
        for alias, keys in candidates.items():
            if alias not in self.excluded_aliases and len(keys) == 1:
                self.alias_targets[alias] = self.by_key[next(iter(keys))]
        for alias, target in self.alias_targets.items():
            target.aliases.append(alias)
        print(f"  registry: {len(self.alias_targets)} aliases, "
              f"{len(self.ambiguous_aliases)} ambiguous, "
              f"{len(self.excluded_aliases)} excluded")

    def resolve(self, key):
        return self.by_key.get(key)

    # -- taxonomy -----------------------------------------------------------
    def classify(self):
        for p in self.pages:
            p.collect_links(self.resolve)

        # Backlink graph.
        for p in self.pages:
            for k in p.links:
                t = self.by_key.get(k)
                if t:
                    t.backlinks.add(p.key)

        hub_keys = {normalize_key(h): k for h, k in HUB_PAGES.items()}
        overrides = {normalize_key(n): c for n, c in CATEGORY_OVERRIDES.items()}

        for p in self.pages:
            if p.is_home:
                p.category = None
                continue
            if p.key in hub_keys:
                p.category = hub_keys[p.key]
                p.is_hub = True
                continue
            p.is_hub = False

            if p.key in overrides:
                p.category = overrides[p.key]
                continue

            cat = None
            for ckey in CLASSIFY_PRIORITY:
                hubname = next(h for k, h, _l, _b in CATEGORIES if k == ckey)
                hub = self.by_key.get(normalize_key(hubname))
                if hub and p.key in hub.links:
                    cat = ckey
                    break
            p.category = cat or FALLBACK_CATEGORY

        for p in self.pages:
            if p.category:
                self.by_category[p.category].append(p)
        for k in self.by_category:
            self.by_category[k].sort(key=lambda x: (not getattr(x, "is_hub", False),
                                                    x.title.lower()))

        for k, v in sorted(self.by_category.items(),
                           key=lambda kv: -len(kv[1])):
            print(f"    {CATEGORY_LABEL[k]:<14} {len(v):>3} pages")

    # -- rendering ----------------------------------------------------------
    def rewrite_links(self, text):
        """Point every wiki link at its generated page."""
        def abs_repl(m):
            key = normalize_key(m.group(1))
            anchor = m.group(2) or ""
            target = self.by_key.get(key)
            if target:
                return f"{target.out}{anchor}"
            return m.group(0)

        text = WIKI_URL_RE.sub(abs_repl, text)

        def rel_repl(m):
            label, dest, anchor = m.group(1), m.group(2), m.group(3) or ""
            if re.match(r"^(https?:|mailto:|#|/)", dest):
                return m.group(0)
            target = self.by_key.get(normalize_key(dest))
            if target:
                return f"[{label}]({target.out}{anchor})"
            return m.group(0)

        text = re.sub(r"\[([^\]]*)\]\(([^):\s]+?)(#[^)]*)?\)", rel_repl, text)

        def html_repl(m):
            dest, anchor = m.group(1), m.group(2) or ""
            if re.match(r"^(https?:|mailto:|#|/)", dest):
                return m.group(0)
            target = self.by_key.get(normalize_key(dest))
            if target:
                return f'<a href="{target.out}{anchor}"'
            return m.group(0)

        text = re.sub(r'<a href="([^":]+?)(#[^"]*)?"', html_repl, text)
        return text

    # Some pages -- the Timeline above all -- mark their sections with ad-hoc
    # plain-text delimiters rather than headings: "-Zanic Age-" for an age and
    # "==SERPENTINE GOLDEN AGE==" for a sub-era. Left alone these render as
    # literal text and the page gets no structure and no contents list.
    # Promoting them to real headings costs nothing and is reversible.
    ERA_MAIN = re.compile(r"^-\s*([^\[\]\n]{2,50}?)\s*-\s*$", re.M)
    ERA_SUB = re.compile(r"^==\s*(.+?)\s*==\s*$", re.M)

    def promote_era_markers(self, text):
        text = self.ERA_SUB.sub(lambda m: f"\n### {m.group(1).title()} "
                                          "{: .era-sub }\n", text)
        text = self.ERA_MAIN.sub(lambda m: f"\n## {m.group(1)} {{: .era }}\n", text)
        return text

    def render_markdown(self, p):
        md = markdown.Markdown(extensions=[
            "extra", "tables", "sane_lists", "toc", "attr_list",
        ], extension_configs={"toc": {"permalink": "¶",
                                     "toc_depth": "2-3"}})
        body = self.rewrite_links(p.body_md)
        body = self.promote_era_markers(body)
        out = md.convert(body)

        # Add contextual links after Markdown conversion so existing Markdown
        # links, headings, code, and other structural elements remain untouched.
        out = self.autolink_html(out, p)

        # External links open in a new tab and get a marker.
        out = re.sub(r'<a href="(https?://[^"]+)"',
                     r'<a href="\1" target="_blank" rel="noopener" class="ext"',
                     out)

        p.toc = getattr(md, "toc_tokens", [])
        p.html = out

        first = re.search(r"<p>(.*?)</p>", out, re.S)
        if first:
            txt = strip_tags(first.group(1)).strip()
            txt = re.sub(r"\s+", " ", txt)
            p.summary = txt
        elif p.epigraph:
            p.summary = p.epigraph
        return out

    AUTOLINK_BLOCKED = {"a", "code", "pre", "script", "style",
                        "h1", "h2", "h3", "h4", "h5", "h6"}

    def autolink_html(self, rendered, page):
        """Link the first meaningful occurrence of every known article alias."""
        aliases = [(alias, target) for alias, target in self.alias_targets.items()
                   if target.key != page.key]
        aliases.sort(key=lambda item: (-len(item[0]), item[0]))
        if not aliases:
            return rendered
        pattern = re.compile(
            r"(?<![\w])(" + "|".join(re.escape(a) for a, _ in aliases) + r")(?![\w])",
            re.I,
        )
        targets = {a: t for a, t in aliases}
        linked_targets = set()
        blocked = []
        parts = re.split(r"(<[^>]+>)", rendered)
        output = []

        for part in parts:
            if part.startswith("<"):
                close = re.match(r"</\s*([a-z0-9]+)", part, re.I)
                opening = re.match(r"<\s*([a-z0-9]+)(?:\s|>|/)", part, re.I)
                if close:
                    tag = close.group(1).lower()
                    if blocked and blocked[-1] == tag:
                        blocked.pop()
                elif opening and not part.rstrip().endswith("/>"):
                    tag = opening.group(1).lower()
                    if tag in self.AUTOLINK_BLOCKED:
                        blocked.append(tag)
                output.append(part)
                continue
            if blocked or not part.strip():
                output.append(part)
                continue

            def replace(match):
                alias = match.group(0).casefold()
                target = targets.get(alias)
                if not target or target.key in linked_targets:
                    return match.group(0)
                linked_targets.add(target.key)
                page.auto_links.add(target.key)
                return f'<a href="{target.out}" class="autolink">{match.group(0)}</a>'

            output.append(pattern.sub(replace, part))
        return "".join(output)

    def plate_for(self, p):
        cat = p.category or "lore"
        plates = CATEGORY_PLATES.get(cat, CATEGORY_PLATES["lore"])
        h = sum(ord(c) * (i + 7) for i, c in enumerate(p.key))
        return plates[h % len(plates)]

    # -- HTML shell ---------------------------------------------------------
    def nav(self, active=None):
        items = []
        for ckey, _hub, label, _b in CATEGORIES:
            cls = ' class="on"' if ckey == active else ""
            items.append(f'<a href="c-{ckey}.html"{cls}>{label}</a>')
        items.append('<a href="all.html#search">Search</a>')
        return "\n        ".join(items)

    def shell(self, *, title, desc, body, active=None, page_class="",
              canonical=""):
        year = _dt.date.today().year
        desc_a = html.escape(re.sub(r"\s+", " ", desc or "")[:180], quote=True)
        og = f"{self.base}/assets/img/sigils-800.jpg" if self.base else \
             "assets/img/sigils-800.jpg"
        return f"""<!DOCTYPE html>
<html lang="en" class="{page_class}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} &mdash; Chronicles of Omniluxia</title>
<meta name="description" content="{desc_a}">
<meta property="og:title" content="{html.escape(title)} — Chronicles of Omniluxia">
<meta property="og:description" content="{desc_a}">
<meta property="og:image" content="{og}">
<meta property="og:type" content="article">
<meta name="theme-color" content="#0a090e">
{f'<link rel="canonical" href="{canonical}">' if canonical else ''}
<link rel="icon" href="assets/img/logo.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;900&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Spectral:ital,wght@0,300;0,400;0,600;1,400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/style.css">
<script src="assets/site.js" defer></script>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>

<header class="topbar">
  <div class="topbar-in">
    <a class="brand" href="index.html" aria-label="Chronicles of Omniluxia — home">
      <img src="assets/img/logo.png" alt="Chronicles of Omniluxia" width="128" height="57">
    </a>
    <input type="checkbox" id="navtoggle" class="navtoggle">
    <label for="navtoggle" class="navburger" aria-label="Menu"><span></span></label>
    <nav class="nav">
        {self.nav(active)}
        <a href="all.html">All Pages</a>
    </nav>
  </div>
</header>

{body}

<footer class="footer">
  <div class="footer-in">
    <div class="footer-cols">
      <div>
        <h4>The World</h4>
        {"".join(f'<a href="c-{c}.html">{l}</a>' for c, _h, l, _b in CATEGORIES)}
      </div>
      <div>
        <h4>The Wiki</h4>
        <a href="index.html">Home</a>
        <a href="all.html">All Pages</a>
        <a href="credits.html">Credits &amp; Licences</a>
        <a href="sources.html">Sources</a>
      </div>
      <div>
        <h4>The Mod</h4>
        <a href="{STEAM_WORKSHOP_URL}" target="_blank" rel="noopener">Steam Workshop</a>
        <a href="https://github.com/dementive/Chronicles-of-Omniluxia" target="_blank" rel="noopener">Source Repository</a>
        <a href="https://github.com/dementive/Chronicles-of-Omniluxia/wiki" target="_blank" rel="noopener">Edit the wiki</a>
        <a href="https://www.youtube.com/watch?v=IlOfHsAzH1U" target="_blank" rel="noopener">Visual History</a>
      </div>
    </div>
    <div class="footer-rule"></div>
    <p class="footer-note">
      <strong>Chronicles of Omniluxia</strong> is a total-conversion fantasy mod for
      <em>Imperator: Rome</em>. This wiki is community-maintained &mdash; corrections and
      additions are welcome.<br>
      Site generated from the project wiki &middot; {year}. Artwork from the mod's own
      assets; see <a href="credits.html">credits</a>.
    </p>
  </div>
</footer>
</body>
</html>
"""

    # -- ornament -----------------------------------------------------------
    RULE = """<div class="rule" aria-hidden="true"><svg viewBox="0 0 240 16" preserveAspectRatio="none">
<path d="M0 8 H92" /><path d="M148 8 H240" />
<path d="M120 2 L128 8 L120 14 L112 8 Z" class="fill"/>
<path d="M104 8 L112 8 M128 8 L136 8"/>
<circle cx="100" cy="8" r="2" class="fill"/><circle cx="140" cy="8" r="2" class="fill"/>
</svg></div>"""

    # -- page types ---------------------------------------------------------
    def render_page(self, p):
        if not p.html:
            self.render_markdown(p)
        plate = self.plate_for(p)
        cat = p.category or "lore"
        catlabel = CATEGORY_LABEL.get(cat, "Lore")

        epi = ""
        if p.epigraph:
            by = (f'<cite>{html.escape(p.epigraph_by)}</cite>'
                  if p.epigraph_by else "")
            epi = (f'<blockquote class="epigraph">'
                   f'<p>{html.escape(p.epigraph)}</p>{by}</blockquote>')

        # Sidebar contents. Prefer H2s, but many long pages (Timeline is 8,000
        # words) are structured entirely with H3s, and those need a TOC most.
        toc_html = ""
        heads = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', p.html, re.S)
        if len(heads) < 3:
            h3s = re.findall(r'<h3 id="([^"]+)">(.*?)</h3>', p.html, re.S)
            if len(h3s) >= 4:
                heads = h3s
        if len(heads) >= 3:
            items = "".join(
                f'<li><a href="#{hid}">{strip_tags(txt).replace("¶","").strip()}</a></li>'
                for hid, txt in heads)
            toc_html = (f'<nav class="toc"><h4>On this page</h4>'
                        f'<ol>{items}</ol></nav>')

        # Explain relationships instead of presenting one opaque mixed list.
        def relation_group(title, keys, limit=10):
            targets = [self.by_key[k] for k in keys
                       if k in self.by_key and not self.by_key[k].is_home]
            targets.sort(key=lambda x: (x.category != p.category, x.title.casefold()))
            targets = targets[:limit]
            if not targets:
                return ""
            chips = "".join(
                f'<a class="chip" href="{t.out}"><span class="chip-c">'
                f'{CATEGORY_LABEL.get(t.category,"Lore")}</span>{html.escape(t.title)}</a>'
                for t in targets)
            return f'<div class="relation-group"><h4>{title}</h4><div class="chips">{chips}</div></div>'

        outgoing = (p.links | p.auto_links) - p.backlinks
        reciprocal = (p.links | p.auto_links) & p.backlinks
        incoming = p.backlinks - (p.links | p.auto_links)
        relation_html = (
            relation_group("Closely related", reciprocal) +
            relation_group("People, places, and ideas in this article", outgoing) +
            relation_group("Mentioned in", incoming)
        )
        rel_html = ""
        if relation_html:
            rel_html = (f'<section class="related">{self.RULE}'
                        f'<h3>Connections across the chronicle</h3>{relation_html}</section>')

        # Sibling navigation within the category.
        sibs = [s for s in self.by_category.get(cat, []) if s is not p]
        prevn = nextn = ""
        allc = self.by_category.get(cat, [])
        if p in allc:
            i = allc.index(p)
            if i > 0:
                q = allc[i - 1]
                prevn = (f'<a class="pn prev" href="{q.out}"><span>Previous</span>'
                         f'<strong>{html.escape(q.title)}</strong></a>')
            if i < len(allc) - 1:
                q = allc[i + 1]
                nextn = (f'<a class="pn next" href="{q.out}"><span>Next</span>'
                         f'<strong>{html.escape(q.title)}</strong></a>')

        rt = reading_time(p.raw)
        connection_count = len((p.links | p.auto_links) - {p.key})
        facts = f"""
      <aside class="article-facts" aria-label="Article summary">
        <h2>At a glance</h2>
        <dl>
          <div><dt>Subject</dt><dd>{catlabel}</dd></div>
          <div><dt>Reading time</dt><dd>{rt} min</dd></div>
          <div><dt>Linked topics</dt><dd>{connection_count}</dd></div>
          <div><dt>Mentioned by</dt><dd>{len(p.backlinks)}</dd></div>
        </dl>
      </aside>"""
        body = f"""
<article class="page">
  <div class="hero{' has-toc-hero' if toc_html else ''}" style="--plate:url('img/hero-{plate:02d}-1280.jpg');--plate-sm:url('img/hero-{plate:02d}-800.jpg')">
    <div class="hero-art" aria-hidden="true"></div>
    <div class="hero-in">
      <a class="eyebrow" href="c-{cat}.html">{catlabel}</a>
      <h1>{html.escape(p.title)}</h1>
      {epi}
      <p class="meta"><span>{p.words:,} words</span><i>&middot;</i><span>{rt} min read</span></p>
    </div>
    <div class="hero-fade" aria-hidden="true"></div>
  </div>

  <div class="wrap{' has-toc' if toc_html else ''}" id="main">
    {toc_html}
    <div class="prose">
      <nav class="breadcrumbs" aria-label="Breadcrumb"><a href="index.html">Home</a>
      <span aria-hidden="true">&rsaquo;</span><a href="c-{cat}.html">{catlabel}</a>
      <span aria-hidden="true">&rsaquo;</span><span>{html.escape(p.title)}</span></nav>
      {facts}
      {p.html}
    </div>
  </div>
  {rel_html}
  <nav class="pagenav">{prevn}{nextn}</nav>
</article>
"""
        return self.shell(title=p.title, desc=p.summary or p.epigraph or "",
                          body=body, active=cat, page_class=f"article category-{cat}")

    def render_category(self, ckey):
        pages = self.by_category[ckey]
        label = CATEGORY_LABEL[ckey]
        blurb = CATEGORY_BLURB[ckey]
        hub = next((p for p in pages if getattr(p, "is_hub", False)), None)
        plate = CATEGORY_PLATES[ckey][0]

        cards = []
        for p in pages:
            self.render_markdown(p) if not p.summary else None
            summ = html.escape((p.summary or "")[:190])
            if len(p.summary or "") > 190:
                summ += "&hellip;"
            hubbadge = '<span class="badge">Overview</span>' if getattr(p, "is_hub", False) else ""
            cards.append(f"""
      <a class="card" href="{p.out}" style="--tex:url('img/tex-{self.plate_for(p):02d}.jpg')">
        <div class="card-tex" aria-hidden="true"></div>
        <div class="card-in">
          <h3>{html.escape(p.title)}{hubbadge}</h3>
          <p>{summ}</p>
          <span class="card-more">Read &rarr;</span>
        </div>
      </a>""")

        body = f"""
<div class="page">
  <div class="hero hero-cat" style="--plate:url('img/hero-{plate:02d}-1280.jpg');--plate-sm:url('img/hero-{plate:02d}-800.jpg')">
    <div class="hero-art" aria-hidden="true"></div>
    <div class="hero-in">
      <span class="eyebrow static">Compendium</span>
      <h1>{label}</h1>
      <p class="lede">{blurb}</p>
      <p class="meta"><span>{len(pages)} entries</span></p>
    </div>
    <div class="hero-fade" aria-hidden="true"></div>
  </div>
  <div class="wrap wide" id="main">
    <div class="cards">{"".join(cards)}
    </div>
  </div>
</div>
"""
        return self.shell(title=label, desc=blurb, body=body, active=ckey,
                          page_class="listing")

    def render_home(self):
        home = self.by_key.get("home")
        total = len(self.pages)
        words = sum(p.words for p in self.pages)

        cats = []
        for ckey, _h, label, blurb in CATEGORIES:
            n = len(self.by_category.get(ckey, []))
            plate = CATEGORY_PLATES[ckey][0]
            cats.append(f"""
        <a class="ccard" href="c-{ckey}.html" style="--tex:url('img/tex-{plate:02d}.jpg')">
          <div class="ccard-tex" aria-hidden="true"></div>
          <div class="ccard-in">
            <h3>{label}</h3>
            <p>{blurb}</p>
            <span class="ccard-n">{n} entries</span>
          </div>
        </a>""")

        feats = []
        for name in FEATURED:
            p = self.by_key.get(normalize_key(name))
            if not p:
                continue
            if not p.summary:
                self.render_markdown(p)
            summ = html.escape((p.summary or "")[:150])
            feats.append(f"""
        <a class="feat" href="{p.out}" style="--tex:url('img/tex-{self.plate_for(p):02d}.jpg')">
          <div class="feat-tex" aria-hidden="true"></div>
          <div class="feat-in">
            <span class="feat-cat">{CATEGORY_LABEL.get(p.category,'Lore')}</span>
            <h3>{html.escape(p.title)}</h3>
            <p>{summ}&hellip;</p>
          </div>
        </a>""")

        body = f"""
<div class="page home">
  <div class="splash">
    <div class="splash-art" aria-hidden="true"></div>
    <div class="splash-in">
      <span class="splash-kicker">A total conversion for Imperator: Rome</span>
      <h1>Chronicles of <em>Omniluxia</em></h1>
      <p class="splash-sub">A world of high elves and gold dwarves, of mana and
      empire &mdash; three centuries after the Great Collapse shattered the old order.</p>
      <div class="splash-cta">
        <a class="btn primary" href="c-history.html">Enter the Chronicle</a>
        <a class="btn" href="all.html">Browse all {total} pages</a>
      </div>
      <div class="splash-stats">
        <div><strong>{total}</strong><span>articles</span></div>
        <div><strong>{words // 1000}k</strong><span>words of lore</span></div>
        <div><strong>670 LC</strong><span>year of the great collapse</span></div>
        <div><strong>1001 LC</strong><span>game start date</span></div>
      </div>
    </div>
    <div class="splash-scroll" aria-hidden="true"><span></span></div>
  </div>

  <section class="wrap wide" id="main">
    <header class="sec-head">
      <h2>The Compendium</h2>
      <p>Nine roads into the world. Follow whichever one calls.</p>
    </header>
    <div class="ccards">{"".join(cats)}
    </div>
  </section>

  <section class="wrap wide feats-sec">
    {self.RULE}
    <header class="sec-head">
      <h2>Start Here</h2>
      <p>The pages most people open first.</p>
    </header>
    <div class="feats">{"".join(feats)}
    </div>
  </section>

  <section class="wrap closing">
    {self.RULE}
    <p class="closing-txt">Chronicles of Omniluxia is a total-conversion fantasy mod for
    <em>Imperator: Rome</em>. Everything here is drawn from the project's own wiki, which
    remains open to contributions.</p>
    <div class="splash-cta">
      <a class="btn primary" href="{STEAM_WORKSHOP_URL}" target="_blank" rel="noopener">Get the mod</a>
      <a class="btn" href="https://github.com/dementive/Chronicles-of-Omniluxia/wiki" target="_blank" rel="noopener">Edit the wiki</a>
    </div>
  </section>
</div>
"""
        return self.shell(
            title="Home",
            desc="The wiki for Chronicles of Omniluxia, a total-conversion fantasy "
                 "mod for Imperator: Rome. Races, religions, empires, magic and "
                 "a thousand years of history.",
            body=body, page_class="homepage")

    def render_all(self):
        groups = []
        for ckey, _h, label, _b in CATEGORIES + [(FALLBACK_CATEGORY, None,
                                                  CATEGORY_LABEL[FALLBACK_CATEGORY],
                                                  None)]:
            pages = self.by_category.get(ckey, [])
            if not pages:
                continue
            links = "".join(
                f'<li data-title="{html.escape(p.title.casefold(), quote=True)}" '
                f'data-category="{ckey}"><a href="{p.out}">{html.escape(p.title)}</a>'
                f'<span>{p.words:,}w</span></li>' for p in pages)
            groups.append(f"""
      <section class="idx-group" data-category-group="{ckey}">
        <h2 id="{ckey}">{label} <span class="idx-n">{len(pages)}</span></h2>
        <ul class="idx-list">{links}</ul>
      </section>""")

        body = f"""
<div class="page">
  <div class="hero hero-cat hero-slim" style="--plate:url('img/hero-05-1280.jpg');--plate-sm:url('img/hero-05-800.jpg')">
    <div class="hero-art" aria-hidden="true"></div>
    <div class="hero-in">
      <span class="eyebrow static">Index</span>
      <h1>All Pages</h1>
      <p class="lede">Every article in the wiki, {len(self.pages)} in total, grouped by subject.</p>
    </div>
    <div class="hero-fade" aria-hidden="true"></div>
  </div>
  <div class="wrap wide" id="main">
    <section class="wiki-search" id="search" aria-labelledby="search-title">
      <label id="search-title" for="wiki-search">Search the chronicle</label>
      <input id="wiki-search" type="search" placeholder="Try Helluvian, Zani, Luxterra&hellip;" autocomplete="off">
      <div class="category-filters" role="group" aria-label="Filter by subject">
        <button type="button" class="on" data-filter="all">All</button>
        {''.join(f'<button type="button" data-filter="{key}">{label}</button>' for key, _hub, label, _blurb in CATEGORIES)}
      </div>
      <p class="search-status" aria-live="polite"></p>
    </section>
    {"".join(groups)}
  </div>
</div>
"""
        return self.shell(title="All Pages", desc="Complete index of the "
                          "Chronicles of Omniluxia wiki.", body=body,
                          page_class="listing")

    def render_credits(self):
        body = f"""
<div class="page">
  <div class="hero hero-cat hero-slim" style="--plate:url('img/hero-08-1280.jpg');--plate-sm:url('img/hero-08-800.jpg')">
    <div class="hero-art" aria-hidden="true"></div>
    <div class="hero-in">
      <span class="eyebrow static">Colophon</span>
      <h1>Credits &amp; Licences</h1>
      <p class="lede">Where the words and the pictures came from.</p>
    </div>
    <div class="hero-fade" aria-hidden="true"></div>
  </div>
  <div class="wrap" id="main">
    <div class="prose">
      <h2 id="text">Text</h2>
      <p>All lore on this site is written by the contributors to the
      <a href="https://github.com/dementive/Chronicles-of-Omniluxia/wiki" target="_blank" rel="noopener" class="ext">Chronicles of Omniluxia wiki</a>
      and is reproduced here unchanged. The wiki remains the canonical source; this
      site is a generated presentation layer over it.</p>
      <p class="callout"><strong>Get the mod.</strong> Chronicles of Omniluxia is
      available on the
      <a href="{STEAM_WORKSHOP_URL}" target="_blank" rel="noopener" class="ext">Steam Workshop</a>.</p>

      <h2 id="wiki-contributors">Wiki Contributors</h2>
      <p>The wiki is primarily maintained by <strong>Zorgoball</strong> and
      <strong>Starlord</strong>, with additions and corrections from the wider
      Omniluxia community.</p>

      <h2 id="mod-contributors">Omniluxia Contributors</h2>
      <p>Credits adapted from the
      <a href="{STEAM_WORKSHOP_URL}" target="_blank" rel="noopener" class="ext">Steam Workshop page</a>.</p>

      <h3 id="current-developers">Current Developers</h3>
      <p>Dementive, Izn, Zorgoball, MurderChicken, Starlord, Primal Aspid,
      Dulac14, Ratatosk.</p>

      <h3 id="original-developers">Original Developers</h3>
      <p>MrAdrianPL and Xangelo as lead developers, with MisterDiego27, POT,
      Pancaked_Src and AlthauSanafu, [REDi]1R CAPT Owlcoholic [A,D],
      MattTheLegoman, Boots, Hispania, anoldretiredelephant, Snowlet, Benjin,
      and Pyrrus.</p>

      <h3 id="invictus-team">Invictus Team</h3>
      <p>Snowlet, Jphiloponus, Mike Bittersteel, Dementive, Erik Erik,
      Hannibal_theCannibal, Izn, OmegaCorps, Palando, Parcipal, Paulus,
      Sealionforever, Thymos, Torugu, Tudhaliya, Aerozona, Diskianterezh,
      gmb360, Idonea, IhateTrains, Olivenkranz, Shocky27, Stallone, Typhion,
      Zorgoball, rickinator9, Acult, TheMadRegent, derekmark, MikeW.</p>

      <h3 id="artists">Artists</h3>
      <p>Aquizar, CrazyZombie, Fildez, Nerdman3000, RetconCrisis, Kailas.</p>

      <h3 id="translators">Translators</h3>
      <p>Apollon, Frank, Juanen, Julianus, Lemon, Machiavello, Pilar, Spikos,
      Vityviktor.</p>

      <h3 id="contributors">Contributors</h3>
      <p>Agamidae, Arkerios, Ben4Peters, DaFoogle, Diego I de Persia, Dustin,
      Hexon, Kalen, MattTheLegoman, Nebular, Pardo, Presidentstorm, Prometheus,
      Licarious, AtomicFission, Sobisonator, IsaacCat, NPK.</p>

      <h3 id="testers">Testers</h3>
      <p>Augustus_Caesar, bla, Brasidas, DDJR, FBI Agent, Jake_P, Jandoski,
      Lil_squindie, llamafanatic, Memer Nener, Pydras, Sav, Somebody, Trewajg,
      Tuko Tuko, Eel, Mateusz, Salt.</p>

      <h3 id="scholars">Scholars</h3>
      <p>Chehrazad, Derek, Felix Amiculus, Herodotus, INKRSN, Manny,
      QuietRustler, Sethos, Trarco.</p>

      <h3 id="special-thanks">Special Thanks</h3>
      <ul>
        <li>Snowlet for balance help and bug finding.</li>
        <li>Pureon for allowing the use of illustrations from the Lord of the Rings mod.</li>
        <li>Terrapass for the chasm terrain shader.</li>
        <li>Turplesmee for the main menu music.</li>
        <li>Agami for Better UI.</li>
      </ul>

      <h2 id="art">Artwork</h2>
      <p>Every image on this site comes from the mod's own files. Nothing is
      sourced from anywhere else. Each one is colour-graded at build time so that
      text stays legible over it, but the underlying art is unaltered:</p>
      <ul>
        <li><strong>Hero backgrounds</strong> &mdash; the fifteen loading screens from
        <code>gfx/loadingscreens/</code>, softened, desaturated and pushed toward the
        site's palette. The same plates, blurred further, fill the card backgrounds.</li>
        <li><strong>The sigil motif</strong> behind the front page &mdash;
        <code>gfx/interface/frontend/main_menu_background.dds</code>, the mod's main
        menu background. The site's entire colour scheme is sampled from it.</li>
        <li><strong>Wordmark</strong>, in the header and as the favicon &mdash;
        <code>gfx/interface/frontend/game_logo_main_menu.dds</code>.</li>
      </ul>
      <p>That is the complete list: three sources, seventeen original files. These
      assets belong to the Chronicles of Omniluxia project and are used here as part
      of it.</p>
      <p class="callout"><strong>Game assets.</strong> Some of this artwork derives
      from <em>Imperator: Rome</em>, which is a trademark of Paradox Interactive.
      Chronicles of Omniluxia is an unofficial fan project, not affiliated with or
      endorsed by Paradox, and is distributed under the terms that apply to mods for
      their games. If you add imagery to the wiki, keep it to the project's own work
      or to material that is public domain or CC0, and record it here.</p>

      <h2 id="type">Typography</h2>
      <ul>
        <li><strong>Cinzel</strong> by Natanael Gama &mdash; SIL Open Font Licence.</li>
        <li><strong>Cormorant Garamond</strong> by Christian Thalmann &mdash; SIL Open Font Licence.</li>
        <li><strong>Spectral</strong> by Production Type &mdash; SIL Open Font Licence.</li>
      </ul>
    </div>
  </div>
</div>
"""
        return self.shell(title="Credits", desc="Art, type, and text credits for "
                          "the Chronicles of Omniluxia wiki site.", body=body,
                          page_class="article")

    def render_404(self):
        body = """
<div class="page">
  <div class="hero" style="--plate:url('img/hero-12-1280.jpg');--plate-sm:url('img/hero-12-800.jpg')">
    <div class="hero-art" aria-hidden="true"></div>
    <div class="hero-in" id="main">
      <span class="eyebrow static">404</span>
      <h1>This road leads nowhere</h1>
      <p class="lede">The page you asked for is not in the chronicle. It may have been
      renamed, or it may never have been written.</p>
      <div class="splash-cta">
        <a class="btn primary" href="index.html">Return to the beginning</a>
        <a class="btn" href="all.html">See every page</a>
      </div>
    </div>
    <div class="hero-fade" aria-hidden="true"></div>
  </div>
</div>
"""
        return self.shell(title="Not Found", desc="Page not found.", body=body,
                          page_class="article")

    # -- write --------------------------------------------------------------
    def write(self):
        os.makedirs(self.out, exist_ok=True)

        # Static assets
        src_assets = os.path.join(HERE, "assets")
        dst_assets = os.path.join(self.out, "assets")
        os.makedirs(dst_assets, exist_ok=True)
        for f in os.listdir(src_assets):
            shutil.copy2(os.path.join(src_assets, f), os.path.join(dst_assets, f))

        # Render every article first. Auto-links then become part of the graph,
        # allowing later pages to display complete generated backlinks.
        for p in self.pages:
            if not p.is_home:
                self.render_markdown(p)
                p.links.update(p.auto_links)
        for p in self.pages:
            for key in p.auto_links:
                target = self.by_key.get(key)
                if target:
                    target.backlinks.add(p.key)

        n = 0
        for p in self.pages:
            if p.is_home:
                continue
            with open(os.path.join(self.out, p.out), "w", encoding="utf-8") as f:
                f.write(self.render_page(p))
            n += 1

        for ckey, _h, _l, _b in CATEGORIES:
            with open(os.path.join(self.out, f"c-{ckey}.html"), "w",
                      encoding="utf-8") as f:
                f.write(self.render_category(ckey))
            n += 1
        if self.by_category.get(FALLBACK_CATEGORY):
            with open(os.path.join(self.out, f"c-{FALLBACK_CATEGORY}.html"), "w",
                      encoding="utf-8") as f:
                f.write(self.render_category(FALLBACK_CATEGORY))
            n += 1

        for fn, fn_render in [("index.html", self.render_home),
                              ("all.html", self.render_all),
                              ("credits.html", self.render_credits),
                              ("404.html", self.render_404)]:
            with open(os.path.join(self.out, fn), "w", encoding="utf-8") as f:
                f.write(fn_render())
            n += 1

        # Tell GitHub Pages not to run Jekyll over this.
        open(os.path.join(self.out, ".nojekyll"), "w").close()

        search_index = [
            {"title": p.title, "url": p.out,
             "category": p.category or FALLBACK_CATEGORY,
             "summary": p.summary, "aliases": sorted(set(p.aliases))}
            for p in self.pages if not p.is_home
        ]
        with open(os.path.join(self.out, "search-index.json"), "w", encoding="utf-8") as f:
            json.dump(search_index, f, ensure_ascii=False, separators=(",", ":"))

        # Sitemap
        if self.base:
            today = _dt.date.today().isoformat()
            urls = ["index.html", "all.html", "credits.html"]
            urls += [f"c-{c}.html" for c, _h, _l, _b in CATEGORIES]
            urls += [p.out for p in self.pages if not p.is_home]
            xml = ['<?xml version="1.0" encoding="UTF-8"?>',
                   '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
            for u in urls:
                xml.append(f"  <url><loc>{self.base}/{u}</loc>"
                           f"<lastmod>{today}</lastmod></url>")
            xml.append("</urlset>")
            with open(os.path.join(self.out, "sitemap.xml"), "w",
                      encoding="utf-8") as f:
                f.write("\n".join(xml))

        print(f"  wrote {n} html files to {self.out}")
        return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", default=os.environ.get("OMNI_WIKI", WIKI_ROOT),
                    help="Directory of wiki markdown (default: the wiki repo "
                         "cloned beside this one)")
    ap.add_argument("--out", default=os.path.join(MOD_ROOT, "docs"),
                    help="Where to write the site (default: docs/)")
    ap.add_argument("--base", default="", help="Canonical base URL for sitemap")
    a = ap.parse_args()

    if not os.path.isdir(a.wiki):
        sys.exit(f"Wiki repo not found: {a.wiki}\nClone it with:\n"
                 f"  git clone https://github.com/dementive/"
                 f"Chronicles-of-Omniluxia.wiki.git\n"
                 f"then pass --wiki PATH or set OMNI_WIKI.")
    if not glob.glob(os.path.join(a.wiki, "*.md")):
        sys.exit(f"No markdown files in {a.wiki}")

    print("Chronicles of Omniluxia - building site")
    print(f"  wiki: {a.wiki}")
    s = Site(a.wiki, a.out, a.base)
    s.load()
    s.classify()
    s.write()
    print("Done.")


if __name__ == "__main__":
    main()
