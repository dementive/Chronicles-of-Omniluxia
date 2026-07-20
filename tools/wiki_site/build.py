#!/usr/bin/env python3
"""
Chronicles of Omniluxia - static wiki site generator.

Reads the project wiki (a flat pile of markdown files) and renders a designed
static site into docs/, which GitHub Pages serves.

The wiki remains the single source of truth. Nothing about authoring changes:
pages are written on github.com as normal. The workflow in
.github/workflows/wiki-site.yml re-runs this on every wiki edit, so the
published site follows the wiki automatically.

Usage:
    python3 tools/wiki_site/build.py [--wiki PATH] [--out PATH] [--base URL]

Locally, clone the wiki beside the mod repo and it will be found automatically:
    git clone https://github.com/dementive/Chronicles-of-Omniluxia.wiki.git
"""

import argparse
import datetime as _dt
import glob
import html
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import defaultdict

try:
    import markdown
except ImportError:
    sys.exit("This script needs python-markdown:  pip install markdown")

HERE = os.path.dirname(os.path.abspath(__file__))
# This script sits in <mod repo>/tools/wiki_site/.
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
# By default look for the wiki cloned as a sibling of the mod repo.
WIKI_ROOT = os.path.join(os.path.dirname(REPO_ROOT), "Chronicles-of-Omniluxia.wiki")

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
    ("items",       "Items",                "Items",
     "Equipment, artifacts, and the markets where characters outfit themselves and their realms."),
    ("bloodlines",  "Bloodlines",           "Bloodlines",
     "Inherited legacies of legendary dynasties, founders, prophets, conquerors, and sacred lineages."),
    ("races",       "Races",                "Races",
     "The peoples of the world, from the near-human to the utterly alien."),
    ("religions",   "Religions",            "Religions",
     "Gods, prophets, pantheons, and the faiths that move nations."),
    ("cultures",    "Cultures",             "Cultures",
     "Tongues, traditions, and the culture groups that carry them."),
    ("continents",  "Continents",           "Continents",
     "Luxterra, Morrigon, Arteon, Eptelon, Horteon, Polaria, and Austropetolia."),
    ("regions",     "Regions",              "Regions",
     "Continents, seas, forests, and the storied places between them."),
    ("countries",   "Countries",            "Countries",
     "The realms of the world, standing and fallen alike — from the Zani "
     "successor states to the ancient dwarven kingdoms."),
    ("characters",  "Important-Characters", "Characters",
     "Gods, demi-gods, emperors, generals, prophets, and rebels who shaped the age."),
]

CLASSIFY_PRIORITY = [
    "races",
    "religions",
    "cultures",
    "characters",
    "continents",
    "regions",     # places before polities: a country page can mention a sea,
    "countries",   # but the sea is a region first
    "history",     # essay hubs last: they link everywhere incidentally
    "magic",
    "items",
    "bloodlines",
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

    # Equipment and item systems
    "Item-Instructions": "items",
    "Artifacts-and-Holy-Site-Items": "items",
    "City-Instructions": "items",

    # Inherited dynasties and legendary lineages
    "Bloodlines": "bloodlines",

    # Polities, standing and fallen
    "Anempanso": "countries",
    "Dwarven-Grandlands": "countries",
    "Eagelian-Kingdom": "countries",
    "Jarenam-Empire": "countries",
    "Kingdom-of-Edis": "countries",
    "Northern-Empire": "countries",
    "Rohenoa-and-Rohevia": "countries",
    "Gevanni": "countries",
    "Zainudian-World": "countries",

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

    # Old World continents and continent-scale lands.
    "Luxterra": "continents",
    "Morrigon": "continents",
    "Arteon": "continents",
    "Eptelon": "continents",
    "Horteon": "continents",
    "Polaria": "continents",
    "Austropetolia": "continents",

    # Places below continental scale.
    "Borderlands": "regions",
    "Silver-Caves": "regions",
    "Eldritch-Forest": "regions",
    "New-World": "regions",

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
    "Great-Wonders": "items",
    "Magic-Instructions": "magic",

    # Explicitly typed pages. Hub links are useful navigation, but must not
    # silently change the subject of a page when a new cross-reference lands.
    "Green-Valley": "regions",
    "Peaceful-Valley": "regions",
    "Marenica-Sea": "regions",
    "Kino": "religions",
    "Theolosius": "characters",
    "Melodias": "characters",
    "Wielkopan": "characters",
    "Wishtheon": "characters",
    "Naathran": "characters",
    "Bachin": "characters",
    "Jaoz": "characters",
    "Zerywani": "races",
    "Character-Interactions": "lore",
    "Getting-Started": "lore",
    "Gameplay-Systems": "lore",
    "Glossary": "lore",
    "Changelog": "history",
}

# Hero art assignment. The loading screens vary by terrain; each category draws
# from plates whose landscape suits it, so a category reads as visually coherent
# without every page looking identical.
CATEGORY_PLATES = {
    "history":    [13, 14, 9, 3],
    "magic":      [8, 12, 7],
    "items":      [5, 11, 1, 12],
    "bloodlines": [13, 2, 10, 9],
    "races":      [6, 0, 8, 7, 4],
    "religions":  [8, 7, 5, 12],
    "cultures":   [6, 10, 5, 2],
    "continents": [0, 11, 4, 12, 2],
    "regions":    [0, 4, 12, 2, 11],
    "countries":  [10, 13, 3, 1, 14],
    "characters": [13, 2, 10, 9],
    "lore":       [5, 11, 1, 12, 0],
}

FEATURED = ["Timeline", "Magic", "Items", "Bloodlines", "Zani-Empire", "Races", "Religions", "Regions"]

ALIASES = {"sumun": "sumun-the-great"}


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
        self.is_alias = self.key in ALIASES
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
        self.words = len(re.findall(r"\w+", self.raw))
        self.mtime = os.path.getmtime(path)
        self.primary_category = None

    # -- parsing ------------------------------------------------------------
    def parse_front(self):
        """Pull the H1 title and a leading blockquote epigraph out of the body."""
        body = self.raw

        # Optional invisible metadata for pages whose subject is broader than
        # the first hub that happens to link to them. Example:
        # <!-- primary-category: regions -->
        category = re.search(r"<!--\s*primary-category:\s*([a-z-]+)\s*-->", body,
                             re.I)
        if category:
            self.primary_category = category.group(1).lower()

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

    def canonical_for(self, output_name):
        return f"{self.base}/{output_name}" if self.base else ""

    def public_pages(self):
        return [p for p in self.pages if not p.is_home and not p.is_alias]

    def clean_output(self):
        """Remove stale generated files before writing a fresh site.

        The output is always a generated directory inside this repository.
        Refusing paths outside the repository prevents an accidental clean of
        an unrelated directory.
        """
        out_abs = os.path.abspath(self.out)
        root_abs = REPO_ROOT
        try:
            inside = os.path.commonpath([out_abs, root_abs]) == root_abs
        except ValueError:
            inside = False
        if not inside or out_abs == root_abs:
            raise SystemExit("Refusing to clean an output directory outside the repository")
        if not os.path.isdir(out_abs):
            return
        # A publish workflow may grade the mod's art into docs/assets/img
        # immediately before this script runs. Preserve that generated bundle
        # while removing every other stale output.
        art_backup = None
        art_path = os.path.join(out_abs, "assets", "img")
        if os.path.isdir(art_path):
            art_backup = tempfile.mkdtemp(prefix="omniluxia-art-")
            shutil.copytree(art_path, os.path.join(art_backup, "img"))
        for name in os.listdir(out_abs):
            path = os.path.join(out_abs, name)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        if art_backup:
            os.makedirs(os.path.join(out_abs, "assets"), exist_ok=True)
            shutil.copytree(os.path.join(art_backup, "img"), art_path)
            shutil.rmtree(art_backup, ignore_errors=True)

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
            if p.is_home or p.is_alias:
                p.category = None
                continue
            if p.key in hub_keys:
                p.category = hub_keys[p.key]
                p.is_hub = True
                continue
            p.is_hub = False

            if p.primary_category in CATEGORY_LABEL:
                p.category = p.primary_category
                continue

            if p.key in overrides:
                p.category = overrides[p.key]
                continue
            if "<!-- country-data:start -->" in p.raw:
                p.category = "countries"
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

        # The data pages contain hand-authored HTML generated from game
        # localization. Normalize the few recurring presentation artifacts at
        # the presentation boundary so the wiki source remains traceable while
        # the public site stays readable.
        out = self.clean_generated_markup(out)
        out = self.enhance_data_entries(out)
        out = self.link_data_references(out, p)

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

    @staticmethod
    def clean_generated_markup(out):
        replacements = {
            "Agressive": "Aggressive",
            "Happyness": "Happiness",
            "negative_gw_workrate_percent_svalue_minor": "Great Work Total Workrate Modifier (minor)",
            "gw_fixed_prestige_svalue_minor": "Great Work Fixed Prestige Modifier (minor)",
            "Â·": "·",
            " Â·": " ·",
        }
        for old, new in replacements.items():
            out = out.replace(old, new)
        out = re.sub(r"\s+([,.;])", r"\1", out)
        return out

    @staticmethod
    def enhance_data_entries(out):
        """Give raw data entries stable anchors and searchable metadata."""
        seen = set()

        def entry_repl(match):
            opening, body = match.group(1), match.group(2)
            title = re.search(r"<h2(?:\s[^>]*)?>(.*?)</h2>", body, re.S)
            key = re.search(r"<code>(.*?)</code>", body, re.S)
            raw_id = strip_tags(key.group(1) if key else (title.group(1) if title else "entry"))
            raw_id = html.unescape(raw_id).strip()
            ident = slugify(raw_id) or "entry"
            if ident in seen:
                n = 2
                while f"{ident}-{n}" in seen:
                    n += 1
                ident = f"{ident}-{n}"
            seen.add(ident)
            opening = opening.replace('class="data-entry"',
                                      f'class="data-entry" id="entry-{ident}" data-entry-key="{html.escape(raw_id, quote=True)}"', 1)
            if title and ' id="' not in title.group(0):
                replacement = title.group(0).replace("<h2", f'<h2 id="entry-{ident}"', 1)
                body = body.replace(title.group(0), replacement, 1)
            return opening + body + "</article>"

        return re.sub(r'(<article\s+class="data-entry">)(.*?)</article>',
                      entry_repl, out, flags=re.S)

    def link_data_references(self, out, page):
        """Link known people, places, and polities inside generated entries.

        Only data-entry headings, descriptions, and location/controller fields
        are touched. Mechanics and localization keys remain literal game data.
        """
        targets = []
        for target in self.pages:
            if target is page or target.is_home or target.is_alias or target.is_hub:
                continue
            label = target.title.strip()
            if len(label) < 5 or label.casefold() in {"overview", "history", "magic"}:
                continue
            targets.append((label, target.out))
        targets.sort(key=lambda item: len(item[0]), reverse=True)

        def field_repl(match):
            tag, attrs, content, closing = match.groups()
            attrs = attrs or ""
            if "mechanics" in attrs:
                return match.group(0)
            if "<a " in content:
                return match.group(0)
            for label, href in targets:
                pattern = re.compile(rf"(?<![\w>])({re.escape(label)})(?![\w<])")
                content = pattern.sub(lambda m: f'<a href="{href}">{m.group(1)}</a>', content)
            return f"<{tag}{attrs}>{content}{closing}"

        entry_re = re.compile(
            r"<(h2|p)(\s[^>]*)?>(.*?)((?:</h2>|</p>))", re.S | re.I)
        return entry_re.sub(field_repl, out)

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
        return "\n        ".join(items)

    def shell(self, *, title, desc, body, active=None, page_class="",
              canonical="", og_type="article"):
        year = _dt.date.today().year
        desc_a = html.escape(re.sub(r"\s+", " ", desc or "")[:180], quote=True)
        full_title = "Chronicles of Omniluxia" if "homepage" in page_class else f"{title} — Chronicles of Omniluxia"
        og = f"{self.base}/assets/img/sigils-800.jpg" if self.base else \
             "assets/img/sigils-800.jpg"
        return f"""<!DOCTYPE html>
<html lang="en" class="{page_class}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(full_title)}</title>
<meta name="description" content="{desc_a}">
<meta property="og:title" content="{html.escape(full_title)}">
<meta property="og:description" content="{desc_a}">
<meta property="og:image" content="{og}">
<meta property="og:type" content="{og_type}">
{f'<meta property="og:url" content="{html.escape(canonical, quote=True)}">' if canonical else ''}
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{html.escape(full_title)}">
<meta name="twitter:description" content="{desc_a}">
<meta name="twitter:image" content="{og}">
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
    <details class="nav-menu" open>
      <summary class="navburger"><span></span><span class="sr-only">Open navigation</span></summary>
      <nav class="nav" id="site-nav" aria-label="Primary">
          {self.nav(active)}
          <a href="all.html">All Pages</a>
      </nav>
    </details>
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
        <a href="https://github.com/dementive/Chronicles-of-Omniluxia" target="_blank" rel="noopener">Repository</a>
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
        self.render_markdown(p)
        plate = self.plate_for(p)
        cat = p.category or "lore"
        catlabel = CATEGORY_LABEL.get(cat, "Lore")
        data_count = len(re.findall(r'<article\s+class="data-entry"', p.html))

        epi = ""
        if p.epigraph:
            by = (f'<cite>{html.escape(p.epigraph_by)}</cite>'
                  if p.epigraph_by else "")
            epi = (f'<blockquote class="epigraph">'
                   f'<p>{html.escape(p.epigraph)}</p>{by}</blockquote>')

        # Sidebar contents. Prefer the parser's TOC tokens, with an HTML
        # fallback for raw headings and generated era markers.
        toc_html = ""
        heads = []

        def flatten_toc(tokens):
            for token in tokens:
                if token.get("level", 0) in (2, 3):
                    heads.append((token.get("id", ""), token.get("name", "")))
                flatten_toc(token.get("children", []))

        if data_count < 20:
            flatten_toc(p.toc)
        if not heads and data_count < 20:
            heads = re.findall(r'<h2\b[^>]*\bid="([^"]+)"[^>]*>(.*?)</h2>',
                               p.html, re.S)
        if len(heads) < 3:
            h3s = re.findall(r'<h3\b[^>]*\bid="([^"]+)"[^>]*>(.*?)</h3>',
                             p.html, re.S)
            if len(h3s) >= 4:
                heads = h3s
        if len(heads) >= 3:
            items = "".join(
                f'<li><a href="#{hid}">{strip_tags(txt).replace("¶","").strip()}</a></li>'
                for hid, txt in heads)
            toc_html = (f'<nav class="toc"><h4>On this page</h4>'
                        f'<ol>{items}</ol></nav>')

        # Related: authored outgoing links first, then useful backlinks. Hub
        # pages are deliberately de-prioritized so cross-references remain
        # semantic rather than becoming a list of index pages.
        rel = []
        seen = set()
        ordered_keys = list(sorted(p.links)) + list(sorted(p.backlinks))
        for k in ordered_keys:
            t = self.by_key.get(k)
            if (t and not t.is_home and not t.is_alias and t.key not in seen
                    and (t.key in p.links or not getattr(t, "is_hub", False))):
                seen.add(t.key)
                rel.append(t)
        rel = rel[:12]
        rel_html = ""
        if rel:
            chips = "".join(
                f'<a class="chip" href="{t.out}"><span class="chip-c">'
                f'{CATEGORY_LABEL.get(t.category,"Lore")}</span>{html.escape(t.title)}</a>'
                for t in rel)
            rel_html = (f'<section class="related">{self.RULE}'
                        f'<h3>Threads leading elsewhere</h3>'
                        f'<div class="chips">{chips}</div></section>')

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
        data_tools = ""
        if data_count >= 20:
            data_tools = f'''<div class="data-tools" role="search">
  <label for="entry-search"><span>Filter this index</span>
    <input id="entry-search" type="search" placeholder="Search names, cultures, effects..." autocomplete="off" data-entry-search>
  </label>
  <span class="data-count" data-entry-count>{data_count:,} entries</span>
</div>'''
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
      {data_tools}
      {p.html}
    </div>
  </div>
  {rel_html}
  <nav class="pagenav">{prevn}{nextn}</nav>
</article>
"""
        return self.shell(title=p.title, desc=p.summary or p.epigraph or "",
                          body=body, active=cat, page_class="article",
                          canonical=self.canonical_for(p.out))

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
            is_hub = getattr(p, "is_hub", False)
            card_title = "Overview" if is_hub else p.title
            action = f"Read {label}" if is_hub else "Read"
            cards.append(f"""
      <a class="card" href="{p.out}" style="--tex:url('img/tex-{self.plate_for(p):02d}.jpg')">
        <div class="card-tex" aria-hidden="true"></div>
        <div class="card-in">
          <h3>{html.escape(card_title)}</h3>
          <p>{summ}</p>
          <span class="card-more">{html.escape(action)} &rarr;</span>
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
      <p class="meta"><span>{len(pages)} {'entry' if len(pages) == 1 else 'entries'}</span></p>
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
                          page_class="listing",
                          canonical=self.canonical_for(f"c-{ckey}.html"),
                          og_type="website")

    def render_home(self):
        home = self.by_key.get("home")
        public = self.public_pages()
        total = len(public)
        words = sum(p.words for p in public)

        cats = []
        for ckey, _h, label, blurb in CATEGORIES:
            n = len(self.by_category.get(ckey, []))
            plate = CATEGORY_PLATES[ckey][0]
            count_word = "entry" if n == 1 else "entries"
            cats.append(f"""
        <a class="ccard" href="c-{ckey}.html" style="--tex:url('img/tex-{plate:02d}.jpg')">
          <div class="ccard-tex" aria-hidden="true"></div>
          <div class="ccard-in">
            <h3>{label}</h3>
            <p>{blurb}</p>
            <span class="ccard-n">{n} {count_word}</span>
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

        guide_cards = []
        for name in ("Getting-Started", "Gameplay-Systems", "Glossary", "Changelog"):
            p = self.by_key.get(normalize_key(name))
            if not p:
                continue
            if not p.summary:
                self.render_markdown(p)
            guide_cards.append(
                f'<a class="quick-link" href="{p.out}"><strong>{html.escape(p.title)}</strong>'
                f'<span>{html.escape((p.summary or "").strip()[:150])}</span></a>')

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
        <div><strong>{words // 1000}k</strong><span>words indexed</span></div>
        <div><strong>670 LC</strong><span>year of the great collapse</span></div>
        <div><strong>1000 LC</strong><span>game start date</span></div>
      </div>
    </div>
    <div class="splash-scroll" aria-hidden="true"><span></span></div>
  </div>

  <section class="wrap wide" id="main">
    <header class="sec-head">
      <h2>The Compendium</h2>
      <p>{len(CATEGORIES)} roads into the world. Follow whichever one calls.</p>
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

  <section class="wrap wide quick-sec">
    {self.RULE}
    <header class="sec-head">
      <h2>Orientation</h2>
      <p>Practical guides for finding your footing.</p>
    </header>
    <div class="quick-links">{"".join(guide_cards)}</div>
  </section>

  <section class="wrap closing">
    {self.RULE}
    <p class="closing-txt">Chronicles of Omniluxia is a total-conversion fantasy mod for
    <em>Imperator: Rome</em>. Everything here is drawn from the project's own wiki, which
    remains open to contributions.</p>
    <div class="splash-cta">
      <a class="btn primary" href="https://steamcommunity.com/workshop/filedetails/?id=3154169256" target="_blank" rel="noopener">Subscribe on Steam</a>
      <a class="btn" href="https://github.com/dementive/Chronicles-of-Omniluxia" target="_blank" rel="noopener">Source &amp; development</a>
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
            body=body, page_class="homepage",
            canonical=self.canonical_for("index.html"), og_type="website")

    def render_all(self):
        groups = []
        for ckey, _h, label, _b in CATEGORIES + [(FALLBACK_CATEGORY, None,
                                                  CATEGORY_LABEL[FALLBACK_CATEGORY],
                                                  None)]:
            pages = self.by_category.get(ckey, [])
            if not pages:
                continue
            links = "".join(
                f'<li data-index-item><a href="{p.out}">{html.escape(p.title)}</a>'
                f'<span>{p.words:,}w</span></li>' for p in pages)
            groups.append(f"""
      <section class="idx-group">
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
  <div class="wrap wide" id="main">{"".join(groups)}
    <div class="index-tools" role="search">
      <label for="index-search"><span>Search the index</span>
        <input id="index-search" type="search" placeholder="Search pages..." autocomplete="off" data-index-search>
      </label>
      <span class="data-count" data-index-count>{len(self.public_pages()):,} pages</span>
    </div>
  </div>
</div>
"""
        return self.shell(title="All Pages", desc="Complete index of the "
                          "Chronicles of Omniluxia wiki.", body=body,
                          page_class="listing",
                          canonical=self.canonical_for("all.html"),
                          og_type="website")

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
      <h2 id="contributors">Omniluxia Contributors</h2>
      <p>Credits adapted from the
      <a href="https://steamcommunity.com/workshop/filedetails/?id=3154169256" target="_blank" rel="noopener" class="ext">Steam Workshop page</a>.</p>
      <p><strong>Current Developers.</strong> Dementive, Izn, Zorgoball,
      MurderChicken, Starlord, Pureon, Anbeeld, Primal Aspid, Dulac14, Ratatosk.</p>

      <h2 id="text">Text</h2>
      <p>All lore on this site is written by the contributors to the
      <a href="https://github.com/dementive/Chronicles-of-Omniluxia/wiki" target="_blank" rel="noopener" class="ext">Chronicles of Omniluxia wiki</a>
      and is reproduced here unchanged. The wiki remains the canonical source; this
      site is a generated presentation layer over it.</p>

      <h2 id="art">Artwork</h2>
      <p>Every image on this site is taken from the mod files used by the project.
      Before redistributing the generated site, confirm the project has permission
      to publish each source image and retain any upstream attribution required by
      the mod or its contributors. Each one is colour-graded at build time so that
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
      <p>That is the current list: three source locations, seventeen files. The
      source files remain subject to the permissions and attribution terms that
      apply to the mod and its credited contributors.</p>
      <p class="callout"><strong>Imperator: Rome.</strong> <em>Imperator: Rome</em>
      is a game developed and published by Paradox Interactive. Imperator: Rome,
      its name, trademarks, and original copyrighted material belong to Paradox
      Interactive and their respective rights holders. Chronicles of Omniluxia is
      an unofficial fan project, not affiliated with or endorsed by Paradox, and is
      distributed under the terms that apply to mods for their games. If you add
      imagery to the wiki, keep it to the project's own work or to material that is
      public domain or CC0, and record it here.</p>

      <h2 id="type">Typography</h2>
      <ul>
        <li><strong>Cinzel</strong> by Natanael Gama &mdash; SIL Open Font Licence.</li>
        <li><strong>Cormorant Garamond</strong> by Christian Thalmann &mdash; SIL Open Font Licence.</li>
        <li><strong>Spectral</strong> by Production Type &mdash; SIL Open Font Licence.</li>
      </ul>

      <h2 id="build">How this site is built</h2>
      <p>A Python script reads the wiki's markdown files, classifies them from the
      hub pages, rewrites the wiki links, and renders them through a single template.
      The site uses a tiny progressive-enhancement script for navigation and search,
      with no build framework. To regenerate after editing the wiki:</p>
      <pre><code>git pull
python3 site/build.py</code></pre>
      <p>The artwork is only regenerated when the mod's own art changes, and needs to
      be pointed at the mod repository, since that is where the source
      <code>.dds</code> files live:</p>
      <pre><code>python3 tools/wiki_site/build_art.py</code></pre>
    </div>
  </div>
</div>
"""
        return self.shell(title="Credits", desc="Art, type, and text credits for "
                          "the Chronicles of Omniluxia wiki site.", body=body,
                          page_class="article",
                          canonical=self.canonical_for("credits.html"),
                          og_type="website")

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
        self.clean_output()
        os.makedirs(self.out, exist_ok=True)

        def save(path, content):
            # Keep generated files clean for diffs and downstream publishing.
            content = content.replace("\ufffd", "-")
            content = re.sub(r"[ \t]+(?=\n)", "", content).rstrip() + "\n"
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

        # Static assets
        src_assets = os.path.join(HERE, "assets")
        dst_assets = os.path.join(self.out, "assets")
        os.makedirs(dst_assets, exist_ok=True)
        for f in os.listdir(src_assets):
            src = os.path.join(src_assets, f)
            dst = os.path.join(dst_assets, f)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        n = 0
        for p in self.pages:
            if p.is_home or p.is_alias:
                continue
            save(os.path.join(self.out, p.out), self.render_page(p))
            n += 1

        # Keep legacy links alive without indexing aliases as duplicate
        # articles.
        for p in self.pages:
            if not p.is_alias:
                continue
            target = self.by_key.get(ALIASES[p.key])
            if not target:
                continue
            canonical = self.canonical_for(target.out)
            location = target.out
            save(os.path.join(self.out, p.out), self.render_redirect(p.title, location, canonical))
            n += 1

        for ckey, _h, _l, _b in CATEGORIES:
            save(os.path.join(self.out, f"c-{ckey}.html"), self.render_category(ckey))
            n += 1
        if self.by_category.get(FALLBACK_CATEGORY):
            save(os.path.join(self.out, f"c-{FALLBACK_CATEGORY}.html"), self.render_category(FALLBACK_CATEGORY))
            n += 1

        for fn, fn_render in [("index.html", self.render_home),
                              ("all.html", self.render_all),
                              ("credits.html", self.render_credits),
                              ("404.html", self.render_404)]:
            save(os.path.join(self.out, fn), fn_render())
            n += 1

        # Tell GitHub Pages not to run Jekyll over this.
        open(os.path.join(self.out, ".nojekyll"), "w").close()

        # Sitemap
        if self.base:
            today = _dt.date.today().isoformat()
            urls = ["index.html", "all.html", "credits.html"]
            urls += [f"c-{c}.html" for c, _h, _l, _b in CATEGORIES]
            if self.by_category.get(FALLBACK_CATEGORY):
                urls.append(f"c-{FALLBACK_CATEGORY}.html")
            urls += [p.out for p in self.public_pages()]
            xml = ['<?xml version="1.0" encoding="UTF-8"?>',
                   '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
            for u in urls:
                xml.append(f"  <url><loc>{self.base}/{u}</loc>"
                           f"<lastmod>{today}</lastmod></url>")
            xml.append("</urlset>")
            with open(os.path.join(self.out, "sitemap.xml"), "w",
                      encoding="utf-8") as f:
                f.write("\n".join(xml))
            with open(os.path.join(self.out, "robots.txt"), "w",
                      encoding="utf-8") as f:
                f.write(f"User-agent: *\nAllow: /\nSitemap: {self.base}/sitemap.xml\n")

        print(f"  wrote {n} html files to {self.out}")
        return n

    def render_redirect(self, title, location, canonical=""):
        body = f'''<div class="page">
  <div class="wrap article-redirect" id="main">
    <div class="prose">
      <h1>{html.escape(title)}</h1>
      <p>This legacy address now lives at <a href="{html.escape(location)}">{html.escape(location)}</a>.</p>
      <p><a class="btn primary" href="{html.escape(location)}">Continue to the article</a></p>
    </div>
  </div>
</div>'''
        return self.shell(title=title, desc="Legacy article address.", body=body,
                          page_class="article", canonical=canonical)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", default=os.environ.get("OMNI_WIKI", WIKI_ROOT),
                    help="Directory of wiki markdown (default: the wiki repo "
                         "cloned beside this one)")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "docs"),
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
