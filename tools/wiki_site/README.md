# Wiki site

Renders the project wiki into a designed static site, published on GitHub Pages
at `https://<owner>.github.io/Chronicles-of-Omniluxia`.

**The wiki stays the source of truth.** Nothing about authoring changes — pages
are written on github.com as normal markdown. This directory only adds a
presentation layer that reads them.

## Why not just style the wiki?

GitHub wikis accept no custom CSS or JavaScript; the grey theme is fixed, and a
wiki repo can neither serve Pages nor run Actions. Generating a site is the only
way to control presentation, so that is what this does — leaving the wiki itself
untouched and still editable in the browser. The two are parallel front doors
onto the same text.

## It builds itself

`.github/workflows/wiki-site.yml` runs on the `gollum` event, which GitHub fires
whenever anyone edits the wiki. An edit on github.com updates the live site about
a minute later. **There is no manual step and nothing generated is committed** —
`docs/` is gitignored and rebuilt by CI on every run.

One-time setup, already done if the site is live: **Settings → Pages → Source →
GitHub Actions.**

The workflow also fails the build if any internal link breaks, so a bad rename in
the wiki surfaces as a red X rather than a dead page.

## Building locally

Requires Python 3.9+, `markdown` and `pillow`.

```bash
pip install markdown pillow

# Clone the wiki beside the mod repo (once); it is then found automatically
git clone https://github.com/dementive/Chronicles-of-Omniluxia.wiki.git ../Chronicles-of-Omniluxia.wiki

python3 tools/wiki_site/build_art.py   # only when gfx/ art changes
python3 tools/wiki_site/build.py
python3 -m http.server -d docs 8000    # then open localhost:8000
```

Point it elsewhere with `--wiki PATH`, or set `OMNI_WIKI`.

## How pages are categorised

Each page is filed under one of nine categories plus a `Lore` catch-all, worked
out from the hub pages: if `Races.md` links to `Drow.md`, Drow is a race.

Hubs are consulted in the order set by `CLASSIFY_PRIORITY`, **not** the order they
appear in the navigation. Curated list hubs (Races, Religions) claim their pages
before essay-style hubs (Magic, History) — those mention half the wiki in passing
and would otherwise hoover up everything.

To fix a miscategorised page, either link it from the correct hub page (better:
it improves the wiki too) or add it to `CATEGORY_OVERRIDES` in `build.py`. A page
landing in `Lore` means no hub links to it.

## Conventions it understands

The generator adapts to how this wiki is actually written:

- **Links** in all three styles in use — full `github.com/.../wiki/Page` URLs,
  bare `[text](Page-Name)`, and raw `<a href="Page">` — are rewritten to local
  pages, anchors preserved.
- **Unicode hyphens.** Some filenames use U+2010 (`Blood‐Stained.md`) while links
  to them use ASCII `-`. Both resolve to the same page.
- **A leading blockquote** becomes the epigraph in the page header.
- **Blockquotes nested in list items** (`* 51,142` / `> what happened`) are the
  wiki's dominant annotation pattern, styled as a definition list.
- **The Timeline's ad-hoc markers** — `-Zanic Age-` and `==SERPENTINE GOLDEN AGE==`
  — are promoted to real headings so the page gets structure and a contents
  sidebar. Lines containing links are skipped, which leaves the
  `-[Level 3](...) Ranking-` lines in the race pages alone.
- **Colliding titles** fall back to the filename: `Magic-Styles.md` is headed
  `# Magic`, same as `Magic.md`.

## Design notes

- Palette is sampled from `gfx/interface/frontend/main_menu_background.dds` — the
  mod's own gold sigils on near-black.
- Hero backgrounds are the fifteen loading screens, softened, desaturated and
  pushed through an obsidian/gold duotone by `build_art.py`. Ungraded they are
  obviously bright gameplay captures and text is illegible over them.
- Each category draws plates from a terrain-appropriate subset
  (`CATEGORY_PLATES`), so a category looks coherent without every page being
  identical.
- **No JavaScript at all.** Scroll reveals use `animation-timeline: view()` behind
  an `@supports` guard, so browsers without it simply show the content. The mobile
  menu is a checkbox. `prefers-reduced-motion` is respected.
- About 1.2 MB of art for the whole site, shared across every page and cached
  after first load. Only sizes the site actually loads are generated: 1280 for
  desktop hero plates, 800 for narrow viewports and the social preview image.

## Artwork and credits

Every image is the mod's own: the fifteen loading screens, the main menu
background, and the wordmark. Seventeen source files, nothing external.

If you add imagery, keep it to the project's own work or to public domain / CC0
material, and record it in the Artwork section of `render_credits()` in
`build.py`. Credit only what is actually used — a credits page listing sources
that appear nowhere on the site is worse than no credits page.

One gotcha: `url()` inside a CSS custom property resolves relative to the
**stylesheet**, not the HTML document. The generator emits `--plate:url('img/...')`,
not `assets/img/...`.
