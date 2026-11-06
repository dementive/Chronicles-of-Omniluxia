# Omniluxia art handoff — instructions for Codex

You are making **texture images only**. Claude (the other agent on this project) converts them to game
format, wires them into the mod and checks them. Follow this document exactly; anything that deviates
from the sizes, names or layouts below cannot be used.

## Ground rules (read first)

1. **Only write inside `content_source/art_inbox/`.** Never create, edit, rename or delete anything in
   `gfx/`, `common/`, `tools/`, `localization/` or any `.asset`, `.mesh`, `.dds` or `.txt` file.
2. **Deliver PNG only**: 8-bit sRGB, RGB (no alpha unless a task says so), no colour profile tricks,
   no embedded text, no watermarks, no borders.
3. **Exact pixel sizes and exact file names** as given per task. Put each task's files in its own folder:
   `content_source/art_inbox/delivered/<task id>/` (e.g. `delivered/T1/`).
4. **Painted albedo only**: no baked lighting, no shadows, no highlights from a light direction, no
   ambient-occlusion darkening except where a task asks for it. The game lights the models itself.
5. **Value range**: keep every pixel between RGB 20 and 235. Pure black or pure white looks wrong in-game.
6. **Seamless where told**: tileable textures must wrap left↔right (and top↔bottom if stated) with no
   visible seam. Test by tiling the image 2×2 before delivering.
7. **Style**: grounded, painterly-realistic ancient-world / low fantasy (think Imperator: Rome, Total War:
   Rome II). No neon, no sci-fi, no cartoon outlines, no logos, no lettering.
8. When finished with a task, write `delivered/<task id>/NOTES.md`: list the files, and one line per file
   on what it depicts. Do not describe how you made it.

Reference images are in `content_source/art_inbox/reference/` (what the game currently uses and how the
models look). Templates are in `content_source/art_inbox/templates/`.

---

## T1 — Prop & weapon material atlas  (highest priority)

**What it's for:** every new weapon, bow, crossbow, quiver and warhammer in the mod samples its colours
from one shared atlas. Right now each region is a flat procedural colour
(`reference/CURRENT_props_atlas_procedural.png`). Real material detail here improves ~150 weapon models at once.

**Deliver:** `delivered/T1/omx_props_atlas_1024.png` — **1024 × 1024**, RGB.

**Layout:** exactly as `templates/TEMPLATE_props_material_atlas_1024.png`. Each magenta-outlined rectangle is
one material; paint the material so it **fills its whole rectangle edge to edge**. Do not paint the
labels, do not leave the magenta lines. Each region must be **seamless left↔right and top↔bottom
within its own rectangle** (models tile and stretch UVs inside a region).

| Region | Pixel box (x, y inclusive) | Paint this |
|---|---|---|
| wood_dark   | 0–255, 0–511     | dark oiled oak/walnut, long grain running **vertically** |
| wood_light  | 256–511, 0–511   | pale ash/beech, grain vertical |
| horn        | 512–767, 0–511   | composite-bow horn: cream-to-amber horn with **horizontal** dark sinew wrap bands every ~40 px |
| steel       | 768–1023, 0–511  | forged iron/steel, subtle **vertical** grind marks, faint mottling; mid-grey (~RGB 150) not mirror |
| brass       | 0–255, 512–767   | cast bronze/brass fittings, warm, light patina in recesses |
| leather     | 256–511, 512–767 | brown leather grip wrap: **diagonal** wrapped strips ~16 px wide |
| string      | 512–639, 512–767 | twisted linen bowstring, cream, fine diagonal twist |
| fletch      | 640–767, 512–767 | white goose feather barbs, fine lines angled ~45° |
| fletch2     | 768–895, 512–767 | same as fletch but dyed muted green |
| leather2    | 896–1023, 512–767| dark green-grey leather, smooth, slight pores |
| darkiron    | 0–255, 768–1023  | blackened/blued iron, dark (~RGB 55–70), faint hammer marks |
| silver      | 256–511, 768–1023| bright silver, fine brushed lines vertical (~RGB 200) |
| gold        | 512–767, 768–1023| gold leaf/gilding, warm yellow (~RGB 215,175,70), soft wear |
| elfwood     | 768–1023, 768–1023| pale silvery birch-like wood, very fine vertical grain |

**Keep the average colour of each region close to the template swatch colour** (within ~15%): Claude
recolours regions programmatically for the five sword finishes (iron, bronze, gilded, dark, elven), so
contrast and *detail* matter more than hue. Medium contrast detail; avoid large blotches — these are
seen at a few pixels' size on the map.

**Acceptance check (do this before delivering):** downscale your image to 128×128 — every region
should still read as its material colour, with no region looking noisy or striped.

---

## T2 — Horsehair crest strip  (portrait helmets, seen close-up)

**What it's for:** the red/white/black/blue/yellow horsehair crests on the new portrait helmets
(`reference/crested_helmet_render.png`, `reference/helmet_set_render.png`). Claude tints this per colour.

**Deliver:** `delivered/T2/crest_horsehair_gray_512.png` — **512 × 512**, **grayscale** (save as RGB with R=G=B).

**Layout:** see `templates/TEMPLATE_crest_strip_512.png`. You paint **only the hair**; Claude places it in
the red area.
- Hair strands run **vertically** (top↔bottom) across the whole image, thousands of fine strands,
  slightly wavy, grouped into soft clumps 6–14 px wide.
- **Brightness profile, top to bottom:** dark roots at the very top edge (value ~40) → brightest in the
  **middle row y=256** (value ~200) → dark roots again at the bottom edge (~40). Smooth gradient, not a band.
- **Seamless left↔right** (the crest wraps around). Not seamless top↔bottom.
- No background, no skin, no helmet metal — hair fills 100% of the image.

---

## T3 — Feather plume  (side plumes on helmets)

**Deliver:** `delivered/T3/plume_feather_gray_256x512.png` — **256 wide × 512 tall**, grayscale (R=G=B).
- One long ostrich/heron-style plume filling the whole image: the shaft (rachis) runs **vertically down
  the exact centre column**, soft barbs fanning out to both side edges at ~40° upward.
- Values: shaft ~220, barbs 120–210, darker toward the barb tips at the left/right edges (~90).
- The tip of the plume is at the **top**; the base (where it joins the helmet) at the **bottom**,
  which should be slightly darker.
- Fill the full rectangle (the 3D shape is cut by geometry, not by alpha). No background colour.

---

## T4 — Holder strip / trim band (optional, after T1–T3)

**Deliver:** `delivered/T4/crest_holder_bronze_512x64.png` — **512 × 64**, RGB.
A bronze crest-holder band: warm bronze with a row of round domed rivets every 42 px centred
vertically, slight edge wear. Seamless left↔right.

---

## What NOT to do

- Do not make sword or ship *pictures* — the 3D shapes are already built; only the materials above are needed.
- Do not deliver DDS, PSD, JPG or WEBP. PNG only.
- Do not resize or "improve" the templates; paint new images at the stated sizes.
- Do not add text, signatures or frames anywhere in a deliverable.
- Do not touch any file outside `content_source/art_inbox/delivered/`.

## Handback

When a task folder is complete, tell Joshua "T<n> delivered". Claude will then convert it, regenerate the
affected textures, render a before/after preview and report back.
