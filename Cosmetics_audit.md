# Character Cosmetics Audit — Omniluxia

Scan of the portrait accessory system: `gfx/portraits/accessories/`, `common/genes/00_genes.txt`, `gfx/portraits/portrait_modifiers/`, `common/ethnicities/`, and the `.asset` model chain. Goal: find cosmetics (helmets, hats, jewelry, clothes, etc.) that are defined but unreachable or wired up incorrectly, so the variety can be recovered.

## How the system works (for reference)
1. Accessory objects (a helmet, a hairstyle) are defined in `gfx/portraits/accessories/*.txt` and point to an `_entity` in a `.asset` file.
2. `00_genes.txt` groups those objects into **templates** (weighted lists) under genes like `hairstyles`, `clothes`, `headgear`.
3. Characters get a template one of two ways: an **ethnicity** assigns it (`name = <template>`), or a **portrait_modifier** forces it (`dna_modifiers { template = <template> }`, gated by a trigger such as `has_culture_group`).
4. Helmets and soldier armor are delivered through the portrait_modifier path (`Gloiratus_modifier_clothes.txt`); civilian hats/jewelry ride along inside hairstyle templates.

A cosmetic is **lost** if its template is never assigned by any ethnicity or modifier, or if its entity/mesh is missing.

---

## 1. Bug — 35 hairstyles reference a missing entity (highest priority)

Every affected hairstyle has an under-helmet variant tagged `required_tags = "no_hair"` that points to **`male_hair_roman_3_entity`**. That entity name **does not exist in any `.asset` file**. The correct name — defined in `gfx/models/portraits/attachments/male_hair/zani/soldier_helmet_hair/soldier_helmet_hair.asset` — is **`soldier_helmet_hair_entity`**.

Effect: when these characters wear a soldier helmet, the flattened under-helmet hair fails to load.

Affected files (replace `male_hair_roman_3_entity` → `soldier_helmet_hair_entity`):

- `forest_elf_hairstyles.txt`
- `navshodia_hairstyles.txt`
- `zani_hairstyles.txt`
- `zaraken_clothes.txt`

This is the same string in all 35 spots — a single find/replace per file fixes it.

---

## 2. Soldier outfits defined but never triggered

The culture-group triggers in `Gloiratus_modifier_clothes.txt` all resolve to real culture groups (no dead triggers), but some complete soldier templates are never referenced by any modifier, so that armor never appears:

- **`zani_soldier_clothes_template`** — orphaned. The `zani_soldier_clothes_modifier` actually applies `roman_soldier_clothes_template` instead. The zani-specific soldier clothing is finished but unused. (Zani do still get *a* helmet via `zani_soldier_helmet_template`.)
- **`hair_soldier_helmet_template`** — unused.
- Leftover real-world soldier-clothes pools wired to no group: `carthaginian_`, `celtic_`, `egyptian_`, `greek_`, `iberian_`, `mauryan_`, `bactrian_`, `ethiopian_`, `germanic_`, `tibetan_` `_soldier_clothes_template`. These are inherited Invictus content — a ready-made pool if you want more soldier looks per culture.

Note: a few cultures reuse another culture's kit rather than their own — e.g. orc soldiers use `dravidian_soldier_clothes_template`, northern Arteon uses `persian_soldier_clothes_template` + `high_elf_soldier_helmet_template`. Working, but worth knowing if you want them visually distinct.

---

## 3. Civilian template pools never assigned to any ethnicity

Defined in `00_genes.txt` but no ethnicity uses them — every accessory inside is unreachable:

- **Hairstyle pools:** `arabian_`, `dravidian_`, `ethiopian_`, `greek_`, `iberian_`, `indian_`, `numidian_`, `persian_`, `sub_saharan_african_`, `tibetean_` hairstyles.
- **Clothes pools:** `roman_clothes`, `germanic_clothes`, `persian_clothes`, `bactrian_clothes`, `ethiopian_clothes`, `horteon_clothes`, `tibetan_clothes`.
- **Top layers:** `dravidian_top_layer`, `roman_top_layer`.
- Catch-alls `all_hairstyles` and `most_clothes` are almost certainly debug/reference lists — safe to leave.

These are the biggest single source of recoverable variety: assign a relevant pool to an ethnicity (or blend a few entries into an existing template) and characters immediately get more hair/clothing options.

---

## 4. Individual accessories in no gene list at all

Defined as accessory objects but referenced by zero templates:

- `male_hair_elven_helmet`
- `female_hair_eptelon_3`
- `female_hair_aegean_2`, `female_hair_aegean_2_v3`
- `male_hair_zaraken_1` (+ `_color2/3/4`) — also one of the broken-entity items in section 1.

Add them to an appropriate template to surface them.

---

## 5. Eyepatch / blindfold never applied

`eye_accessory` defines `eyepatch_1`, `eyepatch_2`, and `blindfold_1`, but nothing references them — no character ever appears with one. In the base game these are applied by a trait-gated portrait_modifier. Wiring them to a trait (e.g. maimed/scarred/blind) would add flavor and variety for free.

(`bust`, `orc_bust`, `silver_dwarf_bust` also show as "unassigned" but are applied by the engine's hardcoded bust gene — not actionable.)

---

## 6. The `headgear` gene slot is nearly empty

The dedicated `headgear` gene only defines `no_headgear` and `zainuddian_headgear`. All other hats/crowns/diadems are delivered through hairstyle templates, which works — diadems, the zainuddian hat, elven capes, and rohenoa noble cloaks are all confirmed reachable. Just noting the dedicated slot is available and mostly unused if you'd rather manage hats there.

---

## Suggested order of work
1. Fix the missing entity (section 1) — it's an active rendering bug, one find/replace across four files.
2. Assign the orphaned civilian pools (section 3) — largest, easiest variety gain.
3. Hook up `zani_soldier_clothes_template` and the loose accessories (sections 2 & 4).
4. Optionally trait-gate the eyepatch/blindfold (section 5).
