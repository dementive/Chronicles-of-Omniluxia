# DLC2 — Northern New World: how to bring it back

The northern half of the eastern New World is **commented out, not deleted**.
Every original line still exists in the repo, prefixed with the marker `#DLC2#`.

Rollback point: git tag **`pre-dlc-split`** (commit `89667b4e`, branch `dev`).

## Current state

| | Provinces | Status |
|---|---|---|
| Red — held for DLC | 827 | impassable wasteland: no names, no colonisation, no conquest |
| Blue — ships now | 312 | fully playable, untouched |
| Green island (`4960` Brinorr, `4979` Brinmel) | 2 | pulled out of red, fully playable |

Red covers `new_world_region_001`–`_029`, `dragon_steppes_west/east_region` and `dragons_point_region`.
Blue covers Jade Island (N/S), Great Mushroom Island, Minor Mushroom Isles, Ant Island (W/C/E), Arame Island (N/W/E) and Middle Intermare Isles.

## Files changed

| File | Change |
|---|---|
| `map_data/default.map` | **added** `impassable_terrain` line with 827 provinces (lines marked `DLC2`) |
| `map_data/ports.csv` | 250 red port entries commented out (island's 2 kept) |
| `setup/provinces/00_new_world_region.txt` | 827 province blocks commented; `4960`/`4979` left live |
| `setup/main/02_countries.txt` | 43 country blocks (`Q00`–`Q39`, `X00`–`X19`) |
| `setup/main/00_great_works.txt` | placements in 424 / 1281 / 1721 / 4724 + database entries 21 / 22 / 23 / 26 |
| `setup/main/02_provinces.txt` | modifiers 79 / 108 / 409 commented; `4960` restored |
| `decisions/tier_1_formables/`, `tier_2_formables/` | 41 `nw_*` formables + `nw_silk_court` (42 files) |
| `setup/provinces/00_new_world_impassable.txt` | **new file** — minimal wasteland setup for the 827 red provinces (see crash note below) |
| `setup/countries/countries.txt` | 43 index lines (`Q00`–`Q39`, `X00`–`X19`) commented with `#DLC2#` |

## ⚠ Crash fix — impassable provinces MUST keep a setup entry

The first version of this split crashed on load (`EXCEPTION_ACCESS_VIOLATION` during
`gamestate.cpp:1417 Updating cached data...`). Cause: the 827 red provinces had **no setup
entry at all**, leaving them with null terrain. Every impassable province must keep a minimal
entry — exactly like the 143 pre-existing wastelands in `00_default.txt`:

```
44={ #Naxka
	terrain="impassable_terrain"
	culture="chivixi"        # original culture/religion kept
	religion="the_hive_concord"
	trade_goods=""
	civilization_value=0
	barbarian_power=0
	province_rank=""
}
```

`setup/provinces/00_new_world_impassable.txt` provides these 827 blocks (cultures/religions
copied from the commented originals). **Delete that file when restoring the New World.**

## Deliberately left alone

- **`areas.txt` and `regions.txt` are untouched.** The red provinces are still assigned to their areas and regions. See the risk note below.
- **The 43 country tags are now commented out of `setup/countries/countries.txt`** with `#DLC2#` markers (files intact in `setup/countries/new_world/`). Note: the original assumption that they'd "never spawn" was wrong — the engine *did* create them as landless countries from the index. Commenting the index lines stops that.
- **Province names** in `provincenames_l_english.yml` — your existing 143 wastelands keep theirs too, so this matches current behaviour.
- **Shared assets** (races, religions, pantheons, heritages, military traditions) — unused definitions are harmless and some are shared with the islands that ship.
- **Formables kept live:** `nw_ash_march`, `nw_pyre_march`, `nw_jade_empire` (island-gated), `nw_coral_empire` (already `always = no`).

## ⚠ Risk to test first

Every one of your existing 143 wasteland provinces sits in **no area at all**. The 827 new ones are in both an area *and* the impassable list, which departs from that convention. This was a deliberate choice to keep the diff small and avoid dangling references, but it is the single most likely thing to throw validator errors or block loading.

**Boot the mod before doing anything else.** If it complains about the New World provinces, the fix is to also comment the red areas out of `areas.txt` and the red regions out of `regions.txt` — trimming `dragons_point_area_1` down to just `4960 4979`. That in turn needs three references patched: `common/scripted_triggers/00_regions.txt` (26 refs in `country_is_in_new_world`), and the `highlight` block in `decisions/tier_1_formables/nw_ash_march.txt` (`new_world_region_020`). Nothing else in the mod references the red regions.

## Restoring

`map_data/default.map` was an addition, so restore it by deleting lines:

```bash
sed -i '/DLC2/d' map_data/default.map
```

Also delete the crash-fix terrain file:

```bash
rm setup/provinces/00_new_world_impassable.txt
```

Everything else was commented in place. Strip the marker and the banners:

```bash
grep -rl '#DLC2#' setup decisions map_data | while read f; do
  sed -i 's/^#DLC2#//' "$f"
  sed -i '/^# =\{20,\}$/,/^# =\{20,\}$/d' "$f"
done
```

Review the banner-stripping pass before committing — it keys off the `# ====…` rules, so check nothing else in a file uses that exact line.

## Verified at time of writing

- All 46 commented files restore byte-for-byte against `pre-dlc-split`; `default.map` restores by deleting its `DLC2` lines.
- 827 added impassables are all red; none of the original 143 wastelands lost; no blue province made impassable.
- Every impassable province retains a minimal live setup entry (`terrain="impassable_terrain"`, empty rank/goods) in `00_new_world_impassable.txt` — **required**, removing these entries is what caused the load crash. None retains pops, a port entry, or a settlement rank.
- Brace balance intact in every edited script file.
- Blue province files, country folders and `map_data` geometry untouched.

Not verified: `imperator-tiger.exe` is Windows-only and could not be run here. Run it, then boot the mod once.
