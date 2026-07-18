# Magic Learning Rework — Design Spec

This document specifies a rework of how spells are *learned* in Omniluxia. It does not change how spells are *cast* (mana economy, terrain scaling, the cast menus themselves), except where the cast-side gate has to read a new variable. The goal is to replace the current "one variable unlocks a whole tier of a discipline" model with a **per-spell** model that supports randomized learning, folk-magic travel, and a magical education path for children.

---

## 1. Goals & principles

- **Every spell is individually learnable.** Knowing a spell is a per-character fact, not a side effect of unlocking a tier.
- **Studying stays the gate.** You still need a trait/variable and an academy (Imperial) or the mage trait + a hedge-wizard (Folk) to begin. The rework changes *what you walk away with*, not the prerequisite to start.
- **Randomized rewards.** Finishing a course of study grants a small number of random spells rather than a fixed pair, and advancing lets you buy additional random spells at a cost.
- **Minimal performance cost.** Gating a cast option must stay a cheap O(1) check.
- **Auto-generated boilerplate.** With 100+ spells, the per-spell plumbing is machine-generated (extending the existing `add_spells.py` codegen), so hand-editing scales.
- **Backward compatible.** Existing saves that hold tier-unlock variables must convert cleanly to the new per-spell flags.

---

## 2. Current system (as built today)

**Schools and disciplines.** Four Imperial schools, each with two disciplines, plus racial/special styles, plus a large Folk pool:

| School | Disciplines | Special styles seen in code |
|---|---|---|
| Omnic | Diviner (D), Hierophant (H) | Magonis, Taanka, Dragonscale, Helluvian Holy |
| Aldic | Cleric (C), Illusionist (I) | Charm, Soul/Astral |
| Amten | Elementalist (E), Druid (D) | Silver Dwarf / Stone Magic |
| Melodian | Necromancer (N), Warlock (W) | Palpan |
| Folk | — (single pool) | ~40 culture-specific spells |

**Tier terminology.** Code uses four tiers. Mapping to the proposal's "Beginner / Intermediate / Advanced (+ archmage)":

| Proposal name | Code tier | Study unlock variable (example, Diviner) | Extra gate |
|---|---|---|---|
| Beginner | T1 | `diviner_t2_unlocked_var` | — |
| Intermediate | T2 | `diviner_t3_unlocked_var` | — |
| Advanced | T3 | `diviner_t4_unlocked_var` | — |
| Master / Archmage | T4 | (highest) | `archmage_trait` |

> Note the off-by-one in the current code: the "tier-1 spell" gate `omni_caster_omnic_t1` is satisfied by the `*_t2_unlocked_var`. This is because the first study rank grants the trait, and the *second* grants the first spells. The rework keeps the same study ranks but stops using these tier flags as the cast gate (see §4).

**How learning works now.** Studying at a magic academy fires the `me_magic_studying.*` chain (`events/magic_studying_events.txt`), routed by `magic_studying_t1_effect` / `t2` / `t3` in `common/scripted_effects/choo_effects.txt`. Completing a rank sets a tier-unlock variable (e.g. `elementalist_t3_unlocked_var`) and grants the discipline trait. Folk magic is a single `folk_unlocked_var`.

**How casting is gated now.** The cast menus (`me_zorgo_magic_focus.*` in `events/zorgo_magic_events.txt`, plus the scripted-GUI buttons `common/scripted_guis/omni_*_button.txt`) show every spell of a tier, gated by the `omni_caster_<school>_t1/t2/t3` triggers in `common/scripted_triggers/OMNI_spell_tier_triggers.txt`. Those triggers just check the tier-unlock variables. T4 options additionally check `has_trait = archmage_trait`.

**Mana model (unchanged by this rework).** Mana is the character variable `magic`, regenerated monthly via `magic_monthly`, capped at `magic_max` (see `common/scripted_effects/0000_magic_effects.txt`, `clamp_character_magic_to_max`). Spend/afford go through `omni_spell_mana_cost_effect` and `omni_can_afford_spell` (`omni_spell_affordable_budget >= BASE`, with terrain/rank scaling in `common/script_values/OMNI_magic_spell_costs.txt`). These are the hooks the Educate Child rewards target (§8).

---

## 3. Spell tracking architecture (the foundation)

**Recommendation: one boolean variable per spell, `<spell_id>_known_var`, managed by codegen.**

Rationale, weighed against a single `variable_list` registry:

- **Performance.** `has_variable = fireball_known_var` is an O(1) hashmap lookup on the character. A `variable_list` membership test (`is_target_in_variable_list`) scans the list every check; with hundreds of cast-option evaluations per open menu across every eligible mage, the boolean flag is strictly cheaper. This is the deciding factor.
- **Ease of implementation.** The cast gate becomes a one-line addition to each option trigger, and grants are one-line `set_variable`. It mirrors the pattern already in `magic_traits.txt` (`*_spell_t1_trait`) and the existing `*_unlocked_var` convention, so it's consistent with the codebase.
- **Boilerplate is generated.** The registry lives in one Python table; `add_spells.py` (already the tool that injects options into the menus) is extended to also emit the per-spell gate, the grant lines, and the random-grant pools. Hand-maintenance stays low despite the spell count.

Trade-off accepted: a save will carry up to ~120 boolean character variables on a fully-trained archmage. Imperator handles this fine (variables are sparse — only *known* spells are stored), and none of them are checked on-tick; they're read only when a magic menu opens.

**Naming convention.** `<spell_id>_known_var`, where `<spell_id>` is the snake_case spell name already used for its modifier (e.g. `fireball`, `light_shields`, `chain_lightning`, `chronostasis_field`). One canonical id per spell, shared by its modifier, its localization key, and its `_known_var`.

**Helper effects/triggers (new, in a dedicated file — see §10):**

- `omni_learn_spell = { SPELL = <id> }` — sets `<id>_known_var = yes`, shows a learn toast, and (optionally) grants the matching `mage`/discipline trait if not present.
- `omni_knows_spell = { SPELL = <id> }` — trigger, wraps `has_variable = <id>_known_var`.
- `omni_grant_random_spell = { SCHOOL = amten TIER = t1 }` — the randomized grant (§6).

---

## 4. Cast-side gating change

Today an option checks `omni_caster_amten_t1 = yes`. After the rework it checks that the caster **knows that specific spell**:

```
# before
trigger = { scope:selected_mage = { omni_can_afford_spell = { BASE = 30 } } ... }

# after
trigger = {
    scope:selected_mage = {
        omni_knows_spell = { SPELL = fireball }
        omni_can_afford_spell = { BASE = 30 }
    }
    ...
}
```

The `omni_caster_<school>_tN` triggers in `OMNI_spell_tier_triggers.txt` are **retired as cast gates** but kept (repurposed) for two things that remain tier-based rather than spell-based: (a) the generic friendly battle buffs and core siege workings that any trained caster can use, and (b) study-eligibility checks (you can't be offered Intermediate study until you've completed Beginner). T4 options keep the additional `has_trait = archmage_trait` check on top of the per-spell flag.

Every option in these locations needs the per-spell line added (mechanical, codegen-driven):

- `events/zorgo_magic_events.txt` — `me_zorgo_magic_focus.3–14`, plus the support/siege/divine-blessing menus.
- `common/scripted_guis/omni_*_button.txt` — the unit/character/siege cast buttons.
- `add_spells.py` — the option strings it injects (already carries the spell id in a comment; extend to emit the gate).

---

## 5. Learning flows

### 5A. Imperial study (academies)

Same entry point as today: you begin a course at a magic academy building, which fires the `me_magic_studying.*` chain. The rework changes the payoff of each rank.

**Completing Beginner <Discipline>** (the first spell-granting rank):
- Grant the `mage_trait` (if not present) and the discipline trait, exactly as now.
- Grant **2 random Beginner spells of that discipline** via `omni_grant_random_spell` (§6), instead of a fixed pair.
- Set the study-progression flag (`<discipline>_t2_unlocked_var`) so the next rank becomes available. This flag now means "eligible to study Intermediate," not "knows all Beginner spells."

**Advancing to Intermediate / Advanced / Master:**
- The study event grants **2 random spells of the newly-completed tier** of that discipline (free, part of finishing the course).
- **Plus** it presents a "catch-up" offer: up to 2 random spells drawn from a tier you have *already* unlocked in that school, purchasable for money + a temporary debuff trait (§7). This is the proposal's "spend some money and a temporary debuff to learn" mechanic, and it's how you fill gaps left by earlier random rolls.

> **Random pool decision (locked): same tier only.** Every random draw — free or purchased — is restricted to the tier being resolved. Finishing Beginner rolls Beginner spells; finishing Intermediate rolls Intermediate spells; the catch-up offer rolls from a chosen already-unlocked tier. Random draws do **not** pull cross-tier or cross-school, and do not include special styles (special styles are unlocked by their own dedicated content, unchanged).

**Special styles (Magonis, Taanka, Dragonscale, Charm, Soul, Stone Magic, Palpan, Helluvian Holy, etc.)** keep their current bespoke unlock paths. Under the new model each becomes its own `_known_var` grant, but they are **not** part of the random tier pools — they are still awarded by their specific events/decisions. This preserves their "special" feel and matches the current `*_unlocked_var` gating.

### 5B. Folk magic (travel)

Entry requirement stays: you must have the `mage_trait`. New acquisition path:

- While a character travels the map (adventure / army movement / a dedicated "wander" activity), random **wizard's-home locations** can trigger an event offering study under a local hedge-wizard.
- Accepting grants one or more Folk spells tied to that location's culture/region (e.g. a Drow enclave teaches `bloodlust`, a Northlander teaches `berserker_rage`), each as its own `_known_var`, and sets `folk_unlocked_var` if not already set.
- Folk practitioners keep tier-1 cast access to the generic buffs (already handled by `omni_caster_tier_1_any_school` including `folk_unlocked_var`).

Open question F1: should folk wizard locations be **fixed** province flags (curated, lore-placed) or **spawned dynamically** as the traveler moves? Fixed is more authored and testable; dynamic is more "anywhere on the map." (Recommend fixed set to start, expandable later.)

---

## 6. The randomized grant mechanic

Core helper, one per school × tier pool:

```
omni_grant_random_spell = {
    # SCHOOL = amten, TIER = t1  ->  rolls one unknown Beginner Amten spell
    random_list = {
        10 = { limit = { NOT = { omni_knows_spell = { SPELL = winds_of_pedentrutzu } } }
               omni_learn_spell = { SPELL = winds_of_pedentrutzu } }
        10 = { limit = { NOT = { omni_knows_spell = { SPELL = fireball } } }
               omni_learn_spell = { SPELL = fireball } }
        # ...one weighted entry per Beginner Amten spell...
        1  = { }   # fallback if everything in the pool is already known
    }
}
```

To grant "2 random," call it twice (the second call sees the first as known, so it won't duplicate). These pool effects are **codegen output**: the Python registry knows every spell's school+tier, so it writes the full `random_list` bodies. When you add a spell to the registry, regenerating updates every pool automatically.

The catch-up purchase (§7) calls the same helper for a tier the character has already unlocked, wrapped in a cost check.

---

## 7. Purchase cost & temporary debuff

The buyable catch-up spells cost:

- **Gold** — a flat or tier-scaled price via the standard `pay_price` / a script value (e.g. Beginner 25, Intermediate 60, Advanced 120).
- **A temporary debuff trait** representing overexertion from cramming. The mod already ships candidates: `magical_sickness_trait` and `magically_inept_trait` (`common/traits/magic_traits.txt`), and a `mana_overexertion` modifier pattern. Recommend a time-limited `magical_sickness_trait` applied for e.g. 365 days (reduced mana regen), removed via an `on_action`/event timer.

Open question P1: confirm the debuff should be `magical_sickness_trait` reused, or a new dedicated `cramming_fatigue_trait` so its magnitude/duration can be tuned independently of existing sickness content.

---

## 8. Educate Child interaction

New character interaction (model on the existing `suggest_training.txt`, which already handles guardian→ward, `pay_price`, `age`/loyalty gating). Target: a child in your realm/family. Five mutually exclusive education focuses:

| Focus | Reward |
|---|---|
| Martial Education | + banked mana points (adds to `magic`) |
| Oratory Education | + banked mana points (adds to `magic`) |
| Economic Education | + banked mana points (adds to `magic`) |
| Religious Education | + banked mana points (adds to `magic`) |
| Magical Education | **+25 `magic_max` and +1 `magic_monthly`** (permanent capacity, not just a one-off top-up) |

Implementation notes:
- The first four `change_variable = { name = magic add = <n> }` then `clamp_character_magic_to_max = yes`.
- Magical Education does `change_variable = { name = magic_max add = 25 }` and `change_variable = { name = magic_monthly add = 1 }`, giving the child a lasting head start as a future mage.
- Guard with a `has_variable = has_been_educated` (mirror of the existing `has_trained` flag) so a child can be educated once.
- Decide whether Magical Education also pre-grants `mage_trait` or merely the capacity (recommend capacity only, so they still have to study — keeps the head-start meaningful without free spells).

Open question E1: are the four non-magical focuses meant to be a small fixed mana bump, or scaled by the guardian's relevant stat (martial/charisma/finesse/zeal)? Stat-scaling makes the choice matter more.

---

## 9. Acquiring mages (context, mostly unchanged)

The proposal keeps the existing acquisition surface: cultural decisions, random events, and "Go on an Adventure" (`OMNI_adventure.txt`, `OMNI_orc_adventure.txt`) can still produce Imperial and Folk mages. These already exist and only need to route their spell grants through `omni_learn_spell` instead of setting tier flags. No new acquisition system is required for v1; the folk-travel path (§5B) is the one new source.

---

## 10. File-by-file work plan

**New files:**
- `common/scripted_effects/OMNI_spell_learning.txt` — `omni_learn_spell`, `omni_grant_random_spell` (all school×tier pools), the study-payoff effects.
- `common/scripted_triggers/OMNI_spell_known_triggers.txt` — `omni_knows_spell` and any convenience "knows any spell in school/tier" rollups.
- `common/character_interactions/OMNI_educate_child.txt` — the Educate Child interaction (§8).
- `events/folk_travel_events.txt` — hedge-wizard location events (§5B).
- `Magic_Learning_Rework_Spec.md` — this document.

**Edited files:**
- `events/magic_studying_events.txt` — change each rank's payoff to call the random-grant helper and set progression-only flags; add the Intermediate+ catch-up purchase branch.
- `common/scripted_effects/choo_effects.txt` — `magic_studying_t1/t2/t3_effect` route to the new grant helpers.
- `events/zorgo_magic_events.txt` — add `omni_knows_spell` to every cast option's trigger.
- `common/scripted_guis/omni_*_button.txt` — same per-spell gate on the button cast paths.
- `common/scripted_triggers/OMNI_spell_tier_triggers.txt` — repurpose from cast gate to study-eligibility + generic-buff gate.
- `add_spells.py` — extend codegen to emit `_known_var` gates, the grant helpers, and the random pools from a single registry table.
- `localization/` — `<spell_id>_known` learn toasts, Educate Child strings, folk-location strings.

**Migration (one-time `on_action`, runs on existing saves):** for each character, read every legacy `*_tN_unlocked_var` and set the `_known_var` for **all** spells of the tiers that flag covered, so no one loses spells they'd already earned. Then the legacy flags can be left in place (harmless) or cleared.

---

## 11. Suggested build order

1. **Registry + codegen.** Put the full spell list (school, discipline, tier, id, mana cost) into `add_spells.py`'s table; generate `_known_var` gates, `omni_learn_spell`/`omni_knows_spell`, and the random pools. Nothing is wired yet — just generated.
2. **Cast gate swap for one school (Amten) end-to-end.** Add `omni_knows_spell` to Amten options; grant Amten spells via the new helper in a test event. Verify a mage who "knows" only Fireball sees only Fireball.
3. **Study payoff rework.** Convert `me_magic_studying.*` to random grants + catch-up purchase.
4. **Migration on_action** so existing saves convert.
5. **Educate Child interaction.**
6. **Folk travel events.**
7. Roll the cast-gate swap across the remaining schools, support/siege/divine menus, and GUI buttons.
8. **Validate** with `imperator-tiger` (the mod already ships `imperator-tiger.exe` / `.conf`) and in-game per the checklist in `SPELLS_WIRING_SUMMARY.md`.

---

## 12. Open questions to resolve before coding

- **F1** — Folk wizard locations: fixed authored provinces or dynamic spawns? (Recommend fixed set first.)
- **P1** — Catch-up debuff: reuse `magical_sickness_trait` or new `cramming_fatigue_trait`? (Recommend new, tunable.)
- **E1** — Educate Child non-magical focuses: flat mana bump or stat-scaled? (Recommend stat-scaled.)
- **T4/Archmage** — keep Master (T4) spells out of the random pools entirely and only ever hand-award them, or let archmages roll T4 in the random grant? (Recommend hand-award only, so archmage stays prestige.)
- **Special styles** — confirm they stay outside the random pools (recommended) and just convert to `_known_var` grants on their existing unlock paths.
- **Duplicate protection** — confirmed handled: the random helper's `limit = { NOT = { omni_knows_spell } }` guarantees no wasted rolls; if the whole tier is known, the fallback entry no-ops (consider refunding/redirecting to gold in that edge case).

---

## Appendix A — Spell registry (from current menus)

Discipline codes in menu comments: D=Diviner, H=Hierophant (Omnic); C=Cleric, I=Illusionist (Aldic); E=Elementalist, D=Druid (Amten); N=Necromancer, W=Warlock (Melodian). Tier = the digit after the code (D1 = Diviner Beginner, etc.). This is the seed for the codegen registry; ids to be normalized to snake_case.

**Omnic — Diviner/Hierophant:** Light Shields (D1), Light Rays (D1), Light Rays II (D2), Light Walls (D2), Spirit Bomb (D3), Blinding Flash (T1), Chain Lightning (T2), Gravity Well (T3), Chronostasis Field (T4/archmage); Exorcism (H1), Revelation (H1), Blessing (H2), Exorcism II (H2), Revelation II (H3). Specials: Summon Animal Auxiliaries, Helluvius' Peace, Hellas' Providence, Summon Eagle of Zanis (Magonis), Battle Avatar (Taanka), Dragonscale Shield, Dragonscale Dome.

**Aldic — Cleric/Illusionist:** Restoration (C1), Healing Light (C1), Healing Light II (C2), Vitality (C2), Pestilence (C3); Altered Inspiration (I1), Inspired Thoughts (I1), Mind Break (I2), Warped Flesh (I2), Warped Flesh II (I3); Arcane Volley (T1), Corrosive Rain (T2), Meteor Strike (T3), Word of Annihilation (T4). Specials: Silver Tongue (Charm), Mended Anima / Astral Search (Soul).

**Amten — Elementalist/Druid:** Detect Minerals (E1), Detect Minerals II (E2), Rock Slide (E2), Summon Earthworks (E3), Winds of Pedentrutzu (E1), Fireball (E1), Aegis of Stone (E); Plant Growth (D1), Summon Dryads (D1), Plant Growth II (D2), Nature's Bounty (D2), Nature's Bounty II (D3), Nature's Vigor (D); Frostbite Wind (T1), Seismic Tremor (T2), Glacial Tomb (T3), Summon Elemental Colossus (T4); battlefield: Blighted Ground, Mudbind. Special: Sculpt Buildings (Stone Magic).

**Melodian — Necromancer/Warlock:** Summon Warborn (N1), Curse (N1), Hex (N2), Curse II (N2), Raise Workers (N3); Rage (W1), Dark Offering (W1), Death Ray (W2), Dark Visions (W2), Dark Offering II (W3), Bestow Rage (W), Withering Touch (N); naval/battle: Cursed Rigging, Fear Hex, Storm Wrack, Hymn of Frailty, Blood Curse. Igniting Spark / Plague Miasma / Cataclysmic Eruption also appear as Melodian battle castings.

**Civic/support (cross-school):** Harbor Ward, Unrest Cleansing, Migration Beacon, Weathercall Growth, Charm.

**Siege:** Breaching Tremor (T1), Wasting Rot (T2), Storm of the Siege (T3), Burning Pitch, Counterward, Illusion Walls, plus siege variants of Igniting Spark / Fireball / Corrosive Rain / Seismic Tremor / Plague Miasma / Glacial Tomb / Meteor Strike / Word of Annihilation / Summon Elemental Colossus / Cataclysmic Eruption.

**Friendly battle buffs (any trained caster, tier-based not spell-based):** Battle Haste, Battle Rally, Battle Discipline, Battle Blessing, Arcane Aegis, Battle Clarity, Conjured Supply, Spectral Boarders, plus Panic Rout / Command Confusion / Blood Price / Turn Feet to Mud and the divine-blessing country spells (Sword of Zanis, Winds of Pedentrutzu, Hammer of Melodias, Feast of Ugone, Light of True Faith, Magical Construction, Golden Fingers of the Pharon, Hand of Adaralu, Divine Ceremonies, Arrows of Lavas).

**Folk pool (~40, culture-keyed, taught by hedge-wizards):** Dreaming (Polaric), Scramble (Copper Halfling), Water Walker (Austropetolian), Unified Blades (Miloni), Golden Hands (Azari), Craftsman's Knowledge (Copper Dwarf), Cafea's Breath/Grace (Deepwood Elf), Bloodlust (Drow), Na'athran's Blessing (Sea Elf), Serpentine Scrying/Swords (Ishtari), Raging Dragon (Dunydurceg), Beastly Transformation/Vampiric Feeding (Vampire), Gozon's Blessing (Orc), Victory's Blessing (Viktriec), Revelations of the Way (Seekers), Kon's Wisdom (Kino), Berserker Rage (Northlander), Channeled Stars (Stellaric), Ancestral Rumination (Ascendic), Explosion (Shoishoni), Blood Boil/Blood Rush (Blood Angel), Coils of the Serpent (Snakefolk), Great Flare of the Setting Sun (Sunset Elf), Blade Storm (Alu-Sarian), Steal (Gnome), Stinging Nettles (Thalmori), Imperial Dragon's Claw (Sovenerian), Sword of Jaoz (Zainudian), Hellas' Light (Helluvian), Spectral Sabres (Revant), Hypnosis (Ishtari), Atheus' Blessing.
