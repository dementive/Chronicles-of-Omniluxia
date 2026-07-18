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
- **Specialization emerges, breadth is earned.** Every mage develops stronger and weaker schools through play; off-specialty schools are slower and costlier to learn and to cast; total mastery of all four schools is a rare, ~10-year achievement, not a default endpoint. See §13.

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
| Master / Archmage | T4 | (highest) | `archmage_trait` (global — replaced by per-discipline mastery in §13) |

> Note the off-by-one in the current code: the "tier-1 spell" gate `omni_caster_omnic_t1` is satisfied by the `*_t2_unlocked_var`. This is because the first study rank grants the trait, and the *second* grants the first spells. The rework keeps the same study ranks but stops using these tier flags as the cast gate (see §4).

> The current single global `archmage_trait` cannot express "archmage of Melodian but not of Amten." §13 replaces it as the T4 *cast gate* with per-discipline mastery; `archmage_trait` survives only as a status/title (granted at the 2-school rung).

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

- **Gold** — `CATCHUP_PRICE[tier]` = 25 / 60 / 120 for Beginner / Intermediate / Advanced (§14.1); T4 spells are never buyable. Charged via `pay_price` / a script value.
- **A temporary debuff trait** — the new `cramming_fatigue_trait` (add to `common/traits/magic_traits.txt`; 180 days; −50% `magic_monthly`), removed via an `on_action`/event timer. Kept separate from the existing `magical_sickness_trait` so its magnitude/duration tune independently (resolves P1). See §14.6.

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
- `common/scripted_triggers/OMNI_spell_known_triggers.txt` — `omni_knows_spell`, `omni_primary_school`, per-school mastery/affinity convenience triggers (§13.1–13.2).
- `common/scripted_effects/OMNI_magic_progression.txt` — affinity increments, `magic_study_progress` handling, mastery grants, `magus_supreme_trait` check, mentor-discount lookup (§13.3–13.7).
- `common/character_interactions/OMNI_educate_child.txt` — the Educate Child interaction (§8).
- `events/folk_travel_events.txt` — hedge-wizard location events (§5B).
- `Magic_Learning_Rework_Spec.md` — this document.

**Edited files:**
- `events/magic_studying_events.txt` — change each rank's payoff to call the random-grant helper and set progression-only flags; add the Intermediate+ catch-up purchase branch; drive advancement off `magic_study_progress` (§13.4); **remove the top-rank trait-stripping** (§13.8).
- `common/scripted_effects/choo_effects.txt` — `magic_studying_t1/t2/t3_effect` route to the new grant + progression helpers.
- `events/zorgo_magic_events.txt` — add `omni_knows_spell` to every cast option's trigger; swap T4 gates to per-school `_mastered_var` (§13.1).
- `common/scripted_guis/omni_*_button.txt` — same per-spell + per-school-mastery gate on the button cast paths.
- `common/scripted_triggers/OMNI_spell_tier_triggers.txt` — repurpose from cast gate to study-eligibility + generic-buff gate.
- `common/script_values/OMNI_magic_spell_costs.txt` — affinity-scaled casting discount (§13.9).
- `common/scripted_effects/omni_city_payment_effects.txt` — `omni_pay_magic_study_cost` becomes income-relative + cross-school multiplier (§13.5, §13.3).
- `events/magic_menu.txt` — study prices/gating read the income-scaled cost and academy/attribute speed factors (§13.5–13.6).
- `common/traits/magic_traits.txt` — add `magus_supreme_trait`; repurpose `archmage_trait` to status-only.
- `add_spells.py` — extend codegen to emit `_known_var` gates, the grant helpers, the random pools, and per-school T4 mastery gates from a single registry table.
- `localization/` — `<spell_id>_known` learn toasts, Educate Child strings, folk-location strings, `magus_supreme_trait` and mastery/affinity strings.
- `common/on_action/` — mastery→`magus_supreme` check and any timed cleanup for the study progress/debuff timers.

**Migration (one-time `on_action`, runs on existing saves):** for each character, read every legacy `*_tN_unlocked_var` and set the `_known_var` for **all** spells of the tiers that flag covered, so no one loses spells they'd already earned. Then the legacy flags can be left in place (harmless) or cleared.

---

## 11. Suggested build order

1. **Registry + codegen.** Put the full spell list (school, discipline, tier, id, mana cost) into `add_spells.py`'s table; generate `_known_var` gates, `omni_learn_spell`/`omni_knows_spell`, and the random pools. Nothing is wired yet — just generated.
2. **Cast gate swap for one school (Amten) end-to-end.** Add `omni_knows_spell` to Amten options; grant Amten spells via the new helper in a test event. Verify a mage who "knows" only Fireball sees only Fireball.
3. **Study payoff rework.** Convert `me_magic_studying.*` to random grants + catch-up purchase.
4. **Progression core (§13.1–13.6, §14).** Add affinity + `<discipline>_mastered_var` + `magic_study_progress`; swap T4 cast gates to per-discipline mastery; make study advancement deterministic with academy/attribute speed and income-scaled + cross-school-multiplied tuition (constants from §14.1). Remove the trait-strip.
5. **Capstone & mentors (§13.7, §13.1).** Add `magus_supreme_trait` + its on_action check and the mentor discount.
6. **Casting discount (§13.9).** Affinity-scaled mana cost in `OMNI_magic_spell_costs.txt`.
7. **Migration on_action** so existing saves convert (legacy tier vars → `_known_var`; existing top-tier holders → matching `_mastered_var` + seeded affinity; existing `archmage_trait` holders keep their disciplines).
8. **Educate Child interaction** (incl. affinity pre-seed).
9. **Folk travel events.**
10. Roll the cast-gate swap across the remaining schools, support/siege/divine menus, and GUI buttons.
11. **Validate** with `imperator-tiger` (the mod already ships `imperator-tiger.exe` / `.conf`) and in-game per the checklist in `SPELLS_WIRING_SUMMARY.md`, then run the §14.8 acceptance tests — including the pacing check that a specialty discipline lands ~1.75 yr and a full `magus_supreme` run ~17 yr (fast path).

---

## 12. Open questions to resolve before coding

**Resolved (locked in):**
- **Specialty model** — emergent (highest affinity), with a full **gradient** of per-school affinities (confirmed cheap: ≤4 vars/character, read only off-tick).
- **Mastery granularity** — tracked per **discipline** (eight `_mastered_var`), not per school, so all eight discipline traits and every spell line up. A school is "mastered" when both its disciplines are.
- **Prestige ladder** — depth vs breadth: 1 full school → `<school>_master_trait` (signature); 2 schools → `archmage_trait`; 3 schools → `grand_archmage_trait`; 4 schools → `magus_supreme_trait`.
- **`lich_trait`** — special undeath transformation only (Jaraharum mission); never granted by study and never required for any rung.
- **Archmage** — no longer granted on first discipline; now the 2-school rung, status-only. Per-discipline `_mastered_var` gates T4 casting; trait-strip removed so discipline traits accumulate; the old archmage mana discount moves to mastery/affinity (§13.1).
- **T4 in random pools** — Master/T4 spells stay out of the random grant and are earned only by mastering the discipline (keeps the capstone prestigious).

**Resolved by §14:**
- **N1** — All pacing constants (friction, progress threshold, academy/attribute bonuses, tuition, affinity→discount) are set in §14.1 and verified in §14.7.
- **P1** — Catch-up debuff = new `cramming_fatigue_trait` (180 days, −50% `magic_monthly`); see §14.6.
- **E1** — Educate Child non-magical focuses = flat `EDU_MANA_BUMP` (+15) for build simplicity; can be swapped to stat-scaled later without touching anything else.

**Still open (small forks, non-blocking):**
- **F1** — Folk wizard locations: fixed authored provinces or dynamic spawns? (Recommend fixed set first.)
- **N2** — Mentor discount: same-school-only (deepens specialization) or any-school (speeds generalists)? (Recommend same-school-only.)
- **S1** — School-signature flavor names for the four `<school>_master_trait`s and any bonuses beyond the identity trait (localization/design polish, not a blocker).
- **Special styles** — confirm they stay outside the random pools (recommended) and just convert to `_known_var` grants on their existing unlock paths.
- **Duplicate protection** — confirmed handled: the random helper's `limit = { NOT = { omni_knows_spell } }` guarantees no wasted rolls; if the whole tier is known, the fallback entry no-ops (consider refunding/redirecting to gold in that edge case).

---

## 13. Progression, pacing & specialization

This section reworks how a character advances from novice to master and how breadth across schools is priced. Design targets: a mage develops a **gradient** of stronger and weaker schools through play; a single *discipline* archmage is a **~1.75 year** investment; mastering **all eight disciplines** (both of all four schools) is a rare **~17-year** life's work (fast path; §14); and your strongest schools cast cheaper than your weakest. It folds in the six improvements identified in review (labelled Imp-1…Imp-6 below).

### 13.1 Per-discipline mastery replaces the global archmage gate

The single global `archmage_trait` cannot say "archmage of Melodian only," so it stops being the T4 cast gate. Mastery is tracked at the **discipline** level, because spells and the discipline traits (§13.11) are per-discipline — there are **eight** disciplines (Diviner, Hierophant, Cleric, Illusionist, Elementalist, Druid, Necromancer, Warlock), two per school.

- Add a per-discipline mastery flag, `<discipline>_mastered_var` (eight total), set when a character completes that discipline's top rank. **A discipline's T4/capstone spells are gated by that discipline's `_mastered_var`**, not by `archmage_trait`. (In §4 terms, the T4 option trigger becomes `omni_knows_spell = { SPELL = ... }` + `has_variable = <discipline>_mastered_var`.)
- A **school** counts as mastered when *both* its disciplines are mastered (convenience trigger `omni_school_mastered = { SCHOOL = <x> }`). This is what the affinity/specialty gradient (§13.2) and the mentor loop (§13.7) read.

**The prestige ladder.** On top of the eight discipline traits (§13.11), whole-*school* mastery drives a graduated set of prestige traits. The theme: **depth is rewarded on its own track (school signatures), breadth climbs the archmage ladder.** A pure specialist earns a signature but never becomes an archmage — and that's correct.

| Rung | Requirement | Trait granted | Meaning |
|---|---|---|---|
| School signature | master 1 full school (both disciplines) | `<school>_master_trait` (×4: `omnic_master_trait`, `aldic_master_trait`, `amten_master_trait`, `melodian_master_trait`) | a unique per-school identity — depth reward, one signature per school |
| Archmage | any **2** schools mastered | `archmage_trait` | commands more than one school — the title now means breadth |
| Grand Archmage | any **3** schools mastered | `grand_archmage_trait` (new) | near-total command of the arts |
| Magus Supreme | **all 4** schools mastered (= all 8 disciplines) | `magus_supreme_trait` (new) | the rare mage who can cast every discipline's capstone |

- All five new traits (`<school>_master_trait` ×4, `grand_archmage_trait`, and `magus_supreme_trait` — plus the re-grant of `archmage_trait`) are awarded via a single `on_action` check that runs after any mastery is gained and evaluates how many schools are complete.
- `archmage_trait` is **no longer granted on first discipline** — it now sits at the 2-school rung and is otherwise pure status (no longer gates casting, no longer strips traits; see Imp-5).
- **Mana-discount migration:** `archmage_trait` currently gives the big affordability discount (`divide = 0.6` in `OMNI_magic_spell_costs.txt`). Since archmage now requires two schools, move the per-caster casting discount onto **discipline/school mastery** and the affinity curve (§13.9) so a dedicated single-school master isn't penalized; leave `archmage_trait`/`grand_archmage_trait` a smaller stacking bonus at most.
- **`lich_trait` is explicitly NOT part of this ladder** — it is a special undeath transformation (Jaraharum mission only, `me_jaraharum`), never required for nor granted by any mastery rung.

### 13.2 School affinity — the gradient (emergent specialty)

- Add `<school>_affinity_var` (integer) per character per school — **five variables maximum per character**, incremented as study sessions and ranks complete in that school. They are read only when a magic menu opens or a study course resolves, **never on-tick**, so the gradient costs effectively nothing at runtime.
- **Specialty is emergent**: your "strongest school" is simply whichever affinity is highest — no explicit choice, no extra UI, and it can shift over a lifetime as investment moves. Convenience trigger `omni_primary_school = { SCHOOL = <x> }` = "this school's affinity is the character's highest."
- **Magical Education** (§8) pre-seeds a starting affinity in a chosen school, giving an educated child a head start toward a specialty without granting spells outright.

### 13.3 Escalating cross-school friction (the pacing lever)

Study time and tuition scale two ways. A **cross-school** multiplier grows with how many *other* schools the character has already entered (spreading thin is expensive). A **same-school familiarity** discount cuts the cost of the *second* discipline within a school you already know. The exact multiplier is computed by the friction formula in §14.3; the table below shows the tuned values and the resulting fast-path times (top academy + gifted mage — the assumptions in §14.1).

| Track (order of study) | Friction ×  | Fast-path time |
|---|---|---|
| School 1, discipline A (specialty) | 1.0 | ~1.75 yr |
| School 1, discipline B (familiarity) | 0.5 | ~0.9 yr |
| School 2, discipline A (new school) | 1.3 | ~2.3 yr |
| School 2, discipline B (familiarity) | 0.8 | ~1.4 yr |
| School 3, discipline A | 1.6 | ~2.8 yr |
| School 3, discipline B (familiarity) | 1.1 | ~1.9 yr |
| School 4, discipline A | 1.9 | ~3.3 yr |
| School 4, discipline B (familiarity) | 1.4 | ~2.45 yr |
| **Total (fast path)** | — | **~16.8 yr** |

So the full magus_supreme path is ~**17 in-game years** for a rich, gifted mage at a top academy — and ~**25+ years** (usually a full lifespan, i.e. never) for a mediocre one (§14.7). **Character mortality is the natural cap.** All constants are in the single tuning table in §14.1.

### 13.4 Deterministic progress track (Imp-1)

The current random-walk study can stall a player for years on bad rolls. Replace the *gate* with a deterministic counter and demote the events to flavor.

- Add `magic_study_progress` (character variable), incremented each study session; the rank is granted when it crosses a threshold, then reset for the next rank.
- The existing `me_magic_studying.*` events remain as **bonuses and setbacks** (extra progress, gold, traits) layered on top — they no longer *are* the advancement mechanism, so a run of bad luck slows but never blocks.

### 13.5 Income-scaled tuition (Imp-2)

Replace the flat `omni_pay_magic_study_cost` prices (50/100) with an **income-relative** base, reusing the income script-values already used by the setback events (`six_months_income_svalue`, yearly-income svalues). This keeps study a meaningful sink for a large empire and survivable for a minor. The §13.3 cross-school multiplier stacks on top of this base.

### 13.6 Study speed from academy & attributes (Imp-3)

Progress-per-session (§13.4) gains a bonus from the **magic academy building level** (`num_of_magic_academy_building`, already referenced in `me_magic_studying.5`) and the studier's **attributes** (finesse/zeal). A clever mage at a high-tier academy needs materially fewer sessions than a dullard at a hedge school — rewarding investment in both the character and the building.

### 13.7 Mentor loop (Imp-4)

A character in your court who has mastered a school (or holds `magus_supreme_trait`) reduces the study cost/time of others learning **that same school** (checked over court characters). This gives archmages a role beyond casting and reinforces specialty lineages — a Melodian master breeds more Melodian mages. Optionally, no mentor discount applies off the mentor's school, so mentorship deepens specialization rather than flattening it.

### 13.8 Archmage identity fix (Imp-5)

Remove the trait-stripping in `me_magic_studying.43` and its sibling top-rank events (they currently `remove_trait` every other discipline before adding `archmage_trait`). Under §13.1 masteries and discipline traits **accumulate**; the capstone is additive. This stops punishing exactly the players who invested in breadth.

### 13.9 Specialty casting discount (Imp-6)

In `common/script_values/OMNI_magic_spell_costs.txt`, scale a spell's mana cost by the caster's affinity in that spell's school: **high-affinity (strong) schools cast cheaper, off-school spells cost full or premium.** This is the carrot that makes breadth a genuine tradeoff rather than pure time-tax — a specialist casts cheaply and deeply; a generalist is broadly capable but pays through the nose to both *learn* and *cast* outside their strengths.

### 13.10 New variables & traits summary

| Name | Scope | Purpose | Runtime cost |
|---|---|---|---|
| `<school>_affinity_var` (×4) | character | gradient / emergent specialty / casting discount | read on menu-open & study-resolve only |
| `<discipline>_mastered_var` (×8) | character | per-discipline T4 cast gate | O(1) `has_variable` |
| `magic_study_progress` | character | deterministic rank advancement | incremented on study session |
| `<school>_master_trait` (×4) | character | school-signature depth reward (1 full school) | on_action check |
| `archmage_trait` (repurposed) | character | prestige — **2 schools** mastered; status only, no cast gate | on_action check |
| `grand_archmage_trait` (new) | character | prestige — **3 schools** mastered | on_action check |
| `magus_supreme_trait` (new) | character | prestige capstone — **4 schools / all 8 disciplines** | on_action check |

### 13.11 Trait-wiring audit (discipline & tier traits)

All discipline traits are already granted inside `events/magic_studying_events.txt` at the appropriate study rank; with the trait-strip removed (§13.8) they now **accumulate** as the student progresses. Confirmed grant sites:

| Trait | School / role | Granted where | On the study path? |
|---|---|---|---|
| `mage_trait` | entry (any discipline, first rank) | `magic_studying_events.txt` (many) | ✅ yes |
| `diviner_trait` | Omnic | `magic_studying_events.txt:1455` | ✅ yes |
| `hierophant_trait` | Omnic | `magic_studying_events.txt:1498` | ✅ yes |
| `cleric_trait` | Aldic | `magic_studying_events.txt:1756` | ✅ yes |
| `illusionist_trait` | Aldic | `magic_studying_events.txt:1713` | ✅ yes |
| `elementalist_trait` | Amten | `magic_studying_events.txt:1670` | ✅ yes |
| `druid_trait` | Amten | `magic_studying_events.txt:1627` | ✅ yes |
| `necromancer_trait` | Melodian | `magic_studying_events.txt:1541` | ✅ yes |
| `warlock_trait` | Melodian | `magic_studying_events.txt:1584` | ✅ yes |
| `archmage_trait` | status (first mastery) | `magic_studying_events.txt:1801+` | ✅ yes (repurposed, additive) |
| `lich_trait` | **special undeath** | `events/mission_events/me_jaraharum.txt:322` only | ❌ **no — excluded from magus_supreme** |

Because trait grants are per-discipline, collecting all eight requires studying **both** disciplines of every school — which is exactly what `magus_supreme` now requires (§13.1). The migration `on_action` (§10) must also grant the matching `_mastered_var` and discipline trait to any existing character already holding a top-tier unlock, so legacy archmages don't lose identity. `lich_trait` is left entirely alone.

---

## 14. Build-ready tuning (concrete constants & formulas)

This section resolves open question N1. Every number below is a plug-in constant; the intent is that a builder can implement §13 without making a single balance decision, and later re-tune by editing only §14.1. Put the constants in `common/script_values/OMNI_magic_progression_values.txt`.

### 14.1 Master constants table

| Constant | Value | Used by |
|---|---|---|
| `STUDY_SESSION_INTERVAL` | 60 days (jitter 45–75) | study event cadence (§14.2) |
| `RANK_THRESHOLD` | 40 progress points | one study rank (§14.2) |
| `RANKS_PER_DISCIPLINE` | 4 | Novice→I→II→III→Mastery |
| `PROGRESS_BASE` | 8 | progress/session (§14.2) |
| `PROGRESS_ACADEMY` | +2 per magic academy in study location, cap +6 | progress/session |
| `PROGRESS_APTITUDE` | (finesse + zeal) × 0.15 | progress/session |
| `FRICTION_CROSS_STEP` | +0.3 per *other* school already entered | friction (§14.3) |
| `FRICTION_FAMILIARITY` | −0.5 if the school's other discipline is already started | friction |
| `FRICTION_FLOOR` | 0.4 | friction clamp |
| `TUITION_BASE[rank]` | 25 / 50 / 75 / 100 (ranks I–IV) | tuition (§14.4) |
| `TUITION_INCOME_FACTOR` | +0.5 × monthly income, per rank | tuition |
| `AFFINITY_PER_RANK` | +1 (max 8 per school = 4 ranks × 2 disc.) | affinity (§14.5) |
| `DISCOUNT_PER_AFFINITY` | −0.04 mana cost per affinity point, floor ×0.68 | casting discount (§14.5) |
| `CATCHUP_PRICE[tier]` | 25 / 60 / 120 (Beg / Int / Adv; T4 not buyable) | catch-up spells (§14.6) |
| `CRAMMING_DEBUFF` | `cramming_fatigue_trait`, 180 days, −50% `magic_monthly` | catch-up spells (§14.6) |
| `EDU_MANA_BUMP` | +15 banked `magic` (non-magical focuses) | Educate Child (§8) |
| `EDU_MAGICAL` | +25 `magic_max`, +1 `magic_monthly`, +2 starting affinity | Educate Child (§8) |

**Fast-path assumptions** used for all "fast-path" times in this doc: a magic academy count giving `PROGRESS_ACADEMY` = +4, and a gifted mage (finesse 12, zeal 9 → aptitude ≈ +3), so the progress numerator at friction ×1.0 is **8 + 4 + 3 = 15/session**.

### 14.2 Study progress (replaces the random walk — Imp-1)

A study course fires a session event every `STUDY_SESSION_INTERVAL`. Each session adds progress; when `magic_study_progress` ≥ `RANK_THRESHOLD`, the rank is granted (spells/trait/affinity) and progress resets. The old `me_magic_studying.*` outcomes remain as *flavor* that can add/subtract a little progress or gold, but never gate advancement.

```
progress_per_session = ( PROGRESS_BASE + PROGRESS_ACADEMY + PROGRESS_APTITUDE ) / friction
# each session:
change_variable = { name = magic_study_progress add = progress_per_session }
if = { limit = { var:magic_study_progress >= RANK_THRESHOLD }
       <grant rank: set next _unlocked_var / at top rank set <discipline>_mastered_var + trait>
       change_variable = { name = magic_study_progress subtract = RANK_THRESHOLD }
       <+AFFINITY_PER_RANK to this school's affinity>
       <run prestige-ladder on_action check> }
```

Sessions per rank at fast path = 40 / (15/friction) = `2.67 × friction`; ×4 ranks × 60 days = **`640 × friction` days ≈ `1.75 × friction` years** per discipline (the §13.3 table).

### 14.3 Cross-school friction (§13.3)

```
friction = 1.0
         + FRICTION_CROSS_STEP × ( count_schools_with_affinity>0  − 1 )   # other schools entered
         − ( FRICTION_FAMILIARITY if this school's other discipline already started else 0 )
friction = max( friction, FRICTION_FLOOR )
```

`count_schools_with_affinity>0` is evaluated when the course begins (count of the four `<school>_affinity_var` that are > 0, including the current one). This yields the exact 1.0 / 0.5 / 1.3 / 0.8 / 1.6 / 1.1 / 1.9 / 1.4 progression in the §13.3 table for a mage who does both disciplines of each school before moving on.

### 14.4 Income-scaled tuition (§13.5)

```
tuition(rank) = ( TUITION_BASE[rank] + TUITION_INCOME_FACTOR × monthly_income ) × friction
```

Charged up front per rank via `omni_pay_magic_study_cost`. Fast-path total across all 8 disciplines (32 ranks) ≈ **250 base × Σfriction(9.6) ≈ 2,350 gold** + a modest income surcharge — for Tavian's wealthy realm, ~**3,000–3,500** in tuition.

### 14.5 Affinity & casting discount (§13.2, §13.9)

- `<school>_affinity_var` gains `AFFINITY_PER_RANK` each rank completed in either of the school's disciplines → 0–8, hitting 8 at full school mastery.
- In `OMNI_magic_spell_costs.txt`, multiply the already-scaled mana cost by `max(0.68, 1 − DISCOUNT_PER_AFFINITY × affinity_of_spell_school)`. So a fully-mastered school casts at **×0.68 (−32%)**, one discipline in → ×0.84 (−16%), untrained → ×1.0. This *replaces* the old flat `archmage_trait` `divide = 0.6` discount, so single-school masters keep a benefit archmage-title-holders used to monopolize.

### 14.6 Catch-up spells & cramming debuff (§7)

Buyable random same-tier spells cost `CATCHUP_PRICE[tier]` gold and apply `cramming_fatigue_trait` (new; 180 days; −50% `magic_monthly`). T4 spells are never buyable — earned only by mastering the discipline. To *know every spell* (not just reach mastery), Tavian buys ~40–60 catch-up spells over his life ≈ **3,000–3,500 gold** and repeated bouts of fatigue.

### 14.7 Pacing verification (worked)

Fast path (numerator 15, threshold 40, 60-day sessions), cumulative age from a start at 16.0:

| Track | Friction | Years | Age at completion | Milestone |
|---|---|---|---|---|
| Melodian – Necromancer | 1.0 | 1.75 | 17.8 | `necromancer_trait` |
| Melodian – Warlock | 0.5 | 0.9 | 18.6 | `warlock_trait`, **`melodian_master_trait`** |
| Amten – Elementalist | 1.3 | 2.3 | 20.9 | `elementalist_trait` |
| Amten – Druid | 0.8 | 1.4 | 22.3 | `druid_trait`, `amten_master_trait`, **`archmage_trait`** (2 schools) |
| Omnic – Diviner | 1.6 | 2.8 | 25.1 | `diviner_trait` |
| Omnic – Hierophant | 1.1 | 1.9 | 27.0 | `hierophant_trait`, `omnic_master_trait`, **`grand_archmage_trait`** (3 schools) |
| Aldic – Cleric | 1.9 | 3.3 | 30.3 | `cleric_trait` |
| Aldic – Illusionist | 1.4 | 2.45 | 32.8 | `illusionist_trait`, `aldic_master_trait`, **`magus_supreme_trait`** (4 schools) |

**Fast path total ≈ 16.8 years → Magus Supreme at ~33.** A mediocre mage (numerator ≈ 10) runs ~1.5× longer — ~25 years, i.e. usually death first. **Total gold ≈ 6,000–7,000** (tuition ~3,000–3,500 + catch-up ~3,000–3,500).

### 14.8 Definition of done (acceptance tests for the builder)

1. A fresh mage studying one discipline at a top academy reaches that discipline's mastery in **~1.6–1.9 years** and gets the discipline trait + `<discipline>_mastered_var`, no trait-strip.
2. Completing a second discipline of the same school grants `<school>_master_trait`; it does **not** grant `archmage_trait`.
3. Mastering a 2nd / 3rd / 4th **school** grants `archmage_trait` / `grand_archmage_trait` / `magus_supreme_trait` respectively, via the on_action check.
4. A caster can only see/cast a spell whose `<id>_known_var` they hold; T4 options additionally require the school's discipline `_mastered_var`.
5. Casting cost falls as school affinity rises, bottoming at ×0.68; no character has the old flat `archmage_trait` mana break.
6. `lich_trait` is never granted or required anywhere in the study/mastery flow.
7. Loading a pre-rework save: every legacy `*_tN_unlocked_var` holder ends with the matching `_known_var`, `_mastered_var`, discipline trait, and prestige rung — and loses no spells.
8. `imperator-tiger` reports no new errors; the pacing check (test 1 + full 8-track run) lands within ±15% of the §14.7 ages.

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
