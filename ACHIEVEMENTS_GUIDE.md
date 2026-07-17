# Omniluxia Custom Achievements — How It Works & How to Add Your Own

This mod ships a self-contained custom achievement system. It exists because
Imperator's real Steam achievements can't be modified by mods, so we fake them
in-game: each achievement is a saved variable, a scripted check, a window entry,
and a side-window notice. The whole thing auto-unlocks like real Steam achievements — no
buttons to press — and can be turned off with an in-game toggle.

This guide explains the architecture, then walks through adding a new achievement
end-to-end.

---

## 1. Architecture at a glance

An achievement is just a **variable on the country** (e.g. `omni_ach_awakening`).
If the variable is set, the achievement is unlocked. Everything else is plumbing
around that idea:

| File | Role |
|------|------|
| `common/scripted_triggers/omni_achievement_triggers.txt` | The **unlock condition** for each achievement, written once. Shared by the check loop and the window. |
| `common/scripted_effects/00_achievement_effects.txt` | `omni_unlock_achievement` (grants + queues the side-window notice, the single choke-point) and `omni_check_achievements` (polls every state-based achievement). |
| `common/on_action/omni_achievements.txt` | Runs the poll monthly **for the human player only**, and hooks battle-won for the event-driven one. |
| `events/omni_achievements_events.txt` | One lightweight minor event per achievement (`omni_ach.1`, `.2`, …), queued one day after unlock for the side window. |
| `common/scripted_guis/sguis_omni_achievements.txt` | One `scripted_gui` per achievement: `is_shown` = unlocked, `is_valid` = still possible. Plus the on/off game-rule toggle. |
| `gui/achievements/jomini_achievements_window.gui` | Overrides the vanilla achievements window to list our achievements in difficulty groups. |
| `gui/shared/ach_font_icons.gui` | The `@ach_<key>!` inline picture used in the minor event. |
| `gui/custom_textformatting.gui` | Defines the `#huge` text tag the minor event uses. |
| `localization/english/omni_achievements_l_english.yml` | All player-facing text. |

### How an unlock actually happens

1. Once a month, for the **player only**, `monthly_country_pulse` calls
   `omni_check_achievements`.
2. That effect walks every achievement: *if the variable isn't set AND the
   shared trigger is true*, it calls `omni_unlock_achievement`.
3. `omni_unlock_achievement` is the **only** thing that ever grants an
   achievement. It checks the achievement isn't already owned, checks the game
   rule isn't disabled, sets the variable, and queues the minor event with a
   one-day delay so the side-window notice appears outside the check pulse.
4. The window reads the variable through the achievement's `scripted_gui`
   (`ScriptedGui.IsShown`) and shows the gold "unlocked" ornament.

Event-driven achievements (like winning a battle led by a child) skip the poll
and call `omni_unlock_achievement` straight from their `on_action`.

### Why it's built this way

- **One source of truth for conditions.** The unlock condition lives only in
  `scripted_triggers`. The check loop and the window's "still possible" state
  both reference it, so they can never disagree. (The original system this was
  ported from duplicated conditions and drifted out of sync.)
- **One choke-point for granting.** Because every unlock goes through
  `omni_unlock_achievement`, a single guard there — "not already owned, and the
  game rule isn't off" — covers polled *and* event-driven achievements and any
  future caller.
- **No manual re-check button, no performance cost.** The poll only ever runs
  for the human player, monthly. AI never touches it.

---

## 2. The enable/disable game rule

A global variable, `game_rule_disable_achievements`, switches the whole system
off. Default is **on** (variable absent). It's toggled by the
`omni_achievements_toggle` scripted_gui, surfaced as a button in the top-left of
the achievements window (next to "Show base-game achievements").

The rule is enforced in exactly one place — inside `omni_unlock_achievement`:

```
NOT = { has_global_variable = game_rule_disable_achievements }
```

So while it's off, nothing new unlocks; achievements already earned are kept, and
flipping it back on resumes automatic unlocking.

---

## 3. Adding a new achievement (step by step)

Say we want **"The Dragon's Hoard" — hold 5,000 gold as a Dwarven realm.**
Pick a unique key: `omni_ach_dragons_hoard`. It needs the next free event id
(look at `events/omni_achievements_events.txt`; if the last is `omni_ach.16`,
use `omni_ach.17`).

### Step 1 — Condition (`common/scripted_triggers/omni_achievement_triggers.txt`)

```
omni_ach_dragons_hoard_trigger = {
    omni_is_dwarven_realm = yes   # reuse an existing helper where you can
    treasury >= 5000
}
```

> Only use triggers you've confirmed exist in Omniluxia's own script. Handy
> country-scope ones already in use: `num_of_cities`, `manpower`, `treasury`,
> `var:magic` (guard with `has_variable = magic` first), `country_culture_group`,
> `tag`, `owns_or_subject_owns_region`, `current_ruler = { ... }`,
> `any_character = { ... }`.

### Step 2 — Check loop (`common/scripted_effects/00_achievement_effects.txt`)

Add one block inside `omni_check_achievements`, before its closing brace:

```
    if = {
        limit = { NOT = { has_variable = omni_ach_dragons_hoard } omni_ach_dragons_hoard_trigger = yes }
        omni_unlock_achievement = { ACH = omni_ach_dragons_hoard EVENT = omni_ach.17 }
    }
```

*(Skip this step only if the achievement is purely event-driven — see §4.)*

### Step 3 — Minor event (`events/omni_achievements_events.txt`)

This is the side-window achievement notice. `omni_unlock_achievement` queues it
with a one-day delay instead of firing it inline.

```
omni_ach.17 = {
    type = minor_country_event
    title = "omni_ach_dragons_hoard_unlocked"
    desc = "omni_ach_dragons_hoard_unlocked_desc"
    option = { name = "CONFIRM" }
}
```

### Step 4 — Scripted GUI (`common/scripted_guis/sguis_omni_achievements.txt`)

```
omni_ach_dragons_hoard = {
    scope = country
    is_shown = { has_variable = omni_ach_dragons_hoard }   # unlocked?
    is_valid = { omni_is_dwarven_realm = yes }             # still possible? (drives the "Not possible" tab)
}
```

Use `is_valid = { always = yes }` if it can never become impossible.

### Step 5 — Window entry (`gui/achievements/jomini_achievements_window.gui`)

Inside the difficulty group you want (e.g. the `hard` group's inner
`flowcontainer`), add:

```
                achievement = {
                    datacontext = "[GetScriptedGui('omni_ach_dragons_hoard')]"
                    blockoverride "icon" { texture = "gfx/interface/icons/achievements/ach_gazophylax.dds" }
                    blockoverride "name" { text = "omni_ach_dragons_hoard" }
                    blockoverride "desc" { text = "omni_ach_dragons_hoard_desc" }
                }
```

### Step 6 — Toast icon (`gui/shared/ach_font_icons.gui`)

```
texticon = {
    icon = ach_dragons_hoard
    iconsize = { texture = "gfx/interface/icons/achievements/ach_gazophylax.dds" size = { 50 50 } offset = { 0 27 } fontsize = 10 }
}
```

### Step 7 — Localization (`localization/english/omni_achievements_l_english.yml`)

```
 omni_ach_dragons_hoard: "The Dragon's Hoard"
 omni_ach_dragons_hoard_desc: "Hold 5,000 gold as a Dwarven realm."
 omni_ach_dragons_hoard_unlocked: "$achievement_unlock$$omni_ach_dragons_hoard$"
 omni_ach_dragons_hoard_unlocked_desc: "$achievement_unlock_desc$@ach_dragons_hoard!\n#bold $omni_ach_dragons_hoard$#!\n$omni_ach_dragons_hoard_desc$"
```

Note the four keys mirror the pattern: name, description (the goal), toast title,
toast body. The `@ach_dragons_hoard!` in the last line must match the `icon` name
from Step 6.

That's it. Save, and the achievement will auto-unlock when the condition is met.

---

## 4. Event-driven achievements (instant, not polled)

For things that happen at a moment rather than a persistent state — winning a
particular battle, completing a mission — skip the check loop (Step 2) and call
`omni_unlock_achievement` from the relevant `on_action` in
`common/on_action/omni_achievements.txt`. The existing "Born to the Blade"
example shows the pattern:

```
on_battle_won_country = {
    effect = {
        if = {
            limit = {
                is_ai = no
                NOT = { has_variable = omni_ach_boy_general }
                scope:actor = { exists = commander commander = { age <= 15 } }
            }
            omni_unlock_achievement = { ACH = omni_ach_boy_general EVENT = omni_ach.3 }
        }
    }
}
```

You still do Steps 1 (optional), 3, 4, 5, 6, and 7. The game-rule guard inside
`omni_unlock_achievement` covers these automatically.

To hook a mission instead, add the `omni_unlock_achievement` call to that
mission task's `on_completion` / bookmark effect.

---

## 5. Icons / art

All achievements currently point at **base-game** achievement `.dds` files as
placeholders so nothing renders as a broken texture. To use custom art:

1. Drop a 64×64 `.dds` (DXT5) into `gfx/interface/icons/achievements/`, e.g.
   `omni_ach_dragons_hoard.dds`.
2. Repoint the two texture references — the `blockoverride "icon"` in the window
   (Step 5) and the `texticon` in `ach_font_icons.gui` (Step 6).

The window and the toast use the same texture, so keep the two in sync.

---

## 6. Testing & validation

- **Syntax:** run the bundled `imperator-tiger.exe` against the mod. Every file
  in this system should come back clean.
- **Quick in-game test:** open the console and set the variable directly, e.g.
  `set_variable omni_ach_dragons_hoard` on your country, then open the
  achievements window — it should show as unlocked. Remove it with
  `remove_variable omni_ach_dragons_hoard`.
- **Reference count sanity:** the achievement files must all agree on the key. A quick
  check that everything is wired:
  `grep -rl "omni_ach_dragons_hoard" common events gui localization` should list
  all the files you edited.
- **Braces:** Paradox is brace-delimited; an unmatched `{`/`}` breaks the file
  silently. Count them if a file stops loading.

---

## 7. Reference: current achievements & event ids

| Group | Key | Event | Unlock condition |
|-------|-----|-------|------------------|
| Very Easy | `omni_ach_court_mage` | `omni_ach.1` | Employ a Mage/Archmage |
| Very Easy | `omni_ach_awakening` | `omni_ach.2` | 50 Magic |
| Easy | `omni_ach_boy_general` | `omni_ach.3` | Win a battle led by a commander ≤15 (event-driven) |
| Easy | `omni_ach_elven_realm` | `omni_ach.4` | Elven realm, 25 cities |
| Easy | `omni_ach_dwarven_hold` | `omni_ach.5` | Dwarven realm, 20 cities |
| Medium | `omni_ach_archmage` | `omni_ach.6` | Have an Archmage |
| Medium | `omni_ach_font_of_power` | `omni_ach.7` | 100 Magic |
| Medium | `omni_ach_serpent_empire` | `omni_ach.8` | Snakefolk realm, 30 cities |
| Medium | `omni_ach_iron_foothills` | `omni_ach.9` | Own the Iron Foothills region |
| Hard | `omni_ach_orc_horde` | `omni_ach.10` | Orcish realm, 50,000 manpower |
| Hard | `omni_ach_sorcerer_king` | `omni_ach.11` | Ruler is an Archmage |
| Hard | `omni_ach_great_realm` | `omni_ach.12` | 100 cities |
| Very Hard | `omni_ach_weave_unbound` | `omni_ach.13` | 200 Magic |
| Very Hard | `omni_ach_arcane_academy` | `omni_ach.14` | 5+ Mages/Archmages at once |
| Very Hard | `omni_ach_unifier` | `omni_ach.15` | 200 cities |
| Very Hard | `omni_ach_zani_empire` | `omni_ach.16` | Form the Zani Empire (tag ZAN) |

Next free event id: **`omni_ach.17`**.
