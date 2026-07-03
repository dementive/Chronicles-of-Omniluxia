# Spell Wiring Summary

This patch wires the new spell icons into the existing magic menus so they are usable in normal play, not just in adventure content.

## What Was Added

### Friendly battle buffs
- `Arcane Aegis`
- `Battle Clarity`
- `Conjured Supply`
- `Spectral Boarders`

### Offensive battlefield spells
- `Blighted Ground`
- `Mudbind`
- `Cursed Rigging`
- `Fear Hex`
- `Storm Wrack`

### Civic and support spells
- `Harbor Ward`
- `Unrest Cleansing`
- `Migration Beacon`
- `Weathercall Growth`

### Siege spells
- `Burning Pitch`
- `Counterward`
- `Illusion Walls`

## Where They Appear

- [Friendly battle menu](C:/Users/Joshua/Documents/Paradox%20Interactive/Imperator/mod/Omniluxia/events/zorgo_magic_events.txt) `me_zorgo_magic_focus.29`
- [Amten offensive menu](C:/Users/Joshua/Documents/Paradox%20Interactive/Imperator/mod/Omniluxia/events/zorgo_magic_events.txt) `me_zorgo_magic_focus.12`
- [Melodian offensive menu](C:/Users/Joshua/Documents/Paradox%20Interactive/Imperator/mod/Omniluxia/events/zorgo_magic_events.txt) `me_zorgo_magic_focus.13`
- [Omnic support menu](C:/Users/Joshua/Documents/Paradox%20Interactive/Imperator/mod/Omniluxia/events/zorgo_magic_events.txt) `me_zorgo_magic_focus.17`
- [Aldic support menu](C:/Users/Joshua/Documents/Paradox%20Interactive/Imperator/mod/Omniluxia/events/zorgo_magic_events.txt) `me_zorgo_magic_focus.18`
- [Amten support menu](C:/Users/Joshua/Documents/Paradox%20Interactive/Imperator/mod/Omniluxia/events/zorgo_magic_events.txt) `me_zorgo_magic_focus.19`
- [Siege menu](C:/Users/Joshua/Documents/Paradox%20Interactive/Imperator/mod/Omniluxia/events/zorgo_magic_events.txt) `me_zorgo_magic_focus.30`

## What Connects To What

- Mana spending uses the shared `omni_spell_mana_cost_effect` helper.
- Affordability checks use the existing `omni_can_afford_spell` trigger.
- Buff/debuff state uses new spell modifiers in [`common/modifiers/SPELLS_mods.txt`](C:/Users/Joshua/Documents/Paradox%20Interactive/Imperator/mod/Omniluxia/common/modifiers/SPELLS_mods.txt).
- Icon mapping is wired in [`common/modifier_icons/00_modifier_icons.txt`](C:/Users/Joshua/Documents/Paradox%20Interactive/Imperator/mod/Omniluxia/common/modifier_icons/00_modifier_icons.txt).

## How To Test In Game

1. Load into a save as a ruler or mage-caster.
2. Open the main character spell button and confirm the new support spells appear.
3. Select an army or navy, open the friendly spell button, and confirm the four new battle buffs appear.
4. Target an enemy army or fleet and open the offensive spell menu to confirm the new hostile spells appear.
5. Start or load a siege, open the siege spell button, and confirm the three new siege spells appear.
6. Cast each spell and verify:
- Mana is reduced from the caster who clicked the spell.
- The correct modifier icon appears on the unit, country, or province.
- The modifier duration matches the spell type.
- Recasting refreshes the same modifier instead of stacking duplicates.
7. Re-open the same menu after casting to make sure the option still shows as long as the caster has enough mana and the required magic unlock.

## Practical Notes

- The new battle, civic, and siege spells are now available through the normal spell menus, so they are not adventure-only content.
- The monthly AI battle-cast pass now considers both armies and fleets, and it can use the new battle-side buffs as well as the existing speed/morale spells.
