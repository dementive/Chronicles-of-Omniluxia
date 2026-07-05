#!/usr/bin/env python3
"""
Regenerate the "currently equipped" hover-tooltip localization for the item
system from the authoritative modifier definitions.

WHY THIS EXISTS
    The Items button on the character window shows, on hover, each equipped
    item and its bonuses (e.g. "Sword:  Martial +2, Light Inf. Offense +5%").
    Imperator cannot expand a static modifier's numbers into tooltip text at
    runtime, so those bonus strings are pre-written localization. This script
    rebuilds that localization from common/modifiers/choo_items_modifiers.txt
    so the tooltip always matches the real modifier values.

WHEN TO RUN
    After you change any item's `item_<id>_modifier` values (or add/remove a
    stat). Run it and the tooltip lines resync automatically.

USAGE
    From the mod root (…/mod/Omniluxia):
        python tools/regen_equipped_tooltip.py
    It rewrites localization/english/omni_equipped_l_english.yml (UTF-8 + BOM).

NOTE
    Only edits to modifier VALUES need a rerun. Item NAMES in the tooltip are
    resolved live in-game and never drift. If you add a BRAND-NEW stat type not
    in STAT_LABELS below, add it here (label + format) or it will be skipped.
"""

import os
import re
import sys

# ---- stat -> (display label, format) -------------------------------------
# format: 'pct' shows value*100 with % ; 'flat' shows the raw value ;
#         'mon' shows the raw value (small monthly additive numbers).
PCT, FLAT, MON = "pct", "flat", "mon"
STAT_LABELS = {
    "martial": ("Martial", FLAT), "finesse": ("Finesse", FLAT),
    "charisma": ("Charisma", FLAT), "zeal": ("Zeal", FLAT), "health": ("Health", FLAT),
    "character_loyalty": ("Loyalty", FLAT), "subject_loyalty": ("Subject Loyalty", FLAT),
    "diplomatic_relations": ("Diplomatic Relations", FLAT),
    "diplomatic_reputation": ("Diplomatic Reputation", FLAT),
    "global_unrest": ("Global Unrest", FLAT), "local_building_slot": ("Local Building Slots", FLAT),
    "monthly_legitimacy": ("Monthly Legitimacy", MON), "monthly_centralization": ("Monthly Centralization", MON),
    "monthly_character_prominence": ("Monthly Prominence", MON), "monthly_character_popularity": ("Monthly Popularity", MON),
    "monthly_character_experience": ("Monthly Character XP", MON), "monthly_military_experience": ("Monthly Military XP", MON),
    "monthly_tyranny": ("Monthly Tyranny", MON),
    "discipline": ("Discipline", PCT),
    "heavy_infantry_offensive": ("Heavy Inf. Offense", PCT), "heavy_infantry_defensive": ("Heavy Inf. Defense", PCT),
    "heavy_infantry_morale": ("Heavy Inf. Morale", PCT), "heavy_infantry_discipline": ("Heavy Inf. Discipline", PCT),
    "light_infantry_offensive": ("Light Inf. Offense", PCT), "light_infantry_defensive": ("Light Inf. Defense", PCT),
    "light_infantry_morale": ("Light Inf. Morale", PCT),
    "heavy_cavalry_offensive": ("Heavy Cav. Offense", PCT), "heavy_cavalry_morale": ("Heavy Cav. Morale", PCT),
    "light_cavalry_offensive": ("Light Cav. Offense", PCT),
    "archers_offensive": ("Archer Offense", PCT), "archers_discipline": ("Archer Discipline", PCT),
    "archers_forest_combat_bonus": ("Archer Forest Combat", PCT),
    "assault_ability": ("Assault Ability", PCT), "siege_ability": ("Siege Ability", PCT),
    "land_morale_modifier": ("Land Morale", PCT), "local_defensive": ("Local Defense", PCT),
    "garrison_size": ("Garrison Size", PCT),
    "army_maintenance_cost": ("Army Maintenance Cost", PCT), "army_movement_speed": ("Army Movement Speed", PCT),
    "military_tech_investment": ("Military Tech Investment", PCT), "oratory_tech_investment": ("Oratory Tech Investment", PCT),
    "civic_tech_investment": ("Civic Tech Investment", PCT), "religious_tech_investment": ("Religious Tech Investment", PCT),
    "global_commerce_modifier": ("Commerce", PCT), "global_tax_modifier": ("Tax Income", PCT),
    "global_monthly_civilization": ("Monthly Civilization", PCT),
    "global_pop_assimilation_speed_modifier": ("Pop Assimilation Speed", PCT),
    "global_pop_conversion_speed": ("Pop Conversion Speed", PCT),
    "global_pop_conversion_speed_modifier": ("Pop Conversion Speed", PCT),
    "global_population_happiness": ("Population Happiness", PCT),
    "happiness_for_same_religion_modifier": ("Same-Religion Happiness", PCT),
    "monthly_political_influence_modifier": ("Political Influence", PCT), "omen_power": ("Omen Power", PCT),
    "war_exhaustion": ("War Exhaustion", PCT), "agressive_expansion_impact": ("Aggressive Expansion Impact", PCT),
    "build_cost": ("Build Cost", PCT), "build_time": ("Build Time", PCT),
    "global_monthly_state_loyalty": ("Monthly Provincial Loyalty", MON),
    "great_work_tribals_workrate_character_modifier": ("Tribal Workrate", PCT),
    "global_defensive": ("Global Defense", PCT),
}


def fmtnum(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def stat_str(key, val):
    label, fm = STAT_LABELS[key]
    sign = "+" if val >= 0 else "-"
    av = abs(val)
    if fm == PCT:
        return f"{label} {sign}{fmtnum(av * 100)}%"
    return f"{label} {sign}{fmtnum(av)}"


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mods_path = os.path.join(root, "common", "modifiers", "choo_items_modifiers.txt")
    artifact_mods_path = os.path.join(root, "common", "modifiers", "omni_artifact_modifiers.txt")
    names_path = os.path.join(root, "localization", "english", "choo_items_l_english.yml")
    artifact_names_path = os.path.join(root, "localization", "english", "omni_artifact_items_l_english.yml")
    equip_path = os.path.join(root, "common", "scripted_guis", "choo_items_gui.txt")
    out_path = os.path.join(root, "localization", "english", "omni_equipped_l_english.yml")

    # item display order = order of equip_item_<id> scripted GUI definitions
    order = []
    seen = set()
    for line in open(equip_path, encoding="utf-8"):
        m = re.match(r"^[ ]{0,2}equip_item_(\w+)\s*=\s*\{\s*$", line)
        if m and m.group(1) != "with_modifier":
            order.append(m.group(1))
            seen.add(m.group(1))
        
        m_art = re.search(r"target = flag:item_(artifact_\w+)", line)
        if m_art and m_art.group(1) not in seen:
            order.append(m_art.group(1))
            seen.add(m_art.group(1))

    # item -> [(stat, value), ...]
    mod_text = open(mods_path, encoding="utf-8").read()
    if os.path.exists(artifact_mods_path):
        mod_text += "\n" + open(artifact_mods_path, encoding="utf-8").read()
        
    item_mods = {}
    for m in re.finditer(r"item_(\w+)_modifier\s*=\s*\{([^}]*)\}", mod_text):
        stats = [(km.group(1), float(km.group(2)))
                 for km in re.finditer(r"(\w+)\s*=\s*(-?[\d.]+)", m.group(2))]
        item_mods[m.group(1)] = stats

    # item -> display name
    names = {}
    names_text = open(names_path, encoding="utf-8-sig").read()
    if os.path.exists(artifact_names_path):
        names_text += "\n" + open(artifact_names_path, encoding="utf-8-sig").read()
        
    for m in re.finditer(r'^\s*item_(\w+?)(?:_modifier)?:0\s*"([^"]*)"', names_text, re.M):
        names[m.group(1)] = m.group(2)

    unknown = set()
    lines = [
        "l_english:",
        r' omni_equipped_header:0 "\n#Y Currently Equipped:#!"',
        r' omni_equipped_none:0 "\n#I Nothing equipped.#!"',
    ]
    for it in order:
        parts = []
        for k, v in item_mods.get(it, []):
            if k in STAT_LABELS:
                parts.append(stat_str(k, v))
            else:
                unknown.add(k)
        summ = ", ".join(parts) if parts else "no bonus"
        nm = names.get(it, it).replace('"', "'")
        lines.append(f' omni_equipped_item_{it}:0 "  #Y {nm}:#!  {summ}"')

    open(out_path, "w", encoding="utf-8-sig", newline="\n").write("\n".join(lines) + "\n")
    print(f"Wrote {out_path} ({len(order)} items).")
    if unknown:
        print("WARNING: unmapped stat(s), add them to STAT_LABELS:", ", ".join(sorted(unknown)))
        sys.exit(1)


if __name__ == "__main__":
    main()
