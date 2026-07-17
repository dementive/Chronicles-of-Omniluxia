from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MODIFIERS = ROOT / "common" / "modifiers" / "omniluxia_generated_items_split.txt"
ITEM_LOC = ROOT / "localization" / "english" / "omniluxia_random_items_l_english.yml"
MOD_LOC = ROOT / "localization" / "english" / "omniluxia_generated_item_modifiers_l_english.yml"

ITEM_RE = re.compile(r"^(item_gen_[A-Za-z0-9_]+)_(personal|army)\s*=\s*\{([^}]*)\}", re.MULTILINE)
LOC_RE = re.compile(r'^\s*(item_gen_[A-Za-z0-9_]+_\d+):0\s+"([^"]+)"', re.MULTILINE)
VALUE_RE = re.compile(r"([A-Za-z0-9_]+)\s*=\s*(-?[0-9.]+)")

DISPLAY_CULTURE_OVERRIDES = {
    "alaguric": "Alagurican",
    "aralans": "Aralansan",
    "aralanic_rohenoa": "Aralanic Rohenoan",
    "arteonian": "Arteonian",
    "austropetolian": "Austropetolian",
    "beatepian": "Beatepian",
    "beatepian_upper": "Upper Beatepian",
    "borderlander": "Borderlander",
    "celtican": "Celtican",
    "colonial_zani": "Colonial Zani",
    "common_dwarves": "Common Dwarven",
    "deep_gnome": "Deep Gnome",
    "dissolved": "Dissolved",
    "dissolved_revant": "Dissolved Revant",
    "dragkhanic": "Dragkhanic",
    "dragkhanic_eastern": "Eastern Dragkhanic",
    "dragkhanic_jarenam": "Jarenam Dragkhanic",
    "dragkhanic_western": "Western Dragkhanic",
    "drow": "Drow",
    "gold_dwarves": "Gold Dwarven",
    "eastern_zerywani": "Eastern Zerywani",
    "eptelon": "Eptelonian",
    "errnorfallian": "Errnorfallian",
    "esquelian": "Esquelian",
    "etaredican": "Etaredican",
    "flusenlander": "Flusenlander",
    "forest_elves": "Forest Elven",
    "gellaiaus_group": "Gellaiaus",
    "gnome": "Gnome",
    "goblin": "Goblin",
    "halfling": "Halfling",
    "hazalars": "Hazalar",
    "high_elves": "High Elven",
    "high_half_elves": "High Half-Elven",
    "hobgoblin": "Hobgoblin",
    "horteonian": "Horteonian",
    "intermarenican": "Intermarenican",
    "kinones": "Kinone",
    "middleonian": "Middleonian",
    "morrigon_forest_elves": "Morrigon Forest Elven",
    "norrfallian": "Norrfallian",
    "northern_arteonian": "Northern Arteonian",
    "northlander": "Northlander",
    "orcish": "Orcish",
    "orcish_humano": "Humano-Orcish",
    "phanician": "Phanician",
    "placeholder": "Local",
    "polarian": "Polarian",
    "polarian_arame": "Polarian Arame",
    "rohenoan": "Rohenoan",
    "sea_elves": "Sea Elven",
    "seeker": "Seeker",
    "selaskusian": "Selaskusian",
    "silver_dwarves": "Silver Dwarven",
    "silver_halfling": "Silver Halfling",
    "snakefolk": "Snakefolk",
    "southern_forest_elves": "Southern Forest Elven",
    "sunset_elves": "Sunset Elven",
    "vetalian": "Vetalian",
    "zani_vetalian": "Zani Vetalian",
    "weagelian_zani": "Weagelian Zani",
    "werhenlander": "Werhenlander",
    "werhenssian": "Werhenssian",
    "westeonian": "Westeonian",
    "western_half_elves": "Western Half-Elven",
    "western_zerywani": "Western Zerywani",
    "yedidyah": "Yedidyah",
    "zarakens": "Zaraken",
    "zinduidian": "Zinduidian",
    "zorg": "Zorg",
    "saurthi": "Saurthi",
    "chivix": "Chivix",
    "ayrith": "Ayrith",
    "mycelar": "Mycelar",
    "tsaalan": "Tsaalan",
    "vothkin": "Vothkin",
    "ulvenar": "Ulvenar",
    "selkanu": "Selkanu",
    "khedrim": "Khedrim",
    "tenmari": "Tenmari",
}

DISPLAY_STAT = {
    "martial": "Martial",
    "finesse": "Finesse",
    "charisma": "Charisma",
    "zeal": "Zeal",
    "health": "Health",
    "discipline": "Discipline",
    "land_morale_modifier": "Land Morale",
    "monthly_character_experience": "Monthly Character XP",
}

UNIT_DISPLAY = {
    "light_infantry": "Light Infantry",
    "heavy_infantry": "Heavy Infantry",
    "archers": "Archers",
    "light_cavalry": "Light Cavalry",
    "heavy_cavalry": "Heavy Cavalry",
    "horse_archers": "Horse Archers",
    "camels": "Camel Cavalry",
    "warelephant": "War Elephants",
    "chariots": "Chariots",
    "tetrere": "Tetreres",
    "hexere": "Hexeres",
    "octere": "Octeres",
    "mega_galley": "Mega-Galleys",
}

TERRAIN_DISPLAY = {
    "forest": "Forest",
    "hills": "Hills",
    "mountain": "Mountain",
    "jungle": "Jungle",
    "desert": "Desert",
    "plains": "Plains",
    "flood_plain": "Flood Plain",
}

OFFENSE_NOUNS = {
    "archers": "Bow",
    "horse_archers": "Recurve",
    "light_infantry": "Blade",
    "heavy_infantry": "Warblade",
    "light_cavalry": "Rider's Spear",
    "heavy_cavalry": "Lance",
    "camels": "Desert Lance",
    "warelephant": "Elephant Goad",
    "chariots": "Charioteer's Spear",
    "tetrere": "Naval Ram",
    "hexere": "Fleet Standard",
    "octere": "Great Ram",
    "mega_galley": "Grand Naval Ram",
}

DEFENSE_NOUNS = {
    "archers": "Pavise",
    "horse_archers": "Rider's Guard",
    "light_infantry": "Shield",
    "heavy_infantry": "Hauberk",
    "light_cavalry": "Rider's Mail",
    "heavy_cavalry": "Cavalry Barding",
    "camels": "Desert Ward",
    "warelephant": "Elephant Armor",
    "chariots": "Chariot Guard",
    "tetrere": "Hull Reinforcement",
    "hexere": "Fleet Guard",
    "octere": "Great Hull Reinforcement",
    "mega_galley": "Grand Hull Reinforcement",
}

QUALITY = {
    range(1, 14): "",
    range(14, 19): "Fine ",
    range(19, 21): "Masterwork ",
}

def values(body):
    return {key: float(value) for key, value in VALUE_RE.findall(body)}

def culture_from_item(item_id):
    match = re.match(r"item_gen_(.+)_(\d+)$", item_id)
    if not match:
        raise ValueError(f"Unexpected generated item id: {item_id}")
    return match.group(1), int(match.group(2))

def display_culture(culture):
    if culture in DISPLAY_CULTURE_OVERRIDES:
        return DISPLAY_CULTURE_OVERRIDES[culture]
    return " ".join(part.capitalize() for part in culture.split("_"))

def quality_prefix(number):
    for span, prefix in QUALITY.items():
        if number in span:
            return prefix
    return ""

def unit_from_modifier(key):
    for unit in sorted(UNIT_DISPLAY, key=len, reverse=True):
        if key.startswith(unit + "_"):
            return unit
    return ""

def primary_army(army):
    typed = []
    for key, value in army.items():
        unit = unit_from_modifier(key)
        if unit:
            typed.append((key, value, unit))
    if typed:
        priority = {"offensive": 0, "defensive": 1, "discipline": 2, "morale": 3}
        def sort_key(row):
            key, value, _unit = row
            kind = next((name for name in priority if key.endswith("_" + name)), "z")
            return (priority.get(kind, 9), -abs(value))
        return sorted(typed, key=sort_key)[0]
    if army:
        return sorted(army.items(), key=lambda row: -abs(row[1]))[0] + ("",)
    return ("", 0.0, "")

def noun_for(item_id, personal, army):
    key, _value, unit = primary_army(army)
    if key.endswith("_defensive") or "_defensive" in key:
        return DEFENSE_NOUNS.get(unit, "Armor")
    if key.endswith("_offensive") or "_offensive" in key:
        return OFFENSE_NOUNS.get(unit, "Weapon")
    if key.endswith("_discipline") or key.endswith("_morale") or key in {"discipline", "land_morale_modifier"}:
        return "Banner"
    if "monthly_character_experience" in personal:
        return "Codex"
    if any(stat in personal for stat in ("finesse", "charisma")):
        return "Signet"
    if "zeal" in personal:
        return "Amulet"
    if "martial" in personal:
        return "War-Totem"
    if "health" in personal:
        return "Ward"
    return "Relic"

def suffix_for(army):
    key, _value, unit = primary_army(army)
    if unit:
        if key.endswith("_offensive"):
            return "of the " + UNIT_DISPLAY[unit]
        if key.endswith("_defensive"):
            return "of the " + UNIT_DISPLAY[unit] + " Guard"
        if key.endswith("_discipline"):
            return "of the " + UNIT_DISPLAY[unit] + " Drill"
        if key.endswith("_morale"):
            return "of the " + UNIT_DISPLAY[unit] + " Host"
    for terrain, display in TERRAIN_DISPLAY.items():
        if any(("_" + terrain + "_combat_bonus") in key for key in army):
            return "of the " + display
    if "land_morale_modifier" in army or "discipline" in army:
        return "of the Host"
    return ""

def item_name(item_id, personal, army):
    culture, number = culture_from_item(item_id)
    prefix = quality_prefix(number)
    noun = noun_for(item_id, personal, army)
    suffix = suffix_for(army)
    name = f"{prefix}{display_culture(culture)} {noun}"
    if suffix and suffix not in name:
        name += f" {suffix}"
    return name

def format_value(key, value):
    sign = "+" if value >= 0 else ""
    if key in {"martial", "finesse", "charisma", "zeal"}:
        return f"{sign}{int(value)} {DISPLAY_STAT[key]}"
    if key == "health":
        return f"{sign}{int(round(value * 100))} Health"
    if abs(value) < 1 and key not in {"monthly_character_experience"}:
        val = int(round(value * 100))
        suffix = "%"
    else:
        val = value
        suffix = ""
    if key == "monthly_character_experience":
        return f"{sign}{value:.2f} {DISPLAY_STAT[key]}"
    unit = unit_from_modifier(key)
    if unit:
        tail = key[len(unit) + 1:]
        if tail in {"offensive", "defensive", "discipline", "morale"}:
            stat = tail.capitalize().replace("Offensive", "Offense").replace("Defensive", "Defense")
            return f"{sign}{val}{suffix} {UNIT_DISPLAY[unit]} {stat}"
        if tail.endswith("_combat_bonus"):
            terrain = tail[:-13]
            terrain_name = TERRAIN_DISPLAY.get(terrain, terrain.replace("_", " ").title())
            return f"{sign}{val}{suffix} {UNIT_DISPLAY[unit]} {terrain_name} Combat"
    label = DISPLAY_STAT.get(key, key.replace("_", " ").title())
    return f"{sign}{val}{suffix} {label}"

def bonus_line(personal, army):
    bits = []
    for source in (personal, army):
        for key, value in source.items():
            bits.append(format_value(key, value))
    return "#G " + ", ".join(bits) + "#!"

def description(item_id, name, personal, army):
    culture, _number = culture_from_item(item_id)
    culture_name = display_culture(culture)
    noun = noun_for(item_id, personal, army).lower()
    key, _value, unit = primary_army(army)
    article = "An" if culture_name[0].lower() in "aeiou" else "A"
    if unit and "offensive" in key:
        first = f"{article} {culture_name} {noun} made for strengthening {UNIT_DISPLAY[unit].lower()} attacks across the realm."
    elif unit and "defensive" in key:
        first = f"{article} {culture_name} {noun} made for protecting {UNIT_DISPLAY[unit].lower()} formations across the realm."
    elif unit and ("discipline" in key or "morale" in key):
        first = f"{article} {culture_name} {noun} carried to steady the {UNIT_DISPLAY[unit].lower()} in battle."
    elif army:
        first = f"{article} {culture_name} {noun} commissioned as a realm-wide military standard."
    else:
        first = f"{article} {culture_name} {noun} carried for the ruler's personal benefit."
    return first + "\\n" + bonus_line(personal, army)

def quote(text):
    return text.replace('"', '\\"')

mods = {}
for item_id, kind, body in ITEM_RE.findall(MODIFIERS.read_text(encoding="utf-8-sig")):
    mods.setdefault(item_id, {})[kind] = values(body)

current = ITEM_LOC.read_text(encoding="utf-8-sig")
order = [match.group(1) for match in LOC_RE.finditer(current)]

missing = sorted(set(mods) - set(order))
if missing:
    order.extend(missing)

item_lines = ["l_english:"]
mod_lines = ["l_english:"]
for item_id in order:
    if item_id not in mods:
        continue
    personal = mods[item_id].get("personal", {})
    army = mods[item_id].get("army", {})
    name = item_name(item_id, personal, army)
    desc = description(item_id, name, personal, army)
    item_lines.append(f' {item_id}:0 "{quote(name)}"')
    item_lines.append(f' {item_id}_desc:0 "{quote(desc)}"')
    mod_lines.append(f' {item_id}_personal:0 "{quote(name)}"')
    mod_lines.append(f' desc_{item_id}_personal:0 "{quote(desc)}"')
    mod_lines.append(f' {item_id}_army:0 "{quote(name)} #I (Army Standard)#!"')
    mod_lines.append(f' desc_{item_id}_army:0 "{quote(desc)}"')
    mod_lines.append(f' {item_id}_modifier:0 "{quote(name)}"')
    mod_lines.append(f' desc_{item_id}_modifier:0 "{quote(desc)}"')

ITEM_LOC.write_text("\n".join(item_lines) + "\n", encoding="utf-8-sig", newline="\r\n")
MOD_LOC.write_text("\n".join(mod_lines) + "\n", encoding="utf-8-sig", newline="\r\n")
print(f"Regenerated {len(order)} generated item localization entries.")
