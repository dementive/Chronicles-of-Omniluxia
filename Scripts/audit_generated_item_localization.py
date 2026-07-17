from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ITEM_LOC = ROOT / "localization" / "english" / "omniluxia_random_items_l_english.yml"
MOD_LOC = ROOT / "localization" / "english" / "omniluxia_generated_item_modifiers_l_english.yml"
MODIFIERS = ROOT / "common" / "modifiers" / "omniluxia_generated_items_split.txt"
ITEM_MENU = ROOT / "events" / "omni_item_menu.txt"

item_text = ITEM_LOC.read_text(encoding="utf-8-sig")
mod_text = MOD_LOC.read_text(encoding="utf-8-sig")
modifier_text = MODIFIERS.read_text(encoding="utf-8-sig")
menu_text = ITEM_MENU.read_text(encoding="utf-8-sig")

names = dict(re.findall(r'^ (item_gen_[A-Za-z0-9_]+_\d+):0 "([^"]+)"', item_text, re.M))
descs = dict(re.findall(r'^ (item_gen_[A-Za-z0-9_]+_\d+)_desc:0 "([^"]+)"', item_text, re.M))
modifier_ids = set(re.findall(r"^(item_gen_[A-Za-z0-9_]+_\d+)_(?:personal|army)\s*=", modifier_text, re.M))

modifier_names = {}
for key, suffix, value in re.findall(r'^ (item_gen_[A-Za-z0-9_]+_\d+)_(personal|army|modifier):0 "([^"]+)"', mod_text, re.M):
    modifier_names.setdefault(key, {})[suffix] = value

nouns = [
    "bow", "recurve", "blade", "warblade", "spear", "lance", "shield", "pavise",
    "hauberk", "mail", "barding", "banner", "codex", "signet", "amulet", "ward",
    "ram", "standard", "guard", "armor",
]

errors = []
if set(names) != modifier_ids:
    errors.append(f"name/id mismatch: names={len(names)} modifier_ids={len(modifier_ids)}")
if set(descs) != modifier_ids:
    errors.append(f"desc/id mismatch: descs={len(descs)} modifier_ids={len(modifier_ids)}")

for item_id, name in names.items():
    desc = descs.get(item_id, "").lower()
    lowered_name = name.lower()
    name_nouns = [noun for noun in nouns if noun in lowered_name]
    desc_nouns = [noun for noun in nouns if noun in desc]
    if name_nouns and desc_nouns and not any(noun in desc_nouns for noun in name_nouns):
        errors.append(f"noun mismatch: {item_id}: {name} :: {descs.get(item_id, '')}")
    mod_entries = modifier_names.get(item_id, {})
    if mod_entries.get("personal") != name:
        errors.append(f"personal modifier name mismatch: {item_id}")
    expected_army = f"{name} #I (Army Standard)#!"
    if "army" in mod_entries and mod_entries["army"] != expected_army:
        errors.append(f"army modifier name mismatch: {item_id}")
    if mod_entries.get("modifier") != name:
        errors.append(f"legacy modifier name mismatch: {item_id}")

for option in re.findall(r"option\s*=\s*\{.*?\n\t\}", menu_text, re.S):
    name = re.search(r"\bname\s*=\s*(item_[A-Za-z0-9_]+)", option)
    tooltip = re.search(r"\bcustom_tooltip\s*=\s*(item_[A-Za-z0-9_]+)_desc", option)
    item_id = re.search(r"\bitem_ID\s*=\s*(item_[A-Za-z0-9_]+)", option)
    if not (name and tooltip and item_id):
        continue
    name_id = name.group(1)
    tooltip_id = tooltip.group(1)
    actual_id = item_id.group(1)
    if name_id != tooltip_id or actual_id != tooltip_id:
        errors.append(f"menu option mismatch: name={name_id} tooltip={tooltip_id}_desc item_ID={actual_id}")

print(f"generated_item_names={len(names)}")
print(f"generated_item_descs={len(descs)}")
print(f"generated_modifier_ids={len(modifier_ids)}")
print(f"errors={len(errors)}")
for error in errors[:25]:
    print(error)
raise SystemExit(1 if errors else 0)
