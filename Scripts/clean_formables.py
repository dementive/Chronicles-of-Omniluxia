import re, os

BASE_DIR = r"C:\Users\Joshua\Documents\Paradox Interactive\Imperator\mod\Omniluxia"

TIER1_PAIRS = {
    "nw_feathered_crown": ("new_world_region_028", "new_world_region_038"),
    "nw_sun_throne": ("new_world_region_019", "new_world_region_029"),
    "nw_jade_kingdom": ("new_world_region_010", "new_world_region_012"),
    "nw_pyre_march": ("new_world_region_036", "new_world_region_034"),
    "nw_obsidian_forge": ("new_world_region_020", "new_world_region_013"),
    "nw_emberpeak": ("new_world_region_009", "new_world_region_011"),
    "nw_deepwave": ("new_world_region_027", "new_world_region_035"),
    "nw_coral_guard": ("new_world_region_024", "new_world_region_019"),
    "nw_saltmere": ("new_world_region_008", "new_world_region_006"),
    "nw_bloodmoon": ("new_world_region_026", "new_world_region_031"),
    "nw_shadowmoon": ("new_world_region_014", "new_world_region_010"),
    "nw_frosthowl": ("new_world_region_007", "new_world_region_003"),
    "nw_thunderhold": ("new_world_region_025", "new_world_region_024"),
    "nw_iron_gate": ("new_world_region_016", "new_world_region_013"),
    "nw_deepstone": ("new_world_region_006", "new_world_region_008"),
    "nw_ash_march": ("new_world_region_034", "new_world_region_032"),
    "nw_sand_crown": ("new_world_region_012", "new_world_region_006"),
    "nw_dunescar": ("new_world_region_005", "new_world_region_002"),
    "nw_duskcap": ("new_world_region_029", "new_world_region_019"),
    "nw_gloomhold": ("new_world_region_013", "new_world_region_007"),
    "nw_sporewood": ("new_world_region_004", "new_world_region_003"),
    "nw_stormpeak": ("new_world_region_023", "new_world_region_020"),
    "nw_windreach": ("new_world_region_015", "new_world_region_014"),
    "nw_cloudwatch": ("new_world_region_003", "new_world_region_004"),
    "nw_silk_court": ("new_world_region_030", "new_world_region_025"),
    "nw_stinger_domain": ("new_world_region_021", "new_world_region_023"),
    "nw_amberhive": ("new_world_region_002", "new_world_region_001"),
    "nw_mistscale": ("new_world_region_022", "new_world_region_020"),
    "nw_scale_march": ("new_world_region_011", "new_world_region_009"),
    "nw_mosshide_dominion": ("new_world_region_001", "new_world_region_002"),
}

TIER2_GROUPS = {
    "nw_great_serpent": ["new_world_region_001", "new_world_region_002", "new_world_region_011", "new_world_region_022"],
    "nw_greathive": ["new_world_region_002", "new_world_region_005", "new_world_region_011", "new_world_region_022", "new_world_region_021"],
    "nw_azure_kingdom": ["new_world_region_003", "new_world_region_006", "new_world_region_007", "new_world_region_010", "new_world_region_015"],
    "nw_fungal_empire": ["new_world_region_004", "new_world_region_007", "new_world_region_008", "new_world_region_013"],
    "nw_dust_empire": ["new_world_region_007", "new_world_region_010", "new_world_region_012", "new_world_region_015"],
    "nw_mountain_realm": ["new_world_region_009", "new_world_region_013", "new_world_region_016", "new_world_region_020", "new_world_region_025"],
    "nw_howling_kingdom": ["new_world_region_010", "new_world_region_012", "new_world_region_014", "new_world_region_017"],
    "nw_tide_empire": ["new_world_region_018", "new_world_region_022", "new_world_region_026", "new_world_region_027"],
    "nw_magma_kingdom": ["new_world_region_009", "new_world_region_011", "new_world_region_016", "new_world_region_020"],
    "nw_obsidian_throne": ["new_world_region_019", "new_world_region_028", "new_world_region_029", "new_world_region_031"],
    "nw_wind_empire": ["new_world_region_023", "new_world_region_029", "new_world_region_031", "new_world_region_033"],
    "nw_dusk_empire": ["new_world_region_017", "new_world_region_024", "new_world_region_027", "new_world_region_029"],
    "nw_moonshadow_empire": ["new_world_region_022", "new_world_region_031", "new_world_region_037", "new_world_region_043"],
    "nw_coral_empire": ["new_world_region_035", "new_world_region_039", "new_world_region_041", "new_world_region_044"],
    "nw_jade_empire": ["new_world_region_035", "new_world_region_038", "new_world_region_042", "new_world_region_044"],
}

def fix_tier1(filepath, base_reg, added_reg):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    # 1. Remove duplicate owns lines - keep only the first 2
    result = []
    owns_seen = 0
    for line in lines:
        if 'owns_or_subject_owns_region' in line:
            owns_seen += 1
            if owns_seen > 2:
                continue
        result.append(line)
    content = '\n'.join(result)
    
    # 2. Fix highlight OR block - remove duplicates, fix indentation
    lines = content.split('\n')
    in_highlight = False
    in_or = False
    regions_found = []
    result = []
    for line in lines:
        if 'highlight = {' in line:
            in_highlight = True
            result.append(line)
            continue
        if in_highlight:
            if 'OR = {' in line:
                in_or = True
                or_indent = re.match(r'^(\s*)', line).group(1)
                result.append(f'{or_indent}OR = {{')
                continue
            if in_or:
                m = re.search(r'is_in_region\s*=\s*(new_world_region_\d+)', line)
                if m:
                    r = m.group(1)
                    if r not in regions_found:
                        regions_found.append(r)
                    continue
                if '}' in line:
                    # End of OR block
                    inner_indent = or_indent + '\t\t'
                    for r in regions_found:
                        result.append(f'{inner_indent}is_in_region = {r}')
                    result.append(f'{or_indent}}}')
                    in_or = False
                    continue
            if 'allow = {' in line:
                in_highlight = False
                result.append(line)
                continue
            if not in_or:
                result.append(line)
                continue
        else:
            result.append(line)
    
    content = '\n'.join(result)
    
    # 3. Check if scope:province closing is correct
    # The issue sometimes is } of OR and } of scope:province
    # Find "OR = {" section and make sure it's properly closed
    lines = content.split('\n')
    result = []
    skip_next_close = False
    for i, line in enumerate(lines):
        if skip_next_close:
            skip_next_close = False
            continue
        if 'OR = {' in line and 'highlight' not in content[:content.find(line)+len(line)]:
            # Check if next non-empty line after OR closing has another }
            pass
        result.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Verify
    with open(filepath, 'r', encoding='utf-8') as f:
        verify = f.read()
    owns_count = verify.count('owns_or_subject_owns_region')
    is_in_count = len(re.findall(r'is_in_region\s*=\s*new_world_region_\d+', verify))
    print(f"  T1 {os.path.basename(filepath)}: {owns_count} owns, {is_in_count} highlight regions")

def fix_tier2(filepath, regions):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # 1. Replace all owns lines with correct set
    result = []
    owns_done = False
    for line in lines:
        if 'owns_or_subject_owns_region' in line:
            if not owns_done:
                indent = re.match(r'^(\s*)', line).group(1)
                for r in regions:
                    result.append(f'{indent}owns_or_subject_owns_region = {r}')
                owns_done = True
            continue
        result.append(line)
    content = '\n'.join(result)
    
    # 2. Fix highlight OR block
    lines = content.split('\n')
    in_highlight = False
    in_or = False
    or_indent = ''
    result = []
    regions_in_or = []
    
    for line in lines:
        if 'highlight = {' in line:
            in_highlight = True
            result.append(line)
            continue
        if in_highlight:
            if 'OR = {' in line:
                in_or = True
                or_indent = re.match(r'^(\s*)', line).group(1)
                result.append(f'{or_indent}OR = {{')
                continue
            if in_or:
                # Check for closing brace of OR
                if '}' in line and 'OR = {' not in line:
                    # Check context - does this close OR or something else?
                    stripped = line.strip()
                    if stripped == '}':
                        # End of OR block
                        inner_indent = or_indent + '\t'
                        for r in regions:
                            result.append(f'{inner_indent}is_in_region = {r}')
                        result.append(f'{or_indent}}}')
                        in_or = False
                        continue
                    else:
                        # Line has more than just }, like "}\n}" - skip it
                        continue
                # Skip is_in_region lines (we'll regenerate them)
                if 'is_in_region' in line:
                    continue
            
            if not in_or:
                result.append(line)
                continue
        else:
            result.append(line)
    
    content = '\n'.join(result)
    
    # 3. Remove duplicate empty closing braces from the highlight section
    content = re.sub(r'\n(\s+)\}\n\s*\}\n(\s+)\}', r'\n\1}\n\2}', content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Verify
    with open(filepath, 'r', encoding='utf-8') as f:
        verify = f.read()
    owns_count = verify.count('owns_or_subject_owns_region')
    is_in_count = len(re.findall(r'is_in_region\s*=\s*new_world_region_\d+', verify))
    print(f"  T2 {os.path.basename(filepath)}: {owns_count} owns, {is_in_count} highlight regions")

print("=== Fixing Tier 1 ===")
for name, (base_reg, add_reg) in sorted(TIER1_PAIRS.items()):
    fp = os.path.join(BASE_DIR, "decisions", "tier_1_formables", f"{name}.txt")
    if os.path.exists(fp):
        fix_tier1(fp, base_reg, add_reg)

print("\n=== Fixing Tier 2 ===")
for name, regions in sorted(TIER2_GROUPS.items()):
    fp = os.path.join(BASE_DIR, "decisions", "tier_2_formables", f"{name}.txt")
    if os.path.exists(fp):
        fix_tier2(fp, regions)

print("\nDone!")
