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

def edit_tier1(filepath, base_region, added_region):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'(^\s+owns_or_subject_owns_region\s*=\s*' + re.escape(base_region) + r'\s*$)'
    replacement = r'\1\n\t\t\towns_or_subject_owns_region = ' + added_region
    content = re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE)
    
    # Use line-based approach for highlight
    lines = content.split('\n')
    in_highlight = False
    in_scope = False
    for i, line in enumerate(lines):
        if 'highlight = {' in line:
            in_highlight = True
            continue
        if in_highlight and 'scope:province = {' in line:
            in_scope = True
            continue
        if in_highlight and in_scope:
            # Check if we have OR already
            if 'OR = {' in line:
                # Find last is_in_region before closing
                for j in range(len(lines)-1, i, -1):
                    if 'is_in_region' in lines[j] and j > i:
                        indent = re.match(r'^(\s*)', lines[j]).group(1)
                        lines.insert(j+1, f'{indent}is_in_region = {added_region}')
                        break
                break
            elif 'is_in_region' in line and 'OR' not in content:
                # Single region - wrap in OR
                indent_match = re.match(r'^(\s*)', line)
                indent = indent_match.group(1) if indent_match else '\t\t\t\t'
                parent_indent = indent[:-4] if len(indent) >= 4 else indent
                old_line = lines[i]
                lines[i] = f'{parent_indent}OR = {{'
                lines.insert(i+1, f'{indent}is_in_region = {base_region}')
                lines.insert(i+2, f'{indent}is_in_region = {added_region}')
                lines.insert(i+3, f'{parent_indent}}}')
                break
    
    content = '\n'.join(lines)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  T1: {os.path.basename(filepath)}")

def edit_tier2(filepath, regions):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace all owns lines
    lines = content.split('\n')
    new_lines = []
    owns_done = False
    for line in lines:
        if 'owns_or_subject_owns_region' in line:
            if not owns_done:
                indent = re.match(r'^(\s*)', line).group(1)
                for r in regions:
                    new_lines.append(f'{indent}owns_or_subject_owns_region = {r}')
                owns_done = True
            continue
        new_lines.append(line)
    content = '\n'.join(new_lines)
    
    # Fix highlight OR block - find the OR block and replace its contents
    # Find the highlight section
    hl_idx = content.find('highlight = {')
    allow_idx = content.find('allow = {')
    
    if hl_idx >= 0 and allow_idx > hl_idx:
        hl_section = content[hl_idx:allow_idx]
        
        # Find OR block within highlight
        or_start = hl_section.find('OR = {')
        if or_start >= 0:
            or_end = hl_section.find('}', or_start)
            # Find matching closing brace (the one after all nested content)
            depth = 1
            pos = or_start + 6  # len('OR = {')
            while depth > 0 and pos < len(hl_section):
                if hl_section[pos] == '{':
                    depth += 1
                elif hl_section[pos] == '}':
                    depth -= 1
                pos += 1
            or_block_end = pos - 1  # position of the closing }
            
            # Get indentation from the OR line
            or_line = hl_section[or_start:or_start + hl_section[or_start:].find('\n')]
            or_indent_match = re.match(r'^(\s*)', or_line)
            or_indent = or_indent_match.group(1) if or_indent_match else '\t\t\t\t'
            inner_indent = or_indent + '\t'
            
            # New OR block
            new_or = f'{or_indent}OR = {{'
            for r in regions:
                new_or += f'\n{inner_indent}is_in_region = {r}'
            new_or += f'\n{or_indent}}}'
            
            # Calculate the position in the full content
            absolute_or_start = hl_idx + or_start
            absolute_or_end = hl_idx + or_block_end
            
            content = content[:absolute_or_start] + new_or + content[absolute_or_end+1:]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  T2: {os.path.basename(filepath)} ({len(regions)} regions)")

# Run
print("=== Tier 1 Formables ===")
for name, (base_reg, add_reg) in sorted(TIER1_PAIRS.items()):
    fp = os.path.join(BASE_DIR, "decisions", "tier_1_formables", f"{name}.txt")
    if os.path.exists(fp):
        edit_tier1(fp, base_reg, add_reg)

print("\n=== Tier 2 Formables ===")
for name, regions in sorted(TIER2_GROUPS.items()):
    fp = os.path.join(BASE_DIR, "decisions", "tier_2_formables", f"{name}.txt")
    if os.path.exists(fp):
        edit_tier2(fp, regions)

print("\nDone!")
