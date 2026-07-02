import math
import re

positions = {}
with open(r'C:\Users\Joshua\Documents\Paradox Interactive\Imperator\mod\Omniluxia\map_data\positions.txt') as f:
    for line in f:
        line = line.strip()
        if not line or '=' not in line: continue
        m = re.match(r'(\d+)\s*=\s*\{\s*(-?\d+)\s+(-?\d+)', line)
        if m:
            positions[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))

area_to_provinces = {}
with open(r'C:\Users\Joshua\Documents\Paradox Interactive\Imperator\mod\Omniluxia\map_data\areas.txt') as f:
    content = f.read()
blocks = re.split(r'\n(?=\w)', content)
for block in blocks:
    lines = block.strip().split('\n')
    m = re.match(r'(\w[\w_]*)\s*=\s*\{', lines[0].strip())
    if not m: continue
    provs = set()
    for ln in lines:
        nums = re.findall(r'\d+', ln.split('#')[0])
        provs.update(int(n) for n in nums)
    provs.discard(0)
    area_to_provinces[m.group(1)] = provs

region_to_areas = {}
with open(r'C:\Users\Joshua\Documents\Paradox Interactive\Imperator\mod\Omniluxia\map_data\regions.txt') as f:
    content = f.read()
rw_blocks = re.split(r'\n(?=new_world_region_\d+)', content)
for block in rw_blocks:
    lines = block.strip().split('\n')
    m = re.match(r'(new_world_region_\d+)\s*=\s*\{', lines[0].strip())
    if not m: continue
    areas = [re.search(r'(new_world_area_\d+)', ln).group(1) for ln in lines if re.search(r'(new_world_area_\d+)', ln)]
    region_to_areas[m.group(1)] = areas

region_provs = {}
for r, areas in region_to_areas.items():
    p = set()
    for a in areas:
        if a in area_to_provinces: p.update(area_to_provinces[a])
    region_provs[r] = p

# Find nearest neighbor for region 030
for r in ['new_world_region_030']:
    rp = [p for p in region_provs[r] if p in positions]
    best = None
    best_dist = 999999
    for r2 in region_provs:
        if r2 == r: continue
        r2p = [p for p in region_provs[r2] if p in positions]
        for p1 in rp:
            x1, y1 = positions[p1]
            for p2 in r2p:
                x2, y2 = positions[p2]
                d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                if d < best_dist:
                    best_dist = d
                    best = r2
    print(f"{r}: closest is {best} at {best_dist:.0f}px")

# Also list positions of 030 provinces
for p in region_provs['new_world_region_030']:
    if p in positions:
        print(f"  Province {p}: {positions[p]}")
