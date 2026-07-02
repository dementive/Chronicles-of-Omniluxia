import math
import re

# 1. Parse positions.txt to get province -> (x, y)
positions = {}
with open(r'C:\Users\Joshua\Documents\Paradox Interactive\Imperator\mod\Omniluxia\map_data\positions.txt') as f:
    for line in f:
        line = line.strip()
        if not line or '=' not in line:
            continue
        m = re.match(r'(\d+)\s*=\s*\{\s*(-?\d+)\s+(-?\d+)', line)
        if m:
            pid = int(m.group(1))
            x = int(m.group(2))
            y = int(m.group(3))
            positions[pid] = (x, y)

print(f"Loaded {len(positions)} province positions")

# 2. Parse areas.txt to get area_name -> set of provinces
area_to_provinces = {}

with open(r'C:\Users\Joshua\Documents\Paradox Interactive\Imperator\mod\Omniluxia\map_data\areas.txt') as f:
    content = f.read()

# Split by area definitions (pattern: `XXXX = {`)
# We'll parse block by block
blocks = re.split(r'\n(?=\w)', content)
current_area = None
for block in blocks:
    lines = block.strip().split('\n')
    first_line = lines[0].strip()
    m = re.match(r'(\w[\w_]*)\s*=\s*\{', first_line)
    if not m:
        continue
    area_name = m.group(1)
    
    # Extract all province IDs from the block
    provs = set()
    for ln in lines:
        nums = re.findall(r'\d+', ln.split('#')[0])
        provs.update(int(n) for n in nums)
    provs.discard(0)  # Remove province 0 if present
    
    area_to_provinces[area_name] = provs

print(f"Loaded {len(area_to_provinces)} areas")

# 3. Parse regions.txt to map new_world_region_XXX -> set of areas
region_to_areas = {}

with open(r'C:\Users\Joshua\Documents\Paradox Interactive\Imperator\mod\Omniluxia\map_data\regions.txt') as f:
    content = f.read()

# Split by new_world_region definitions
rw_blocks = re.split(r'\n(?=new_world_region_\d+)', content)
for block in rw_blocks:
    lines = block.strip().split('\n')
    first_line = lines[0].strip()
    m = re.match(r'(new_world_region_\d+)\s*=\s*\{', first_line)
    if not m:
        continue
    region_name = m.group(1)
    
    areas = []
    for ln in lines:
        am = re.search(r'(new_world_area_\d+)', ln)
        if am:
            areas.append(am.group(1))
    
    region_to_areas[region_name] = areas

print(f"Found {len(region_to_areas)} new world regions")

# 4. Build region -> set of provinces
region_to_provinces = {}
for region, areas in region_to_areas.items():
    provs = set()
    for area in areas:
        if area in area_to_provinces:
            provs.update(area_to_provinces[area])
    region_to_provinces[region] = provs
    print(f"  {region}: {len(provs)} provinces")

# 5. Determine adjacency
adjacency = {r: set() for r in region_to_provinces}
regions_list = list(region_to_provinces.keys())

THRESHOLD = 100  # pixels

region_centroids = {}
for r, provs in region_to_provinces.items():
    if provs:
        xs = [positions[p][0] for p in provs if p in positions]
        ys = [positions[p][1] for p in provs if p in positions]
        if xs and ys:
            region_centroids[r] = (sum(xs)/len(xs), sum(ys)/len(ys))

print("\n--- Region Adjacency Analysis ---")
for i, r1 in enumerate(regions_list):
    for j, r2 in enumerate(regions_list):
        if i >= j:
            continue
        if r1 not in region_centroids or r2 not in region_centroids:
            continue
        cx1, cy1 = region_centroids[r1]
        cx2, cy2 = region_centroids[r2]
        centroid_dist = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
        
        if centroid_dist > 800:
            continue
        
        provs1 = [p for p in region_to_provinces[r1] if p in positions]
        provs2 = [p for p in region_to_provinces[r2] if p in positions]
        
        found_adjacent = False
        min_dist = 999999
        for p1 in provs1:
            x1, y1 = positions[p1]
            for p2 in provs2:
                x2, y2 = positions[p2]
                d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                if d < min_dist:
                    min_dist = d
                if d < THRESHOLD:
                    found_adjacent = True
                    break
            if found_adjacent:
                break
        
        if found_adjacent:
            adjacency[r1].add(r2)
            adjacency[r2].add(r1)
            print(f"  {r1} <-> {r2}  (min dist: {min_dist:.0f}px, centroid dist: {centroid_dist:.0f}px)")

print("\n--- Final Adjacency Lists ---")
for r in sorted(adjacency.keys()):
    if adjacency[r]:
        print(f"  {r}: {', '.join(sorted(adjacency[r]))}")
    else:
        print(f"  {r}: (isolated)")
