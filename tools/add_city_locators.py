#!/usr/bin/env python3
"""
Add the missing city locator entries that the game logs as
    "Could not find locator transform for city, <id> : gfx/map/map_object_data/city_locators.txt"

Positions are derived from map_data/provinces.png the same way the game's own
generator does: locator position = (pixel_x, 0, imageHeight - pixel_y).

The transform was verified empirically against all 5230 existing city locators
(median ratio 0.999 on both axes). Rather than using the raw centroid - which can
land outside a concave or crescent-shaped province and trip the
"locator is too far outside of its province's bounding box" warning - we snap to
the province pixel nearest the centroid.
"""
import re
import sys

import numpy as np
from PIL import Image

MOD = "/sessions/serene-cool-pascal/mnt/Omniluxia"
LOCATORS = MOD + "/gfx/map/map_object_data/city_locators.txt"

MISSING = [
    658, 955, 1474, 1483, 1493, 1496, 1723, 2502, 2506, 2531, 2536, 2537, 2539,
    2545, 2579, 2585, 2586, 2601, 2606, 2608, 2616, 2618, 2619, 2635, 2645,
    2661, 2722, 2723, 2724, 2730, 2747, 2754, 2765, 2766, 2777, 2783, 2807,
    2823, 2836, 2842, 2857, 2858, 2876, 2897, 2898, 2903, 2911, 2925, 2930,
    2959, 2960, 4598, 4724,
]


def load_colors():
    colors = {}
    with open(MOD + "/map_data/definition.csv", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split(";")
            try:
                colors[int(p[0])] = (int(p[1]), int(p[2]), int(p[3]))
            except (ValueError, IndexError):
                continue
    return colors


def main():
    colors = load_colors()

    src = open(LOCATORS, encoding="utf-8-sig", errors="replace").read()
    existing = set(int(m.group(1)) for m in re.finditer(r"id=(\d+)", src))
    todo = [p for p in MISSING if p not in existing]
    if not todo:
        print("nothing to do - all ids already present")
        return
    print("adding %d locators" % len(todo))

    img = np.asarray(Image.open(MOD + "/map_data/provinces.png").convert("RGB"))
    H, W, _ = img.shape
    key = (img[:, :, 0].astype(np.uint32) << 16) | \
          (img[:, :, 1].astype(np.uint32) << 8) | img[:, :, 2].astype(np.uint32)

    blocks = []
    skipped = []
    for pid in todo:
        c = colors.get(pid)
        if c is None:
            skipped.append((pid, "no colour in definition.csv"))
            continue
        k = (c[0] << 16) | (c[1] << 8) | c[2]
        ys, xs = np.nonzero(key == k)
        if len(xs) == 0:
            skipped.append((pid, "colour not present in provinces.png"))
            continue
        cx, cy = xs.mean(), ys.mean()
        # snap to the province pixel nearest the centroid so the locator is
        # guaranteed to sit inside the province
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        i = int(np.argmin(d2))
        px, py = float(xs[i]), float(ys[i])
        pos_x, pos_z = px + 0.5, (H - py) - 0.5
        blocks.append(
            "\t\t{\n"
            "\t\t\tid=%d\n"
            "\t\t\tposition={ %.6f 0.000000 %.6f }\n"
            "\t\t\trotation={ 0.000000 0.000000 0.000000 1.000000 }\n"
            "\t\t\tscale={ 1.000000 1.000000 1.000000 }\n"
            "\t\t}\n" % (pid, pos_x, pos_z)
        )
        print("  %5d  %d px  ->  x=%.1f z=%.1f" % (pid, len(xs), pos_x, pos_z))

    for pid, why in skipped:
        print("  SKIP %5d  %s" % (pid, why), file=sys.stderr)

    # splice before the closing braces of the instances={ } block.
    # File tail is: <last instance>"\n\t\t}"  "\n\t\t}"(instances)  "\n}"(locator)
    s = src.rstrip()
    for _ in range(2):
        if not s.endswith("}"):
            sys.exit("unexpected file tail: %r" % s[-40:])
        s = s[:-1].rstrip()
    out = s + "\n" + "".join(blocks) + "\t\t}\n}\n"

    with open(LOCATORS, "w", encoding="utf-8") as fh:
        fh.write(out)
    print("wrote %s (%d -> %d entries)" % (
        LOCATORS, len(existing), len(existing) + len(blocks)))


if __name__ == "__main__":
    main()
