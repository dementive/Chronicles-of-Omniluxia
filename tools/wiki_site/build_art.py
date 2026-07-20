#!/usr/bin/env python3
"""
Chronicles of Omniluxia - wiki site art pipeline.

Reads the mod's own .dds art (loading screens, main menu background, logo,
coats of arms) and produces web-ready, colour-graded imagery for the static
wiki site.

The loading screens are raw in-game captures: bright, high-detail, obviously
gameplay. To serve as cinematic heroes behind text they are graded down into
atmospheric plates -- softened, desaturated, crushed toward an obsidian-and-gold
duotone, and vignetted. The result reads as mood, not as a screenshot.

The source .dds files live in this repository, so this needs no configuration.
It only needs re-running when the mod's own artwork changes.

Usage:  python3 tools/wiki_site/build_art.py [--mod PATH] [--out PATH]
"""

import argparse
import os
import sys
from PIL import Image, ImageFilter, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__))

# This script sits in <mod repo>/tools/wiki_site/; the art is in <mod repo>/gfx/.
MOD_ROOT = os.environ.get(
    "OMNI_MOD", os.path.abspath(os.path.join(HERE, "..", "..")))
OUT = os.path.join(MOD_ROOT, "docs", "assets", "img")

# --- palette -----------------------------------------------------------------
# Sampled from gfx/interface/frontend/main_menu_background.dds: the mod's own
# glowing sigil rings. Shadow end is a cold near-black, highlight end warm gold.
SHADOW = (10, 9, 14)
MIDTONE = (74, 52, 38)
HIGHLIGHT = (240, 186, 108)

N_LOADSCREENS = 15


def duotone_ramp():
    """256-entry RGB lookup ramp: obsidian -> warm brown -> gold."""
    ramp = []
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            u = t / 0.5
            a, b = SHADOW, MIDTONE
        else:
            u = (t - 0.5) / 0.5
            a, b = MIDTONE, HIGHLIGHT
        ramp.append(tuple(int(a[c] + (b[c] - a[c]) * u) for c in range(3)))
    return ramp


RAMP = duotone_ramp()


def apply_duotone(img, strength=0.55):
    """Blend the image toward the obsidian/gold duotone ramp."""
    lum = img.convert("L")
    flat = lum.point(lambda v: v)
    r = flat.point([RAMP[i][0] for i in range(256)])
    g = flat.point([RAMP[i][1] for i in range(256)])
    b = flat.point([RAMP[i][2] for i in range(256)])
    toned = Image.merge("RGB", (r, g, b))
    return Image.blend(img, toned, strength)


def vignette(img, power=0.85):
    """Radial darkening so text sits legibly over the centre and edges recede."""
    w, h = img.size
    # Build a small radial mask and upscale - far faster than per-pixel work.
    small = 96
    mask = Image.new("L", (small, small), 0)
    px = mask.load()
    cx = cy = (small - 1) / 2.0
    maxd = (cx ** 2 + cy ** 2) ** 0.5
    for y in range(small):
        for x in range(small):
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 / maxd
            v = 1.0 - (d ** 1.7) * power
            px[x, y] = max(0, min(255, int(v * 255)))
    mask = mask.resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(w / 40))
    black = Image.new("RGB", (w, h), (4, 4, 7))
    return Image.composite(img, black, mask)


def grade_hero(src_path):
    """Full grade for a 1920x1080 cinematic hero plate."""
    im = Image.open(src_path).convert("RGB")

    # Soften the low-poly gameplay detail without erasing the landscape shape.
    im = im.filter(ImageFilter.GaussianBlur(2.2))

    # Pull saturation down, then re-tint through the duotone.
    im = ImageEnhance.Color(im).enhance(0.42)
    im = apply_duotone(im, strength=0.58)

    # Crush toward the dark end so overlaid text always wins on contrast.
    im = ImageEnhance.Brightness(im).enhance(0.46)
    im = ImageEnhance.Contrast(im).enhance(1.14)

    im = vignette(im, power=0.9)
    return im


def grade_texture(hero):
    """Heavily blurred plate for section backgrounds and card fills."""
    t = hero.resize((480, 270), Image.LANCZOS)
    t = t.filter(ImageFilter.GaussianBlur(14))
    t = ImageEnhance.Brightness(t).enhance(0.8)
    return t


def save_jpg(im, name, quality=82, widths=(1280, 800)):
    """Write responsive widths; returns list of written filenames.

    Every file is width-suffixed, with no bare-name variant. Only sizes the
    site actually loads are produced: 1280 for the desktop hero plates, 800 for
    narrow viewports and the social preview image. These are heavily blurred,
    vignetted backgrounds, so a 1280 plate stretched across a wider display is
    indistinguishable from a native-resolution one and costs a third as much.
    """
    written = []
    for w in widths:
        h = int(im.height * (w / im.width))
        r = im.resize((w, h), Image.LANCZOS)
        fn = f"{name}-{w}.jpg"
        path = os.path.join(OUT, fn)
        r.save(path, "JPEG", quality=quality, optimize=True, progressive=True)
        written.append((fn, os.path.getsize(path)))
    return written


def main():
    global MOD_ROOT, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", default=MOD_ROOT,
                    help="Path to the Omniluxia mod repository (source of the "
                         ".dds artwork)")
    ap.add_argument("--out", default=OUT, help="Where to write graded images")
    a = ap.parse_args()
    MOD_ROOT, OUT = a.mod, a.out

    if not os.path.isdir(os.path.join(MOD_ROOT, "gfx", "loadingscreens")):
        sys.exit(f"No gfx/loadingscreens/ under: {MOD_ROOT}\n"
                 f"This script needs the mod repo for its source art. "
                 f"Pass --mod PATH or set OMNI_MOD.")

    os.makedirs(OUT, exist_ok=True)
    total = 0

    # --- loading screens -> hero plates --------------------------------------
    ls_dir = os.path.join(MOD_ROOT, "gfx", "loadingscreens")
    for i in range(N_LOADSCREENS):
        src = os.path.join(ls_dir, f"load_{i}.dds")
        if not os.path.exists(src):
            print(f"  ! missing {src}", file=sys.stderr)
            continue
        hero = grade_hero(src)
        for fn, size in save_jpg(hero, f"hero-{i:02d}"):
            total += size
        tex = grade_texture(hero)
        tp = os.path.join(OUT, f"tex-{i:02d}.jpg")
        tex.save(tp, "JPEG", quality=70, optimize=True)
        total += os.path.getsize(tp)
        print(f"  hero-{i:02d}  graded")

    # --- main menu background: already dark and gold, needs only a light touch
    mm = os.path.join(MOD_ROOT, "gfx", "interface", "frontend",
                      "main_menu_background.dds")
    if os.path.exists(mm):
        im = Image.open(mm).convert("RGB")
        im = ImageEnhance.Brightness(im).enhance(0.92)
        im = vignette(im, power=0.45)
        for fn, size in save_jpg(im, "sigils", quality=86):
            total += size
        print("  sigils   graded (main menu background)")

    # --- logo -----------------------------------------------------------------
    lg = os.path.join(MOD_ROOT, "gfx", "interface", "frontend",
                      "game_logo_main_menu.dds")
    if os.path.exists(lg):
        im = Image.open(lg).convert("RGBA")
        im = im.crop(im.getbbox() or (0, 0, im.width, im.height))
        im.save(os.path.join(OUT, "logo.png"), optimize=True)
        print("  logo     trimmed")

    print(f"\nWrote art to {OUT}  ({total / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
