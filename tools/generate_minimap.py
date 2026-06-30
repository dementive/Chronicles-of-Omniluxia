from __future__ import annotations

import csv
import re
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
MAP_DATA = ROOT / "map_data"
OUT_PATH = ROOT / "gfx" / "interface" / "minimap" / "minimap.dds"

OUT_SIZE = (384, 192)
WORK_SCALE = 4
WORK_SIZE = (OUT_SIZE[0] * WORK_SCALE, OUT_SIZE[1] * WORK_SCALE)

SEA_COLOR = (28, 44, 49)
LAND_COLOR = (169, 159, 134)
COAST_COLOR = (86, 81, 69)


def rgb_to_565(color: tuple[int, int, int]) -> int:
    r, g, b = color
    return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)


def rgb_from_565(value: int) -> tuple[int, int, int]:
    r = (value >> 11) & 0x1F
    g = (value >> 5) & 0x3F
    b = value & 0x1F
    return ((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2))


def color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def encode_dxt1_block(colors: list[tuple[int, int, int]]) -> bytes:
    darkest = min(colors, key=lambda c: c[0] + c[1] + c[2])
    lightest = max(colors, key=lambda c: c[0] + c[1] + c[2])
    color0 = rgb_to_565(lightest)
    color1 = rgb_to_565(darkest)

    if color0 <= color1:
        color0, color1 = color1, color0

    c0 = rgb_from_565(color0)
    c1 = rgb_from_565(color1)
    palette = [
        c0,
        c1,
        tuple((2 * c0[i] + c1[i]) // 3 for i in range(3)),
        tuple((c0[i] + 2 * c1[i]) // 3 for i in range(3)),
    ]

    indices = 0
    for index, color in enumerate(colors):
        closest = min(range(4), key=lambda palette_index: color_distance(color, palette[palette_index]))
        indices |= closest << (2 * index)

    return struct.pack("<HHI", color0, color1, indices)


def save_dxt1_dds(image: Image.Image, path: Path) -> None:
    image = image.convert("RGB")
    width, height = image.size
    if width % 4 != 0 or height % 4 != 0:
        raise ValueError("DXT1 DDS output dimensions must be divisible by 4.")

    pixels = image.load()
    blocks = bytearray()
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            colors = [pixels[x + bx, y + by] for by in range(4) for bx in range(4)]
            blocks.extend(encode_dxt1_block(colors))

    header = bytearray()
    header.extend(b"DDS ")
    header.extend(
        struct.pack(
            "<7I44x",
            124,
            0x00021007,
            height,
            width,
            len(blocks),
            0,
            0,
        )
    )
    header.extend(struct.pack("<II4s5I", 32, 0x00000004, b"DXT1", 0, 0, 0, 0, 0))
    header.extend(struct.pack("<5I", 0x00001000, 0, 0, 0, 0))

    path.write_bytes(header + blocks)


def parse_ranges(default_map: str, key: str) -> set[int]:
    ids: set[int] = set()

    for match in re.finditer(rf"\b{re.escape(key)}\s*=\s*LIST\s*\{{([^}}]*)\}}", default_map):
        ids.update(int(value) for value in re.findall(r"\d+", match.group(1)))

    for match in re.finditer(rf"\b{re.escape(key)}\s*=\s*RANGE\s*\{{\s*(\d+)\s+(\d+)\s*\}}", default_map):
        start, end = (int(match.group(1)), int(match.group(2)))
        ids.update(range(start, end + 1))

    return ids


def load_land_colors() -> set[tuple[int, int, int]]:
    default_map = (MAP_DATA / "default.map").read_text(encoding="utf-8-sig")
    water_ids = parse_ranges(default_map, "sea_zones") | parse_ranges(default_map, "lakes")

    land_colors: set[tuple[int, int, int]] = set()
    with (MAP_DATA / "definition.csv").open(newline="", encoding="utf-8-sig") as definition:
        for row in csv.reader(definition, delimiter=";"):
            if not row or row[0].startswith("#"):
                continue

            province_id = int(row[0])
            if province_id == 0 or province_id in water_ids:
                continue

            land_colors.add((int(row[1]), int(row[2]), int(row[3])))

    return land_colors


def build_land_mask(land_colors: set[tuple[int, int, int]]) -> Image.Image:
    provinces = Image.open(MAP_DATA / "provinces.png").convert("RGB")
    provinces = provinces.resize(WORK_SIZE, Image.Resampling.NEAREST)

    pixels = provinces.get_flattened_data() if hasattr(provinces, "get_flattened_data") else provinces.getdata()
    mask = Image.new("L", WORK_SIZE, 0)
    mask.frombytes(bytes(255 if pixel in land_colors else 0 for pixel in pixels))

    return mask.resize(OUT_SIZE, Image.Resampling.LANCZOS).filter(ImageFilter.MaxFilter(3))


def make_background() -> Image.Image:
    image = Image.new("RGB", OUT_SIZE, SEA_COLOR)
    draw = ImageDraw.Draw(image, "RGBA")

    for x in range(0, OUT_SIZE[0], 32):
        alpha = 18 if x % 64 == 0 else 9
        draw.line((x, 0, x, OUT_SIZE[1]), fill=(108, 122, 119, alpha), width=1)
    for y in range(0, OUT_SIZE[1], 32):
        alpha = 18 if y % 64 == 0 else 9
        draw.line((0, y, OUT_SIZE[0], y), fill=(108, 122, 119, alpha), width=1)

    draw.rectangle((0, 0, OUT_SIZE[0] - 1, OUT_SIZE[1] - 1), outline=(42, 54, 53, 165), width=2)
    draw.rectangle((4, 4, OUT_SIZE[0] - 5, OUT_SIZE[1] - 5), outline=(18, 28, 31, 110), width=1)
    return image


def compose_minimap(mask: Image.Image) -> Image.Image:
    image = make_background()

    coast = mask.filter(ImageFilter.MaxFilter(5))
    image.paste(COAST_COLOR, mask=coast.point(lambda value: min(180, value)))
    image.paste(LAND_COLOR, mask=mask)

    shade = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
    shade_draw = ImageDraw.Draw(shade, "RGBA")
    for y in range(OUT_SIZE[1]):
        shade_draw.line((0, y, OUT_SIZE[0], y), fill=(0, 0, 0, int(32 * y / OUT_SIZE[1])))
    image = Image.alpha_composite(image.convert("RGBA"), shade)

    return image.convert("RGBA")


def main() -> None:
    mask = build_land_mask(load_land_colors())
    minimap = compose_minimap(mask)
    save_dxt1_dds(minimap, OUT_PATH)
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
