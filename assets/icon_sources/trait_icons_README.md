# Missing trait icons

Generated with the built-in imagegen tool using existing mod trait icons as style references. Temporary reference montages and preview exports are excluded from the repository.

Sources are the five matching PNG files in this directory. Runtime assets are `gfx/interface/icons/traits/<trait identifier>.dds`. Imperator's `TraitItem.GetIcon` resolves these names automatically; no trait statistics or GUI overrides are needed.

All runtime icons are 54 × 54 RGBA DDS, matching the existing uncompressed trait textures. Sources are reduced with Pillow's Lanczos resampler while retaining alpha.

Prompt set (shared style plus the indicated subject):

- Shared: One fantasy Imperator trait icon, circular antique gold-rimmed medallion, painterly sculpted emblem, weathered gold highlights, simple strong silhouette readable at 54 × 54. Centered on a square canvas with fully transparent surroundings. Match the existing reference icons. No text or additional icons.
- `arame_trait`: Proud mystical feline head in profile, warm bronze and ivory fur, subtle turquoise dreaming eye, deep muted teal enamel.
- `blood_of_kadjka`: Antique gold dragon head in profile, curled neck and clear horns, dark burgundy enamel, matching existing bloodline dragon medallions. Represents the royal Arame guardians of Tutkamud.
- `golem_trait`: Carved stone golem head with a softly glowing blue soul rune, dark slate enamel.
- `tsaalan_trait`: Proud bronze dragon-kin head in profile, warm sand enamel.
- `ulvenar_trait`: Fierce silver-gray wolf head in profile, deep midnight blue enamel.

Validation: open each DDS as RGBA, verify 54 × 54 dimensions and transparent corners, and check that every identifier in `racial_traits.txt` and `00_status_bloodlines.txt` has a matching texture. Runtime appearance still needs an in-game check.
