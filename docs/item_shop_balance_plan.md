# Item Shop Balance Plan

This plan is for a numbers-only balance pass. It does not add new mechanics, new loops, new recurring checks, new UI flows, or any feature that should affect performance. The goal is to preserve a meaningful midgame and late-game economy for 200+ year campaigns by making better shops and better items stay valuable for longer.

## Goals

1. Make early shopping useful without letting players finish their equipment progression too quickly.
2. Make midgame cities feel meaningfully better than early cities.
3. Make fully developed cities produce better shop inventories, as intended.
4. Keep rare and unique items exciting in the late game rather than common in ordinary developed cities.
5. Make high-tier generated and fixed items remain meaningful gold sinks.
6. Avoid all performance-sensitive systems.

## Non-Goals

1. Do not add new mechanics.
2. Do not add new events, pulses, province scans, character scans, or recurring background logic.
3. Do not change item effects or create new item tiers.
4. Do not redesign the shop UI.
5. Do not manually edit generated files unless there is no safer source file to change.

## Current Item Quality Structure

The shop appears to have more than four practical purchase tiers, even though generated item rarity itself is simpler.

Generated item rarities:

1. Common
2. Uncommon
3. Rare

Fixed items also include unique or named items, which function as a fourth practical quality class in shops.

Generated item price tiers:

1. `omni_price_tier_2`
2. `omni_price_tier_3`
3. `omni_price_tier_4`
4. `omni_price_tier_5`
5. `omni_price_tier_6`
6. `omni_price_tier_7`

So the practical answer is: there are three generated rarity tiers, a fixed unique/named class above them, and six generated price tiers.

## Main Risk

The current system likely lets strong cities reach high shop quality too early. Players can probably fill item slots before the late game, then only occasionally replace items. That weakens the incentive to keep building and shopping across a 200+ year campaign.

The fix should be to stretch the numbers:

1. Raise shop quality thresholds.
2. Reduce rare and unique frequency in non-maxed cities.
3. Make the best items more expensive.
4. Keep maxed cities rewarding, but make the best results less automatic.

## Target File 1: Shop Quality and Rarity Weights

File:

`common/scripted_effects/choo_items_effects.txt`

Primary effects to inspect:

1. `omni_build_item_shop_stock`
2. `omni_roll_market_shop_item`
3. `omni_build_ai_country_item_shop_stock`

### Step 1: Audit Current Quality Sources

Review every source that adds `omni_market_quality`, especially:

1. Commerce buildings
2. Ports
3. Workshop
4. Foundry
5. Library
6. Academy
7. Used building slots
8. Population
9. Civilization value
10. Capital bonus
11. Holy site bonus
12. Trade good bonuses
13. Merchant trait bonuses

Current issue to watch for:

Developed capitals with good buildings and trade goods can likely reach the top quality band before the true late game.

### Step 2: Raise Market Quality Bands

Current approximate bands:

```text
<45      low shop
45-74    medium shop
75-104   high shop
105+     top shop
```

Recommended first-pass bands:

```text
<55      low shop
55-89    medium shop
90-129   high shop
130+     top shop
```

Stricter alternative:

```text
<60      low shop
60-94    medium shop
95-134   high shop
135+     top shop
```

Expected result:

Ordinary cities still improve, but only truly excellent cities reliably reach the top band.

### Step 3: Reduce Unique Frequency

Current top market roll appears generous:

```text
10 common / 35 uncommon / 40 rare / 15 unique
```

Recommended first-pass top band:

```text
15 common / 42 uncommon / 35 rare / 8 unique
```

Stricter alternative:

```text
20 common / 45 uncommon / 30 rare / 5 unique
```

Recommended high band adjustment:

Current high band includes a small unique chance. Consider lowering it or removing it.

```text
20 common / 50 uncommon / 25 rare / 5 unique
```

Possible replacement:

```text
25 common / 55 uncommon / 20 rare / 0 unique
```

Expected result:

Rare items remain visible in good cities, but uniques mostly belong to exceptional cities.

### Step 4: Raise Extra Roll Thresholds

Current extra roll thresholds:

```text
>=55  extra market roll
>=90  extra rare roll
>=120 extra unique roll
```

Recommended first-pass thresholds:

```text
>=75  extra market roll
>=115 extra rare roll
>=160 extra unique roll
```

Stricter alternative:

```text
>=80  extra market roll
>=125 extra rare roll
>=175 extra unique roll
```

Expected result:

Good cities get more inventory, but extra rare and unique rolls become late-game rewards.

### Step 5: Consider Reducing Some Quality Bonuses

Only adjust these if threshold changes are not enough.

Possible conservative changes:

```text
Commerce building bonus: +10 each to +8 each
Trade good bonus: +15/+20 to +10/+15
Capital bonus: keep +5
Holy site bonus: keep +10
Workshop/foundry/library/academy: keep initially
```

Expected result:

The biggest city-quality spikes become less explosive without making buildings feel irrelevant.

### Step 6: Keep Stock Size Values Stable

Current shop max sizes appear to use:

```text
4 / 6 / 8 / 10
```

Recommendation:

Keep those exact values unless absolutely necessary.

Reason:

Other helper logic appears to expect these values. Changing the values themselves may require more file edits. Raising the thresholds for 6, 8, and 10 stock is safer than inventing new stock counts like 5, 7, or 9.

### Step 7: Mirror Player/AI Shop Tuning

If `omni_build_ai_country_item_shop_stock` uses parallel logic, mirror the same threshold and rarity tuning there.

Expected result:

The player and AI use comparable shop economy rules.

## Target File 2: Generated Item Prices

File:

`common/scripted_effects/omni_army_loadout.txt`

Current generated price curve:

```text
130 / 200 / 300 / 440 / 640 / 920
```

Recommended first-pass curve:

```text
130 / 220 / 360 / 560 / 850 / 1250
```

Stricter alternative:

```text
150 / 240 / 380 / 600 / 900 / 1350
```

### Step 8: Preserve Early Accessibility

Do not overprice the first two tiers.

Reason:

Early shops should still feel useful. The balance problem is mainly the speed of finishing the best gear, not the existence of cheap starter gear.

### Step 9: Steepen Late Prices

Focus most of the price increase on tiers 5, 6, and 7.

Expected result:

Players can fill slots, but replacing good items with excellent items remains a major economic decision.

### Step 10: Update Sell Refunds

If buy prices change, update sell/refund values consistently.

Recommended rule:

Keep sell value at roughly half of purchase price unless the current system intentionally uses a different ratio.

## Target File 3: Shop Menu Price Checks and Fixed Item Prices

File:

`events/omni_item_menu.txt`

### Step 11: Sync Generated Item Price Checks

Generated shop purchase options have hardcoded affordability checks. If generated prices change in `omni_army_loadout.txt`, update the matching checks here.

Expected result:

The UI and purchase effect agree about what the player can afford.

### Step 12: Audit Fixed Item Prices

Review every fixed item purchase option.

Suggested target ranges:

```text
Basic fixed gear: keep mostly accessible
Good fixed gear: moderate increase if needed
Minor named items: 900-1200
Strong uniques: 1200-1800
Top artifacts: 1800-2500+
```

Expected result:

Fixed unique items become long-term gold sinks instead of midgame impulse purchases.

### Step 13: Avoid Broad Fixed-Price Sweeps

Do not blindly multiply every fixed price.

Reason:

Some fixed items may be flavor items, weaker items, or intentionally early options. Price changes should reflect power and rarity.

## Files to Avoid Unless Needed

### Generated Price Mapping

File:

`common/scripted_effects/omni_gen_item_prices.txt`

Avoid manual edits unless changing which generated items belong to which price tier.

Reason:

This file appears generated.

### Generator Script

File:

`Scripts/regenerate_generated_items.py`

Only touch this if changing the generation source of price tiers or helper output.

Reason:

The current shop path appears to use tiered army purchase logic, while parts of this generator may still contain older rarity-price assumptions.

### Generated Slot Helpers

File:

`common/scripted_effects/omni_gen_slot_helpers.txt`

Only touch this if changing shop max sizes away from `4 / 6 / 8 / 10`.

Reason:

Keeping those stock sizes avoids unnecessary edits and reduces risk.

## Recommended Pass Order

1. Tune shop quality thresholds in `choo_items_effects.txt`.
2. Tune market rarity weights in `choo_items_effects.txt`.
3. Tune extra roll thresholds in `choo_items_effects.txt`.
4. Mirror equivalent AI shop logic if present.
5. Raise generated item prices in `omni_army_loadout.txt`.
6. Sync generated item price checks in `events/omni_item_menu.txt`.
7. Audit fixed item prices in `events/omni_item_menu.txt`.
8. Test by comparing several city archetypes.

## City Archetypes to Test

Use these as balance checkpoints.

### Early City

Expected outcome:

Mostly common items, occasional uncommon items, no regular rare or unique access.

### Normal Developed City

Expected outcome:

Common and uncommon items, rare items possible but not frequent.

### Strong Capital

Expected outcome:

Reliable uncommon items, meaningful rare access, uniques possible only rarely.

### Maxed Capital or Sacred Trade Hub

Expected outcome:

Best shop class should appear here. Rare items should be common enough to notice. Unique items should be possible, but still exciting.

## Success Criteria

The balance pass succeeds if:

1. Players can still buy useful items early.
2. Players cannot reliably complete high-quality loadouts too early.
3. Maxed cities clearly have better shops than ordinary cities.
4. Unique items remain rare enough to feel special.
5. High-tier gear creates a real late-game money sink.
6. The changes are limited to constants, thresholds, weights, and prices.
7. No new recurring logic or performance-sensitive mechanics are added.

## Proposed First-Pass Numbers

Use this set if we want a balanced but not punitive first pass:

```text
Market bands:
<55
55-89
90-129
130+

Extra roll thresholds:
>=75  extra market roll
>=115 extra rare roll
>=160 extra unique roll

Top market roll:
15 common / 42 uncommon / 35 rare / 8 unique

Generated item prices:
130 / 220 / 360 / 560 / 850 / 1250
```

Use this set if we want a harsher late-game stretch:

```text
Market bands:
<60
60-94
95-134
135+

Extra roll thresholds:
>=80  extra market roll
>=125 extra rare roll
>=175 extra unique roll

Top market roll:
20 common / 45 uncommon / 30 rare / 5 unique

Generated item prices:
150 / 240 / 380 / 600 / 900 / 1350
```

## Final Recommendation

Start with the balanced first-pass numbers, then test several city archetypes. If maxed cities still produce too many unique items, lower the top unique weight before raising thresholds again. If early shopping feels bad, lower only the first two generated item prices or leave early shop bands untouched.

The most important principle is to stretch the top end, not punish the bottom end.
