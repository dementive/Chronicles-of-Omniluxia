# Magic Plan

Going to override 3 different modifiers that do nothing and repurpose them for magic mechanic

1. great_work_tribals_workrate_character_modifier - All uses of this will be replaced with great_work_total_workrate_character_modifier. This will be the **character** modifier used to measure magic.
2. global_cohort_recruit_speed - Unused in vanilla and does nothing. This will be the **country** modifier used to measure magic.
3. local_cohort_recruit_speed - Unused in vanilla and does nothing. This will be the **province** modifier used to measure magic.

Things to do to get it to work:

1. Override the 3 modifiers with new icons and localization for magic mechanic. Each modifier will represent the yearly magic gain for each scope.

2. Magic itself will be a variable named "magic" stored on countries, characters, and provinces. Each of these will need to have a way to display the current magic value.

3. Implement character focus tree hybrid magic thing

4. Add a new "office" for a "court mage" that will be able to cast various kinds of spells depending on what kind of mage they are (UI particle effects???) (anbennar like spell casting menu???)

5. Make some kind of magic interaction at the province level.
