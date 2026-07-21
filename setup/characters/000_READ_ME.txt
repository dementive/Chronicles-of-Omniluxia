##UPDATE THIS IF CHANGED
## First available character id is: 331
##
## HOW CHARACTER IDS WORK (do not break this):
## The game instantiates characters in ascending numeric ID order, globally,
## regardless of which file they are in or where they sit inside that file.
## A character may therefore only reference IDs LOWER than its own.
## This applies to: father, mother, marry_character, set_as_ruler.
## Referencing a higher ID gives "Failed to scope to character via ID 'N'".
##
## Consequence: parents must always have a lower ID than their children, and
## ancestors get the lowest IDs. This mirrors vanilla, which reserves IDs 2-9
## for the dead ancestors in 00_hades.txt / 00_heaven.txt / 00_nirvana.txt.
##
## Reference syntax is char:N  -  e.g. father="char:127" or set_as_ruler=char:113
## A bare number (father=127) will not resolve.
##
## Characters whose death_date is before START_DATE (1001.1.1) still load and
## can be used as father/mother, but cannot be set_as_ruler or married to a
## living character.
##
## DEAD ANCESTORS CARRY NO EFFECTS (this is the #1 source of log spam):
## A character whose death_date is on or before 1001.1.1 is already dead when
## setup effects run, so add_gold, add_popularity, marry_character and
## give_office all fail on them with "... : scope was dead". Vanilla proves the
## convention: across all 76 dead-before-start characters in the base game,
## ZERO carry any of those four. Dead ancestors get only: first_name, family /
## family_name, female, birth_date, death_date, culture, religion, father,
## mother, no_stats / stat lines, no_traits / add_trait, and dna.
## Do not "marry" two dead ancestors to each other - it does nothing and it
## logs an error every single load.
##
## SET_AS_RULER IS ALWAYS A SELF-REFERENCE:
## The ruler block is written INSIDE that character's own block, exactly once
## per country, and must name that same character:
##     13={ ... c:MJR={ set_as_ruler=char:13 } }
## Copying a country file and forgetting to renumber this is how a country ends
## up silently installing ANOTHER country's ruler, or erroring with
## "set_as_ruler effect [ Target Character ... is not alive ]".
##
## BEFORE COMMITTING, RUN:  python tools/validate_characters.py
## It checks every rule in this file and exits non-zero on any violation.
## It is also clean against vanilla, so it encodes vanilla's conventions.
