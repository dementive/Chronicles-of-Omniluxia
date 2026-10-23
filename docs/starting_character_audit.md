# Starting character, bloodline, and event audit

Audited 22 October 2026 against the working tree, with a campaign start of 1001.1.1.

## Coverage

- 332 unique scripted character IDs in `setup/characters`; no duplicate IDs.
- 65 explicitly appointed starting rulers, all born and alive at the start date.
- 170 parent links: existing targets, appropriate parent sex, plausible minimum parental age, no ancestry cycles, and no births implausibly after a parent's death. Spouse references resolve. This is a structural check, not proof of every lore relationship.
- 96 setup families; every explicit character family reference resolves to its owner and key.
- 535 numeric character references across events, decisions, common scripts, GUI, main setup, and English localization. Of these, 465 references in 27 custom files have reviewed identity and binding-order contracts; 70 inherited vanilla references are separately identified below.
- Existing bloodline audit: 26 country-based starting seeds, 117 race-constrained explicit bloodline assignments, 36 inheritable bloodlines, and 32 race restrictions pass. Event, decision, and mission `blood_of_*` rewards use the validated reward helper.

The pre-renumbering tree at `66373f3d^`, current character setup, country roles, parent sex, and surrounding event text were used to reconstruct intent. Numeric references were corrected per file and role, not by applying one global ID offset. Several lines had already been updated correctly; those bindings were retained.

## Corrections

| Content | Incorrect behavior | Correct binding |
| --- | --- | --- |
| Saptarum restoration and missions | Sea-elf rulers or unrelated families selected as claimant and merchant families | Krstoff Aspatta 79; Sigerdd/Kvaran 80; Ingredd/Otensonn 81; Anggrm/Segurrvelk 105 |
| Saptarum emperor confrontation | Targeted Ap-Tron 225, dead since 946, while text named Jarenam's Sakhir | Starting Nazani emperor Dasrunp 326, matching the original Western Emperor role |
| Saptarum historical text | Named hobgoblins/orcs in place of dwarf ancestors | Galmarr 73, Hthgart 64, Ferja 74 |
| Duggmannrum | Ivarr portrait replaced by ancient Zanis; Sennmoggen recruitment and friendship targeted hobgoblins/orcs; localization named sea elves | Ivarr 160, Hartbaltt 77, Gerrheltt 78; historical Gerodd 76 and Hefdarr 65 |
| Jaraharum | Lich transformation, spouse death, banishment, marriage requirement, and loyalist revolt targeted ancient Zanis | Ivarr 160 |
| Dune marriage and childbirth | Portraits named unrelated people; newborn's parents pointed at Valeiae and Trinia, with the mother/father roles reversed relative to the intended royal couple | Current AGE ruler as father, current DUN ruler as mother; Eudocia 1 and Xacob 47 in starting lore |
| AGE setup | Xacob existed but was never explicitly appointed ruler, leaving the intended royal marriage partner dependent on automatic ruler selection | Explicit `set_as_ruler=char:47` |
| Zechi elven visitors | Recruitment and return effects targeted dead Herulian ancestors | Meriliar 131 and Dandalonia 132 |
| Razani governor/revolt setup | Province appointments and three theme revolts targeted long-dead pedigree characters | Renze 232, Ok-Uunn 233, O-Sarzen 234, Etedunn 235 |
| Razani adoption event | Western rebel adopted into a family through dead Ap-Tron 225 | Dasrunp 326's Nazani family |
| Legion histories and Herue startup | Command histories named unrelated elven ancestors; Herue setup cleared positions on an ancient goblin | Dobpunn 125, O-Bantenn 229, Siaan 127; Eirdorsnor 144 |
| Character-specific death event | Child-triggered scripted deaths targeted two ancient emperors | Original intended roles: Etedunn 235 and Uriel 161 |
| Snakefolk warfare event | Trigger targeted dead dwarf Oten | Ssytu 59 |
| Jarenam civil-war follow-up | War target used bare `JQF`, which Tiger rejected as an unknown token | Explicit country scope `c:JQF` |
| Starting lore and mission text | Reused old IDs for Soventeir, Zakara, Mekian, Duny, Jarenam, Phanician, High Elven, Polarian, Ytalian, and Herulian characters | Matched current setup identities; current Soventeir ruler replaces late Kotemaske in the opening portrait |
| Razani succession story | Said O-Bantenn was an adopted commoner without Zani blood, contradicting his current parents and explicit `zani` trait | Text acknowledges his bloodline and describes opposition based on military reputation; parentage and bloodline retained |
| Blood of the Prophet | Country conversion bonus placed in unsupported province scope | `global_pop_conversion_speed` moved into the existing country block |
| Blood of the Sea Elves | Naval speed bonus placed in unsupported province scope | `navy_movement_speed` moved into the existing country block |

The numeric repairs in the first row include the immediately preceding Saptarum fix. Existing unrelated working-tree changes were retained.

## Repeatable checks

Run from the repository root:

```powershell
python Scripts/audit_starting_characters.py --export
```

This also runs `Scripts/audit_bloodlines.py` and refreshes:

- `docs/starting_character_roster.csv`: setup identities, parents, spouses, families, explicit bloodlines, and ruler appointments.
- `docs/starting_character_references.csv`: each numeric reference, its resolved identity, location, death date, and source line.

`Scripts/starting_character_reference_roles.json` stores reviewed identities and the ordered numeric bindings for each custom file. A newly added, removed, reordered, renumbered, or repurposed binding requires review. Do not regenerate the manifest simply to make a failing audit pass. Dynamic scopes still require contextual review; this audit is not a full engine interpreter.

Five injected regressions were checked in memory: substituting sea elf 87 for Saptarum's claimant, repurposing claimant ID 79 to a different name, a nonexistent father, a dead starting ruler, and a female father. All five were rejected without changing game files.

Tiger validation completed against the installed Imperator 2.0.5. The report changed from 253 errors to 250 after correcting the two bloodline modifier scopes and Jarenam's war-target scope. No remaining Tiger errors name the changed character/event/mission/localization files or bloodline trait file. The remaining repository-wide errors are outside this audit; a successful process exit does not mean the entire mod is clean.

`git diff --check` passes (Git may report line-ending normalization notices).

## Remaining limitations and review notes

1. `localization/english/event_l_english.yml` retains 68 inherited vanilla character references, including ID 385, which has no mod setup definition. They describe Alexander and the Diadochi and have no corresponding Omniluxia identities. No matching startup/Diadochi event definitions were found in this mod's event scripts. They were inventoried rather than assigned invented fantasy equivalents; vanilla event reachability was not proven.
2. `common/on_action/00_specific_from_code.txt` retains the vanilla regent exception involving IDs 279 and 281. In this mod they resolve to Thetima and Sostrate, both dead centuries before the start date; the exception cannot match the starting living consort/ruler pair. It is inherited behavior, not an intended Omniluxia relationship.
3. Two historical setup records named Zanis remain: legacy ID 37 (100–177) and pedigree founder ID 168 (20–97). Active custom founder references now use 168. ID 37 remains defined rather than deleting an existing historical identity or changing chronology without a lore decision. This is a known historical-data discrepancy, not a duplicate numeric ID.
4. Grombar's explicit vampire trait differs from his culture's normal orc race. His starting lore explicitly supports the supernatural identity, so it was retained.
5. English numeric references were reviewed. Other localization languages and arbitrary dynamically selected characters are not covered by the identity manifest. Static validation cannot prove every future succession, event timing, or saved-scope lifetime.
6. No in-game campaign was run. These edits prevent incorrect future script targets; they do not undo completed migrations, deaths, marriages, births, rulership changes, or bloodline awards in existing saves. The separate existing bloodline repair action only removes race-incompatible bloodlines and cannot reconstruct past event outcomes.
