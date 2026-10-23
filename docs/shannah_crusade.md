# Shannah succession and crusades

Implemented in the local Omniluxia scripts. Static checks pass; the runtime
acceptance cases below still require Imperator. No Invictus files were changed.

## Gameplay

- Every incoming ruler of a landed Helluvian Shahanat receives `shannah_trait`.
  This follows the game's actual successor, rather than giving the trait to
  children before they inherit. Living former rulers lose the trait on ruler
  change. Monthly repair covers existing saves. A non-Helluvian current ruler
  loses the trait. The dead ruler's historical trait is retained.
- Claim the Shahanat requires a Helluvian country owning Ddu-Nuinop (2706)
  and an existing ruler. It transfers the office to that ruler: other holders
  lose their current ruler's trait, their Shahanat government becomes the
  existing feudal_monarchy, and a deposed flag prevents succession repair from
  recreating their office. No ownership-change hook strips an incumbent merely
  for losing the city. The reigning Shannah cannot repeatedly take the decision.
- Former holders, including IF3, can reclaim the office after retaking the
  city. `new_pope_var` now records prior tenure/rewards rather than blocking
  reclamation: the 50 legitimacy and 100 influence grant is first-time only.
  Successful reclamation removes the deposed flag and restores succession.
- Taking the office scans other countries once; there is no recurring global
  claimant scan. Existing duplicate offices are revoked when a new claim is
  taken. This does not automatically consolidate duplicates on loading a save.
- A displaced Shannah's unlaunched crusade is cancelled and refunded through
  the existing cancellation routine. Active crusade administration is closed
  without forcing peace, removing armies, or removing countries from the war.
  A declaration already registered during the one-day launch transition is
  treated as an ongoing war rather than refunded as an unlaunched expedition.
- Call For A Crusade requires the Shannah trait, with no territorial-rank
  requirement. A one-territory Shahanat may call a crusade. The host must be
  independent, at peace, and free of civil war. Cooldown and the worldwide
  expedition lock are availability conditions, not visibility conditions;
  unmet requirements are shown on the disabled decision.
- A free council offers up to three distinct eligible non-Helluvian
  nearby countries within diplomatic range by name, following the Invictus war council's saved-country event
  choices. The player chooses one and then confirms or dismisses the proposal.
  Where more than three are eligible, the council samples three; it is not a
  full country dropdown. Cancel and reopen to hear a new set of proposals.
  Allied countries, subjects, truce
  partners, countries outside diplomatic range, and civil-war countries are
  excluded. With only one or two eligible countries, only those choices appear.
- Confirmation costs 250 gold and 50 political influence. Preparation takes
  180 days. The 3,650-day country cooldown begins at confirmation and survives
  succession and cancellation. A global flag permits one expedition at a time.
- Invitations go to independent Helluvian countries in diplomatic range of
  both the host and target. Civil wars, conflicting wars, target alliances,
  and target truces exclude a country. This is a diplomatic-range restriction,
  not a guarantee that the AI can transport an army to the target.
- Invitees can pledge, donate a size-scaled offering, or decline. Military pledges require
  peace; donations can be made while fighting an unrelated war. AI choice
  weights consider opinion, an alliance with the host, treasury, and war.
  Military acceptance now also uses a strength snapshot taken at confirmation:
  score = territories + 2 * current cohorts, minimum 1. This is a credibility
  proxy, not an estimate of unraised levy capacity or battlefield quality.
  The free expedition army and donations cannot inflate the snapshot.
  Scores below 25 multiply military weight by 0.1; 25-74 by 0.35; 75-149
  by 1; 150-299 by 1.5; 300+ by 2.5. An invitee stronger than the host
  multiplies it by 0.35; a target stronger than the host by 0.5. These modify
  the original weight of 40, alongside relations and treasury. Donations
  retain weight 25 and refusal 35 before their own modifiers. These are
  relative AI weights, not guaranteed percentages or a coalition cap.
- The confirmed target receives an early warning (.20) and may send one wave
  of defensive appeals. Eligible recipients are independent, in range of
  both principals, without civil war, hostile war, alliance with the host,
  or a truce with the host. Existing target allies qualify; other recipients
  must border the target and have opinion >=25, the same religion, or shared
  race affinity. Offensive pledgers and donors cannot switch commitments.
  Unanswered offensive invitations are withdrawn when a defensive appeal is sent.
- Defense has two choices: pledge or decline. Pledging requires peace.
  AI weights are 20 versus 40, with pledge multipliers: target ally x4,
  opinion >=50 x2, opinion <=-25 x0.25, shared faith x2, shared race x1.5,
  opinion of Shannah <=-50 x1.5, treasury below 100 x0.5. Race follows primary
  culture and the existing choo_effects racial-assimilation groupings, with
  sunset elves explicitly excluded from the legacy default-human bucket.
- The target's Review the Defense against the Crusade decision (.24) can
  send the appeal later during preparation if it was initially declined,
  or inspect defensive replies. Only one appeal wave is allowed. Pledges
  create no permanent alliance. After the war is confirmed, valid peaceful
  defenders are added with attacker=no, leader=no, before launch news.
  Existing defensive war members are recognized without duplicate entry.
  Commitments and view state are cleared with the expedition or orphan repair.
- Peaceful, otherwise eligible refusal costs 10 opinion with the Shahanat;
  countries already at war or no longer eligible are excused. Donating earns
  15 opinion. Verified military entry earns 30 opinion.
- At the end of preparation the Shannah can launch, postpone by 30 days, or cancel. Launching
  revalidates the host and target, then declares an ordinary conquest war for
  the target's capital province. The dedicated war goal is available only
  during scripted declaration and only against the recorded target.
- Monthly preparation checks only cancel for lost religious leadership or
  independence, a missing/landless target, a converted target, or a target
  becoming a subject. Temporary wars, civil wars, range changes, alliances,
  and truces do not discard preparations. Full launch eligibility is still
  required before departure; postponement preserves pledges and funding.
- After declaration, a separate event on the next in-game day identifies the war by its
  attacking host, defending target, and host leadership. There is no fallback
  to another current war. Pledgers are revalidated before being added.
- The declaration authorization stays set until expedition cleanup. Monthly
  preparation validation and voluntary cancellation skip the registration
  transition, so the new war is not itself treated as a reason to cancel.
- A pledger already on the attacking side through ordinary alliance behavior
  is counted too. Only verified attacking participants receive the five-year
  Answered the Shannah's Call modifier: +5% omen power and +0.02 monthly
  legitimacy. A pledge alone does not earn it.
- The host receives the original expedition army only after a real war is
  identified: three heavy infantry, two light infantry, two heavy cavalry,
  two supply trains, 100 food, and +2 morale. There is no maintenance exemption
  or automatic dismissal.
- The new war uses negotiated conquest, not civil-war-style instant conquest.
  War leadership, peace deals, occupation, and AI movement use the game's
  ordinary systems. There is no scripted land distribution.
- Once the war is confirmed and pledgers have been recruited, one global
  announcement wave sends a single launch event to each landed country:
  the host receives The Swords of Hellas (.11), the main target receives A
  Crusade Against Us! (.18), other actual participants on either side receive
  The Crusade Begins! (.17), and outsiders receive A Holy War Abroad (.19).
  This uses actual war membership, not religion or promised support. Donors
  outside the war receive outside news. An unfulfilled pledger's news includes
  a note explaining its absence. The previous pledge-entry/release popups are
  no longer dispatched, avoiding duplicate launch messages. News has no gameplay
  effects, runs once per expedition, and adds no recurring global scan.
- Review the Crusade reports pledge, donor, and verified participant counts.
  Click its "Read the roll of those summoned" response for categories:
  military pledges, offerings, declines, and pending replies. Defensive replies
  and verified war entries have separate category menus. Each category lists
  fifteen country names per page beneath one shared heading, without repeating
  a response paragraph or blank paragraph between names. Empty categories are
  hidden. Launch results distinguish actual entries from unfulfilled pledges
  on both sides. A pledge remains a promise, even after departure.
  Next/previous buttons browse one category; category buttons switch rolls.
  Page construction uses one filtered scan when opened, with no monthly scan.
  Entry records describe historical entry, not continued war participation.
  During preparation it also offers cancellation at a cost of 3 stability.
  Cancelling from the deadline event has the same stability cost.
- Cancelled preparation returns the host's entry cost and surviving donors'
  recorded payments. Donor repayments are deducted from the host. Automatic
  invalidation has no stability penalty. The cooldown is retained. If the
  host is annexed, orphaned records are cleared without creating refunds from
  a nonexistent treasury.
- War-ending callbacks schedule a one-day delayed check; monthly checks are
  a fallback. The crusade's administrative state closes only if no matching
  host-led war remains. Losing war leadership also ends that mandate, without
  forcing participants out of an ongoing war or imposing a peace settlement.
- Old pending `me_zorgoomni.55`/`.56` expeditions are retired and refunded via
  a migration event; they cannot run the former random-target declaration.

## Code precedents

These are local Imperator examples, not APIs borrowed from other games.

| Component | Reference file and entry |
| --- | --- |
| Ruler change, former ruler, current ruler | Invictus `common/on_action/on_ruler_change_inv_.txt`, `on_ruler_change`; Omniluxia `common/on_action/00_specific_from_code.txt` |
| Trait assignment | Omniluxia `decisions/flavor_decisions.txt`, `become_pope` |
| Save repair on monthly pulse | Omniluxia `common/on_action/omni_nazani_imperial_government.txt` |
| Intervention, neutrality, opinion, pledge flags | Invictus `events/mission_events/me_judea_inv.txt`, `judea_inv.1` |
| Opinion-based AI response weights | Same file, `judea_inv.3` options |
| Declare, then find/save war and recruit | Invictus `events/flavor_cap.txt`, `flavor_cap.2` and `.3` |
| Attacker/defender membership checks | Same file, `flavor_cap.3` |
| Join a selected war | Omniluxia `events/mission_events/me_cymeva.txt`, `me_cymeva.4` |
| War-ended hook | Invictus `common/on_action/on_ending_war_inv.txt` |
| Country references stored as variables | Omniluxia `common/scripted_effects/omni_dynastic_union_effects.txt` |
| Country variable compared to defender | Omniluxia `common/wargoals/omni_bastard_claim_wargoal.txt` |
| Flag-gated war goal | Omniluxia `common/wargoals/00_default.txt`, `horde_conquest` |
| Ordinary negotiated conquest settings | Same file, `conquer_wargoal` |
| Named candidate country choices | Invictus `events/war_council.txt`, `war_council.1` |
| Expedition army and delayed preparation | Omniluxia former `me_zorgoomni.55`/`.56` implementation |
| Throttled global scan | Omniluxia `common/scripted_effects/ai_war_effects.txt`, `ai_recruit_mercenaries` |
| Localization tone and color conventions | Omniluxia `localization/english/event_chains/choo_ularianexplorer_l_english.yml`; crusade prose has its own sacred and martial imagery |

The Judea chain contains an explicitly disabled Antigonid option. Its response
structure is a precedent; that disabled branch was not copied as working code.
The complete generic coalition is a new composition of these patterns.

## Validation

Run `python Scripts/audit_shannah_crusade.py`. It parses the new scripts, checks
event/helper/localization references and text markup, and checks integration,
funding/refund wiring, succession hooks, war guards, cooldown retention, legacy
entry points, and transient-variable cleanup. It does not execute game effects.

### Required in-game acceptance cases

Use a disposable save with a Shahanat of any territorial rank, a valid neighboring
non-Helluvian target, and several eligible independent Helluvian countries.

1. Kill the Shannah through the game's testing tools. Verify the actual new
   ruler receives the trait, and repeat with a non-child successor. Verify a
   living replaced ruler loses it. Load an older save lacking the trait and
   advance through a monthly pulse to check repair.
   For office transfer, have IF3 lose ownership of Ddu-Nuinop while surviving:
   verify the office is retained until a Helluvian conqueror takes Claim the
   Shahanat. Then verify IF3 becomes a feudal monarchy and loses the trait.
   Advance a monthly pulse and change its ruler: neither may restore its
   office. Retake the city as IF3, reclaim the title, and verify succession
   works again without repeat legitimacy/influence rewards. Repeat transfer
   once during preparation (refund expected) and once during an active war
   (war and armies must survive). Historical dead Shannahs retain their traits.
2. Propose and dismiss a target: no money, influence, cooldown, or units gained.
   Verify the council shows up to three different eligible country names and
   that confirmation retains exactly the country chosen. Check one-, two-,
   and three-candidate cases; invalidated choices should disappear. Choosing a
   candidate alone does not spend the entry cost. Direct console event `.1`
   remains the random-proposal test entry; `.15` opens the new council.
   Confirm: exactly 250 gold and 50 influence are spent; preparation and the
   cooldown start. A second Shahanat cannot start a concurrent expedition.
3. Use different invitees to pledge, donate, and refuse. Verify the donor loses
   100 and the host gains 100 only once. Refusal does not put a country at war.
   A country fighting an unrelated war may donate or refuse without penalty.
4. Save/reload during preparation and with an invitation unanswered. Verify
   the target, host, counts, pledge, and donation persist. Replace the Shannah
   during preparation; the successor must retain the expedition and cooldown.
5. Cancel voluntarily: verify -3 stability, entry-cost return, donor repayment,
   no army creation, and retained cooldown. Advance past day 180: no stale
   deadline event may launch a cancelled expedition.
6. Convert or annex the target during preparation. Advance the monthly pulse:
   preparations should cancel, refund, and release the global lock. The
   cancellation event should explicitly identify its reason. Separately start
   an unrelated war involving the host: preparations must survive monthly
   checks and the deadline must offer postponement. Postpone by 30 days,
   restore peace, and verify launch becomes available again. Also test range,
   alliance, truce, and civil-war blockers without automatic cancellation.
   An annexed host must not leave a permanent lock.
7. Launch with two eligible pledgers and a refuser. Verify the correct target,
   capital-province objective, host leadership, attacking participants, nine
   subunits, and verified-entry rewards. The refuser remains outside.
   Check the news as host, main target, attacking ally, defending ally, donor,
   refuser, and an uninvited outsider. Each must receive its one appropriate
   launch event. No launch announcements should fire for a cancelled expedition.
8. Change a pledger's religion, subject status, war status, or diplomatic
   obligations before launch. Verify no forced invalid entry and no blessing.
   Also check an ordinary ally who joins automatically is counted once.
9. Exercise the engine's declaration failure cases. No matching war means no
   expedition army and a refunded cancellation. This specifically verifies the
   flag-gated custom war goal and its recorded-defender comparison at runtime.
10. End an unrelated war while the crusade continues. Its records must remain.
    End the crusade, annex the target, or change war leadership: records should
    clear without altering other wars, deleting armies, or forcing peace.
11. Save/reload during the active war; verify monthly and peace-triggered
    cleanup still identify the matching participants. Test a later expedition
    after the cooldown without old donations or pledges carrying forward.
12. Inspect localization in the event windows and decision tooltips: dynamic
    names, numeric counters, paragraph breaks, colors, and the legacy refund
    message. Check the game's script/error logs for unknown commands or scopes.
    In Review the Crusade, click "Read the roll of those summoned" and check
    actual country names and replies in the new event body. Verify next and
    previous pages, the return-to-summary button, the final partial page, and
    an empty roll. Another expedition's records must be excluded. Reading
    pages must not change money, pledges, or war membership. Recheck after
    departure for fulfilled/unfulfilled pledges and unanswered invitations.

### Retesting a launch that previously cancelled

Restart Imperator completely and load a save from before launch (preferably
before the deadline event opens). Launch and advance at least one in-game day;
the expedition army and coalition are confirmed on that following day. A save
already cancelled retains its cooldown, so use a pre-cancellation save.

Search `logs/debug.log` and `logs/game.log` for `OMNI_CRUSADE:`. Stage messages
distinguish monthly host/target invalidation, rejected launch eligibility,
declaration failure with the host still at peace, and a war that exists but
does not match the recorded attacker/defender/leadership. The previous tests'
logs do not contain those new diagnostics, so their precise cancellation path
has not been established. The timing repair requires in-game confirmation.

## Deliberately outside this implementation

No contribution rankings, automated territorial allocation, beneficiary system,
new crusader countries, defensive religious coalition, or custom crusade GUI.
Those were postponed in the agreed plan. AI war membership does not guarantee
useful army movement; runtime observation must guide invitation-range tuning.

## Acceptance, defense, and compact-roster regression cases

1. Start a new expedition after restarting the game to load scripts. Check that
   the target gets its warning at confirmation, not on selecting a candidate.
2. Compare several expeditions from a tiny host and a strong host. Use enough
   invitations to see a trend; any one AI response is probabilistic. Check score
   boundaries 24/25, 74/75, 149/150, and 299/300. Unraised levies are not counted
   as cohorts; territories provide the peacetime component.
3. As target, appeal immediately, then check the defense review decision.
   Decline the warning once and confirm the decision allows one later appeal.
   Once sent, the appeal option must disappear. Same-faith/race neighbors and
   allies should be eligible; unrelated unfriendly neighbors should not.
4. Pledge defense as an eligible neighbor, then launch. Verify actual defender
   entry and the participant news. Joining an unrelated war, becoming a subject,
   acquiring a host truce/alliance, or civil war before launch must block
   scripted entry. Countries already defending through normal alliances should
   be recognized. No country should be scripted onto both sides.
5. Leave an offensive invitation open before receiving a defensive appeal.
   The stale offensive popup must no longer offer a usable offensive pledge
   or donation, and dismissal must not overwrite the defensive reply.
6. Cancel preparation and verify all defensive records and the target's review
   decision disappear. No permanent alliance should have been created.
7. Browse each category with 0, 1, 15, 16, and 74 names. Verify compact names,
   same-category pagination, no stale names on the last page, and no more than
   six buttons in any menu. Check very long country names visually.
8. Save/reload during preparation and while reviewing the target's roster.
   Reopen through its decision; verify the viewer is not mistaken for the host.

Static checks cannot validate actual event layout, AI choice execution, diplomatic
engine restrictions, or game reload behavior. Those cases require Imperator.

Targets must be within diplomatic range AND either share a border with the
host or have a capital within 250 map-distance units of the host capital.
The cap is omni_crusade_target_distance_svalue; it does not grow with ordinary
diplomatic-range bonuses. Direct neighbors bypass the capital-distance cap,
so a large neighboring realm is not excluded because its capital is distant.
The decision, council, direct-event fallback and launch use matching rules.
Capital distance is geographical proximity, not a guarantee of a passable army
route. Moving capitals during preparation can make a non-border target fail
launch validation; preparations can be postponed under the existing rules.

## Size-scaled offerings

New expeditions snapshot the host's territory count at confirmation. Each donor
pays 2 gold for 1-9 host territories, 5 for 10-24, 10 for 25-49, 25 for 50-99,
50 for 100-199, and 100 for 200+. Army strength does not inflate offerings.
Each donation records its exact payment; cancellation returns that amount and
subtracts the same amount from the host. Existing expeditions without a price
snapshot keep their previous 100-gold price; old donations without a ledger
refund 100. Already-received gold is not retroactively removed.

Test a new expedition on both sides of every territory threshold; check donor
and host treasury changes match, later territorial changes do not alter the
price, and cancellation returns precisely the original payment. Check an old
save's 100-gold donations still refund 100. The invitation tooltip quotes the
current offering; review/refund prose no longer assumes every offering is 100.
