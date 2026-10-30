"""Regression checks for the 2026-10-29 Helluvian/Shannah audit fixes. Bounded; not an engine test."""
import re
from pathlib import Path
from validate_shannah_favor import Model, realm, read, T, F
ROOT=Path(__file__).resolve().parents[1]
def src(p):return (ROOT/p).read_text(encoding='utf-8-sig')
F.update(read('common/scripted_effects/omni_helluvian_court.txt'))
T.update(read('common/scripted_triggers/omni_helluvian_court.txt'))
HOOK=read('common/on_action/omni_shahanat_authority.txt')

def run():
    # 1. The Shannah trait follows the office: anyone outside it loses the trait.
    repair=src('common/scripted_effects/omni_shannah_crusade_effects.txt')
    block=repair[repair.index('omni_shannah_repair_office_effect'):repair.index('omni_shannah_revoke_office_effect')]
    assert re.search(r'else_if = \{\s*limit = \{\s*exists = current_ruler\s*current_ruler = \{ has_trait = shannah_trait \}',block)
    # 2. First accession reads its announced-holder variable behind a guard.
    acc=src('events/omniluxia/omni_shannah_accession.txt')
    assert 'NOT = { var:omni_accession_announced_holder' not in acc
    # 3. Favorable powers grant gratitude; condemnation does not.
    pap=src('common/scripted_effects/omni_shannah_papal_actions.txt')
    for ev in (11,12,13,14,15):assert f'omni_shannah_grace_effect = yes scope:actor = {{ omni_hc_pastoral_authority_effect = yes omni_shannah_announce_effect = {{ EVENT = omni_shannah_powers.{ev} }}' in pap,ev
    assert 'omni_shannah_grace_effect = yes scope:actor = { omni_hc_pastoral_authority_effect = yes omni_shannah_announce_effect = { EVENT = omni_shannah_powers.10' not in pap
    assert 'omni_shannah_grace_opinion' in src('common/opinions/omni_shannah_papal_actions.txt')
    # 4. Yearly Authority upkeep: -2 above 75, -1 above 50, nothing at or below 50; registered yearly.
    assert 'omni_hc_authority_yearly' in src('common/on_action/00_yearly_country.txt')
    hook=dict(HOOK['omni_hc_authority_yearly'])
    for start,end in ((80,78),(76,74),(75,74),(60,59),(51,50),(50,50),(10,10)):
        h=realm('host',True);h['government']='papacy';h['variables']['omni_hc_authority_base']=start
        m=Model([h]);m.root=h
        assert m.check(hook['trigger'],h)
        m.fx(hook['effect'],h);assert h['variables']['omni_hc_authority_base']==end,(start,h['variables']['omni_hc_authority_base'])
    # 5. AI recognition odds respond to Authority in both directions.
    rec=src('common/scripted_effects/omni_helluvian_court.txt')
    assert rec.count('omni_hc_authority >= 75')>=2 and rec.count('omni_hc_authority < 25')>=2
    # 6. Crusade offerings are no longer token amounts.
    off=src('common/script_values/omni_shannah_crusade.txt')
    body=off[off.index('omni_crusade_offering_svalue'):off.index('omni_crusade_current_offering_svalue')]
    vals=[int(v) for v in re.findall(r'value = (\d+)',body)]
    assert min(vals)>=25 and vals[0]==25,vals
    print('PASS: trait follows office, guarded accession read, grace opinion on favorable powers only, yearly Authority upkeep bands, Authority-weighted AI recognition, crusade offering floor. Not an engine test.')
if __name__=='__main__':run()
