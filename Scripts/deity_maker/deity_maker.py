sSS

import codecs
import random
from jinja2 import Environment, FileSystemLoader

def add_utf8_bom(filename):
	with codecs.open(filename, 'r', 'utf-8') as f:
		content = f.read()
	with codecs.open(filename, 'w', 'utf-8') as f2:
		f2.write('\ufeff')
		f2.write(content)
	return

religions_todo = [
	"zorg_religion",
]

used_deity_names = ["Atmis","Yhdite","Deteus","Zunmjir","Shulios","Itenar","Otreus","Nuius","Solagi","Shurnir","Rhuldir","Khygius","Luzdum","Yruer","Vatyx","Phodarr","Qhedon","Ybium","Chezmes","Mothos","Roruer","Axnir","Banas","Kaion","Uderin","Nirsin","Lotis","Tuton","Rias","Udes","Breona","Gholuna","Nezva","Cerra","Brumtia","Xuarae","Xena","Eneas","Gumis","Aona","Qidas","Aien","Dearin","Rymris","Ydes","Vensyn","Qhidis","Oion","Nogasis","Xeesis","Rhaveus","Phadrasil","Rumbris","Xagbium","Faseyr","Chimir","Oenar","Ibium","Phudite","Tumir","Delene","Qhekuna","Runhena","Baphion","Azone","Cigona","Naotl","Adione","Touna","Nova","Elysus","Uneus","Chuvldir","Itses","Khavlios","Ohthar","Eros","Theenar","Cekdros","Ulotl","Xudone","Uone","Phozien","Qimis","Nodall","Bierin","Cunren","Dosus","Omlos","Bixaldin","Xoruer","Unas","Oxbris","Ubium","Zuros","Rotar","Rotar","Devses","Gydes","Zeglous","Ovdos","Phadis","Vimies","Qhiais","Cozerin","Eteus","Bearin","Inas","Phornos","Kadis","Tophin","Drilmis","Xyella","Grinneas","Krahena","Nugsyn","Grorena","Phogyn","Kosta","Qihena","Agena","Unos","Cokysus","Gialdir","Dueyr","Axjun","Aeyr","Modarr","Phuvtar","Ydohr","Boborh","Lihdon","Shadur","Toveus","Ekdes","Unir","Rheesis","Xedbris","Adros","Amdohr","Veaos","Igaldin","Bidtin","Iohr","Riktar","Buras","Buras","Ruris","Pheasis","Lophin","Indos","Dhusus","Measis","Bazra","Ighena","Onja","Zydia","Kugmis","Nuventa","Itrix","Qavara","Deara","Alos","Tidon","Lootl","Gostris","Idaldin","Kusin","Uxses","Thizphin","Culerin","Giean","Qhateus","Noses","Hokdes","Ilous","Qheruer","Vadar","Khidos","Khidos","Zumher","Tiborh","Chotnia","Draone","Uthys","Zexla","Remasis","Uhenta","Dramsoi","Cyana","Inja","Xomtia","Remasis","Krerphion","Kradone","Qudona","Greneas","Breraura","Tuharae","Taena","Qilena","Uona","Uona","Quelia","Wigaldir","Nixbris","Nulenar","Udall","Yxton","Araldir","Exdarr","Roesis","Nuzotl","Obium","Adsin","Xidall","Iesis","Uzies","Phabin","Aton","Odos","Diasis","Thyhaldin","Iktin","Meara","Orsyn","Thoena","Xuxanke","Qrirva","Zinanke","Itune","Qodite","Qinera","Aana","Eagi","Rojun","Edbium","Fyvldir","Voenar","Asdur","Deses","Fuzotl","Senir","Teher","Qhosher","Dyesis","Teen","Tahbin","Phavros","Talos","Nuvian","Iies","Phoslos","Inir","Uhotl","Brutris","Krylella","Thyta","Gridla","Drutone","Ziarae","Iona","Zasta","Dyphine","Wusdite","Khudlios","Abium","Qilses","Shuysus","Sytyx","Xiione","Uxzotz","Haarus","Dhineyr","Alion","Lidite","Zagone","Nonaldin","Gialdin","Dhossin","Avgen","Idas","Tinir","Gutia","Odtix","Phikdione","Unone","Xudasis","Thilene","Ota","Danlena","Aaris","Motella","Dinja","Phedmes","Guxmjir","Choemis","Fatia","Cizotz","Azotl","Ulios","Rhises","Rhises","Ythos","Lugther","Dhoglan","Kuxzotl","Dhelos","Limdis","Arlir","Sulphin","Roldall","Thotin","Nivdall","Qragtuna","Ghadione","Ghiyja","Drerphy","Xira","Thomis","Quva","Krosoi","Thamis","Vedelia","Calios","Ogeyr","Qutohr","Rhodar","Phamir","Hiohr","Nalmus","Phetyx","Hamdur","Quarus","Dhejun","Codros","Futton","Tuton","Qhetyx","Khakmus","Phinia","Fundos","Shadius","Todis","Qhoen","Ruton","Nuzmis","Baion","Ovmos","Molos","Gaarin","Qadon","Qhalgen","Druarae","Girla","Eta","Iella","Akhena","Qhorena","Druuna","Druuna","Qhakdione","Vetia","Brukmis","Urdros","Cereus","Loione","Hoysus","Qhuddur","Veaos","Febris","Udos","Ilios","Reemis","Nireus","Rhadite","Edis","Hityx","Sherohr","Ekstus","Liglous","Vizjun","Udarr","Nidite","Gyeyar","Annos","Aeyar","Gilan","Xyohr","Igen","Xyohr","Qhiias","Qhylos","Artia","Ynsin","Iphy","Iris","Ghizheia","Xugyja","Ghiana","Krezdia","Drartrix","Ghiheia","Aella","Azena","Khienar","Xovaos","Phidite","Ykenar","Vomemis","Tothar","Aldir","Udis","Rhedmjir","Dimnas","Eanh","Vunaris","Lozgen","Zalgen","Qhodaldin","Phesotl","Uphin","Eias","Qimdes","Maotz","Dana","Kruyja","Qruena","Gegtix","Cayja","Kylphy","Vovanke","Kryasis","Iztix","Ogzotz","Onas","Iton","Xugius","Ekros","Ixtyx","Sudon","Godur","Uton","Egdur","Cythar","Cijun","Retmjir","Phidon","Oenar","Khehbium","Oslios","Phudar","Ados","Ados","Onir","Ledall","Idlin","Seton","Ikmis","Toen","Toen","Vytia","Avgen","Qizdite","Iztos","Vatlene","Iuna","Byzva","Enena","Ghanaris","Ghanaris","Urphion","Illena","Qremera","Theelia","Uenar","Walthos","Enir","Thomos","Monysus","Waaos","Torenar","Motarus","Rhuzreus","Dhaion","Izanh","Thaone","Ohotl","Aies","Yton","Kexohr","Emais","Namos","Utis","Aius","Datia","Thahmjir","Inos","Sevmir","Oohr","Chereus","Chereus","Ledon","Avdros","Athos","Manmera","Badea","Thyulla","Ciotz","Nogara","Igla","Mirlena","Taana","Ikona","Ihtuna","Xembium","Phaagi","Qhulmus","Hizther","Hoslotl","Thexarus","Uhdarr","Funos","Okeyar","Wodite","Zozros","Maros","Bunlan","Zyris","Koros","Kedall","Niltin","Detos","Tybin","Ugen","Qata","Xudara","Ugone","Votix","Imdione","Alene","Gaana","Vunera","Kysyja","Qredione","Taros","Tiara","Qhaxdon","Nezotl","Iher","Vatos","Teses","Adon","Dhahzotl","Earis","Vomir","Dhazotl","Umir","Byxton","Lomos","Relios","Chedis","Kheruer","Zotmos","Daeyr","Wemir","Boros","Yarus","Dhilotl","Chuesis","Phednos","Dhogher","Ybus","Ixdos","Cebris","Uher","Osyn","Itdis","Phozaldin","Cysyn","Aara","Lierin","Idall","Cyanh","Aas","Anona","Emsoi","Amera","Talona","Rephine","Qhuktris","Kiris","Rotulla","Bootl","Atia","Aton","Oean","Yzmos","Abris","Gudbin","Ekdall","Anir","Atdes","Ezotl","Otlos","Fiarus","Vuton","Minas","Itris","Yion","Razmera","Naneas","Qhaone","Gikthys","Zirthar","Qelo","Yerin","Zidite","Iaias","Anandeix","Ishial","Saririyash","Anegip","Paruparyo","Jarhaukri","Sibipulay","Uaruong","Akuretsag","Yiyipag","Payulurkas","Bayelako","Laluyop","Nenataulon","Tarube","Iyanetu","akanpuro","Tiyamdir","Karikakek","Yuyatampa","Tayaskari","Maluatuang","Pektusk","Pulukaset","Herabarchiya","Olkiya","Tangkep","Sitiyadram","Banadutong","Yiyogang","Sangtutla","Penurugusk","Anenjur","Adalauska","Kahartauya","Atiyankur","Adapandar","Esantua","Koriks","Buyotay","Muaratim","Palamat","Gansarti","Anduk","Anahbrak","Lulinagla","Atuea","Abekom","Nagarutiya","Kuangkortah","Kapuyubah","Ayuposkiyu","Anitshempa","Paluaruak","'Amedayul","Parimades","Achrianoc","Asazantur","Palpadvaf","Atachupio","Balbatoc","Panichuros","Oruhalla","Sadinupa","Garapagod","Siphoriam","Erandas","Pichosinma","Anichures","Elogus","Palchvas","Chirochatsem","Fediandes","Daialoba","Aielupter","Anipandre","Abaxias","Povabanis","Chalchichi","Leasintrecha","Apanachus","Povadini","Paireros","Chaiesachui","Feragontus","Berhadesia","Neripadog","Iamaduh","Huhiater","Adihaxes","Agon","Iendos","Pirutanta","Achuscha","Lelandi","Penrachi","Chaiapald","Fiagatri","Burog","Ialag","Aeiloch","Sulpugro","Icholu","Ugoscutra","Anaphaer","Piasulog","Achanter","Anaphendiag","Vouiaran","Piochora","Amadresian"]
new_deity_names = []
deity_names = [x for x in new_deity_names if x not in used_deity_names]

deity_and_religion_dict = dict()
deity_categories = ["war", "culture", "fertility", "economy"]

apotheosis_effect_war = [
	"military_apotheosis_manpower_effect = yes",
	"military_apotheosis_defensive_effect = yes",
	"military_apotheosis_capital_freemen_effect = yes",
	"military_apotheosis_military_experience_effect = yes",
	"war_apotheosis_martial_tech_effect = yes",
	"naval_apotheosis_effect = yes",
]

apotheosis_effect_fertility = [
	"fertility_apotheosis_capital_effect = yes",
	"fertility_apotheosis_capital_slaves_effect = yes",
	"fertility_apotheosis_food_effect = yes",
]

apotheosis_effect_economy = [
	"economy_apotheosis_province_improvement_effect = yes",
	"economy_apotheosis_capital_citizens_effect = yes",
	"economy_income_effect = yes",
	"economy_apotheosis_capital_noble_effect = yes",
]

apotheosis_effect_culture = [
	"culture_apotheosis_civic_tech_effect = yes",
	"culture_apotheosis_oratory_tech_effect = yes",
	"culture_apotheosis_assimilate_effect = yes",
	"culture_apotheosis_capital_effect = yes",
	"culture_apotheosis_characters_effect = yes",
	"culture_apotheosis_rel_tech_effect = yes",
]

war_deity_passive = [
	"discipline = deity_discipline_svalue",
	"land_morale_modifier = deity_land_morale_modifier_svalue",
	"war_score_cost = deity_war_score_cost_svalue",
	"army_maintenance_cost = deity_army_maintenance_cost_svalue",
	"agressive_expansion_impact = deity_aggressive_expansion_impact_svalue",
	"war_breaking_truce_cost_modifier = -0.125",
	"monthly_military_experience_modifier = deity_monthly_military_experience_modifier_svalue",
	"global_manpower_modifier = deity_global_manpower_modifier_svalue",
	"experience_decay = deity_experience_decay_svalue",
	"global_supply_limit_modifier = deity_global_supply_limit_modifier_svalue",
	"global_monthly_state_loyalty = deity_global_monthly_state_loyalty_svalue",
	"global_defensive = omen_global_defensive_svalue",
	"agressive_expansion_monthly_change = deity_aggressive_expansion_monthly_change_svalue",
	"war_exhaustion = deity_war_exhaustion_svalue",
	"assault_ability = 0.05",
	"global_start_experience = deity_global_start_experience_svalue",
	"manpower_recovery_speed = deity_global_manpower_recovery_speed_svalue",
	"global_ship_start_experience = deity_global_ship_start_experience_svalue",
	"navy_maintenance_cost = deity_navy_maintenance_cost_svalue",
]

war_deity_omen = [
	"discipline = omen_discipline_svalue",
	"land_morale_modifier = omen_land_morale_modifier_svalue",
	"naval_morale_modifier = omen_naval_morale_modifier_svalue",
	"manpower_recovery_speed = omen_manpower_recovery_speed",
	"agressive_expansion_monthly_change = omen_aggressive_expansion_monthly_change_svalue",
	"naval_damage_done = omen_naval_damage_done_svalue",
	"naval_damage_taken = omen_naval_damage_taken_svalue",
	"global_defensive = omen_global_defensive_svalue",
	"war_no_cb_cost_modifier = omen_war_no_cb_cost_modifier_svalue",
	"assault_ability = omen_assault_ability_svalue",
	"fabricate_claim_speed = omen_fabricate_claim_speed_svalue",
]

economy_deity_omen = [
	"build_cost = omen_build_cost_svalue",
	"global_commerce_modifier = omen_global_commerce_modifier_svalue",
	"global_tax_modifier = omen_global_tax_modifier_svalue",
	"religious_tech_investment = omen_religious_tech_investment_svalue",
	"civic_tech_investment = omen_civic_tech_investment_svalue",
	"military_tech_investment = omen_military_tech_investment",
	"oratory_tech_investment = omen_oratory_tech_investment",
	"fort_maintenance_cost = omen_fort_maintenance_cost_svalue",
	"mercenary_land_maintenance_cost = omen_mercenary_land_maintenance_cost_svalue",
	"monthly_wage_modifier = omen_monthly_wage_modifier_svalue",
]

economy_deity_passive = [
	"build_cost = deity_build_cost_svalue",
	"build_time = deity_build_time_svalue",
	"global_capital_trade_routes = deity_global_capital_trade_routes_svalue",
	"global_commerce_modifier = deity_global_commerce_modifier_svalue",
	"global_tax_modifier = deity_global_tax_modifier_svalue",
	"global_nobles_output = deity_global_nobles_output_svalue",
	"global_nobles_happyness = deity_global_nobles_happiness_svalue",
	"global_citizen_output = deity_global_citizen_output_svalue",
	"global_citizen_happyness = deity_global_citizen_happiness_svalue",
	"diplomatic_relations = deity_diplomatic_relations_svalue",
	"diplomatic_reputation = deity_diplomatic_reputation_svalue",
	"civic_tech_investment = deity_civic_tech_investment_svalue",
	"military_tech_investment = deity_military_tech_investment_svalue",
	"oratory_tech_investment = deity_oratory_tech_investment_svalue",
	"religious_tech_investment = deity_religious_tech_investment_svalue",
	"monthly_wage_modifier = deity_monthly_wage_modifier_svalue",
]

culture_deity_passive = [
	"monthly_political_influence_modifier = deity_monthly_political_influence_modifier_svalue",
	"ruler_popularity_gain = deity_ruler_popularity_gain_svalue",
	"global_monthly_state_loyalty = deity_global_monthly_state_loyalty_svalue",
	"research_points_modifier = deity_research_points_modifier_svalue",
	"monthly_corruption = deity_monthly_corruption_svalue",
	"monthly_tyranny = deity_monthly_tyranny_svalue",
	"global_monthly_civilization = deity_global_monthly_civilization_svalue",
	"global_slaves_happyness = deity_global_slaves_happiness_svalue",
	"global_serfs_happyness = deity_global_slaves_happiness_svalue",
	"global_slaves_output = deity_global_slaves_output_svalue",
	"global_serfs_output = deity_global_slaves_output_svalue",
	"global_pop_assimilation_speed_modifier = deity_global_pop_assimilation_speed_modifier_svalue",
	"global_pop_conversion_speed_modifier = deity_global_pop_conversion_speed_modifier_svalue",
	"happiness_for_wrong_culture_group_modifier = deity_happiness_for_wrong_culture_group_modifier_svalue",
	"happiness_for_wrong_culture_modifier = deity_happiness_for_wrong_culture_modifier_svalue",
	"happiness_for_same_culture_modifier = deity_happiness_for_same_culture_modifier_svalue",
	"happiness_for_same_religion_modifier = deity_happiness_for_same_religion_modifier",
	"stability_cost_modifier = deity_stability_cost_modifier",
	"stability_monthly_change = deity_stability_monthly_change",
	#"stability_monthly_decay = deity_stability_monthly_decay",
]

culture_deity_omen = [
	"ruler_popularity_gain = omen_ruler_popularity_gain_svalue",
	"research_points_modifier = omen_research_points_modifier_svalue",
	"stability_monthly_change = omen_stability_monthly_change_svalue",
	"monthly_tyranny = omen_monthly_tyranny_svalue",
	"war_score_cost = omen_war_score_cost_svalue",
	"global_monthly_civilization = omen_global_monthly_civilization_svalue",
	"global_pop_conversion_speed_modifier = omen_global_pop_conversion_speed_modifier_svalue",
	"global_pop_assimilation_speed_modifier = omen_global_pop_assimilation_speed_modifier_svalue",
	"happiness_for_same_culture_modifier = omen_happiness_for_same_culture_modifier_svalue",
	"happiness_for_same_religion_modifier = omen_happiness_for_same_religion_modifier_svalue",
	"happiness_for_wrong_culture_group_modifier = omen_happiness_for_wrong_culture_group_modifier_svalue",
	"happiness_for_wrong_culture_modifier = omen_happiness_for_wrong_culture_modifier_svalue",
	"monthly_corruption = omen_monthly_corruption",
	"war_exhaustion = omen_war_exhaustion",
	"global_monthly_state_loyalty = omen_global_monthly_state_loyalty",
	"global_population_happiness = omen_global_population_happiness",
	"global_serfs_output = omen_global_slaves_output_svalue",
]

fertility_deity_passive = [
	"global_monthly_civilization = deity_global_monthly_civilization_svalue",
	"happiness_for_same_culture_modifier = deity_happiness_for_same_culture_modifier_svalue",
	"happiness_for_same_religion_modifier = deity_happiness_for_same_religion_modifier",
	"global_population_capacity_modifier = deity_global_population_capacity_modifier_svalue",
	"global_population_growth = deity_global_population_growth_svalue",
	"global_monthly_food_modifier = deity_global_monthly_food_modifier_svalue",
	"global_food_capacity = deity_global_food_capacity_svalue",
	"global_supply_limit_modifier = deity_global_supply_limit_modifier_svalue",
]

fertility_deity_omen = [
	"research_points_modifier = omen_research_points_modifier_svalue",
	"global_monthly_civilization = omen_global_monthly_civilization_svalue",
	"happiness_for_wrong_culture_group_modifier = omen_happiness_for_wrong_culture_group_modifier_svalue",
	"hostile_attrition = omen_hostile_attrition_svalue",
	"global_nobles_happyness = omen_global_nobles_happiness_svalue",
	"global_nobles_output = omen_global_nobles_output_svalue",
	"global_citizen_happyness = omen_global_citizen_happiness_svalue",
	"global_citizen_output = omen_global_citizen_output_svalue",
	"global_freemen_happyness = omen_global_freemen_happiness_svalue",
	"global_freemen_output = omen_global_freemen_output_svalue",
	"global_slaves_happyness = omen_global_slaves_happiness_svalue",
	"global_slaves_output = omen_global_slaves_output_svalue",
	"global_population_happiness = omen_global_population_happiness",
	"global_monthly_food_modifier = omen_global_monthly_food_modifier",
	"global_population_growth = omen_global_population_growth",
	"army_weight_modifier = omen_army_weight_modifier",
	"manpower_recovery_speed = omen_manpower_recovery_speed",
	"ruler_popularity_gain = omen_ruler_popularity_gain_svalue",
]

# Load and render the template
env = Environment(loader=FileSystemLoader('./'))
template = env.get_template('deity.jinja2')

def get_passive(category):
	if category == "war":
		return random.choice(war_deity_passive)
	if category == "culture":
		return random.choice(culture_deity_passive)
	if category == "fertility":
		return random.choice(fertility_deity_passive)
	if category == "economy":
		return random.choice(economy_deity_passive)

def get_omen(category):
	if category == "war":
		return random.choice(war_deity_omen)
	if category == "culture":
		return random.choice(culture_deity_omen)
	if category == "fertility":
		return random.choice(fertility_deity_omen)
	if category == "economy":
		return random.choice(economy_deity_omen)

def get_apotheosis(category):
	if category == "war":
		return random.choice(apotheosis_effect_war)
	if category == "culture":
		return random.choice(apotheosis_effect_culture)
	if category == "fertility":
		return random.choice(apotheosis_effect_fertility)
	if category == "economy":
		return random.choice(apotheosis_effect_economy)

def get_deity_output(religion):
	global deity_and_religion_dict
	deity_output = str()
	deity_and_religion_dict.update({religion: list()})
	for k in range(3):
		for j in deity_categories:
			category = j
			deity = random.choice(deity_names)
			deity_info = (deity.lower(), category)
			deity_and_religion_dict[religion].append(deity_info)
			deity_names.remove(deity)
			passive = get_passive(category)
			omen = get_omen(category)
			effect = get_apotheosis(category)
			deity_output += template.render(deity=deity.lower(), religion=religion, category=category, passive=passive, omen=omen, effect=effect)

	return deity_output

def write_setup_output(religion, current_free_key):
	with open(f"output/setup/00_{religion}.txt", "w") as file:
		file.write(f"deity_manager = {{\n\tdeities_database = {{ ### KEYS {current_free_key} - {current_free_key + 99}")
	with open(f"output/setup/00_{religion}.txt", "a") as file:
		for j in deity_and_religion_dict[religion]:
			file.write(f"\n\t\t{current_free_key} = {{\n\t\t\tkey = omen_{j[0]}\n\t\t\tdeity = deity_{j[0]}\n\t\t}}")
			current_free_key += 1
		file.write("\n\t}\n}")

def write_localzation_output(religion):
	with open(f"output/localization/deities_{religion}_l_english.yml", "w") as file:
		file.write(f"l_english:\n ## {religion} ##\n")
	with open(f"output/localization/deities_{religion}_l_english.yml", "a") as file:
		for j in deity_and_religion_dict[religion]:
			file.write(f'deity_{j[0]}:0 "$omen_{j[0]}$"\n')
			file.write(f'omen_{j[0]}:0 "{j[0].title()}"\n')
			file.write(f'omen_{j[0]}_desc:0 "{j[0].title()} is a deity of {j[1].title()}"\n\n')
		add_utf8_bom(file.name)

if __name__ == '__main__':
	for i in religions_todo:
		with open(f"output/deities/00_{i}.txt", "w") as file:
			file.write("") # clear file
		with open(f"output/deities/00_{i}.txt", "a") as file:
			file.write(get_deity_output(i))
			add_utf8_bom(file.name)

	current_free_key = 3200 # each religion gets 100 free keys, currently 30 are done.


	for i in deity_and_religion_dict:
		write_setup_output(i, current_free_key)
		write_localzation_output(i)
		current_free_key += 100
