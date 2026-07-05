import sys

file_path = r'c:\Users\Joshua\Documents\Paradox Interactive\Imperator\mod\Omniluxia\events\zorgo_magic_events.txt'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# We need to insert the new spell options into me_zorgo_magic_focus.10, 11, 12, 13, and 30

omnic_spells = '''
    option = { # Blinding Flash
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 15 } }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 15 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        scope:target = {
            every_character_unit = {
                limit = { is_army = yes }
                if = { limit = { has_unit_modifier = omni_blinding_flash_mod } remove_unit_modifier = omni_blinding_flash_mod }
                add_unit_modifier = { name = omni_blinding_flash_mod duration = 30 }
            }
        }
        custom_tooltip = tier_one_spell_tt
        custom_tooltip = 15_mana_tt
        name = "me_zorgo_magic_focus.new.omnic1"
    }
    option = { # Chain Lightning
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 40 } }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 40 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        scope:target = {
            every_character_unit = {
                limit = { is_army = yes }
                if = { limit = { has_unit_modifier = omni_chain_lightning_mod } remove_unit_modifier = omni_chain_lightning_mod }
                add_unit_modifier = { name = omni_chain_lightning_mod duration = 30 }
            }
        }
        custom_tooltip = tier_two_spell_tt
        custom_tooltip = 40_mana_tt
        name = "me_zorgo_magic_focus.new.omnic2"
    }
    option = { # Gravity Well
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 100 } }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 100 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        scope:target = {
            every_character_unit = {
                limit = { is_army = yes }
                if = { limit = { has_unit_modifier = omni_gravity_well_mod } remove_unit_modifier = omni_gravity_well_mod }
                add_unit_modifier = { name = omni_gravity_well_mod duration = 30 }
            }
        }
        custom_tooltip = tier_three_spell_tt
        custom_tooltip = 100_mana_tt
        name = "me_zorgo_magic_focus.new.omnic3"
    }
    option = { # Chronostasis Field
        trigger = {
            scope:selected_mage = { 
                has_trait = archmage_trait
                omni_can_afford_spell = { BASE = 175 } 
            }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 175 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        scope:target = {
            every_character_unit = {
                limit = { is_army = yes }
                if = { limit = { has_unit_modifier = omni_chronostasis_field_mod } remove_unit_modifier = omni_chronostasis_field_mod }
                add_unit_modifier = { name = omni_chronostasis_field_mod duration = 30 }
            }
        }
        custom_tooltip = tier_four_spell_tt
        custom_tooltip = 175_mana_tt
        name = "me_zorgo_magic_focus.new.omnic4"
    }
'''

aldic_spells = '''
    option = { # Arcane Volley
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 5 } }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 5 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        scope:target = {
            every_character_unit = {
                limit = { is_army = yes }
                if = { limit = { has_unit_modifier = omni_arcane_volley_mod } remove_unit_modifier = omni_arcane_volley_mod }
                add_unit_modifier = { name = omni_arcane_volley_mod duration = 30 }
            }
        }
        custom_tooltip = tier_one_spell_tt
        custom_tooltip = 5_mana_tt
        name = "me_zorgo_magic_focus.new.aldic1"
    }
    option = { # Corrosive Rain
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 25 } }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 25 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        scope:target = {
            every_character_unit = {
                limit = { is_army = yes }
                if = { limit = { has_unit_modifier = omni_corrosive_rain_mod } remove_unit_modifier = omni_corrosive_rain_mod }
                add_unit_modifier = { name = omni_corrosive_rain_mod duration = 30 }
            }
        }
        custom_tooltip = tier_two_spell_tt
        custom_tooltip = 25_mana_tt
        name = "me_zorgo_magic_focus.new.aldic2"
    }
    option = { # Meteor Strike
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 95 } }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 95 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        scope:target = {
            random_character_unit = {
                damage_unit_percent = 0.20
            }
        }
        custom_tooltip = tier_three_spell_tt
        custom_tooltip = 95_mana_tt
        name = "me_zorgo_magic_focus.new.aldic3"
    }
    option = { # Word of Annihilation
        trigger = {
            scope:selected_mage = { 
                has_trait = archmage_trait
                omni_can_afford_spell = { BASE = 140 } 
            }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 140 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        scope:target = {
            every_character_unit = {
                limit = { is_army = yes }
                if = { limit = { has_unit_modifier = omni_word_of_annihilation_mod } remove_unit_modifier = omni_word_of_annihilation_mod }
                add_unit_modifier = { name = omni_word_of_annihilation_mod duration = 30 }
            }
        }
        custom_tooltip = tier_four_spell_tt
        custom_tooltip = 140_mana_tt
        name = "me_zorgo_magic_focus.new.aldic4"
    }
'''

amten_spells = '''
    option = { # Frostbite Wind
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 15 } }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 15 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        scope:target = {
            every_character_unit = {
                limit = { is_army = yes }
                if = { limit = { has_unit_modifier = omni_frostbite_wind_mod } remove_unit_modifier = omni_frostbite_wind_mod }
                add_unit_modifier = { name = omni_frostbite_wind_mod duration = 30 }
            }
        }
        custom_tooltip = tier_one_spell_tt
        custom_tooltip = 15_mana_tt
        name = "me_zorgo_magic_focus.new.amten1"
    }
    option = { # Seismic Tremor
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 35 } }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 35 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        scope:target = {
            every_character_unit = {
                limit = { is_army = yes }
                if = { limit = { has_unit_modifier = omni_seismic_tremor_mod } remove_unit_modifier = omni_seismic_tremor_mod }
                add_unit_modifier = { name = omni_seismic_tremor_mod duration = 30 }
            }
        }
        custom_tooltip = tier_two_spell_tt
        custom_tooltip = 35_mana_tt
        name = "me_zorgo_magic_focus.new.amten2"
    }
    option = { # Glacial Tomb
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 85 } }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 85 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        scope:target = {
            every_character_unit = {
                limit = { is_army = yes }
                if = { limit = { has_unit_modifier = omni_glacial_tomb_mod } remove_unit_modifier = omni_glacial_tomb_mod }
                add_unit_modifier = { name = omni_glacial_tomb_mod duration = 30 }
            }
        }
        custom_tooltip = tier_three_spell_tt
        custom_tooltip = 85_mana_tt
        name = "me_zorgo_magic_focus.new.amten3"
    }
    option = { # Summon Elemental Colossus
        trigger = {
            scope:selected_mage = { 
                has_trait = archmage_trait
                omni_can_afford_spell = { BASE = 150 } 
            }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 150 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        root = {
            every_character_unit = {
                limit = { 
                    is_army = yes
                    unit_owner = { this = root }
                    has_combat = yes
                }
                if = { limit = { has_unit_modifier = omni_elemental_colossus_mod } remove_unit_modifier = omni_elemental_colossus_mod }
                add_unit_modifier = { name = omni_elemental_colossus_mod duration = 30 }
            }
        }
        custom_tooltip = tier_four_spell_tt
        custom_tooltip = 150_mana_tt
        name = "me_zorgo_magic_focus.new.amten4"
    }
'''

melodian_spells = '''
    option = { # Igniting Spark
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 10 } }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 10 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        scope:target = {
            every_character_unit = {
                limit = { is_army = yes }
                if = { limit = { has_unit_modifier = omni_igniting_spark_mod } remove_unit_modifier = omni_igniting_spark_mod }
                add_unit_modifier = { name = omni_igniting_spark_mod duration = 30 }
            }
        }
        custom_tooltip = tier_one_spell_tt
        custom_tooltip = 10_mana_tt
        name = "me_zorgo_magic_focus.new.melodian1"
    }
    option = { # Fireball
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 30 } }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 30 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        scope:target = {
            random_character_unit = {
                damage_unit_percent = 0.05
            }
            every_character_unit = {
                limit = { is_army = yes }
                if = { limit = { has_unit_modifier = omni_fireball_mod } remove_unit_modifier = omni_fireball_mod }
                add_unit_modifier = { name = omni_fireball_mod duration = 30 }
            }
        }
        custom_tooltip = tier_two_spell_tt
        custom_tooltip = 30_mana_tt
        name = "me_zorgo_magic_focus.new.melodian2"
    }
    option = { # Plague Miasma
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 75 } }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 75 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        scope:target = {
            every_character_unit = {
                limit = { is_army = yes }
                if = { limit = { has_unit_modifier = omni_plague_miasma_mod } remove_unit_modifier = omni_plague_miasma_mod }
                add_unit_modifier = { name = omni_plague_miasma_mod duration = 30 }
            }
        }
        custom_tooltip = tier_three_spell_tt
        custom_tooltip = 75_mana_tt
        name = "me_zorgo_magic_focus.new.melodian3"
    }
    option = { # Cataclysmic Eruption
        trigger = {
            scope:selected_mage = { 
                has_trait = archmage_trait
                omni_can_afford_spell = { BASE = 165 } 
            }
            any_character_unit = {
                is_army = yes
                unit_owner = { this = root }
                has_combat = yes
            }
            NOT = { root = { has_variable = omni_battle_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 165 } }
        root = { set_variable = { name = omni_battle_spell_cooldown days = 30 } }
        scope:target = {
            random_character_unit = {
                damage_unit_percent = 0.30
            }
            every_character_unit = {
                limit = { is_army = yes }
                if = { limit = { has_unit_modifier = omni_cataclysmic_eruption_mod } remove_unit_modifier = omni_cataclysmic_eruption_mod }
                add_unit_modifier = { name = omni_cataclysmic_eruption_mod duration = 30 }
            }
        }
        custom_tooltip = tier_four_spell_tt
        custom_tooltip = 165_mana_tt
        name = "me_zorgo_magic_focus.new.melodian4"
    }
'''

siege_spells = '''
    option = { # Igniting Spark (Siege)
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 10 } }
            NOT = { root = { has_variable = omni_siege_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 10 } }
        root = { set_variable = { name = omni_siege_spell_cooldown days = 30 } }
        scope:target = {
            random_character_unit = {
                limit = { is_army = yes unit_location = { has_siege = yes } }
                unit_location = {
                    if = { limit = { has_province_modifier = omni_igniting_spark_mod } remove_province_modifier = omni_igniting_spark_mod }
                    add_province_modifier = { name = omni_igniting_spark_mod duration = 30 }
                }
            }
        }
        custom_tooltip = tier_one_spell_tt
        custom_tooltip = 10_mana_tt
        name = "me_zorgo_magic_focus.new.siege1"
    }
    option = { # Corrosive Rain (Siege)
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 25 } }
            NOT = { root = { has_variable = omni_siege_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 25 } }
        root = { set_variable = { name = omni_siege_spell_cooldown days = 30 } }
        scope:target = {
            random_character_unit = {
                limit = { is_army = yes unit_location = { has_siege = yes } }
                unit_location = {
                    if = { limit = { has_province_modifier = omni_corrosive_rain_mod } remove_province_modifier = omni_corrosive_rain_mod }
                    add_province_modifier = { name = omni_corrosive_rain_mod duration = 365 }
                }
            }
        }
        custom_tooltip = tier_two_spell_tt
        custom_tooltip = 25_mana_tt
        name = "me_zorgo_magic_focus.new.siege2"
    }
    option = { # Fireball (Siege)
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 30 } }
            NOT = { root = { has_variable = omni_siege_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 30 } }
        root = { set_variable = { name = omni_siege_spell_cooldown days = 30 } }
        scope:target = {
            random_character_unit = {
                limit = { is_army = yes unit_location = { has_siege = yes } }
                unit_location = {
                    if = { limit = { has_province_modifier = omni_fireball_mod } remove_province_modifier = omni_fireball_mod }
                    add_province_modifier = { name = omni_fireball_mod duration = 365 }
                }
            }
        }
        custom_tooltip = tier_two_spell_tt
        custom_tooltip = 30_mana_tt
        name = "me_zorgo_magic_focus.new.siege3"
    }
    option = { # Seismic Tremor (Siege)
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 35 } }
            NOT = { root = { has_variable = omni_siege_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 35 } }
        root = { set_variable = { name = omni_siege_spell_cooldown days = 30 } }
        scope:target = {
            random_character_unit = {
                limit = { is_army = yes unit_location = { has_siege = yes } }
                unit_location = {
                    if = { limit = { has_province_modifier = omni_seismic_tremor_mod } remove_province_modifier = omni_seismic_tremor_mod }
                    add_province_modifier = { name = omni_seismic_tremor_mod duration = 365 }
                    siege = { add_breach = 1 }
                }
            }
        }
        custom_tooltip = tier_two_spell_tt
        custom_tooltip = 35_mana_tt
        name = "me_zorgo_magic_focus.new.siege4"
    }
    option = { # Plague Miasma (Siege)
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 75 } }
            NOT = { root = { has_variable = omni_siege_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 75 } }
        root = { set_variable = { name = omni_siege_spell_cooldown days = 30 } }
        scope:target = {
            random_character_unit = {
                limit = { is_army = yes unit_location = { has_siege = yes } }
                unit_location = {
                    if = { limit = { has_province_modifier = omni_plague_miasma_mod } remove_province_modifier = omni_plague_miasma_mod }
                    add_province_modifier = { name = omni_plague_miasma_mod duration = 365 }
                }
            }
        }
        custom_tooltip = tier_three_spell_tt
        custom_tooltip = 75_mana_tt
        name = "me_zorgo_magic_focus.new.siege5"
    }
    option = { # Glacial Tomb (Siege)
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 85 } }
            NOT = { root = { has_variable = omni_siege_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 85 } }
        root = { set_variable = { name = omni_siege_spell_cooldown days = 30 } }
        scope:target = {
            random_character_unit = {
                limit = { is_army = yes unit_location = { has_siege = yes } }
                unit_location = {
                    if = { limit = { has_province_modifier = omni_glacial_tomb_mod } remove_province_modifier = omni_glacial_tomb_mod }
                    add_province_modifier = { name = omni_glacial_tomb_mod duration = 365 }
                }
            }
        }
        custom_tooltip = tier_three_spell_tt
        custom_tooltip = 85_mana_tt
        name = "me_zorgo_magic_focus.new.siege6"
    }
    option = { # Meteor Strike (Siege)
        trigger = {
            scope:selected_mage = { omni_can_afford_spell = { BASE = 95 } }
            NOT = { root = { has_variable = omni_siege_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 95 } }
        root = { set_variable = { name = omni_siege_spell_cooldown days = 30 } }
        scope:target = {
            random_character_unit = {
                limit = { is_army = yes unit_location = { has_siege = yes } }
                unit_location = {
                    if = { limit = { has_province_modifier = omni_meteor_strike_mod } remove_province_modifier = omni_meteor_strike_mod }
                    add_province_modifier = { name = omni_meteor_strike_mod duration = 365 }
                    siege = { add_breach = 1 }
                }
            }
        }
        custom_tooltip = tier_three_spell_tt
        custom_tooltip = 95_mana_tt
        name = "me_zorgo_magic_focus.new.siege7"
    }
    option = { # Word of Annihilation (Siege)
        trigger = {
            scope:selected_mage = { 
                has_trait = archmage_trait
                omni_can_afford_spell = { BASE = 140 } 
            }
            NOT = { root = { has_variable = omni_siege_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 140 } }
        root = { set_variable = { name = omni_siege_spell_cooldown days = 30 } }
        scope:target = {
            random_character_unit = {
                limit = { is_army = yes unit_location = { has_siege = yes } }
                unit_location = {
                    add_fort_level = -1
                    siege = { add_breach = 1 }
                }
            }
        }
        custom_tooltip = tier_four_spell_tt
        custom_tooltip = 140_mana_tt
        name = "me_zorgo_magic_focus.new.siege8"
    }
    option = { # Summon Elemental Colossus (Siege)
        trigger = {
            scope:selected_mage = { 
                has_trait = archmage_trait
                omni_can_afford_spell = { BASE = 150 } 
            }
            NOT = { root = { has_variable = omni_siege_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 150 } }
        root = { set_variable = { name = omni_siege_spell_cooldown days = 30 } }
        root = {
            every_character_unit = {
                limit = { 
                    is_army = yes
                    unit_owner = { this = root }
                    unit_location = { has_siege = yes }
                }
                if = { limit = { has_unit_modifier = omni_elemental_colossus_mod } remove_unit_modifier = omni_elemental_colossus_mod }
                add_unit_modifier = { name = omni_elemental_colossus_mod duration = 365 }
            }
        }
        custom_tooltip = tier_four_spell_tt
        custom_tooltip = 150_mana_tt
        name = "me_zorgo_magic_focus.new.siege9"
    }
    option = { # Cataclysmic Eruption (Siege)
        trigger = {
            scope:selected_mage = { 
                has_trait = archmage_trait
                omni_can_afford_spell = { BASE = 165 } 
            }
            NOT = { root = { has_variable = omni_siege_spell_cooldown } }
        }
        scope:selected_mage = { omni_spell_mana_cost_effect = { BASE = 165 } }
        root = { set_variable = { name = omni_siege_spell_cooldown days = 30 } }
        scope:target = {
            random_character_unit = {
                limit = { is_army = yes unit_location = { has_siege = yes } }
                unit_location = {
                    add_civilization_value = -10
                    add_city_status = -1
                    siege = { add_breach = 1 }
                    if = { limit = { has_province_modifier = omni_cataclysmic_eruption_mod } remove_province_modifier = omni_cataclysmic_eruption_mod }
                    add_province_modifier = { name = omni_cataclysmic_eruption_mod duration = 3650 }
                }
            }
        }
        custom_tooltip = tier_four_spell_tt
        custom_tooltip = 165_mana_tt
        name = "me_zorgo_magic_focus.new.siege10"
    }
'''

def insert_spells(event_name, spells, text):
    event_start = text.find(event_name + ' = {')
    if event_start == -1: return text
    
    end_of_event = text.find('\\n}', event_start)
    cancel_idx = text.rfind('me_zorgo_magic_focus.2.z.2"', event_start, end_of_event)
    
    insert_point = text.rfind('    option = {', event_start, cancel_idx)
    if insert_point == -1: return text
    
    return text[:insert_point] + spells + text[insert_point:]

content = insert_spells('me_zorgo_magic_focus.10', omnic_spells, content)
content = insert_spells('me_zorgo_magic_focus.11', aldic_spells, content)
content = insert_spells('me_zorgo_magic_focus.12', amten_spells, content)
content = insert_spells('me_zorgo_magic_focus.13', melodian_spells, content)
content = insert_spells('me_zorgo_magic_focus.30', siege_spells, content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
