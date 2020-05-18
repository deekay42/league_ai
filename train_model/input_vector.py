from constants import game_constants


class Input:

        pos_start = 0
        pos_end = pos_start + 1
        champs_start = pos_end
        champs_half = champs_start + game_constants.CHAMPS_PER_TEAM
        champs_end = champs_start + game_constants.CHAMPS_PER_GAME
        items_start = champs_end
        items_half = items_start + game_constants.MAX_ITEMS_PER_CHAMP * 2 * game_constants.CHAMPS_PER_TEAM
        items_end = items_start + game_constants.MAX_ITEMS_PER_CHAMP * 2 * game_constants.CHAMPS_PER_GAME
        total_gold_start = items_end
        total_gold_half = total_gold_start + game_constants.CHAMPS_PER_TEAM
        total_gold_end = total_gold_start + game_constants.CHAMPS_PER_GAME
        cs_start = total_gold_end
        cs_half = cs_start + game_constants.CHAMPS_PER_TEAM
        cs_end = cs_start + game_constants.CHAMPS_PER_GAME
        neutral_cs_start = cs_end
        neutral_cs_half = neutral_cs_start + game_constants.CHAMPS_PER_TEAM
        neutral_cs_end = neutral_cs_start + game_constants.CHAMPS_PER_GAME
        xp_start = neutral_cs_end
        xp_half = xp_start + game_constants.CHAMPS_PER_TEAM
        xp_end = xp_start + game_constants.CHAMPS_PER_GAME
        lvl_start = xp_end
        lvl_half = lvl_start + game_constants.CHAMPS_PER_TEAM
        lvl_end = lvl_start + game_constants.CHAMPS_PER_GAME
        kda_start = lvl_end
        kda_half = kda_start + game_constants.CHAMPS_PER_TEAM * 3
        kda_end = kda_start + game_constants.CHAMPS_PER_GAME * 3
        current_gold_start = kda_end
        current_gold_half = current_gold_start + game_constants.CHAMPS_PER_TEAM
        current_gold_end = current_gold_start + game_constants.CHAMPS_PER_GAME
        baron_start = current_gold_end
        baron_half = baron_start + 1
        baron_end = baron_start + 2
        elder_start = baron_end
        elder_half = elder_start + 1
        elder_end = elder_start + 2
        dragons_killed_start = elder_end
        dragons_killed_half = dragons_killed_start + 4
        dragons_killed_end = dragons_killed_start + 2*4
        dragon_soul_start = dragons_killed_end
        dragon_soul_half = dragon_soul_start + 1
        dragon_soul_end = dragon_soul_start + 2
        dragon_soul_type_start = dragon_soul_end
        dragon_soul_type_half = dragon_soul_type_start + 4
        dragon_soul_type_end = dragon_soul_type_start + 8
        turrets_start = dragon_soul_type_end
        turrets_half = turrets_start + 1
        turrets_end = turrets_start + 2
        first_team_blue_start = turrets_end
        first_team_blue_end = first_team_blue_start + 1
        len = first_team_blue_end
