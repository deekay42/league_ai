APPROX_PRO_GAMES_A_DAY = 4

CHAMPS_PER_TEAM = 5
MAX_ITEMS_PER_CHAMP = 6
CHAMPS_PER_GAME = 10
SPELLS_PER_CHAMP = 2
SPELLS_PER_GAME = SPELLS_PER_CHAMP * CHAMPS_PER_GAME
NUM_FEATURES = CHAMPS_PER_GAME + CHAMPS_PER_GAME * SPELLS_PER_CHAMP + CHAMPS_PER_GAME * MAX_ITEMS_PER_CHAMP
BARON_DURATION = 180
ELDER_DURATION = 150
ROLE_ORDER = ["top", "jg", "mid", "adc", "sup"]

ELITE_LEAGUES = ["Challenger", "Grandmaster", "Master"]
LOWER_LEAGUES = ["Diamond_1", "Diamond_2", "Diamond_3", "Diamond_4",
         "Platinum_1", "Platinum_2", "Platinum_3", "Platinum_4",
         "Gold_1", "Gold_3", "Gold_3", "Gold_4",
         "Silver_1", "Silver_2", "Silver_3", "Silver_4",
         "Bronze_1", "Bronze_2", "Bronze_3", "Bronze_4",
         "Iron_1", "Iron_2", "Iron_3", "Iron_4"]

NUM_LEAGUES = 27
NUM_ELITE_LEAGUES = 3

dragon2index = {"AIR_DRAGON":0, "EARTH_DRAGON":1, "FIRE_DRAGON":2, "WATER_DRAGON":3}

max_clip = {"kills": 20,"deaths": 20,"assists": 30, "total_gold": 20000,
                                          "cs": 400,
                                          "lvl": 18,
                                          "dragons_killed": 4.0,
                                          "turrets_destroyed": 11.0,
                               "current_gold": 4000.0}
min_clip = {"kills": 0, "deaths": 0, "assists": 0,
                                          "total_gold": 500,
                                          "cs": 0,
                                          "lvl": 1,
                                          "dragons_killed": 0.0,
                                          "turrets_destroyed": 0.0,
                                          "current_gold": 0.0}



