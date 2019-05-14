APPROX_PRO_GAMES_A_DAY = 4

CHAMPS_PER_TEAM = 5
MAX_ITEMS_PER_CHAMP = 6
CHAMPS_PER_GAME = 10
SPELLS_PER_CHAMP = 2
SPELLS_PER_GAME = SPELLS_PER_CHAMP * CHAMPS_PER_GAME
NUM_FEATURES = CHAMPS_PER_GAME + CHAMPS_PER_GAME * SPELLS_PER_CHAMP + CHAMPS_PER_GAME * MAX_ITEMS_PER_CHAMP

ROLE_ORDER = ["top", "jg", "mid", "adc", "sup"]