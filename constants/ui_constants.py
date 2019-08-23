SCOREBOARD_SCALING = 0.5
STD_SCOREBOARD_WIDTH = 870
STD_SCOREBOARD_HEIGHT = 352
SCOREBOARD_LEFT_X_OFFSET = 285
SCOREBOARD_Y_OFFSET = 174
SCOREBOARD_BORDER_WIDTH = 8
SCOREBOARD_INNER_LEFT_X_OFFSET = 10
SCOREBOARD_INNER_RIGHT_X_OFFSET = 437
SCOREBOARD_INNER_Y_BOT_OFFSET = 287
SCOREBOARD_INNER_TILE_WIDTH = 425
SCOREBOARD_INNER_TILE_HEIGHT = 57
SCOREBOARD_TOP_TILE_Y_OFFSET = 10
SCOREBOARD_ITEM_BORDER_WIDTH = 10
SCOREBOARD_INNER_BORDER_WIDTH = 3
SCOREBOARD_HEAD_Y_OFFSET = 44
SCOREBOARD_HEAD_LEFT_X_OFFSET = 27

CHAMP_CIRCLE = 34
CHAMP_SMALL_CIRCLE = 17
SMALL_CIRCLE_X_OFFSET = 5
SMALL_CIRCLE_Y_OFFSET = 6
CHAMP_IMG_BLACK_BORDER_WIDTH = 4
CHAMP_MIN_SIZE = 20
CHAMP_MAX_SIZE = 50

ITEM_MIN_SIZE = 15
ITEM_MAX_SIZE = 40

SPELL_MIN_SIZE = 15
SPELL_MAX_SIZE = 40

NETWORK_ITEM_IMG_CROP = (24, 24)
NETWORK_CHAMP_IMG_CROP = (20, 20)
NETWORK_SELF_IMG_CROP = (20, 20)


# different resolutions

# 1440x900
class Res_8_5:
    ITEM_INNER_OFFSET = 2

    CHAMP_LEFT_X_OFFSET = 335
    CHAMP_RIGHT_X_OFFSET = 813
    CHAMP_Y_OFFSET = 275
    CHAMP_Y_DIFF = 64
    CHAMP_SIZE = 23

    ITEM_LEFT_X_OFFSET = 516
    ITEM_RIGHT_X_OFFSET = 992
    ITEM_Y_DIFF = 64
    ITEM_Y_OFFSET = 276
    ITEM_X_DIFF = 28.4
    ITEM_SIZE = 24
    SUMM_NAMES_DIS_X_OFFSET = -7
    SUMM_NAMES_DIS_Y_OFFSET = 8

    SPELL_LEFT_X_OFFSET = 276
    SPELL_RIGHT_X_OFFSET = 754
    SPELL_Y_DIFF = 41
    SPELL_Y_DIFF_LARGE = 64
    SPELL_Y_OFFSET = 267
    SPELL_SIZE = 22

    SELF_INDICATOR_Y_DIFF = 64

    SELF_INDICATOR_LEFT_X_OFFSET = 247
    SELF_INDICATOR_RIGHT_X_OFFSET = 723

    SELF_INDICATOR_Y_OFFSET = 300
    SELF_INDICATOR_SIZE = 15

    STD_WIDTH = 1440
    STD_HEIGHT = 900


class Res_16_9:
    CHAMP_LEFT_X_OFFSET = 497
    CHAMP_RIGHT_X_OFFSET = 1071
    CHAMP_Y_OFFSET = 331
    CHAMP_Y_DIFF = 76
    CHAMP_SIZE = 28

    ITEM_LEFT_X_OFFSET = 715
    ITEM_RIGHT_X_OFFSET = 1286
    ITEM_Y_DIFF = 76
    ITEM_Y_OFFSET = 332
    ITEM_X_DIFF = 34
    ITEM_SIZE = 29
    ITEM_INNER_OFFSET = 2

    SUMM_NAMES_DIS_X_OFFSET = -9
    SUMM_NAMES_DIS_Y_OFFSET = 10

    SPELL_LEFT_X_OFFSET = 356
    SPELL_RIGHT_X_OFFSET = 834
    SPELL_Y_DIFF = 41
    SPELL_Y_DIFF_LARGE = 64
    SPELL_Y_OFFSET = 267
    SPELL_SIZE = 22

    SELF_INDICATOR_LEFT_X_OFFSET = 392
    SELF_INDICATOR_RIGHT_X_OFFSET = 963
    SELF_INDICATOR_Y_DIFF = 76
    SELF_INDICATOR_Y_OFFSET = 360
    SELF_INDICATOR_SIZE = 17

    STD_WIDTH = 1920
    STD_HEIGHT = 1080


class Res_4_3:
    CHAMP_LEFT_X_OFFSET = 183
    CHAMP_RIGHT_X_OFFSET = 592
    CHAMP_Y_OFFSET = 235
    CHAMP_Y_DIFF = 55
    CHAMP_SIZE = 20

    ITEM_LEFT_X_OFFSET = 337
    ITEM_RIGHT_X_OFFSET = 743
    ITEM_Y_DIFF = 55
    ITEM_Y_OFFSET = 236
    ITEM_X_DIFF = 24
    ITEM_SIZE = 20
    ITEM_INNER_OFFSET = 2
    SUMM_NAMES_DIS_X_OFFSET = -6
    SUMM_NAMES_DIS_Y_OFFSET = 6

    SELF_INDICATOR_LEFT_X_OFFSET = 108
    SELF_INDICATOR_RIGHT_X_OFFSET = 514
    SELF_INDICATOR_Y_DIFF = 55
    SELF_INDICATOR_Y_OFFSET = 254
    SELF_INDICATOR_SIZE = 11

    STD_WIDTH = 1024
    STD_HEIGHT = 768


class Res_1366_768:
    CHAMP_LEFT_X_OFFSET = 352
    CHAMP_RIGHT_X_OFFSET = 761
    CHAMP_Y_OFFSET = 234
    CHAMP_Y_DIFF = 55
    CHAMP_SIZE = 21

    ITEM_LEFT_X_OFFSET = 508
    ITEM_RIGHT_X_OFFSET = 915
    ITEM_Y_DIFF = 55
    ITEM_Y_OFFSET = 236
    ITEM_X_DIFF = 24
    ITEM_SIZE = 19
    ITEM_INNER_OFFSET = 2
    SUMM_NAMES_DIS_X_OFFSET = -6
    SUMM_NAMES_DIS_Y_OFFSET = 7

    SELF_INDICATOR_LEFT_X_OFFSET = 278
    SELF_INDICATOR_RIGHT_X_OFFSET = 686
    SELF_INDICATOR_Y_DIFF = 55
    SELF_INDICATOR_Y_OFFSET = 254
    SELF_INDICATOR_SIZE = 13

    STD_WIDTH = 1366
    STD_HEIGHT = 768


class ResConverter:
    instance = None


    def __init__(self, x, y):
        if not ResConverter.instance:
            ResConverter.instance = ResConverter.__ResConverter(int(x), int(y))

    def set_res(self, x, y):
        ResConverter.instance = ResConverter.__ResConverter(int(x), int(y))

    def __getattr__(self, name):
        return getattr(self.instance, name)


    class __ResConverter:

        def __init__(self, x, y):

            if round(x / y, 4) == 1.7786:
                self.selected_res = Res_1366_768()
            elif round(x / y, 2) == round(16 / 9, 2):
                self.selected_res = Res_16_9()
            elif round(x / y, 2) == round(8 / 5, 2):
                self.selected_res = Res_8_5()
            elif round(x / y, 2) == round(4 / 3, 2):
                self.selected_res = Res_4_3()
            else:
                raise Exception("Screen resolution not supported: " + str(x) + " " + str(y))

            self.CHAMP_LEFT_X_OFFSET = (x * self.selected_res.CHAMP_LEFT_X_OFFSET / self.selected_res.STD_WIDTH)

            self.CHAMP_RIGHT_X_OFFSET = (x * self.selected_res.CHAMP_RIGHT_X_OFFSET / self.selected_res.STD_WIDTH)

            self.CHAMP_Y_OFFSET = (y * self.selected_res.CHAMP_Y_OFFSET / self.selected_res.STD_HEIGHT)

            self.CHAMP_Y_DIFF = (y * self.selected_res.CHAMP_Y_DIFF / self.selected_res.STD_HEIGHT)

            self.CHAMP_SIZE = round(x * self.selected_res.CHAMP_SIZE / self.selected_res.STD_WIDTH)

            self.ITEM_LEFT_X_OFFSET = (x * self.selected_res.ITEM_LEFT_X_OFFSET / self.selected_res.STD_WIDTH)

            self.ITEM_RIGHT_X_OFFSET = (x * self.selected_res.ITEM_RIGHT_X_OFFSET / self.selected_res.STD_WIDTH)

            self.ITEM_Y_OFFSET = (y * self.selected_res.ITEM_Y_OFFSET / self.selected_res.STD_HEIGHT)

            self.ITEM_Y_DIFF = (y * self.selected_res.ITEM_Y_DIFF / self.selected_res.STD_HEIGHT)

            self.ITEM_SIZE = round(x * self.selected_res.ITEM_SIZE / self.selected_res.STD_WIDTH)

            self.ITEM_X_DIFF = (x * self.selected_res.ITEM_X_DIFF / self.selected_res.STD_WIDTH)

            self.ITEM_INNER_OFFSET = (x * self.selected_res.ITEM_INNER_OFFSET / self.selected_res.STD_WIDTH)

            self.SUMM_NAMES_DIS_X_OFFSET = (x * self.selected_res.SUMM_NAMES_DIS_X_OFFSET / self.selected_res.STD_WIDTH)

            self.SUMM_NAMES_DIS_Y_OFFSET = (
                        y * self.selected_res.SUMM_NAMES_DIS_Y_OFFSET / self.selected_res.STD_HEIGHT)

            self.SELF_INDICATOR_LEFT_X_OFFSET = (
                    x * self.selected_res.SELF_INDICATOR_LEFT_X_OFFSET / self.selected_res.STD_WIDTH)

            self.SELF_INDICATOR_RIGHT_X_OFFSET = (
                    x * self.selected_res.SELF_INDICATOR_RIGHT_X_OFFSET / self.selected_res.STD_WIDTH)

            self.SELF_INDICATOR_Y_OFFSET = (
                        y * self.selected_res.SELF_INDICATOR_Y_OFFSET / self.selected_res.STD_HEIGHT)

            self.SELF_INDICATOR_Y_DIFF = (y * self.selected_res.SELF_INDICATOR_Y_DIFF / self.selected_res.STD_HEIGHT)

            self.SELF_INDICATOR_SIZE = round(x * self.selected_res.SELF_INDICATOR_SIZE / self.selected_res.STD_WIDTH)
