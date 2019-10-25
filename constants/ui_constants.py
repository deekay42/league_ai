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
NETWORK_CURRENT_GOLD_IMG_CROP = (8,8)
NETWORK_KDA_IMG_CROP = (14,9)
NETWORK_CS_IMG_CROP = (14,9)
NETWORK_LVL_IMG_CROP = (11,7)


# different resolutions


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

    CURRENT_GOLD_LEFT_X = 1140
    CURRENT_GOLD_TOP_Y = 1057.5
    CURRENT_GOLD_X_OFFSET = 0
    CURRENT_GOLD_Y_OFFSET = 2
    CURRENT_GOLD_DIGIT_WIDTH = 9
    CURRENT_GOLD_SIZE = 8

    KDA_X_START = 610
    KDA_Y_START = 335
    KDA_HEIGHT = 24
    KDA_WIDTH = 90
    KDA_X_DIFF = 574
    KDA_Y_DIFF = 76
    KDA_X_CROP = 9
    KDA_Y_CROP = 14

    CS_X_START = 556
    CS_Y_START = KDA_Y_START
    CS_HEIGHT = KDA_HEIGHT
    CS_WIDTH = 50
    CS_X_DIFF = KDA_X_DIFF
    CS_Y_DIFF = KDA_Y_DIFF
    CS_X_CROP = KDA_X_CROP
    CS_Y_CROP = KDA_Y_CROP

    LVL_X_START = 523
    LVL_Y_START = 363
    LVL_HEIGHT = 11
    LVL_WIDTH = 14
    LVL_X_DIFF = KDA_X_DIFF
    LVL_Y_DIFF = KDA_Y_DIFF
    LVL_X_CROP = 7
    LVL_Y_CROP = 11


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

    CURRENT_GOLD_LEFT_X = 870
    CURRENT_GOLD_TOP_Y = 881
    CURRENT_GOLD_X_OFFSET = 0
    CURRENT_GOLD_Y_OFFSET = 1
    CURRENT_GOLD_DIGIT_WIDTH = 8
    CURRENT_GOLD_SIZE = 7

    KDA_X_START = 430
    KDA_Y_START = 278
    KDA_HEIGHT = 22
    KDA_WIDTH = 70
    KDA_X_DIFF = 479
    KDA_Y_DIFF = 64
    KDA_X_CROP = 7
    KDA_Y_CROP = 10

    CS_X_START = 382
    CS_Y_START = KDA_Y_START
    CS_HEIGHT = KDA_HEIGHT
    CS_WIDTH = 44
    CS_X_DIFF = KDA_X_DIFF
    CS_Y_DIFF = KDA_Y_DIFF
    CS_X_CROP = KDA_X_CROP
    CS_Y_CROP = KDA_Y_CROP

    LVL_X_START = 355
    LVL_Y_START = 302
    LVL_HEIGHT = 10
    LVL_WIDTH = 13
    LVL_X_DIFF = 480
    LVL_Y_DIFF = KDA_Y_DIFF
    LVL_X_CROP = 5
    LVL_Y_CROP = 10



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

    CURRENT_GOLD_LEFT_X = 811
    CURRENT_GOLD_TOP_Y = 752
    CURRENT_GOLD_X_OFFSET = 0
    CURRENT_GOLD_Y_OFFSET = 1
    CURRENT_GOLD_DIGIT_WIDTH = 6
    CURRENT_GOLD_SIZE = 5

    KDA_X_START = 436
    KDA_Y_START = 236
    KDA_HEIGHT = 20
    KDA_WIDTH = 58
    KDA_X_DIFF = 408
    KDA_Y_DIFF = 55
    KDA_X_CROP = 5
    KDA_Y_CROP = 8

    CS_X_START = 396
    CS_Y_START = KDA_Y_START
    CS_HEIGHT = KDA_HEIGHT
    CS_WIDTH = 30
    CS_X_DIFF = KDA_X_DIFF
    CS_Y_DIFF = KDA_Y_DIFF
    CS_X_CROP = KDA_X_CROP
    CS_Y_CROP = KDA_Y_CROP

    LVL_X_START = 372
    LVL_Y_START = 258
    LVL_HEIGHT = 9
    LVL_WIDTH = 10
    LVL_X_DIFF = KDA_X_DIFF
    LVL_Y_DIFF = KDA_Y_DIFF
    LVL_X_CROP = 4
    LVL_Y_CROP = 7


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

    CURRENT_GOLD_LEFT_X = 640
    CURRENT_GOLD_TOP_Y = 752
    CURRENT_GOLD_X_OFFSET = 0
    CURRENT_GOLD_Y_OFFSET = 1
    CURRENT_GOLD_DIGIT_WIDTH = 6
    CURRENT_GOLD_SIZE = 5

    KDA_X_START = 267
    KDA_Y_START = 236
    KDA_HEIGHT = 22
    KDA_WIDTH = 56
    KDA_X_DIFF = 409
    KDA_Y_DIFF = 55
    KDA_X_CROP = 5
    KDA_Y_CROP = 8

    CS_X_START = 230
    CS_Y_START = KDA_Y_START
    CS_HEIGHT = KDA_HEIGHT
    CS_WIDTH = 26
    CS_X_DIFF = KDA_X_DIFF
    CS_Y_DIFF = KDA_Y_DIFF
    CS_X_CROP = KDA_X_CROP
    CS_Y_CROP = KDA_Y_CROP

    LVL_X_START = 200
    LVL_Y_START = 259
    LVL_HEIGHT = 7
    LVL_WIDTH = 12
    LVL_X_DIFF = KDA_X_DIFF
    LVL_Y_DIFF = KDA_Y_DIFF
    LVL_X_CROP = 5
    LVL_Y_CROP = 7


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

            self.CURRENT_GOLD_LEFT_X = round(x * self.selected_res.CURRENT_GOLD_LEFT_X / self.selected_res.STD_WIDTH)

            self.CURRENT_GOLD_TOP_Y = round(y * self.selected_res.CURRENT_GOLD_TOP_Y / self.selected_res.STD_HEIGHT)

            self.CURRENT_GOLD_X_OFFSET = round(x * self.selected_res.CURRENT_GOLD_X_OFFSET / self.selected_res.STD_WIDTH)

            self.CURRENT_GOLD_Y_OFFSET = round(y * self.selected_res.CURRENT_GOLD_Y_OFFSET /
                                               self.selected_res.STD_HEIGHT)

            self.CURRENT_GOLD_DIGIT_WIDTH = round(
                x * self.selected_res.CURRENT_GOLD_DIGIT_WIDTH / self.selected_res.STD_WIDTH)

            self.CURRENT_GOLD_SIZE = round(
                x * self.selected_res.CURRENT_GOLD_SIZE / self.selected_res.STD_WIDTH)

            self.KDA_X_START = round(
                x * self.selected_res.KDA_X_START / self.selected_res.STD_WIDTH)

            self.KDA_Y_START = round(y * self.selected_res.KDA_Y_START /
                                     self.selected_res.STD_HEIGHT)

            self.KDA_HEIGHT = round(y * self.selected_res.KDA_HEIGHT /
                                               self.selected_res.STD_HEIGHT)

            self.KDA_WIDTH = round(
                x * self.selected_res.KDA_WIDTH / self.selected_res.STD_WIDTH)

            self.KDA_X_DIFF = round(
                x * self.selected_res.KDA_X_DIFF / self.selected_res.STD_WIDTH)

            self.KDA_Y_DIFF = round(y * self.selected_res.KDA_Y_DIFF /
                               self.selected_res.STD_HEIGHT)

            self.KDA_X_CROP = round(
                x * self.selected_res.KDA_X_CROP / self.selected_res.STD_WIDTH)

            self.KDA_Y_CROP = round(y * self.selected_res.KDA_Y_CROP /
                                    self.selected_res.STD_HEIGHT)


            self.CS_X_START = round(
                x * self.selected_res.CS_X_START / self.selected_res.STD_WIDTH)

            self.CS_Y_START = round(y * self.selected_res.CS_Y_START /
                                     self.selected_res.STD_HEIGHT)

            self.CS_HEIGHT = round(y * self.selected_res.CS_HEIGHT /
                                    self.selected_res.STD_HEIGHT)

            self.CS_WIDTH = round(
                x * self.selected_res.CS_WIDTH / self.selected_res.STD_WIDTH)

            self.CS_X_DIFF = round(
                x * self.selected_res.CS_X_DIFF / self.selected_res.STD_WIDTH)

            self.CS_Y_DIFF = round(y * self.selected_res.CS_Y_DIFF /
                                    self.selected_res.STD_HEIGHT)

            self.CS_X_CROP = round(
                x * self.selected_res.CS_X_CROP / self.selected_res.STD_WIDTH)

            self.CS_Y_CROP = round(y * self.selected_res.CS_Y_CROP /
                                    self.selected_res.STD_HEIGHT)

