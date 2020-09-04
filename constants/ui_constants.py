import numpy as np

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


class ResolutionBase:

    def __init__(self):
        self.coords = {"champs": dict(),
                       "items": dict(),
                       "self": dict(),
                       "kda": dict(),
                       "current_gold": dict(),
                       "cs": dict(),
                       "lvl": dict()}


class Res_16_9(ResolutionBase):

    def __init__(self):
        super().__init__()
        self.STD_WIDTH = 1920
        self.STD_HEIGHT = 1080

        self.coords["champs"]["x_start"] = 497
        self.coords["champs"]["y_start"] = 331
        self.coords["champs"]["x_diff"] = 574
        self.coords["champs"]["y_diff"] = 76
        self.coords["champs"]["x_crop"] = 28
        self.coords["champs"]["y_crop"] = 28

        self.coords["items"]["x_start"] = 714
        self.coords["items"]["y_start"] = 331
        self.coords["items"]["x_diff"] = 571
        self.coords["items"]["y_diff"] = 76
        self.coords["items"]["x_crop"] = 29
        self.coords["items"]["y_crop"] = 29
        self.coords["items"]["x_inner_offset"] = 2
        self.coords["items"]["x_item_spacing"] = 34
        self.coords["items"]["summ_names_x_offset"] = -9
        self.coords["items"]["summ_names_y_offset"] = 10

        self.coords["self"]["x_start"] = 390
        self.coords["self"]["y_start"] = 356
        self.coords["self"]["x_diff"] = 571
        self.coords["self"]["y_diff"] = 76
        self.coords["self"]["x_crop"] = 17
        self.coords["self"]["y_crop"] = 17

        self.coords["current_gold"]["x_crop"] = 8
        self.coords["current_gold"]["y_crop"] = 12
        self.coords["current_gold"]["min_x_start"] = 1120.5
        self.coords["current_gold"]["max_x_start"] = 1200
        self.coords["current_gold"]["min_y_start"] = 1057.5
        self.coords["current_gold"]["max_y_start"] = 1045.5
        self.coords["current_gold"]["min_x_width"] = 60
        self.coords["current_gold"]["max_x_width"] = 90
        self.coords["current_gold"]["min_y_height"] = 15
        self.coords["current_gold"]["max_y_height"] = 24

        self.coords["kda"]["x_start"] = 610
        self.coords["kda"]["y_start"] = 337
        self.coords["kda"]["y_height"] = 17
        self.coords["kda"]["x_width"] = 90
        self.coords["kda"]["x_diff"] = 574
        self.coords["kda"]["y_diff"] = 76
        self.coords["kda"]["x_crop"] = 9
        self.coords["kda"]["y_crop"] = 14

        self.coords["cs"]["x_start"] = 556
        self.coords["cs"]["y_start"] = self.coords["kda"]["y_start"]
        self.coords["cs"]["y_height"] = self.coords["kda"]["y_height"]
        self.coords["cs"]["x_width"] = 50
        self.coords["cs"]["x_diff"] = self.coords["kda"]["x_diff"]
        self.coords["cs"]["y_diff"] = self.coords["kda"]["y_diff"]
        self.coords["cs"]["x_crop"] = self.coords["kda"]["x_crop"]
        self.coords["cs"]["y_crop"] = self.coords["kda"]["y_crop"]


        self.coords["lvl"]["x_start"] = 521
        self.coords["lvl"]["y_start"] = 360
        self.coords["lvl"]["y_height"] = 13
        self.coords["lvl"]["x_width"] = 18
        self.coords["lvl"]["x_diff"] = self.coords["kda"]["x_diff"]
        self.coords["lvl"]["y_diff"] = self.coords["kda"]["y_diff"]
        self.coords["lvl"]["x_crop"] = 7
        self.coords["lvl"]["y_crop"] = 11


# 1440x900
class Res_8_5(ResolutionBase):

    def __init__(self):
        super().__init__()
        self.STD_WIDTH = 1440
        self.STD_HEIGHT = 900

        self.coords["champs"]["x_start"] = 335
        self.coords["champs"]["y_start"] = 275
        self.coords["champs"]["x_diff"] = 478
        self.coords["champs"]["y_diff"] = 62.5
        self.coords["champs"]["x_crop"] = 23
        self.coords["champs"]["y_crop"] = 23

        self.coords["items"]["x_start"] = 516
        self.coords["items"]["y_start"] = 276
        self.coords["items"]["x_diff"] = 476
        self.coords["items"]["y_diff"] = self.coords["champs"]["y_diff"]
        self.coords["items"]["x_crop"] = 24
        self.coords["items"]["y_crop"] = 24
        self.coords["items"]["x_inner_offset"] = 2
        self.coords["items"]["x_item_spacing"] = 28.4
        self.coords["items"]["summ_names_x_offset"] = -7
        self.coords["items"]["summ_names_y_offset"] = 8

        self.coords["self"]["x_start"] = 245
        self.coords["self"]["y_start"] = 296
        self.coords["self"]["x_diff"] = 476
        self.coords["self"]["y_diff"] = self.coords["champs"]["y_diff"]
        self.coords["self"]["x_crop"] = 15
        self.coords["self"]["y_crop"] = 15

        self.coords["current_gold"]["x_crop"] = 5
        self.coords["current_gold"]["y_crop"] = 9
        self.coords["current_gold"]["min_x_start"] = 854
        self.coords["current_gold"]["max_x_start"] = 922
        self.coords["current_gold"]["min_y_start"] = 881
        self.coords["current_gold"]["max_y_start"] = 871
        self.coords["current_gold"]["min_x_width"] = 50
        self.coords["current_gold"]["max_x_width"] = 72
        self.coords["current_gold"]["min_y_height"] = 14
        self.coords["current_gold"]["max_y_height"] = 18

        self.coords["kda"]["x_start"] = 435
        self.coords["kda"]["y_start"] = 282
        self.coords["kda"]["y_height"] = 11
        self.coords["kda"]["x_width"] = 70
        self.coords["kda"]["x_diff"] = 479
        self.coords["kda"]["y_diff"] = self.coords["champs"]["y_diff"]
        self.coords["kda"]["x_crop"] = 7
        self.coords["kda"]["y_crop"] = 10

        self.coords["cs"]["x_start"] = 382
        self.coords["cs"]["y_start"] = self.coords["kda"]["y_start"]
        self.coords["cs"]["y_height"] = self.coords["kda"]["y_height"]
        self.coords["cs"]["x_width"] = 44
        self.coords["cs"]["x_diff"] = self.coords["kda"]["x_diff"]
        self.coords["cs"]["y_diff"] = self.coords["kda"]["y_diff"]
        self.coords["cs"]["x_crop"] = self.coords["kda"]["x_crop"]
        self.coords["cs"]["y_crop"] = self.coords["kda"]["y_crop"]


        self.coords["lvl"]["x_start"] = 355
        self.coords["lvl"]["y_start"] = 301
        self.coords["lvl"]["y_height"] = 11
        self.coords["lvl"]["x_width"] = 12
        self.coords["lvl"]["x_diff"] = 479
        self.coords["lvl"]["y_diff"] = self.coords["kda"]["y_diff"]



class Res_1366_768(ResolutionBase):

    def __init__(self):
        super().__init__()

        self.STD_WIDTH = 1366
        self.STD_HEIGHT = 768

        self.coords["champs"]["x_start"] = 352
        self.coords["champs"]["y_start"] = 234
        self.coords["champs"]["x_diff"] = 409
        self.coords["champs"]["y_diff"] = 55
        self.coords["champs"]["x_crop"] = 21
        self.coords["champs"]["y_crop"] = 21

        self.coords["items"]["x_start"] = 508
        self.coords["items"]["y_start"] = 236
        self.coords["items"]["x_diff"] = 407
        self.coords["items"]["y_diff"] = 55
        self.coords["items"]["x_crop"] = 19
        self.coords["items"]["y_crop"] = 19
        self.coords["items"]["x_inner_offset"] = 2
        self.coords["items"]["x_item_spacing"] = 24
        self.coords["items"]["summ_names_x_offset"] = -6
        self.coords["items"]["summ_names_y_offset"] = 7

        self.coords["self"]["x_start"] = 278
        self.coords["self"]["y_start"] = 253
        self.coords["self"]["x_diff"] = 408
        self.coords["self"]["y_diff"] = 55
        self.coords["self"]["x_crop"] = 13
        self.coords["self"]["y_crop"] = 13

        self.coords["current_gold"]["x_start"] = 808
        self.coords["current_gold"]["y_start"] = 750
        self.coords["current_gold"]["y_height"] = 12
        self.coords["current_gold"]["x_width"] = 30
        self.coords["current_gold"]["x_crop"] = 5
        self.coords["current_gold"]["y_crop"] = 7
        self.coords["current_gold"]["min_x_start"] = 797
        self.coords["current_gold"]["max_x_start"] = 856
        self.coords["current_gold"]["min_y_start"] = 752
        self.coords["current_gold"]["max_y_start"] = 744
        self.coords["current_gold"]["min_x_width"] = 42
        self.coords["current_gold"]["max_x_width"] = 60
        self.coords["current_gold"]["min_y_height"] = 11
        self.coords["current_gold"]["max_y_height"] = 17

        self.coords["kda"]["x_start"] = 436
        self.coords["kda"]["y_start"] = 241
        self.coords["kda"]["y_height"] = 10
        self.coords["kda"]["x_width"] = 58
        self.coords["kda"]["x_diff"] = 408
        self.coords["kda"]["y_diff"] = 55
        self.coords["kda"]["x_crop"] = 5
        self.coords["kda"]["y_crop"] = 8

        self.coords["cs"]["x_start"] = 394
        self.coords["cs"]["y_start"] = self.coords["kda"]["y_start"]
        self.coords["cs"]["y_height"] = self.coords["kda"]["y_height"]
        self.coords["cs"]["x_width"] = 34
        self.coords["cs"]["x_diff"] = self.coords["kda"]["x_diff"]
        self.coords["cs"]["y_diff"] = self.coords["kda"]["y_diff"]
        self.coords["cs"]["x_crop"] = self.coords["kda"]["x_crop"]
        self.coords["cs"]["y_crop"] = self.coords["kda"]["y_crop"]


        self.coords["lvl"]["x_start"] = 372
        self.coords["lvl"]["y_start"] = 255
        self.coords["lvl"]["y_height"] = 11
        self.coords["lvl"]["x_width"] = 10
        self.coords["lvl"]["x_diff"] = self.coords["kda"]["x_diff"]
        self.coords["lvl"]["y_diff"] = self.coords["kda"]["y_diff"]
        self.coords["lvl"]["x_crop"] = 4
        self.coords["lvl"]["y_crop"] = 7


class Res_4_3(ResolutionBase):

    def __init__(self):
        super().__init__()
        self.STD_WIDTH = 1024
        self.STD_HEIGHT = 768

        self.coords["champs"]["x_start"] = 183
        self.coords["champs"]["y_start"] = 235
        self.coords["champs"]["x_diff"] = 409
        self.coords["champs"]["y_diff"] = 55
        self.coords["champs"]["x_crop"] = 20
        self.coords["champs"]["y_crop"] = 20

        self.coords["items"]["x_start"] = 337
        self.coords["items"]["y_start"] = 236
        self.coords["items"]["x_diff"] = 406
        self.coords["items"]["y_diff"] = 55
        self.coords["items"]["x_crop"] = 20
        self.coords["items"]["y_crop"] = 20
        self.coords["items"]["x_inner_offset"] = 2
        self.coords["items"]["x_item_spacing"] = 24
        self.coords["items"]["summ_names_x_offset"] = -6
        self.coords["items"]["summ_names_y_offset"] = 6

        self.coords["self"]["x_start"] = 107
        self.coords["self"]["y_start"] = 253
        self.coords["self"]["x_diff"] = 406
        self.coords["self"]["y_diff"] = 55
        self.coords["self"]["x_crop"] = 11
        self.coords["self"]["y_crop"] = 11

        self.coords["current_gold"]["x_crop"] = 5
        self.coords["current_gold"]["y_crop"] = 7
        self.coords["current_gold"]["min_x_start"] = 627
        self.coords["current_gold"]["max_x_start"] = 686
        self.coords["current_gold"]["min_y_start"] = 752
        self.coords["current_gold"]["max_y_start"] = 744
        self.coords["current_gold"]["min_x_width"] = 42
        self.coords["current_gold"]["max_x_width"] = 60
        self.coords["current_gold"]["min_y_height"] = 11
        self.coords["current_gold"]["max_y_height"] = 17

        self.coords["kda"]["x_start"] = 267
        self.coords["kda"]["y_start"] = 241
        self.coords["kda"]["y_height"] = 10
        self.coords["kda"]["x_width"] = 56
        self.coords["kda"]["x_diff"] = 409
        self.coords["kda"]["y_diff"] = 55
        self.coords["kda"]["x_crop"] = 5
        self.coords["kda"]["y_crop"] = 8

        self.coords["cs"]["x_start"] = 228
        self.coords["cs"]["y_start"] = self.coords["kda"]["y_start"]
        self.coords["cs"]["y_height"] = self.coords["kda"]["y_height"]
        self.coords["cs"]["x_width"] = 30
        self.coords["cs"]["x_diff"] = self.coords["kda"]["x_diff"]
        self.coords["cs"]["y_diff"] = self.coords["kda"]["y_diff"]
        self.coords["cs"]["x_crop"] = self.coords["kda"]["x_crop"]
        self.coords["cs"]["y_crop"] = self.coords["kda"]["y_crop"]

        self.coords["lvl"]["x_start"] = 200
        self.coords["lvl"]["y_start"] = 256
        self.coords["lvl"]["y_height"] = 9
        self.coords["lvl"]["x_width"] = 12
        self.coords["lvl"]["x_diff"] = self.coords["kda"]["x_diff"]
        self.coords["lvl"]["y_diff"] = self.coords["kda"]["y_diff"]
        self.coords["lvl"]["x_crop"] = 5
        self.coords["lvl"]["y_crop"] = 7


class ResConverter:
    network_crop = dict()
    network_crop["items"] = (24, 24)
    network_crop["champs"] = (20, 20)
    network_crop["self"] = (20, 20)
    network_crop["current_gold"] = (8, 8)
    network_crop["kda"] = (14, 9)
    network_crop["cs"] = (14, 9)
    network_crop["lvl"] = (11, 7)


    def __init__(self, x, y, hud_scale=None, summ_names_displayed=False):
        self.x = int(x)
        self.y = int(y)
        self.hud_scale = hud_scale
        self.summ_names_displayed = summ_names_displayed

        if round(self.x / self.y, 4) == 1.7786:
            self.selected_res = Res_1366_768()
        elif round(self.x / self.y, 2) == round(16 / 9, 2):
            self.selected_res = Res_16_9()
        elif round(self.x / self.y, 2) == round(8 / 5, 2):
            self.selected_res = Res_8_5()
        elif round(self.x / self.y, 2) == round(4 / 3, 2):
            self.selected_res = Res_4_3()
        else:
            raise Exception("Screen resolution not supported: " + str(self.x) + " " + str(self.y))


    def generate_std_coords(self, elements):
        for team_offset in [0, self.lookup(elements, "x_diff")]:
            for row in range(5):
                yield round(self.lookup(elements, "x_start") + team_offset), \
                      round(row * self.lookup(elements, "y_diff") + self.lookup(elements, "y_start"))


    def generate_item_coords(self):
        item_x_offset = self.lookup("items", "x_inner_offset")
        item_y_offset = item_x_offset
        if self.summ_names_displayed:
            item_x_offset += self.lookup("items", "summ_names_x_offset")
            item_y_offset += self.lookup("items", "summ_names_y_offset")


        for team_offset in [0, self.lookup("items", "x_diff")]:
            for row in range(5):
                for item in range(7):
                    yield round(self.lookup("items", "x_start") + team_offset + item_x_offset + item * self.lookup(
                        "items", "x_item_spacing")), \
                          round(row * self.lookup("items", "y_diff") + self.lookup("items", "y_start") + item_y_offset)



    def generate_current_gold_coords(self):
        if self.hud_scale is None:
            raise Exception("No value given for hud_scale")
        return self.lookup("current_gold", "min_x_start") + self.hud_scale * (self.lookup("current_gold", "max_x_start") -
                                                                          self.lookup(
            "current_gold", "min_x_start")), \
                self.lookup("current_gold", "min_y_start") - self.hud_scale * (self.lookup("current_gold", "min_y_start") - self.lookup(
            "current_gold", "max_y_start")), \
                self.lookup("current_gold", "min_x_width") + self.hud_scale * (self.lookup("current_gold", "max_x_width") - self.lookup(
            "current_gold", "min_x_width")), \
                self.lookup("current_gold", "min_y_height") + self.hud_scale * (self.lookup("current_gold", "max_y_height") - self.lookup(
            "current_gold", "min_y_height"))


    def lookup(self, elements, attribute):
        if "x_" in attribute:
            return (self.x * self.selected_res.coords[elements][attribute] / self.selected_res.STD_WIDTH)
        elif "y_" in attribute:
            return (self.y * self.selected_res.coords[elements][attribute] / self.selected_res.STD_HEIGHT)
