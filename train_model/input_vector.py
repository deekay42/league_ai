import json

import numpy as np

from constants import game_constants, app_constants
from utils import heavy_imports
from utils.artifact_manager import ChampManager


class Input:
    instance = None

    indices = dict()
    indices["start"] = dict()
    indices["half"] = dict()
    indices["end"] = dict()
    indices["mid"] = dict()

    indices["start"]["gameid"] = 0
    indices["end"]["gameid"] = indices["start"]["gameid"] + 1

    indices["start"]["pos"] = indices["end"]["gameid"]
    indices["end"]["pos"] = indices["start"]["pos"] + 1

    indices["start"]["champs"] = indices["end"]["pos"]
    indices["half"]["champs"] = indices["start"]["champs"] + game_constants.CHAMPS_PER_TEAM
    indices["end"]["champs"] = indices["start"]["champs"] + game_constants.CHAMPS_PER_GAME

    indices["start"]["items"] = indices["end"]["champs"]
    indices["half"]["items"] = indices["start"][
                                   "items"] + game_constants.MAX_ITEMS_PER_CHAMP * 2 * game_constants.CHAMPS_PER_TEAM
    indices["end"]["items"] = indices["start"][
                                  "items"] + game_constants.MAX_ITEMS_PER_CHAMP * 2 * game_constants.CHAMPS_PER_GAME

    indices["start"]["total_gold"] = indices["end"]["items"]
    indices["half"]["total_gold"] = indices["start"]["total_gold"] + game_constants.CHAMPS_PER_TEAM
    indices["end"]["total_gold"] = indices["start"]["total_gold"] + game_constants.CHAMPS_PER_GAME

    indices["start"]["cs"] = indices["end"]["total_gold"]
    indices["half"]["cs"] = indices["start"]["cs"] + game_constants.CHAMPS_PER_TEAM
    indices["end"]["cs"] = indices["start"]["cs"] + game_constants.CHAMPS_PER_GAME

    indices["start"]["lvl"] = indices["end"]["cs"]
    indices["half"]["lvl"] = indices["start"]["lvl"] + game_constants.CHAMPS_PER_TEAM
    indices["end"]["lvl"] = indices["start"]["lvl"] + game_constants.CHAMPS_PER_GAME

    indices["start"]["kills"] = indices["end"]["lvl"]
    indices["half"]["kills"] = indices["start"]["kills"] + game_constants.CHAMPS_PER_TEAM
    indices["end"]["kills"] = indices["start"]["kills"] + game_constants.CHAMPS_PER_GAME

    indices["start"]["deaths"] = indices["end"]["kills"]
    indices["half"]["deaths"] = indices["start"]["deaths"] + game_constants.CHAMPS_PER_TEAM
    indices["end"]["deaths"] = indices["start"]["deaths"] + game_constants.CHAMPS_PER_GAME

    indices["start"]["assists"] = indices["end"]["deaths"]
    indices["half"]["assists"] = indices["start"]["assists"] + game_constants.CHAMPS_PER_TEAM
    indices["end"]["assists"] = indices["start"]["assists"] + game_constants.CHAMPS_PER_GAME

    indices["start"]["current_gold"] = indices["end"]["assists"]
    indices["half"]["current_gold"] = indices["start"]["current_gold"] + game_constants.CHAMPS_PER_TEAM
    indices["end"]["current_gold"] = indices["start"]["current_gold"] + game_constants.CHAMPS_PER_GAME

    indices["start"]["baron"] = indices["end"]["current_gold"]
    indices["half"]["baron"] = indices["start"]["baron"] + 1
    indices["end"]["baron"] = indices["start"]["baron"] + 2

    indices["start"]["elder"] = indices["end"]["baron"]
    indices["half"]["elder"] = indices["start"]["elder"] + 1
    indices["end"]["elder"] = indices["start"]["elder"] + 2

    indices["start"]["dragons_killed"] = indices["end"]["elder"]
    indices["half"]["dragons_killed"] = indices["start"]["dragons_killed"] + 4
    indices["end"]["dragons_killed"] = indices["start"]["dragons_killed"] + 8

    indices["start"]["dragon_soul_type"] = indices["end"]["dragons_killed"]
    indices["half"]["dragon_soul_type"] = indices["start"]["dragon_soul_type"] + 4
    indices["end"]["dragon_soul_type"] = indices["start"]["dragon_soul_type"] + 8

    indices["start"]["turrets_destroyed"] = indices["end"]["dragon_soul_type"]
    indices["half"]["turrets_destroyed"] = indices["start"]["turrets_destroyed"] + 1
    indices["end"]["turrets_destroyed"] = indices["start"]["turrets_destroyed"] + 2

    indices["start"]["blue_side"] = indices["end"]["turrets_destroyed"]
    indices["half"]["blue_side"] = indices["start"]["blue_side"] + 1
    indices["end"]["blue_side"] = indices["half"]["blue_side"] + 1

    len = indices["end"]["blue_side"]

    numeric_slices = {'total_gold',
                      'cs',
                      'lvl',
                      'kills',
                      'deaths',
                      'assists',
                      'current_gold',
                      'turrets_destroyed',
                      'dragons_killed'}

    all_slices = {"gameid", "pos", "champs", "items", "total_gold", "cs",
                  "lvl", "kills", "deaths", "assists",
                  "current_gold", "baron", "elder", "dragons_killed",
                  "dragon_soul_type", "turrets_destroyed", "blue_side"}

    nonsymmetric_slices = {"gameid", "pos"}


    @staticmethod
    def fit(input_slice, slice_name):
        # d = np.reshape(input_slice, (-1, 1))
        d = input_slice
        d = np.clip(d, game_constants.min_clip[slice_name], game_constants.max_clip[slice_name])
        #this is a nonlinear transformation, afterwards the following equation no longer necessarily holds true:
        # sum([topkills, jgkills, midkills, adckills, suppkills]) == team1_kills
        # pt = heavy_imports.PowerTransformer(method='yeo-johnson', standardize=False)
        # d_t = pt.fit_transform(d)
        ss = heavy_imports.StandardScaler()
        d = ss.fit_transform(d)
        # mm = heavy_imports.MinMaxScaler()
        # d = mm.fit_transform(d)
        return {
                    # "yeo_lambdas": pt.lambdas_.tolist(),
                    "standard": {"mean": ss.mean_.tolist(), "scale": ss.scale_.tolist(), "var": ss.var_.tolist()},
                    # "minmax": {"min": mm.min_.tolist(), "scale": mm.scale_.tolist()}
                }


    #slices must be scaled uniformly or else you cannot perform derivative features since the scales are different
    @staticmethod
    def fit_scale_inputs(X):
        params = dict()
        kda_slice_names = {"kills", "deaths", "assists"}
        for slice_name in Input.numeric_slices - kda_slice_names:
            input_slice = X[:, Input.indices["start"][slice_name]:Input.indices["end"][slice_name]]
            # diff = Input.indices["half"][slice_name] - Input.indices["start"][slice_name]
            # input_slice_concat = [np.concatenate([input_slice[:, i], input_slice[:, i + diff]],
            #                                      axis=0) for i in
            #                       range(diff)]
            # input_slice_concat = np.transpose(input_slice_concat, (1, 0))
            params[slice_name] = Input.fit(np.reshape(input_slice, (-1,1)), slice_name)

        kda_slices = np.reshape([X[:, Input.indices["start"][slice_name]:Input.indices["end"][slice_name]] for
                               slice_name in kda_slice_names], (-1,1))
        kda_params = Input.fit(kda_slices, slice_name)
        for slice_name in kda_slice_names:
            params[slice_name] = kda_params

        return params


    @staticmethod
    def dict2vec(input_dict):

        x = np.zeros(shape=Input.input_len, dtype=np.float32)
        for slice_name in Input.all_slices:
            x[:, Input.indices["start"][slice_name]:Input.indices["end"][slice_name]] = input_dict[slice_name] if\
                slice_name in input_dict else 0
        return x


        # dragons_int = [0, 0, 0, 0, 0, 0, 0, 0]
        # for dragon_type in dragons:
        #     team_dragon_kills = dragons[dragon_type]
        #     dragon_index = game_constants.dragon2index[dragon_type]
        #     dragons_int[dragon_index] += team_dragon_kills[0]
        #     dragons_int[dragon_index + 4] += team_dragon_kills[1]
        #
        # dragon_soul = [dragon_soul_type[0] != "NONE", dragon_soul_type[1] != "NONE"]
        # dragon_soul_type_ints = [
        #     game_constants.dragon2index[soul_team] if soul_team in game_constants.dragon2index
        #     else 0
        #     for soul_team in dragon_soul_type]
        # dragon_soul_type = [0, 0, 0, 0, 0, 0, 0, 0]
        # if dragon_soul_type_ints[0] != 0:
        #     dragon_soul_type[dragon_soul_type_ints[0]] = 1
        # if dragon_soul_type_ints[1] != 0:
        #     dragon_soul_type[4 + dragon_soul_type_ints[1]] = 1



        # x[Input.indices["start"]["champs"]:Input.indices["end"]["champs"]] = \
        #     [ChampManager().lookup_by("name",chstr)["int"] for
        #                                                                       chstr in
        #                                                                       champs_str]
        # x[Input.indices["start"]["cs"]:Input.indices["end"]["cs"]] = cs
        # x[Input.indices["start"]["lvl"]:Input.indices["end"]["lvl"]] = lvl
        # x[Input.indices["start"]["kda"]:Input.indices["end"]["kda"]] = np.ravel(kda)
        # x[Input.indices["start"]["champs"]:Input.indices["end"]["champs"]] = total_gold
        # x[Input.indices["start"]["baron"]:Input.indices["end"]["baron"]] = baron_active
        # x[Input.indices["start"]["elder"]:Input.indices["end"]["elder"]] = elder_active
        # x[Input.indices["start"]["dragons_killed"]:Input.indices["end"]["dragons_killed"]] = dragons_int
        # x[Input.indices["start"]["dragon_soul_type"]:Input.indices["end"]["dragon_soul_type"]] = \
        #     dragon_soul_type
        # x[Input.indices["start"]["dragon_soul_type_start"]:Input.indices["start"][
        #     "dragon_soul_type_end"]] = \
        #     dragon_soul
        # x[Input.indices["start"]["turrets_destroyed"]:Input.indices["end"]["turrets_destroyed"]] = \
        #     turrets
        # x[Input.indices["start"]["first_team_blue"]:Input.indices["end"]["first_team_blue"]] = \
        #     first_team_blue_start

        return x


    @staticmethod
    def scale(scale_dict):
        result = dict(scale_dict)
        X = np.zeros(Input.len)
        for slice_name in Input.numeric_slices:
            X[Input.indices["start"][slice_name]] = scale_dict[slice_name]
        X = Input().scale_inputs(X[np.newaxis, :])
        for slice_name in Input.numeric_slices:
            result[slice_name] = X[0][Input.indices["start"][slice_name]]
        return result


    @staticmethod
    def flip_teams(X):
        X_copy = np.copy(X)
        for slice in Input.all_slices - Input.nonsymmetric_slices:
            X_copy[:, Input.indices["start"][slice]:Input.indices["half"][slice]] = \
                X[:, Input.indices["half"][slice]:Input.indices["end"][slice]]
            X_copy[:, Input.indices["half"][slice]:Input.indices["end"][slice]] = \
                X[:, Input.indices["start"][slice]:Input.indices["half"][slice]]
        return X_copy


    def __init__(self):
        if not Input.instance:
            Input.instance = Input.__Input()


    def __getattr__(self, name):
        return getattr(self.instance, name)


    class __Input:

        def __init__(self):
            with open(app_constants.asset_paths["input_scales"], "r") as f:
                params = json.load(f)

            self.game_config = None

            self.power_transformers = dict()
            self.standard_scalers = dict()
            self.minmax_scalers = dict()


            for slice_name in Input.numeric_slices:
                slice_len = Input.indices["end"][slice_name] - Input.indices["start"][slice_name]
                # lambdas = np.tile(params[slice_name]["yeo_lambdas"], [slice_len])
                scale_norm = np.tile(params[slice_name]["standard"]["scale"], [slice_len])
                mean = np.tile(params[slice_name]["standard"]["mean"], [slice_len])
                var = np.tile(params[slice_name]["standard"]["var"], [slice_len])
                # min_mm = np.tile(params[slice_name]["minmax"]["min"], [slice_len])
                # scale_mm = np.tile(params[slice_name]["minmax"]["scale"], [slice_len])

                # self.power_transformers[slice_name] = heavy_imports.PowerTransformer(method='yeo-johnson', standardize=False)
                self.standard_scalers[slice_name] = heavy_imports.StandardScaler()
                # self.minmax_scalers[slice_name] = heavy_imports.MinMaxScaler()

                # self.power_transformers[slice_name].lambdas_ = lambdas
                self.standard_scalers[slice_name].mean_ = mean
                self.standard_scalers[slice_name].var_ = var
                self.standard_scalers[slice_name].scale_ = scale_norm
                # self.minmax_scalers[slice_name].min_ = min_mm
                # self.minmax_scalers[slice_name].scale_ = scale_mm


        def scale_inputs(self, X):
            result = np.copy(X).astype(np.float32)
            for slice_name in Input.numeric_slices:
                d = result[:, Input.indices["start"][slice_name]:Input.indices["end"][slice_name]]
                d = np.clip(d, game_constants.min_clip[slice_name], game_constants.max_clip[slice_name])
                # d = self.power_transformers[slice_name].transform(d)
                d = self.standard_scalers[slice_name].transform(d)
                # d = self.minmax_scalers[slice_name].transform(d)
                result[:, Input.indices["start"][slice_name]:Input.indices["end"][slice_name]] = d

            return result
