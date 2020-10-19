import json

import numpy as np

from constants import game_constants, app_constants
from utils import heavy_imports


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
        # this is a nonlinear transformation, afterwards the following equation no longer necessarily holds true:
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


    # slices must be scaled uniformly or else you cannot perform derivative features since the scales are different
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
            params[slice_name] = Input.fit(np.reshape(input_slice, (-1, 1)), slice_name)

        kda_slices = np.reshape([X[:, Input.indices["start"][slice_name]:Input.indices["end"][slice_name]] for
                                 slice_name in kda_slice_names], (-1, 1))
        kda_params = Input.fit(kda_slices, slice_name)
        for slice_name in kda_slice_names:
            params[slice_name] = kda_params

        return params


    @staticmethod
    def dict2vec(input_dict):
        x = np.zeros(shape=Input.len, dtype=np.uint64)
        for slice_name in Input.all_slices:
            if slice_name in input_dict:
                assert np.all(np.array(input_dict[slice_name]) >= 0)
                x[Input.indices["start"][slice_name]:Input.indices["end"][slice_name]] = input_dict[slice_name]
            else:
                x[Input.indices["start"][slice_name]:Input.indices["end"][slice_name]] = 0
        return x


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
            Input.instance = Input.__Input(Input.numeric_slices, Input.indices, Input.len)


    def __getattr__(self, name):
        return getattr(self.instance, name)


    class __Input:

        def get_params_path(self):
            return app_constants.asset_paths["input_scales"]


        def __init__(self, numeric_slices, indices, length):
            with open(self.get_params_path(), "r") as f:
                params = json.load(f)

            self.game_config = None
            self.numeric_slices = numeric_slices
            self.indices = indices
            self.length = length
            self.power_transformers = dict()
            self.standard_scalers = dict()
            self.minmax_scalers = dict()

            for slice_name in self.numeric_slices:
                slice_len = self.indices["end"][slice_name] - self.indices["start"][slice_name]
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
            if X.size == 0:
                return np.empty((1, self.length))
            result = np.copy(X).astype(np.float32)
            for slice_name in self.numeric_slices:
                d = result[:, self.indices["start"][slice_name]:self.indices["end"][slice_name]]
                d = np.clip(d, game_constants.min_clip[slice_name], game_constants.max_clip[slice_name])
                # d = self.power_transformers[slice_name].transform(d)
                d = self.standard_scalers[slice_name].transform(d)
                # d = self.minmax_scalers[slice_name].transform(d)
                result[:, self.indices["start"][slice_name]:self.indices["end"][slice_name]] = d

            return result


class InputWinPred():
    instance = None

    indices = dict()
    indices["start"] = dict()
    indices["half"] = dict()
    indices["end"] = dict()
    indices["mid"] = dict()



    indices["start"]["champs"] = 0
    indices["half"]["champs"] = indices["start"]["champs"] + game_constants.CHAMPS_PER_TEAM
    indices["end"]["champs"] = indices["start"]["champs"] + game_constants.CHAMPS_PER_GAME

    indices["start"]["total_gold"] = indices["end"]["champs"]
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

    indices["start"]["baron"] = indices["end"]["assists"]
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

    indices["start"]["current_health"] = indices["end"]["blue_side"]
    indices["half"]["current_health"] = indices["start"]["current_health"] + game_constants.CHAMPS_PER_TEAM
    indices["end"]["current_health"] = indices["half"]["current_health"] + game_constants.CHAMPS_PER_TEAM

    indices["start"]["max_health"] = indices["end"]["current_health"]
    indices["half"]["max_health"] = indices["start"]["max_health"] + game_constants.CHAMPS_PER_TEAM
    indices["end"]["max_health"] = indices["half"]["max_health"] + game_constants.CHAMPS_PER_TEAM

    indices["start"]["dragon_countdown"] = indices["end"]["max_health"]
    indices["half"]["dragon_countdown"] = indices["start"]["dragon_countdown"] + 1
    indices["end"]["dragon_countdown"] = indices["half"]["dragon_countdown"]

    indices["start"]["baron_countdown"] = indices["end"]["dragon_countdown"]
    indices["half"]["baron_countdown"] = indices["start"]["baron_countdown"] + 1
    indices["end"]["baron_countdown"] = indices["half"]["baron_countdown"]

    indices["start"]["elder_countdown"] = indices["end"]["baron_countdown"]
    indices["half"]["elder_countdown"] = indices["start"]["elder_countdown"] + 1
    indices["end"]["elder_countdown"] = indices["half"]["elder_countdown"]

    indices["start"]["baron_time_left"] = indices["end"]["elder_countdown"]
    indices["half"]["baron_time_left"] = indices["start"]["baron_time_left"] + 1
    indices["end"]["baron_time_left"] = indices["half"]["baron_time_left"]

    indices["start"]["elder_time_left"] = indices["end"]["baron_time_left"]
    indices["half"]["elder_time_left"] = indices["start"]["elder_time_left"] + 1
    indices["end"]["elder_time_left"] = indices["half"]["elder_time_left"]

    indices["start"]["team_odds"] = indices["end"]["elder_time_left"]
    indices["half"]["team_odds"] = indices["start"]["team_odds"] + 1
    indices["end"]["team_odds"] = indices["start"]["team_odds"] + 2

    # competitive 5, wr 2, early_late 7, syn 4, counter 5,
    indices["start"]["champ_wr"] = indices["end"]["team_odds"]
    indices["half"]["champ_wr"] = indices["start"]["champ_wr"] + game_constants.CHAMPS_PER_TEAM*23
    indices["end"]["champ_wr"] = indices["half"]["champ_wr"] + game_constants.CHAMPS_PER_TEAM*23

    len = indices["end"]["champ_wr"]

    numeric_slices = {'total_gold',
                      'cs',
                      'lvl',
                      'kills',
                      'deaths',
                      'assists',
                      'team_odds',
                      'turrets_destroyed',
                      'dragons_killed',
                      # "current_health", "max_health", "baron_countdown", "dragon_countdown","elder_time_left",
                      # "elder_countdown",
                      # "baron_time_left", }
                      }

    all_slices = {"champs", "total_gold", "cs",
                  "lvl", "kills", "deaths", "assists",
                  "baron", "elder", "dragons_killed", "team_odds",
                  "dragon_soul_type", "turrets_destroyed", "blue_side", "current_health", "max_health",
                  "baron_countdown", "dragon_countdown","elder_countdown",
                      "baron_time_left", "elder_time_left", "champ_wr"}

    nonsymmetric_slices = {"baron_countdown", "dragon_countdown","elder_countdown",
                      "baron_time_left", "elder_time_left"}


    # slices must be scaled uniformly or else you cannot perform derivative features since the scales are different
    @staticmethod
    def fit_scale_inputs(X):
        params = dict()
        for slice_name in InputWinPred.numeric_slices:
            input_slice = X[:, InputWinPred.indices["start"][slice_name]:InputWinPred.indices["end"][slice_name]]
            params[slice_name] = Input.fit(np.reshape(input_slice, (-1, 1)), slice_name)
        return params


    @staticmethod
    def scale_abs(scale_dict):
        result = dict(scale_dict)
        X = np.zeros(InputWinPred.len)
        for slice_name in InputWinPred.numeric_slices:
            X[InputWinPred.indices["start"][slice_name]] = scale_dict[slice_name]
        X = InputWinPred().scale_inputs(X[np.newaxis, :])
        for slice_name in InputWinPred.numeric_slices:
            result[slice_name] = X[0][InputWinPred.indices["start"][slice_name]]
        return result


    @staticmethod
    def scale_rel(scale_dict):
        result = dict(scale_dict)
        X = np.zeros(InputWinPred.len)
        for slice_name in InputWinPred.numeric_slices:
            X[InputWinPred.indices["start"][slice_name]] = scale_dict[slice_name]
            X[InputWinPred.indices["start"][slice_name]:InputWinPred.indices["end"][
                slice_name]] = X[InputWinPred.indices["start"][slice_name]:InputWinPred.indices["end"][
                slice_name]] / InputWinPred().standard_scalers[
                                   slice_name].scale_
            result[slice_name] = X[InputWinPred.indices["start"][slice_name]]

        return result

    @staticmethod
    def dict2vec(input_dict):
        x = np.zeros(shape=InputWinPred.len, dtype=np.float32)
        for slice_name in InputWinPred.all_slices:
            if slice_name in input_dict:
                assert np.all(np.array(input_dict[slice_name]) >= 0)
                x[InputWinPred.indices["start"][slice_name]:InputWinPred.indices["end"][slice_name]] = input_dict[slice_name]
            else:
                x[InputWinPred.indices["start"][slice_name]:InputWinPred.indices["end"][slice_name]] = 0
        return x

    @staticmethod
    def flip_teams(X):
        X_copy = np.copy(X)
        for slice in InputWinPred.all_slices - InputWinPred.nonsymmetric_slices:
            X_copy[:, InputWinPred.indices["start"][slice]:InputWinPred.indices["half"][slice]] = \
                X[:, InputWinPred.indices["half"][slice]:InputWinPred.indices["end"][slice]]
            X_copy[:, InputWinPred.indices["half"][slice]:InputWinPred.indices["end"][slice]] = \
                X[:, InputWinPred.indices["start"][slice]:InputWinPred.indices["half"][slice]]
        return X_copy


    def __init__(self):
        if not InputWinPred.instance:
            InputWinPred.instance = InputWinPred.__Input(InputWinPred.numeric_slices, InputWinPred.indices, InputWinPred.len)


    def __getattr__(self, name):
        return getattr(self.instance, name)

    class __Input(Input._Input__Input):

        def get_params_path(self):
            return app_constants.asset_paths["input_scales_winpred"]


        # def input2inputdelta(self, in_vec):
        #     in_vec_cont = []
        #     prev_x = np.zeros((InputWinPred.len))
        #     slices = ["total_gold", "kills", "deaths", "assists", "cs", "lvl", "baron", "elder", "dragons_killed",
        #               "dragon_soul_type", "turrets_destroyed"]
        #     while in_vec.shape[0] > 0:
        #         x = in_vec[0]
        #         cont_x = np.concatenate([x, np.zeros(InputDelta.len - x.shape[0], dtype=np.float64)], axis=0)
        #         for slice in slices:
        #             cont_x[InputDelta.indices["start"][slice + "_diff"]: InputDelta.indices["end"][slice + "_diff"]] = \
        #                 x[InputDelta.indices["start"][slice]: InputDelta.indices["end"][slice]] - \
        #                 prev_x[InputDelta.indices["start"][slice]: InputDelta.indices["end"][slice]]
        #
        #         in_vec_cont.append(cont_x)
        #         prev_x = x
        #         in_vec = in_vec[1:]
        #     return in_vec_cont


def Input2InputWinPred():
    X = np.load("training_data/next_items/processed/elite/sorted/uninf/train_sorted_processed.npz")['arr_0']
    # X = X[::1000]
    result = []
    result_gameids = []
    print("now starting the loop")
    i = 0
    while X.shape[0] > 0:
        row = X[0]
        new_x = np.zeros(InputWinPred.len, dtype=np.float32)
        for slice in InputWinPred.all_slices:
            try:
                new_x[InputWinPred.indices["start"][slice]:InputWinPred.indices["end"][slice]] = \
                    row[Input.indices["start"][slice]:Input.indices["end"][slice]]
            except KeyError:
                continue
        result_gameids.append(row[Input.indices["start"]["gameid"]])

        result.append(new_x)
        X = X[1:]
        # print(X.shape)
        i += 1

    with open("training_data/win_pred/train_elite_winpred.npz", "wb") as writer:
        np.savez_compressed(writer, result)
    with open("training_data/win_pred/train_elite_winpred_gameids.npz", "wb") as writer:
        np.savez_compressed(writer, result_gameids)

# def run_init_fit_scale():
#     X = np.load("training_data/win_pred/train_winpred_odds.npz")['arr_0']
#     params = InputWinPred.fit_scale_inputs(X)
#     with open(app_constants.asset_paths["input_scales_winpred"], "w") as f:
#         f.write(json.dumps(params))

if __name__ == "__main__":
    Input2InputWinPred()
    # run_init_fit_scale()

    # def run_init_fit_scale():
    #         dataloader_elite = SortedNextItemsDataLoader(app_constants.train_paths[
    #                                                                          "next_items_processed_elite_sorted_inf"])
    #         X_elite, _ = dataloader_elite.get_train_data()
    #         params = Input.fit_scale_inputs(X_elite)
    #         with open(app_constants.asset_paths["input_scales_win"], "w") as f:
    #                 f.write(json.dumps(params))


