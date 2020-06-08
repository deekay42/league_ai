import os
import platform

base_asset_path = "assets" + os.sep
train_imgs_path = base_asset_path + "train_imgs" + os.sep
asset_paths = dict()



asset_paths["champs"] = train_imgs_path + "champs" + os.sep
asset_paths["items"] = train_imgs_path + "items" + os.sep
asset_paths["spells"] = train_imgs_path + "self" + os.sep
asset_paths["self"] = train_imgs_path + "self" + os.sep
asset_paths["current_gold"] = train_imgs_path + "current_gold" + os.sep
asset_paths["kda"] = train_imgs_path + "kda" + os.sep
asset_paths["cs"] = train_imgs_path + "cs" + os.sep
asset_paths["lvl"] = train_imgs_path + "lvl" + os.sep
asset_paths["jsons"] = base_asset_path + "data" + os.sep
asset_paths["champs_json"] = asset_paths["jsons"] + "champ2id.json"
asset_paths["items_json"] = asset_paths["jsons"] + "item2id.json"
asset_paths["self_json"] = asset_paths["jsons"] + "self2id.json"
asset_paths["spells_json"] = asset_paths["jsons"] + "spell2id.json"
asset_paths["current_gold_json"] = asset_paths["jsons"] + "current_gold2id.json"
asset_paths["kda_json"] = asset_paths["jsons"] + "kda2id.json"
asset_paths["cs_json"] = asset_paths["jsons"] + "cs2id.json"
asset_paths["lvl_json"] = asset_paths["jsons"] + "lvl2id.json"
asset_paths["xp_table"] = asset_paths["jsons"] + "xp_table.json"
asset_paths["diamond_league_ids"] = asset_paths["jsons"] + os.sep + "diamond_league_ids"
if platform.system() == "Windows":
    asset_paths["tesseract_tmp_files"] = os.path.join(os.getenv('LOCALAPPDATA'),"League IQ", "tesseract")
    asset_paths["tesseract_list_file"] = os.path.join(asset_paths["tesseract_tmp_files"], "list")
    tess_path = "tessdata"
else:
    asset_paths["tesseract_list_file"] = base_asset_path + "tesseract/list.txt"
    asset_paths["tesseract_tmp_files"] = base_asset_path + "tesseract/"
    tess_path = "/usr/local/Cellar/tesseract/4.1.0/share/tessdata/"
asset_paths["tesseract_separator"] = base_asset_path + "tesseract/sep.png"



base_train_path = "training_data" + os.sep
train_paths = dict()
train_paths["next_items"] = base_train_path + "next_items" + os.sep
train_paths["win_pred"] = base_train_path + "win_pred" + os.sep
train_paths["positions"] = base_train_path + "positions" + os.sep
train_paths["next_items_processed_elite_unsorted_inf"] = train_paths["next_items"] + "processed" + os.sep + "elite" + \
                                                         os.sep + "unsorted" + \
                                                   os.sep + "inf" + os.sep
train_paths["next_items_processed_elite_unsorted_uninf"] = train_paths["next_items"] + "processed" + os.sep + "elite"\
                                                           + os.sep + \
                                                           "unsorted" + \
                                                   os.sep + "uninf" + os.sep
train_paths["next_items_processed_elite_unsorted_complete"] = train_paths["next_items"] + "processed" + os.sep + \
                                                              "elite" + os.sep + "unsorted" + \
                                                   os.sep + "complete" + os.sep
train_paths["next_items_processed_lower_unsorted_inf"] = train_paths["next_items"] + "processed" + os.sep + "lower" + \
                                                         os.sep + "unsorted"\
                                                        + \
                                                   os.sep + "inf" + os.sep
train_paths["next_items_processed_lower_unsorted_uninf"] = train_paths["next_items"] + "processed" + os.sep + "lower"\
                                                           + os.sep + "unsorted" + \
                                                   os.sep + "uninf" + os.sep
train_paths["next_items_processed_lower_unsorted_complete"] = train_paths["next_items"] + "processed" + os.sep + \
                                                              "lower" + os.sep + "unsorted" + \
                                                   os.sep + "complete" + os.sep

train_paths["next_items_processed_elite_sorted_inf"] = train_paths["next_items"] + "processed" + os.sep + "elite" + \
                                                     os.sep + \
                                                 "sorted" + os.sep\
                                                 + "inf" + os.sep
train_paths["next_items_processed_elite_sorted_uninf"] = train_paths["next_items"] + "processed" + os.sep + "elite" + os.sep\
                                                   + "sorted" + \
                                                   os.sep + "uninf" + os.sep
train_paths["next_items_processed_elite_sorted_complete"] = train_paths["next_items"] + "processed" + os.sep + "elite" + \
                                                      os.sep + "sorted" + \
                                                   os.sep + "complete" + os.sep
train_paths["next_items_processed_lower_sorted_inf"] = train_paths["next_items"] + "processed" + os.sep + "lower" + \
                                                     os.sep + \
                                                 "sorted" + os.sep\
                                                 + "inf" + os.sep
train_paths["next_items_processed_lower_sorted_uninf"] = train_paths["next_items"] + "processed" + os.sep + "lower" + os.sep\
                                                   + "sorted" + \
                                                   os.sep + "uninf" + os.sep
train_paths["next_items_processed_lower_sorted_complete"] = train_paths["next_items"] + "processed" + os.sep + "lower" + \
                                                      os.sep + "sorted" + \
                                                   os.sep + "complete" + os.sep

train_paths["positions_processed"] = train_paths["positions"] + "processed" + os.sep
train_paths["positions_to_be_pred"] = train_paths["positions"] + "to_be_pred" + os.sep

train_paths["elite_matchids"] = train_paths["next_items"] + "elitematchids"
train_paths["lower_matchids"] = train_paths["next_items"] + "lowermatchids"

train_paths["accountids"] = train_paths["next_items"] + "summoner_account_ids"

train_paths["sorted_matches_path"] = train_paths["next_items"] + "matches_sorted.json"
train_paths["presorted_matches_path"] = train_paths["next_items"] + "matches_presorted.json"

asset_paths["champ_vs_roles"] = asset_paths["jsons"] + "champ_vs_roles.json"
asset_paths["my_champs_embeddings"] = asset_paths["jsons"] + "my_champ_embs_normed.npy"
asset_paths["opp_champs_embeddings"] = asset_paths["jsons"] + "opp_champ_embs_normed.npy"


jq_base_path = "jq_scripts" + os.sep
jq_script_names = ["itemUndos_robust", "sortEqualTimestamps.jq", "buildAbsoluteItemTimeline.jq",
                   "extractNextItemsForWinningTeam.jq"]

base_model_path = "models" + os.sep
model_paths = dict()
for label in ["train", "best"]:
    paths = {"base": base_model_path + label + os.sep}
    paths.update({
        "next_items_standard": paths["base"] + "next_items" + os.sep + "standard" + os.sep,
        "next_items_late": paths["base"] + "next_items" + os.sep + "late" + os.sep,
        "next_items_starter": paths["base"] + "next_items" + os.sep + "starter" + os.sep,
        "next_items_first_item": paths["base"] + "next_items" + os.sep + "first_item" + os.sep,
        "next_items_boots": paths["base"] + "next_items" + os.sep + "boots" + os.sep,
        "win_pred_standard": paths["base"] + "win_pred" + os.sep + "standard" + os.sep,
        "win_pred_init": paths["base"] + "win_pred" + os.sep + "init" + os.sep,
        "positions": paths["base"] + "positions" + os.sep})
    paths.update({"imgs": paths["base"] + "imgs" + os.sep})
    paths.update({"champs": paths["imgs"] + "champs" + os.sep,
                  "items": paths["imgs"] + "items" + os.sep,
                  "self": paths["imgs"] + "self" + os.sep,
                  "kda": paths["imgs"] + "kda" + os.sep,
                  "cs": paths["imgs"] + "cs" + os.sep,
                  "lvl": paths["imgs"] + "lvl" + os.sep,
                  "current_gold": paths["imgs"] + "current_gold" + os.sep})
    model_paths[label] = paths

