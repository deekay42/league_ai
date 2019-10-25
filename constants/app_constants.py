import os

base_asset_path = "assets" + os.sep
asset_paths = dict()
asset_paths["champ_imgs"] = base_asset_path + "champ_imgs" + os.sep
asset_paths["item_imgs"] = base_asset_path + "item_imgs_train" + os.sep
asset_paths["self_imgs"] = base_asset_path + "self_indicator_imgs" + os.sep
asset_paths["current_gold_imgs"] = base_asset_path + "current_gold_imgs" + os.sep
asset_paths["kda_imgs"] = base_asset_path + "kda_imgs" + os.sep
asset_paths["cs_imgs"] = base_asset_path + "kda_imgs" + os.sep
asset_paths["lvl_imgs"] = base_asset_path + "kda_imgs" + os.sep
asset_paths["jsons"] = base_asset_path + "data" + os.sep
asset_paths["champ_json"] = asset_paths["jsons"] + "champ2id.json"
asset_paths["item_json"] = asset_paths["jsons"] + "item2id.json"
asset_paths["self_json"] = asset_paths["jsons"] + "self2id.json"
asset_paths["spell_json"] = asset_paths["jsons"] + "spell2id.json"
asset_paths["current_gold_json"] = asset_paths["jsons"] + "current_gold2id.json"
asset_paths["kda_json"] = asset_paths["jsons"] + "kda2id.json"
asset_paths["cs_json"] = asset_paths["jsons"] + "cs2id.json"
asset_paths["lvl_json"] = asset_paths["jsons"] + "lvl2id.json"
asset_paths["xp_table"] = asset_paths["jsons"] + "xp_table.json"
asset_paths["diamond_league_ids"] = asset_paths["jsons"] + os.sep + "diamond_league_ids"

base_train_path = "training_data" + os.sep
train_paths = dict()
train_paths["next_items"] = base_train_path + "next_items" + os.sep
train_paths["positions"] = base_train_path + "positions" + os.sep
train_paths["next_items_processed_unsorted"] = train_paths["next_items"] + "processed" + os.sep + "unsorted" + os.sep
train_paths["next_items_processed_sorted"] = train_paths["next_items"] + "processed" + os.sep + "sorted" + os.sep
train_paths["positions_processed"] = train_paths["positions"] + "processed" + os.sep

train_paths["matchids"] = train_paths["next_items"] + "matchids.json"

train_paths["accountids"] = train_paths["next_items"] + "summoner_account_ids"

train_paths["sorted_matches_path"] = train_paths["next_items"] + "matches_sorted.json"
train_paths["presorted_matches_path"] = train_paths["next_items"] + "matches_presorted.json"

jq_base_path = "jq_scripts" + os.sep
jq_script_names = ["itemUndos_robust", "sortEqualTimestamps.jq", "buildAbsoluteItemTimeline.jq",
                   "extractNextItemsForWinningTeam.jq"]

base_model_path = "models" + os.sep
model_paths = dict()
for label in ["train", "best"]:
    paths = {"base": base_model_path + label + os.sep}
    paths.update({
        "next_items_early": paths["base"] + "next_items" + os.sep + "early" + os.sep,
        "next_items_late": paths["base"] + "next_items" + os.sep + "late" + os.sep,
        "positions": paths["base"] + "positions" + os.sep})
    paths.update({"imgs": paths["base"] + "imgs" + os.sep})
    paths.update({"champ_imgs": paths["imgs"] + "champs" + os.sep,
                  "item_imgs": paths["imgs"] + "items" + os.sep,
                  "self_imgs": paths["imgs"] + "self" + os.sep,
                  "kda_imgs": paths["imgs"] + "kda" + os.sep,
                  "cs_imgs": paths["imgs"] + "cs" + os.sep,
                  "lvl_imgs": paths["imgs"] + "lvl" + os.sep,
                  "current_gold_imgs": paths["imgs"] + "current_gold" + os.sep})
    model_paths[label] = paths

