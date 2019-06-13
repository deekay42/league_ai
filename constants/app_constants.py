import os

base_asset_path = "assets" + os.sep
asset_paths = dict()
asset_paths["champ_imgs"] = base_asset_path + "champ_imgs" + os.sep
asset_paths["item_imgs"] = base_asset_path + "item_imgs" + os.sep
asset_paths["self_imgs"] = base_asset_path + "self_imgs" + os.sep
asset_paths["jsons"] = base_asset_path + "data" + os.sep
asset_paths["champ_json"] = asset_paths["jsons"] + "champ2id.json"
asset_paths["item_json"] = asset_paths["jsons"] + "item2id.json"
asset_paths["self_json"] = asset_paths["jsons"] + "self2id.json"
asset_paths["spell_json"] = asset_paths["jsons"] + "spell2id.json"

base_train_path = "training_data" + os.sep
train_paths = dict()
train_paths["next_items"] = base_train_path + "next_items" + os.sep
train_paths["positions"] = base_train_path + "positions" + os.sep
train_paths["next_items_processed"] = train_paths["next_items"] + "processed" + os.sep
train_paths["positions_processed"] = train_paths["positions"] + "processed" + os.sep

train_paths["matchids"] = train_paths["next_items"] + "matchids"
train_paths["diamond_league_ids"] = "res" + os.sep +"diamond_league_ids"
train_paths["accountids"] = train_paths["next_items"] + "summoner_account_ids"

train_paths["sorted_matches_path"] = train_paths["next_items"] + "matches_sorted.json"
train_paths["presorted_matches_path"] = train_paths["next_items"] + "matches_presorted.json"

jq_base_path = "jq_scripts" + os.sep
jq_script_names = ["itemUndos_robust", "sortEqualTimestamps", "buildAbsoluteItemTimeline",
                   "extractNextItemsForWinningTeam"]

base_model_path = "models" + os.sep
model_paths = dict()
for label in ["train", "best"]:
    paths = {"base": base_model_path + label + os.sep}
    paths.update({
        "next_items": paths["base"] + "next_items" + os.sep,
        "positions": paths["base"] + "positions" + os.sep})
    paths.update({"imgs": paths["base"] + "imgs" + os.sep})
    paths.update({"champ_imgs": paths["imgs"] + "champs" + os.sep,
                  "item_imgs": paths["imgs"] + "items" + os.sep,
                  "self_imgs": paths["imgs"] + "self" + os.sep})
    model_paths[label] = paths