base_asset_path = "../assets/"
asset_paths = dict()
asset_paths["champ_imgs"] = base_asset_path + "champ_imgs/"
asset_paths["item_imgs"] = base_asset_path + "item_imgs/"
asset_paths["self_imgs"] = base_asset_path + "self_imgs/"
asset_paths["jsons"] = base_asset_path + "data/"
asset_paths["champ_json"] = asset_paths["jsons"] + "champ2id.json"
asset_paths["item_json"] = asset_paths["jsons"] + "item2id.json"
asset_paths["self_json"] = asset_paths["jsons"] + "self2id.json"
asset_paths["spell_json"] = asset_paths["jsons"] + "spell2id.json"

base_train_path = "training_data/"
train_paths = dict()
train_paths["next_items"] = base_train_path + "next_items/"
train_paths["positions"] = base_train_path + "positions/"
train_paths["next_items_processed"] = train_paths["next_items"] + "processed/"
train_paths["positions_processed"] = train_paths["positions"] + "processed/"

train_paths["matchids"] = train_paths["next_items"] + "matchids"
train_paths["diamond_league_ids"] = "res/diamond_league_ids"
train_paths["accountids"] = train_paths["next_items"] + "summoner_account_ids"

train_paths["sorted_matches_path"] = train_paths["next_items"] + "matches_sorted.json"
train_paths["presorted_matches_path"] = train_paths["next_items"] + "matches_presorted.json"

jq_base_path = "jq_scripts/"
jq_script_names = ["itemUndos_robust", "sortEqualTimestamps", "buildAbsoluteItemTimeline",
                   "extractNextItemsForWinningTeam"]

base_model_path = "models/"
model_paths = dict()
for label in ["train", "best"]:
    paths = {"base": base_model_path + label + "/"}
    paths.update({
        "next_items": paths["base"] + "next_items/",
        "positions": paths["base"] + "positions/"})
    paths.update({"imgs": paths["base"] + "imgs/"})
    paths.update({"champ_imgs": paths["imgs"] + "champs/",
                  "item_imgs": paths["imgs"] + "items/",
                  "self_imgs": paths["imgs"] + "self/"})
    model_paths[label] = paths