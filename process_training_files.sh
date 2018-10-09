<matches jq -f ../jq/itemUndos_robust >undone1 &&
<undone1 jq -f ../jq/sortEqualTimestamps >undone_1_sorted && <undone_1_sorted jq -f ../jq/buildAbsoluteItemTimeline >absolute_1 && python ../getRoles/inflate_items absolute_1 && <absolute_1_inflated jq -f ../jq/extractNextItemsForWinningTeam >next_1 && python process_training_data.py absolute_1_inflated next_1
