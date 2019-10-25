[.[] | { "gameId": .gameId,
        "participants": .participants,
        "spells": .spells,
        "itemsTimeline":
        [ .itemsTimeline |
          ([.[] | .timestamp] |
          unique) as $ts |
          range(0,$ts | length) as $index |
          map(select(.timestamp==$ts[$index])) |
          group_by(.participantId) as $eq_ts_grp_by_parts |
            [$eq_ts_grp_by_parts[] |
              map(select(.type=="ITEM_SOLD"))
              +map(select(.type=="ITEM_DESTROYED"))
              +map(select(.type=="ITEM_PURCHASED"))
            ]
        ] | flatten
}]
