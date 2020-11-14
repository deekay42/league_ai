#this script inserts a new itemstate if a potion is consumed(ITEM_DESTROYED) or more than 1 potion is bought

def uniq:
  [range(0;length) as $i
   | .[$i] as $x
   | if $i == 0 or $x != .[$i-1] then $x else empty end];

[.[] | { "gameId": .gameId,
        "participants": .participants,
        "spells": [.participants[] | [.spell1Id, .spell2Id]],
        "itemsTimeline": [([.itemsTimeline[]] as $item_transactions |
          [.participants[]] as $participants |
          (($item_transactions | length)-1) as $transactions_len |
          foreach range(0; $transactions_len) as $index
            ([[],[],[],[],[],[],[],[],[],[]];
              $item_transactions[$index] as $current |
              ($current.participantId | tonumber) as $participantsIdKey |
              ([$participants[].participantId] | index($participantsIdKey)) as $key |
              if $current.type=="ITEM_PURCHASED" then
                if $current.itemId==3901 or $current.itemId==3902 or $current.itemId==3903 or $current.itemId==0 or
                $current.itemId>7000 then
                    .
                else
                    .[$key] |= .+ [$current.itemId]
                end
              elif $current.type=="ITEM_DESTROYED" then
                #stopwatch is only being removed if built into upgraded item, otherwise transforms into 2421
                if $current.itemId==2420 or $current.itemId==2421 then
                    .[$key][[2420]][0] as $del_index |
                        if $del_index == null then
                          .
                        else
                          del(.[$key] | .[$del_index])
                        end |
                        .[$key][[2421]][0] as $del_index |
                        if $del_index == null then
                          .
                        else
                          del(.[$key] | .[$del_index])
                        end |
                    if ($item_transactions[$index-1].itemId==3157 or $item_transactions[$index-1].itemId==3193 or
                        $item_transactions[$index-1].itemId==3026) then
                        .
                    else
                        .[$key] |= .+ [2421]
                    end
                elif $current.itemId==3004 or $current.itemId==3003 or $current.itemId==2138 or $current.itemId==2139 or
                 $current.itemId==2140 or $current.itemId==6664 or $current.itemId==6662 or $current.itemId==3068 or
                 $current.itemId==2065 or $current.itemId==4005 or $current.itemId==3190 or $current.itemId==6672 or $current.itemId==6673 or $current.itemId==4636 or $current.itemId==3152 or $current.itemId==6653 or $current.itemId==4633 or $current.itemId==6655 or $current.itemId==6656 or $current.itemId==3078 or $current.itemId==6631 or $current.itemId==6632 or $current.itemId==6630 or $current.itemId==6692 or $current.itemId==6693 or $current.itemId==6691 then
                    .
                else
                    .[$key][[$current.itemId]][0] as $del_index |
                    if $del_index == null then
                      .
                    else
                      del(.[$key] | .[$del_index])
                    end
                end
              elif $current.type=="ITEM_SOLD" then
                if $current.itemId==3040 then
                  .[$key][[3003]][0] as $del_index |
                  if $del_index == null then
                    .
                  else
                    del(.[$key] | .[$del_index])
                  end
                elif $current.itemId==3042 then
                  .[$key][[3004]][0] as $del_index |
                  if $del_index == null then
                    .
                  else
                    del(.[$key] | .[$del_index])
                  end
                else
                  .[$key][[$current.itemId]][0] as $del_index |
                  if $del_index == null then
                    .
                  else
                    del(.[$key] | .[$del_index])
                  end
                end
              else
                .
              end;
              if $index == 0 then
                $item_transactions[0] + {"absolute_items":[[],[],[],[],[],[],[],[],[],[]]},
                $item_transactions[1] + {"absolute_items":.}
              elif $item_transactions[$index].timestamp == $item_transactions[$index+1].timestamp and
                $item_transactions[$index].participantId == $item_transactions[$index+1].participantId and
                $item_transactions[$index+1].type=="ITEM_DESTROYED"
              then
                empty
              else
                if $index < $transactions_len - 2 and $item_transactions[$index+1].timestamp ==
                $item_transactions[$index+2].timestamp and
                    $item_transactions[$index+1].participantId == $item_transactions[$index+2].participantId and
                    $item_transactions[$index+2].type=="ITEM_DESTROYED" then
                     if $item_transactions[$index+2].itemId==2422 then
                        .[([$participants[].participantId] | index(($item_transactions[$index+2].participantId | tonumber)))][[2422]][0] as $item_index |
                        if $item_index == null then
                            .[([$participants[].participantId] | index(($item_transactions[$index+2].participantId | tonumber)))] |= .+ [2422] |
                            $item_transactions[$index+1] + {"absolute_items":.}
                        else
                            $item_transactions[$index+1] + {"absolute_items":.}
                        end
                     elif $item_transactions[$index+2].itemId==2419 then
                        .[([$participants[].participantId] | index(($item_transactions[$index+2].participantId | tonumber)))
                            ] |= .+ [2419] |
                        $item_transactions[$index+1] + {"absolute_items":.}
                     else
                        $item_transactions[$index+1] + {"absolute_items":.}
                     end
                else
                    $item_transactions[$index+1] + {"absolute_items":.}
                end
              end
             )
           )] | uniq
}]
