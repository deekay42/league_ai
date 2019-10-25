#does NOT detect multiple items unfortunately
#EDIT: now it does

def set_diff_multi($b):
    if $b==null or $b==[] then
        .
    else

        [foreach range(0;$b | length) as $b_index
        ( .;
        .[[$b[$b_index]]][0] as $del_index |
                        if $del_index == null then
                          .
                        else
                          del(.[$del_index])
                        end
                        ;
        .)][-1] | if . == null then [] else . end
    end
;


def winningTeamNextItems:
  . as $in
  | length as $length
  | [foreach range(length-2; -1; -1) as $index
    ([[],[],[],[],[]];
      [range(0;5) as $playerIndex |
      ($in[$index+1][$playerIndex] | set_diff_multi($in[$index][$playerIndex])) as $item_diff |
      if ($item_diff | length) >1 then
        $item_diff | debug | $item_diff + 5
      elif $item_diff == [] then
        .[$playerIndex]
      else
        $item_diff[0]
      end];
      .)] | reverse
;

[.[] | { "gameId": .gameId,
        "winningTeamNextItems": ([.itemsTimeline[] | .absolute_items] | winningTeamNextItems)}
]
