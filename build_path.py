import cassiopeia as cass
import json
import copy
import random
from collections import Counter
import sys

MAX_ITEMS_PER_SUMM = 6

config = cass.get_default_config()
config['pipeline']['ChampionGG'] = {
    "package": "cassiopeia_championgg",
    "api_key": "496b4c2f287421a51a41aeb51a808f74"  # Your api.champion.gg API key (or an env var containing it)
  }

config['pipeline']['RiotAPI'] = {
    "api_key": "RGAPI-39e77a14-355a-4826-8cfb-6f5cc68bdc7c",
    "limiting_share": 1.0,
    "request_error_handling": {
        "404": {
            "strategy": "throw"
        },
        "429": {
            "service": {
                "strategy": "exponential_backoff",
                "initial_backoff": 1.0,
                "backoff_factor": 2.0,
                "max_attempts": 4
            },
            "method": {
                "strategy": "retry_from_headers",
                "max_attempts": 5
            },
            "application": {
                "strategy": "retry_from_headers",
                "max_attempts": 5
            }
        },
        "500": {
            "strategy": "exponential_backoff",
            "initial_backoff": 1.0,
            "backoff_factor": 2.0,
            "max_attempts": 4
        },
        "503": {
            "strategy": "exponential_backoff",
            "initial_backoff": 1.0,
            "backoff_factor": 2.0,
            "max_attempts": 4
        },
        "timeout": {
            "strategy": "exponential_backoff",
            "initial_backoff": 1.0,
            "backoff_factor": 2.0,
            "max_attempts": 4
        },
        "403": {
            "strategy": "throw"
        }
    }
  }

cass.set_default_region("KR")
cass.apply_settings(config)

def list_diff(first, second):
    diff = Counter()
    for item in first:
        diff[item] += 1
    for item in second:
        diff[item] -= 1
    diff = list(diff.elements())
    assert len(diff) <= 1
    if diff == []:
        return []
    else:
        return diff[0]

def main():
    if len(sys.argv) != 2:
        print("specify filename")
        exit(-1)
    filename = sys.argv[1]
    print("loading input file")
    with open(filename) as f:
        content = json.load(f)
    print("loading input file complete")
    with open(filename+"_inflated", "w") as f:
        f.write('[')
        for i, game in enumerate(content):
            print("{0:.0%} complete".format(i/len(content)))
            out_itemsTimeline = []
            if i > 0:
                f.write(',')
            prev_state = copy.deepcopy(game['itemsTimeline'][0])
            out_itemsTimeline += [copy.deepcopy(prev_state)]
            for item_state in game['itemsTimeline']:
                summ_index = -1
                for summ_items, prev_summ_items in zip(item_state, prev_state):
                    summ_index += 1
                    if summ_items == prev_summ_items:
                        continue
                    else:
                        new_item = list_diff(summ_items, prev_summ_items)
                        if new_item:
                            new_item = cass.Item(id=int(new_item), region="KR")
                            l = new_item.name
                            insert_item_states = build_path(prev_state[summ_index], new_item)
                            if len(insert_item_states) > 1:
                                for summ_item_state in insert_item_states:
                                    if len(summ_item_state) <= MAX_ITEMS_PER_SUMM:
                                        out_itemsTimeline += [copy.deepcopy(prev_state)]
                                        out_itemsTimeline[-1][summ_index] = summ_item_state
                            else:
                                out_itemsTimeline += [item_state]
                            break
                        else:
                            out_itemsTimeline += [item_state]
                prev_state = copy.deepcopy(item_state)
            f.write(json.dumps({"gameId": game['gameId'], "participants": game['participants'],
                                "itemsTimeline": out_itemsTimeline}))
            f.write("\n")
            f.flush()
        f.write(']')

def get_item_score(comp, curr_used):
    total_discount = 0
    for i in curr_used:
        # curr_used_score += i.tier
        total_discount += i.gold.total
    if total_discount == comp.gold.total:
        return 0
    else:
        return comp.gold.total / (comp.gold.total - total_discount)

def _build_path(prev_avail_items, next_i, abs_items):
    if next_i.id in prev_avail_items:
        return [], [next_i], [], prev_avail_items
    l = next_i.name
    comps_l = next_i.builds_from
    comps = Counter(list(comps_l))
    comps_cpy = Counter([item.id for item in comps_l])
    if not comps:
        return [next_i], [], [Counter([next_i.id]) + copy.deepcopy(abs_items[-1])], prev_avail_items
    result_buy_seq, result_ex_i_used, to_remove, result_abs, best_prev_avail_items_result = [], [], [], [], []
    while comps:
        max_comp_score = -1
        max_ex_i_used = buy_seq = best_next_cmp = best_abs = prev_avail_items_result = None
        for comp in comps.elements():
            curr_seq, curr_used, curr_abs, prev_avail_items_result = _build_path(copy.deepcopy(prev_avail_items), comp, abs_items)
            # curr_used_score = len(curr_used)
            comp_score = get_item_score(comp, curr_used)
            if comp_score > max_comp_score or comp_score == max_comp_score and random.random() > 0.5:
                max_comp_score = comp_score
                max_ex_i_used = curr_used
                buy_seq = curr_seq
                best_next_cmp = comp
                best_abs = curr_abs
                best_prev_avail_items_result = prev_avail_items_result
        result_buy_seq += buy_seq
        result_ex_i_used += max_ex_i_used
        # we already own this component. now it's being used in a recipe, so delete it from our items list
        if not best_abs:
            prev_avail_items = prev_avail_items - Counter([best_next_cmp.id])
        else:
            result_abs += best_abs
            abs_items = copy.deepcopy(result_abs)
            prev_avail_items = best_prev_avail_items_result
        comps = comps - Counter([best_next_cmp])

    result_buy_seq += [next_i]
    if not result_abs:
        result_abs += [copy.deepcopy(abs_items[-1])]
    else:
        result_abs += [copy.deepcopy(result_abs[-1])]
    result_abs[-1] = result_abs[-1] + Counter([next_i.id])
    result_abs[-1] = result_abs[-1] - comps_cpy

    return result_buy_seq, result_ex_i_used, result_abs, prev_avail_items


def build_path(prev_avail_items, next_item):
    result_buy_seq, result_ex_i_used, result_abs, prev_avail_items = _build_path(Counter(prev_avail_items), next_item, [copy.deepcopy(Counter(prev_avail_items))])
    result_abs = [list(item_state.elements()) for item_state in result_abs]
    return result_buy_seq, result_ex_i_used, result_abs, prev_avail_items

# while True:
#     new_item = cass.Item(id=3078, region="KR")
#     j = build_path([], new_item)
#     print("lololo")

if __name__ == "__main__":
    main()
