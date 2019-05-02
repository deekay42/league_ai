import copy
import random
from collections import Counter


def get_item_score(comp, curr_used):
    total_discount = 0
    for i in curr_used:
        # curr_used_score += i.tier
        total_discount += i.gold.total
    if total_discount == comp.gold.total:
        return 0
    else:
        return comp.gold.total / (comp.gold.total - total_discount)


# this method is not deterministic. there may be multiple paths to a given build
def _build_path(prev_avail_items, next_i, abs_items):
    if next_i.id in prev_avail_items:
        return [], [next_i], [], prev_avail_items

    # what components is next_i made of?
    l = next_i.name
    comps_l = next_i.builds_from
    comps = Counter(list(comps_l))
    comps_cpy = Counter([item.id for item in comps_l])

    # next_i doesn't have any subcomponents
    if not comps:
        return [next_i], [], [Counter([next_i.id]) + copy.deepcopy(abs_items[-1])], prev_avail_items

    result_buy_seq, result_ex_i_used, to_remove, result_abs, best_prev_avail_items_result = [], [], [], [], []
    while comps:
        max_comp_score = -1
        max_ex_i_used = buy_seq = best_next_cmp = best_abs = prev_avail_items_result = None

        for comp in comps.elements():
            curr_seq, curr_used, curr_abs, prev_avail_items_result = _build_path(copy.deepcopy(prev_avail_items), comp,
                                                                                 abs_items)

            # is the current component closest to completion?
            comp_score = get_item_score(comp, curr_used)
            if comp_score > max_comp_score or comp_score == max_comp_score and random.random() > 0.5:
                max_comp_score = comp_score
                max_ex_i_used = curr_used
                buy_seq = curr_seq
                best_next_cmp = comp
                best_abs = curr_abs
                best_prev_avail_items_result = prev_avail_items_result

        # the component closest to completion has been found
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
    occ = prev_avail_items.count(next_item.id)
    prev_avail_items = Counter(prev_avail_items)
    if occ:
        del prev_avail_items[next_item.id]
    result_buy_seq, result_ex_i_used, result_abs, prev_avail_items = _build_path(prev_avail_items, next_item, [
        copy.deepcopy(Counter(prev_avail_items))])
    prev_avail_items[next_item.id] += occ
    result_abs = [list(item_state.elements()) + [next_item.id] * occ for item_state in result_abs]
    return result_buy_seq, result_ex_i_used, result_abs, prev_avail_items
