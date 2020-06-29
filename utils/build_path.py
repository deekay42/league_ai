import copy
import random
from collections import Counter
import itertools
from utils.artifact_manager import ItemManager
import numpy as np
# from utils import cass_configured as cass
# from cassiopeia.core.staticdata import Item
from utils.misc import itemslots_left, iditem2intitems

from utils import heavy_imports

# def get_item_score(comp, curr_used, current_gold=None):
#     total_discount = 0
#     for i in curr_used:
#         # curr_used_score += i.tier
#         total_discount += i.gold.total
#     if total_discount == comp.gold.total:
#         return 0
#     else:
#         return comp.gold.total / (comp.gold.total - total_discount)

class InsufficientGold(Exception):
    def __init__(self, item):
        self.item = item

class NoPathFound(Exception):
    def __init__(self, item):
        self.item = item

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
        return [next_i], [], [Counter([next_i.id]) + Counter(abs_items[-1])], prev_avail_items

    result_buy_seq, result_ex_i_used, to_remove, result_abs, best_prev_avail_items_result = [], [], [], [], []
    while comps:
        max_comp_score = -1
        max_ex_i_used = buy_seq = best_next_cmp = best_abs = prev_avail_items_result = None

        for comp in comps.elements():
            curr_seq, curr_used, curr_abs, prev_avail_items_result = _build_path(Counter(prev_avail_items), comp,
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
            abs_items = [Counter(i) for i in result_abs]
            prev_avail_items = best_prev_avail_items_result
        comps = comps - Counter([best_next_cmp])

    result_buy_seq += [next_i]
    if not result_abs:
        result_abs += [Counter(abs_items[-1])]
    else:
        result_abs += [Counter(result_abs[-1])]
    result_abs[-1] = result_abs[-1] + Counter([next_i.id])
    result_abs[-1] = result_abs[-1] - comps_cpy

    return result_buy_seq, result_ex_i_used, result_abs, prev_avail_items


def build_path_nogold(prev_avail_items, next_item):
    occ = prev_avail_items.count(next_item.id)
    prev_avail_items = Counter(prev_avail_items)
    if occ:
        del prev_avail_items[next_item.id]
    result_buy_seq, result_ex_i_used, result_abs, prev_avail_items = _build_path(prev_avail_items, next_item, [
        Counter(prev_avail_items)])
    prev_avail_items[next_item.id] += occ
    result_abs = [list(item_state.elements()) + [next_item.id] * occ for item_state in result_abs]
    return result_buy_seq, result_ex_i_used, result_abs, prev_avail_items


def get_item_score(comp, curr_used):
    total_discount = 0
    for i in curr_used:
        # curr_used_score += i.tier
        total_discount += i.gold.total
    if total_discount == comp.gold.total:
        return 0
    else:
        return comp.gold.total / (comp.gold.total - total_discount)


def flatten(arr):

  for i in arr:
    if isinstance(i, tuple):
      yield from flatten(i)
    else:
      yield i


# def build_item_tree(item):
#     c = item.name
#     comps = list(item.builds_from)
#     if not comps:
#         yield item
#     else:
#         for comp in comps:
#             yield from build_item_tree(comp)
#         yield item

#
# def build_cost(item, existing_items):
#
#     result_buy_seq, result_ex_i_used, result_abs, prev_avail_items = _build_path(existing_items.copy(), item, [
#         Counter(existing_items)])
#     cost = 0
#     full_ex_i_tree = Counter()
#     for ex_i, qty in existing_items.items():
#         for _ in range(qty):
#             full_ex_i_tree += Counter(full_item_trees[ex_i])
#     comps = Counter(full_item_trees[item.id])
#     cost_counter = comps - full_ex_i_tree
#     # items_used = full_ex_i_tree.keys() & comps.keys()
#     # items_used = Counter({i:comps[i] for i in items_used})
#     items_used = comps - cost_counter
#     items_used_deflated = items_used.copy()
#     for i in items_used:
#         inf_i = Counter(full_item_trees[i])
#         inf_i -= Counter({i:1})
#         items_used_deflated -= inf_i
#
#
#     for item, qty in cost_counter.items():
#         cost += cass.Item(id=item, region="EUW").gold.base * qty
#     return cost, items_used
#
#
# def generate_abs_items(ex_items, next_item):
#     if next_item.id in ex_items:
#         return None
#     else:
#         comps = next_item.builds_from
#         if not comps:
#             return ex_items + next_item.id
#         else:
#             for comp in comps:
#                 if comp.id not in ex_items:
#                     next_abs_items = generate_abs_items(ex_items, comp)
#                     next_abs_items_last = next_abs_items[-1].copy()
#                     comp_comps = Counter([item.id for item in comp.builds_from])
#                     next_abs_items_last -= comp_comps
#                     return next_abs_items + next_abs_items_last
#
#


# print("lulz")

# full_item_trees = dict()
# def brute_force_build_paths(item, existing_items):
#     c = item.name
#     comps = list(item.builds_from)
#     if not comps:
#         if item.name not in full_item_trees:
#             full_item_trees[item.id] = [item.id]
#         return [item.id, None]
#     comps_build_paths = [brute_force_build_paths(comp, existing_items) for comp in comps if comp.id not in
#                          existing_items]
#     if item.id not in full_item_trees:
#         all_comp_trees = []
#         for comp in comps:
#             all_comp_trees.extend(full_item_trees[comp.id])
#         full_item_trees[item.id] = all_comp_trees + [item.id]
#
#     result = list(itertools.product(*comps_build_paths)) + [item.id]
#     return result

def brute_force_build_paths(item, existing_items):
    c = item.name
    comps = list(item.builds_from)

    if not comps:
        return [item.id, None]
    comps_counter = Counter([comp.id for comp in comps])
    intersec = comps_counter & existing_items
    comps_counter -= intersec
    existing_items -= intersec
    comps_build_paths = [brute_force_build_paths(heavy_imports.Item(id=compid, region="EUW"), existing_items) for compid,
                                                                                                    qty in comps_counter.items() for _
                         in range(qty)]
    result = list(itertools.product(*comps_build_paths)) + [item.id]
    return result


def build_path_nogold_v2(prev_avail_items, next_item):
    occ = prev_avail_items[next_item.id]
    if occ:
        del prev_avail_items[next_item.id]
    result_buy_seq, result_ex_i_used, result_abs, prev_avail_items = _build_path(prev_avail_items, next_item, [
        Counter(prev_avail_items)])
    prev_avail_items += Counter({next_item.id:occ})
    result_abs = [Counter(item_state.elements()) + Counter({next_item.id: occ}) for item_state in result_abs]
    return result_buy_seq, result_ex_i_used, result_abs, prev_avail_items



def score_build_path(build_path, existing_items, current_gold):
    existing_items_copy = Counter(existing_items)
    result_buy_seq, result_ex_i_used, result_abs, prev_avail_items = [], [], [existing_items_copy.copy()], []
    already_used_items = Counter()
    already_used_items_add_to_abs = [Counter()]
    score = 0

    for item in build_path:
        if not item:
            continue
        cass_item = heavy_imports.Item(id=item, region="EUW")
        # if item in existing_items_copy:
        #     result_abs[-1] -= Counter({item:1})
        #     already_used_items += Counter({item:1})
        #     already_used_items_add_to_abs += [already_used_items.copy()]
        #     existing_items_copy -= Counter({item:1})
        # else:
        buy_seq, ex_i_used, abs_items, prev_avail_items = build_path_nogold_v2(result_abs[-1].copy(), cass_item)
        result_buy_seq += buy_seq
        existing_items_copy -= Counter([i.id for i in ex_i_used])
        result_abs += abs_items
        already_used_items_add_to_abs += [already_used_items.copy()] * len(abs_items)

        item_buy_cost = 0
        for bi in buy_seq:
            item_buy_cost += bi.gold.base
        #
        #
        # item_recipe_cost, items_used = build_cost(cass_item, existing_items)
        # existing_items_copy -= items_used
        current_gold -= item_buy_cost


        multiplier = 1 + (cass_item.tier-1)/10
        # jg items must be finished first before enchantment
        if item==3706 or item==3715 or item==1400 or item==1401 or item==1402 or item==1412 or item==1413 or \
                item==1414 or item==1416 or item==1419 or item==1039 or item==1041:
            multiplier *= 2
        score += cass_item.gold.total * multiplier

    score += sum([heavy_imports.Item(id=item, region="EUW").gold.total for item in existing_items_copy])
    if current_gold >= 0:
        return score, result_buy_seq, [ai+aai for ai, aai in zip(result_abs, already_used_items_add_to_abs)]
    else:
        return current_gold, result_buy_seq, [ai+aai for ai, aai in zip(result_abs, already_used_items_add_to_abs)]

# def is_super_path(path, current_items, final_item):
#     full_comp_tree = Counter()
#     for item in path:
#         full_comp_tree += Counter(full_item_trees[item])

#     for current_item, qty in current_items.items():
#         if current_item not in full_item_trees[final_item.id]:
#             continue
#         elif current_item in full_comp_tree:
#             full_comp_tree -= Counter({current_item:qty})
#     else:
#         return full_comp_tree != Counter({})


def normalize_bps(bps):
    bps = [list(flatten((path,))) for path in bps]
    bps = [[i for i in bp if i] for bp in bps]
    bps = [bp for bp in bps if bp]
    bps = [sorted(bp, key=lambda a: heavy_imports.Item(id=a, region="EUW").gold.total, reverse=True) for bp in bps]
    try:
        return np.unique(bps, axis=0).tolist()
    except:
        return np.unique(bps).tolist()


def build_path_for_gold(item, current_items, gold):
    bps = brute_force_build_paths(item, current_items.copy())
    bps = normalize_bps(bps)
    scores = [score_build_path(path, current_items, gold) for path in bps]
    scores = [score for score in scores if itemslots_left(iditem2intitems(score[2][-1])) >= 0]
    if not scores:
        # raise NoPathFound(item)
        return [], []
    if np.all(np.array(scores)[:,0]<0):
        # raise InsufficientGold(item)
        return [], []
    max_score = -9999999
    max_score_index = -1
    for i, c_score in enumerate(scores):
        if c_score[0] > max_score:
            max_score_index = i
            max_score = c_score[0]
    score, buy_seq, abs_items = scores[max_score_index]

    return buy_seq, abs_items[1:]

# lol = build_path_for_gold(cass.Item(id=3153, region="EUW"), Counter({1036:1, 3133:2, 1042:1, 1043:1}), 1450)
# bc = build_cost(cass.Item(id=1401, region="EUW"), Counter({1039:1}))
