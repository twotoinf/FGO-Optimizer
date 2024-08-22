# encoding = utf-8
# v0.3
# 1. 支持配置任意数量加成
# 2. 支持选择倾向的一个加成配置
import numpy as np
from scipy.optimize import linprog
from typing import List
from input import *
CARD_UPGRADE_N, FULL_REWARD=5, 2

class GainVec():
    def __init__(self, gain:np.ndarray, space:np.ndarray):
        assert(len(gain.shape) == 1)
        assert(gain.shape == space.shape)
        self.gain = gain
        self.space = space
    def __add__(self, others):
        return GainVec(self.gain + others.gain, self.space + others.space)
    def __mul__(self, set_indx:np.ndarray):
        return GainVec(self.gain * set_indx, self.space * set_indx)
    def __len__(self) -> int:
        return len(self.gain)

def generate_vecs_my(my_cards_limit_full:np.ndarray, my_cards_limit_un:np.ndarray) -> List[GainVec]:
    my_cards_num_limit = my_cards_limit_full + my_cards_limit_un
    vecs_my = []
    for i in range(POS_LIMIT_OWN + 1):
        if (i > my_cards_num_limit[0]):
            break
        for j in range(POS_LIMIT_OWN + 1):
            if j > my_cards_num_limit[1]:
                break
            k = POS_LIMIT_OWN - i - j
            if k > my_cards_num_limit[2]:
                continue
            if k < 0:
                break
            space = np.array([i ,j, k])
            set_idx = (space <= my_cards_limit_full)
            gain = space + space * set_idx + (my_cards_limit_full + 
                                  (my_cards_limit_full + my_cards_limit_un - space) // (CARD_UPGRADE_N - 1)
                                  ) * (1 - set_idx)
            vecs_my.append(GainVec(gain, space))
    return vecs_my

def generate_vecs(vecs_my, support:GainVec) -> List[GainVec]:
    vecs_raw = []
    set_indx = np.array([0] * len(support))
    for vec_my in vecs_my:
        for i in range(len(support)):            
            set_indx[i] = 1
            vecs_raw.append(vec_my + support * set_indx)
            set_indx[i] = 0

    vecs = []
    for i in range(len(vecs_raw)):
        flag = True
        for j in range(len(vecs_raw)):
            if (i == j): continue
            if ((vecs_raw[i].gain <= vecs_raw[j].gain).all()):
                flag = False
                break
        if flag:
            vecs.append(vecs_raw[i])
    return vecs


def optimize(vecs,
             a,
             b,
             now_demond,
             w_bias = 0,
             gain_bias = None
             ):
    T = np.array([vec.gain for vec in vecs]).T
    A = np.ones_like(T)
    B = np.hstack([np.diag(a[i]) @ A + np.diag(b[i]) @ T
        for i in range(a.shape[0])      ])    

    if w_bias != 0:
        # bias 
        for i in range(len(vecs)):
            if (vecs[i].gain == gain_bias).all():
                pos = i
                break
        w = [1] * len(vecs)
        w[pos] = 1 - w_bias
        weights =  np.array(w * a.shape[0])
    else:
        weights = np.ones(len(vecs)*a.shape[0])

    res = linprog(c = weights, 
                  A_ub = -B, 
                  b_ub = -now_demond, 
                  method = "highs", 
                  integrality = np.ones_like(weights))
    return res

if __name__ == "__main__":
    # convert form
    my_cards_limit_full, my_cards_limit_un = np.array(my_cards_limit_full), np.array(my_cards_limit_un)
    support_available, full_support_available = np.array(support_available), np.array(full_support_available)
    a = np.array(a)
    b = np.array(b)
    all_demond = np.array(all_demond)
    now_own = np.array(now_own)   
    if (len(a.shape) == 1): 
        a = a.reshape(1, -1)
    if (len(b.shape) == 1): 
        b = b.reshape(1, -1)
    if "POS_LIMIT_OWN" not in locals():
        POS_LIMIT_OWN = 5 
    if "gain_bias" in locals():
        gain_bias = np.array(gain_bias)

    # check
    assert((my_cards_limit_full >= 0).all())
    assert((my_cards_limit_un >= 0).all())
    assert(my_cards_limit_full.dtype == int)
    assert(my_cards_limit_un.dtype == int)
    assert(((support_available == 0) + (support_available == 1)).all())
    assert(((full_support_available == 0) + (full_support_available == 1)).all())
    assert(type(POS_LIMIT_OWN) == int)
    assert(0<=POS_LIMIT_OWN<=5)
    assert((a >= 0).all())
    assert((b >= 0).all())
    assert(a.shape == b.shape)
    assert(a.shape[0] == len(level_name))
    assert(a.shape[1] == 3)
    assert(b.shape[1] == 3)
    assert(len(my_cards_limit_full) == 3)
    assert(len(my_cards_limit_un) == 3)
    assert(len(support_available) == 3)
    assert(len(full_support_available) == 3)
    assert(len(all_demond) == 3)
    assert(len(now_own) == 3)
    if ("gain_bias") in locals():
        assert(len(gain_bias) == 3)

    # generate vecs
    vecs_my = generate_vecs_my(my_cards_limit_full, my_cards_limit_un)
    support = GainVec(
        np.maximum(support_available, full_support_available * FULL_REWARD),
        support_available
    )
    vecs = generate_vecs(vecs_my, support)

    # optimize (iteration)
    now_demond = all_demond - now_own
    if "force" in locals() and "gain_bias" in locals():
        res = optimize(vecs,
                    a,
                    b,
                    now_demond,
                    w_bias=0.1,
                    gain_bias=gain_bias
                    )        
    else:
        if "gain_bias" in locals():
            w_bias_l, w_bias_r = 0, 0.5
            res_l =optimize(vecs,
                        a,
                        b,
                        now_demond,
                        w_bias_l,
                        gain_bias
                        )
            res_r =optimize(vecs,
                        a,
                        b,
                        now_demond,
                        w_bias_r,
                        gain_bias
                        )
            w_eps = 1e-2
            while (w_bias_r - w_bias_l > w_eps):
                w_bias_m = (w_bias_l + w_bias_r) * 0.5
                res_m = optimize(vecs,
                            a,
                            b,
                            now_demond,
                            w_bias_m,
                            gain_bias
                            )
                if (res_m.x.sum() == res_l.x.sum()):
                    w_bias_l = w_bias_m
                else:
                    w_bias_r = w_bias_m
            res = res_l
            print(w_bias_l, w_bias_r)
        else:
            res = optimize(vecs,
                    a,
                    b,
                    now_demond)

    
    # output
    with open("result.txt","w") as f:
        if res.status == 0:
            f.write("Success\n")
            f.write(f"Minimum number of times: {res.x.sum()}\n\n")
            f.write("Summary:\n")
            for i in range(len(res.x)):
                split, vec_id = i // len(vecs), i % len(vecs)
                if (vec_id == 0):
                    f.write(f"level {level_name[split]}\t a={a[split]}\t b={b[split]}\n")
                if res.x[i] != 0:
                    f.write(f"nums: {res.x[i]}\t gain: {vecs[vec_id].gain}\t num_card: {vecs[vec_id].space}\n")
                if (vec_id == len(vecs) - 1):
                    f.write('\n')
        elif res.status == 1:
            f.write("Iteration limit reached.")
        elif res.status == 2:
            f.write("Problem appears to be infeasible.")
        elif res.status == 3:
            f.write("Problem appears to be unbounded.")
        elif res.status == 4:
            f.write("Numerical difficulties encountered.")



