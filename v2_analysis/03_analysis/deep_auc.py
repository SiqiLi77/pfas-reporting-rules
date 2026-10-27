import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from pilot_common import *

warnings.filterwarnings("ignore")
C = pickle.load(open("cache.pkl", "rb"))
wide, meta, codes = C["wide"], C["meta"], C["codes"]
lm = wide.groupby(level=0).mean()
w3 = wide[SIX].copy()
for c in SIX:
    w3.loc[w3[c] < U3MRL[c], c] = 0
lm3 = w3.groupby(level=0).mean()
det = pd.Series({l: codes["PotentialPFASSourcesDetail"].get(l, set()) for l in lm.index})
ps = pd.Series({l: codes["PotentialPFASSources"].get(l, set()) for l in lm.index})
AFFF = {"MB", "FT", "AO", "PS"}
IND = {"MF", "CC", "TA", "PP", "FM", "PR", "CE", "MM", "PC", "FP", "FF"}
none_ = ps.map(lambda s: s == {"No"})

def feats(M, kind):
    tot = M.sum(axis=1)
    if kind == "magnitude":
        return pd.DataFrame({"logtot": np.log1p(tot), "ndet": (M > 0).sum(axis=1)})
    if kind == "composition":
        sh = M.div(tot.replace(0, np.nan), axis=0).fillna(0)
        return sh
    return np.log1p(M)

def cv_auc(X, y, groups):
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_leaf_nodes=15, random_state=0)
    p = cross_val_predict(clf, X.values, y, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]
    return round(roc_auc_score(y, p), 3)

out = {}
for task, pos in [("AFFF", det.map(lambda s: bool(s & AFFF))),
                  ("Industrial", det.map(lambda s: bool(s & IND) and not (s & AFFF)))]:
    idx = lm.index[(pos | none_) & ~(pos & none_)]
    y = pos[idx].astype(int).values
    g_pws = meta.PWSID.reindex(idx).values
    g_st = meta.State.reindex(idx).values
    r = {"n_pos": int(y.sum()), "n_pws_pos": int(pd.Series(g_pws[y == 1]).nunique())}
    for reg, M in [("29@U5", lm.loc[idx, ALL29]), ("6@U5", lm.loc[idx, SIX]), ("6@U3", lm3.loc[idx, SIX])]:
        for kind in ["magnitude", "composition", "full"]:
            for gname, g in [("byPWS", g_pws), ("byState", g_st)]:
                r[f"{reg}|{kind}|{gname}"] = cv_auc(feats(M, kind), y, g)
        print(task, reg, {k: v for k, v in r.items() if k.startswith(reg)}, flush=True)
    # composition-only among detected EPs at U5 (does fingerprint add beyond detection?)
    dmask = (lm.loc[idx, ALL29].sum(axis=1) > 0).values
    for reg, M in [("29@U5", lm.loc[idx, ALL29]), ("6@U5", lm.loc[idx, SIX])]:
        r[f"detectedOnly|{reg}|composition|byPWS"] = cv_auc(feats(M[dmask], "composition"), y[dmask], g_pws[dmask])
    print(task, "detected-only", {k: v for k, v in r.items() if k.startswith("detectedOnly")}, flush=True)
    out[task] = r
json.dump(out, open("deep_auc.json", "w"), indent=1)
print("done")
