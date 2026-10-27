import json
import pickle
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pilot_common import *

warnings.filterwarnings("ignore")
exec(open("deep_ej.py").read().split("res = {}")[0])

n_ep = meta.groupby("PWSID").size()
D["log_nep"] = np.log(n_ep.reindex(D.index).fillna(1))
D["log_pop2"] = D.log_pop ** 2
y_old = "Any detection, UCMR3 rule (6 @ old MRLs)"
y_new = "Any detection, UCMR5 rule (29 @ new MRLs)"

def stacked(data, covs, v="pct_black"):
    dd = data.dropna(subset=[y_old, y_new]).copy()
    keep = dd.groupby("state")[y_old].transform(lambda s: s.nunique() > 1) & dd.groupby("state")[y_new].transform(lambda s: s.nunique() > 1)
    dd = dd[keep]
    S = pd.concat([dd.assign(y=dd[y_old].astype(float), new=0.0), dd.assign(y=dd[y_new].astype(float), new=1.0)])
    Xm = S[["pct_black", "pct_hisp", "pct_pov"] + covs].copy()
    Xm[["pct_black", "pct_hisp", "pct_pov"]] /= 10.0
    for c in ["pct_black", "pct_hisp", "pct_pov"] + covs:
        Xm[c + "_x_new"] = Xm[c] * S.new
    # No main effect of the rule: the state x rule dummies span it, and the rank-deficient design made the
    # cluster-robust errors depend on row order and BLAS rounding.
    st_d = pd.get_dummies(S.state + "_" + S.new.astype(int).astype(str), drop_first=True).astype(float)
    Xm = sm.add_constant(pd.concat([Xm, st_d], axis=1))
    assert np.linalg.matrix_rank(Xm.values) == Xm.shape[1]
    m = sm.GLM(S.y, Xm, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(S.index)[0]})
    k = v + "_x_new"
    return {"OR_old": float(np.exp(m.params[v])), "OR_new": float(np.exp(m.params[v] + m.params[k])),
            "ratio": float(np.exp(m.params[k])), "p": float(m.pvalues[k]), "n": len(dd), "pos_old": int(dd[y_old].sum())}

out = {}
base = ["log_pop", "any_sw"]
specs = {"base": base, "+log n EP": base + ["log_nep"], "+log n EP +log_pop^2": base + ["log_nep", "log_pop2"]}
for name, cv in specs.items():
    r = stacked(D, cv)
    out[name] = r
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

loso = {}
for s in sorted(D.state.unique()):
    sub = D[D.state != s]
    if (sub[y_old].sum() == D[y_old].sum()):
        continue
    r = stacked(sub, base)
    loso[s] = r
    print("drop", s, "pos_old", r["pos_old"], "ratio", round(r["ratio"], 3), "p", round(r["p"], 4))
rat = [v["ratio"] for v in loso.values()]
ps = [v["p"] for v in loso.values()]
out["loso"] = {"n_states_dropped": len(loso), "ratio_min": min(rat), "ratio_max": max(rat), "p_max": max(ps),
               "worst_state": max(loso, key=lambda s: loso[s]["p"]), "detail": loso}
print("LOSO ratio range", round(min(rat), 3), round(max(rat), 3), "max p", round(max(ps), 4), "worst", out["loso"]["worst_state"])
json.dump(out, open("deep_ej3.json", "w"), indent=1)
