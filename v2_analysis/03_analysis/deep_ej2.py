import json
import pickle
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pilot_common import *

warnings.filterwarnings("ignore")
exec(open("deep_ej.py").read().split("res = {}")[0])

out = {}
def stacked(y1, y2, data, label):
    dd = data.dropna(subset=[y1, y2]).copy()
    keep = dd.groupby("state")[y1].transform(lambda s: s.nunique() > 1) & dd.groupby("state")[y2].transform(lambda s: s.nunique() > 1)
    dd = dd[keep]
    A = dd.assign(y=dd[y1].astype(float), new=0.0)
    B = dd.assign(y=dd[y2].astype(float), new=1.0)
    S = pd.concat([A, B])
    Xm = S[["pct_black", "pct_hisp", "pct_pov", "log_pop", "any_sw"]].copy()
    Xm[["pct_black", "pct_hisp", "pct_pov"]] /= 10.0
    for v in ["pct_black", "pct_hisp", "pct_pov"]:
        Xm[v + "_x_new"] = Xm[v] * S.new
    # No main effect of the rule: the state x rule dummies span it, and the rank-deficient design made the
    # cluster-robust errors depend on row order and BLAS rounding.
    st_d = pd.get_dummies(S.state + "_" + S.new.astype(int).astype(str), drop_first=True).astype(float)
    Xm = sm.add_constant(pd.concat([Xm, st_d], axis=1))
    assert np.linalg.matrix_rank(Xm.values) == Xm.shape[1]
    m = sm.GLM(S.y, Xm, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(S.index)[0]})
    r = {}
    for v in ["pct_black", "pct_hisp", "pct_pov"]:
        k = v + "_x_new"
        r[v] = {"OR_old": float(np.exp(m.params[v])),
                "OR_new": float(np.exp(m.params[v] + m.params[k])),
                "ratio_of_OR": float(np.exp(m.params[k])),
                "p_diff": float(m.pvalues[k])}
    print(label, "n_PWS", len(dd), {v: {a: round(b, 4) for a, b in d.items()} for v, d in r.items()})
    return r

y_old = "Any detection, UCMR3 rule (6 @ old MRLs)"
y_new = "Any detection, UCMR5 rule (29 @ new MRLs)"
out["all_PWS"] = stacked(y_old, y_new, D, "All PWS (same 2023-25 water, two rules)")
Dl = D[meta.drop_duplicates("PWSID").set_index("PWSID").Size.reindex(D.index) == "L"]
out["large_PWS"] = stacked(y_old, y_new, Dl, "Large PWS only")
both = D.dropna(subset=["Actual UCMR3 detection (2013-15)"])
out["matched_actualU3_vs_U5"] = stacked("Actual UCMR3 detection (2013-15)", y_new, both, "PWS in both cycles: actual UCMR3 vs UCMR5 rule")
out["matched_recensored_vs_U5"] = stacked(y_old, y_new, both, "PWS in both cycles: re-censored vs UCMR5 rule")
json.dump(out, open("deep_ej2.json", "w"), indent=1)
