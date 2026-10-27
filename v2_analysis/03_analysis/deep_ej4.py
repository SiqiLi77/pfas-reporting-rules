import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pilot_common import *

warnings.filterwarnings("ignore")
exec(open("deep_ej.py").read().split("res = {}")[0])

DIV = {"New England": "CT ME MA NH RI VT", "Mid-Atlantic": "NJ NY PA", "ENC": "IL IN MI OH WI",
       "WNC": "IA KS MN MO NE ND SD", "South Atlantic": "DE DC FL GA MD NC SC VA WV",
       "ESC": "AL KY MS TN", "WSC": "AR LA OK TX", "Mountain": "AZ CO ID MT NV NM UT WY",
       "Pacific": "AK CA HI OR WA"}
s2d = {s: d for d, ss in DIV.items() for s in ss.split()}
D["div"] = D.state.map(s2d).fillna("Territory/Tribal")
n_ep = meta.groupby("PWSID").size()
D["log_nep"] = np.log(n_ep.reindex(D.index).fillna(1))
y_old = "Any detection, UCMR3 rule (6 @ old MRLs)"
y_new = "Any detection, UCMR5 rule (29 @ new MRLs)"
DEM = ["pct_black", "pct_hisp", "pct_pov"]

def design(dd, covs, fe):
    S = pd.concat([dd.assign(y=dd[y_old].astype(float), new=0.0), dd.assign(y=dd[y_new].astype(float), new=1.0)])
    Xm = S[DEM + covs].copy()
    Xm[DEM] /= 10.0
    for c in DEM + covs:
        Xm[c + "_x_new"] = Xm[c] * S.new
    # No main effect of the rule: the FE x rule dummies span it, and the rank-deficient design made the
    # cluster-robust errors depend on row order and BLAS rounding.
    fd = pd.get_dummies(S[fe] + "_" + S.new.astype(int).astype(str), drop_first=True).astype(float)
    return S, sm.add_constant(pd.concat([Xm, fd], axis=1))

def fit(dd, covs, fe, robust=True):
    S, X = design(dd, covs, fe)
    assert np.linalg.matrix_rank(X.values) == X.shape[1]
    m = sm.GLM(S.y, X, family=sm.families.Binomial())
    r = m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(S.index)[0]}) if robust else m.fit()
    return r

out = {}
base = ["log_pop", "any_sw"]
for fe in ["div", "state"]:
    for covs in [base, base + ["log_nep"]]:
        dd = D.dropna(subset=[y_old, y_new])
        r = fit(dd, covs, fe)
        k = "pct_black_x_new"
        key = f"FE={fe} covs={'+'.join(covs)}"
        out[key] = {"ratio": float(np.exp(r.params[k])), "p_cluster": float(r.pvalues[k]),
                    "OR_old": float(np.exp(r.params["pct_black"])), "OR_new": float(np.exp(r.params["pct_black"] + r.params[k]))}
        print(key, {a: round(b, 4) for a, b in out[key].items()})

# PWS bootstrap of the ratio, division FE (stable) and state FE
rng = np.random.default_rng(7)
dd = D.dropna(subset=[y_old, y_new])
idx = np.arange(len(dd))
for fe in ["div", "state"]:
    for covs in [base, base + ["log_nep"]]:
        bs = []
        for b in range(200):
            sub = dd.iloc[rng.choice(idx, len(idx), replace=True)].copy()
            sub.index = [f"b{i}" for i in range(len(sub))]
            try:
                r = fit(sub, covs, fe, robust=False)
                bs.append(float(r.params["pct_black_x_new"]))
            except Exception:
                pass
        bs = np.array(bs)
        bs = bs[np.isfinite(bs) & (np.abs(bs) < 5)]
        key = f"boot FE={fe} covs={'+'.join(covs)}"
        out[key] = {"n_boot": len(bs), "ratio_ci": [float(np.exp(np.percentile(bs, 2.5))), float(np.exp(np.percentile(bs, 97.5)))],
                    "p_one_sided_le_1": float((bs <= 0).mean())}
        print(key, out[key])
json.dump(out, open("deep_ej4.json", "w"), indent=1)
