import json
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import optimize, stats
from pilot_common import *

warnings.filterwarnings("ignore")
C = pickle.load(open("cache.pkl", "rb"))
wide, meta, mrl, codes = C["wide"], C["meta"], C["mrl"], C["codes"]
out = {}
rng = np.random.default_rng(7)

# 1) Regression-to-the-mean baseline: within UCMR5, persistence of "detected at UCMR3 MRLs"
w5c = wide[SIX].copy()
for c in SIX:
    w5c.loc[w5c[c] < U3MRL[c], c] = 0
d = (w5c > 0).any(axis=1).unstack()
pairs = []
for i in d.columns:
    for j in d.columns:
        if i != j:
            ok = d[i].notna() & d[j].notna() & (d[i] == True)
            pairs.append((int(ok.sum()), float(d.loc[ok, j].mean())))
n = sum(p[0] for p in pairs)
rtm = sum(p[0] * p[1] for p in pairs) / n
print(f"within-UCMR5 persistence P(detected at U3 MRLs in another event | detected in one) = {rtm:.3f}")
out["rtm_persistence"] = rtm

# 2) Censored lognormal per analyte x source type (GW vs surface-influenced)
wt = meta.FacilityWaterType.reindex(wide.index.get_level_values(0)).values
grp = np.where(wt == "GW", "GW", "SW")
fits = {}
for a in ALL29:
    for g in ["GW", "SW"]:
        x = wide[a].values[grp == g]
        det = x[x > 0]
        nc = int((x == 0).sum())
        L = mrl[a]
        if len(det) < 15:
            fits[(a, g)] = None
            continue
        ld = np.log(det)
        def nll(p):
            mu, ls = p
            s = np.exp(ls)
            return -(stats.norm.logpdf(ld, mu, s).sum() + nc * stats.norm.logcdf((np.log(L) - mu) / s))
        r = optimize.minimize(nll, [np.log(L) - 2, 0.3], method="Nelder-Mead")
        mu, s = r.x[0], np.exp(r.x[1])
        fits[(a, g)] = (mu, s)
summ = []
for a in ["PFOA", "PFOS", "PFHxS", "PFNA", "PFBA", "PFPeA", "PFHxA", "PFBS", "HFPO-DA"]:
    for g in ["GW", "SW"]:
        f = fits[(a, g)]
        if f:
            mu, s = f
            pL = stats.norm.cdf((np.log(mrl[a]) - mu) / s)
            summ.append([a, g, round(np.exp(mu), 3), round(s, 2), round(pL, 3)])
print(pd.DataFrame(summ, columns=["analyte", "grp", "median_ng_L", "sdlog", "P(<MRL)"]).to_string())
out["fits"] = {f"{a}|{g}": v for (a, g), v in fits.items() if v}

def draw_below(a, g, k, u):
    f = fits[(a, g)]
    if f is None:
        return np.zeros(k)
    mu, s = f
    pL = stats.norm.cdf((np.log(mrl[a]) - mu) / s)
    return np.exp(mu + s * stats.norm.ppf(np.clip(u * pL, 1e-12, 1 - 1e-12)))

# 3) Monte Carlo: impute NDs, recompute location means, standard status
def standards(m):
    hi = m["PFHxS"] / 10 + m["PFNA"] / 10 + m["HFPO-DA"] / 10 + m["PFBS"] / 2000
    return pd.DataFrame({
        "US2026_PFOA_PFOS": (m.PFOA >= 4.05) | (m.PFOS >= 4.05),
        "HI_EPA": hi >= 1.5,
        "Canada_sum25_30": m[M533].sum(axis=1) > 30,
        "Sweden_PFAS4_4": m[PFAS4].sum(axis=1) > 4,
        "Denmark_PFAS4_2": m[PFAS4].sum(axis=1) > 2,
        "MA_PFAS6_20": m[MA6].sum(axis=1) > 20,
        "Germany_PFAS4_20": m[PFAS4].sum(axis=1) > 20,
        "EU_sum20_100": m[EU20_MEASURED].sum(axis=1) > 100,
    })
base = standards(wide.groupby(level=0).mean())
R = 1000
res = {mode: [] for mode in ["independent", "comonotone"]}
nd = (wide == 0).values
for mode in res:
    acc = None
    for r in range(R):
        W = wide.values.copy()
        if mode == "comonotone":
            U = np.repeat(rng.random((W.shape[0], 1)), W.shape[1], axis=1)
        else:
            U = rng.random(W.shape)
        for j, a in enumerate(ALL29):
            for g in ["GW", "SW"]:
                sel = nd[:, j] & (grp == g)
                W[sel, j] = draw_below(a, g, int(sel.sum()), U[sel, j])
        m = pd.DataFrame(W, index=wide.index, columns=ALL29).groupby(level=0).mean()
        s = standards(m).astype(int)
        acc = s if acc is None else acc + s
    p = acc / R
    for k in base.columns:
        flip = (~base[k]) & (p[k] >= 0.5)
        res[mode].append({"standard": k, "nd0_exceed": int(base[k].sum()),
                          "hidden_exceed_p50": int(flip.sum()),
                          "expected_hidden": round(float(p[k][~base[k]].sum()), 1),
                          "expected_hidden_mcse": round(float(np.sqrt((p[k] * (1 - p[k]))[~base[k]].sum() / R)), 2),
                          "uncertain_0.1_0.9": int(((p[k] > 0.1) & (p[k] < 0.9)).sum())})
    print(f"{mode} ({R} replicates)"); print(pd.DataFrame(res[mode]).to_string())
out["mc"] = res
out["mc_replicates"] = R

# 4) Entry points without any PFAS-4 report share one imputation distribution per source group and number of samples,
#    so their exceedance probabilities coincide. When that common probability lies near 0.1, 0.5 or 0.9, the counts
#    'p >= 0.5' and 'undecidable' of thousands of entry points move together with Monte Carlo noise.
nd4 = (wide[PFAS4] == 0).all(axis=1).groupby(level=0).all()
ep = pd.DataFrame({"nd4": nd4, "n": wide.groupby(level=0).size(), "g": pd.Series(grp, index=wide.index).groupby(level=0).first()})
rng_t = np.random.default_rng(1)

def p_group(g, k, limit, mode, N=200000):
    shape = (N, k, 1) if mode == "comonotone" else (N, k, len(PFAS4))
    U = np.broadcast_to(rng_t.random(shape), (N, k, len(PFAS4)))
    tot = sum(draw_below(a, g, None, U[:, :, j]) for j, a in enumerate(PFAS4))
    return float((tot.mean(axis=1) > limit).mean())

ties = {"n_ep_no_pfas4_report": int(ep.nd4.sum()), "n_draws": 200000, "groups": []}
for (g, k), n in ep[ep.nd4].groupby(["g", "n"]).size().sort_values(ascending=False).head(2).items():
    row = {"source": g, "samples": int(k), "n_ep": int(n)}
    for lim, key in ((2, "Denmark_PFAS4_2"), (4, "Sweden_PFAS4_4")):
        for mode in res:
            row[f"{key}|{mode}"] = round(p_group(g, int(k), lim, mode), 3)
    ties["groups"].append(row)
print("entry points without any PFAS-4 report:", ties["n_ep_no_pfas4_report"]); print(pd.DataFrame(ties["groups"]).to_string())
out["no_pfas4_report_groups"] = ties
json.dump(out, open("deep_decide.json", "w"), indent=1, default=float)
