import json
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from pilot_common import *

warnings.filterwarnings("ignore")
C = pickle.load(open("cache.pkl", "rb"))
wide, meta, codes, dates = C["wide"], C["meta"], C["codes"], C["dates"]
lm = wide.groupby(level=0).mean()
out = {}

# ---- Treatment: within-PWS comparison
pt = codes["PFASTreatment"]
po = codes["PFASOccurrence"]
L = lm.index
treated = pd.Series([bool(pt.get(l, set()) & {"GAC", "IEX", "NRO"}) for l in L], index=L)
untreated = pd.Series([(not treated[l]) and bool(pt.get(l, set()) & {"NMT"} or po.get(l, set()) & {"No"}) for l in L], index=L)
pws = meta.PWSID.reindex(L)
tot = lm[ALL29].sum(axis=1)
short = lm[["PFBA", "PFPeA", "PFHxA", "PFBS", "PFPeS", "PFHpA"]].sum(axis=1)
oaos = lm[["PFOA", "PFOS"]].sum(axis=1)
both = set(pws[treated]) & set(pws[untreated])
rows = []
for p in sorted(both):
    t = L[(pws == p) & treated]; u = L[(pws == p) & untreated]
    rows.append({"pws": p, "n_t": len(t), "n_u": len(u),
                 "oaos_t": oaos[t].mean(), "oaos_u": oaos[u].mean(),
                 "short_t": short[t].mean(), "short_u": short[u].mean(),
                 "tot_t": tot[t].mean(), "tot_u": tot[u].mean(),
                 "exc_t": ((lm.loc[t, "PFOA"] >= 4.05) | (lm.loc[t, "PFOS"] >= 4.05)).mean(),
                 "exc_u": ((lm.loc[u, "PFOA"] >= 4.05) | (lm.loc[u, "PFOS"] >= 4.05)).mean()})
D = pd.DataFrame(rows)
D["sf_t"] = D.short_t / D.tot_t.replace(0, np.nan)
D["sf_u"] = D.short_u / D.tot_u.replace(0, np.nan)
print("PWS with both treated and untreated EPs:", len(D))
wp = {}
for a, b in [("oaos_t", "oaos_u"), ("short_t", "short_u"), ("tot_t", "tot_u"), ("exc_t", "exc_u")]:
    w = stats.wilcoxon(D[a], D[b], zero_method="zsplit")
    wp[a[:-2]] = float(w.pvalue)
    print(f"  {a} median {D[a].median():.2f} vs {b} {D[b].median():.2f}  mean {D[a].mean():.2f} vs {D[b].mean():.2f}  p={w.pvalue:.3g}")
dd = D.dropna(subset=["sf_t", "sf_u"])
w = stats.wilcoxon(dd.sf_t, dd.sf_u)
print(f"  short-chain fraction (both >0): n={len(dd)} median {dd.sf_t.median():.2f} vs {dd.sf_u.median():.2f} p={w.pvalue:.3g}")
out["within_pws"] = {"n": len(D), "oaos_med": [D.oaos_t.median(), D.oaos_u.median()],
                     "oaos_mean": [D.oaos_t.mean(), D.oaos_u.mean()],
                     "short_mean": [D.short_t.mean(), D.short_u.mean()],
                     "exc_mean": [D.exc_t.mean(), D.exc_u.mean()],
                     "sf_n": len(dd), "sf_med": [dd.sf_t.median(), dd.sf_u.median()], "sf_p": w.pvalue}
# direction of the treated-minus-untreated difference in each system, with the paired Wilcoxon p value above
rt = {}
for k in ("oaos", "exc", "short"):
    t, u = D[f"{k}_t"].to_numpy(), D[f"{k}_u"].to_numpy()
    rt[k] = {"lower": int((t < u).sum()), "equal": int((t == u).sum()), "higher": int((t > u).sum()),
             "equal_both_zero": int(((t == 0) & (u == 0)).sum()), "wilcoxon_p": wp[k]}
    print(f"  {k}: lower at treated {rt[k]['lower']}, equal {rt[k]['equal']} (both zero {rt[k]['equal_both_zero']}), "
          f"higher {rt[k]['higher']}")
out["within_pws"]["direction"] = rt
# treated exceeders: how high?
te = L[treated & ((lm.PFOA >= 4.05) | (lm.PFOS >= 4.05))]
print("treated exceeders:", len(te), "median PFOA+PFOS", round(oaos[te].median(), 1), "water type", meta.FacilityWaterType.reindex(te).value_counts().to_dict())

# ---- Seasonality robustness (surface-influenced 4-event locations)
ev = wide.groupby(level=0).size()
L4 = ev[ev == 4].index
L4 = L4[meta.FacilityWaterType.reindex(L4).isin(["SW", "GU", "MX"])]
print("surface-water 4-event locations", len(L4))
w4 = wide.loc[L4]
d = pd.to_datetime(dates.loc[L4], format="%m/%d/%Y")
df = pd.DataFrame({"loc": [i[0] for i in w4.index], "ev": [i[1] for i in w4.index],
                   "q": d.dt.quarter.values, "yr": d.dt.year.values, "mon": d.dt.month.values,
                   "short": w4[["PFBA", "PFPeA", "PFHxA", "PFBS"]].sum(axis=1).values,
                   "long": w4[["PFOA", "PFOS", "PFHxS"]].sum(axis=1).values})
res = {}
for comp in ["short", "long"]:
    ok = df.groupby("loc")[comp].transform(lambda s: (s > 0).all())
    s = df[ok].copy()
    s["lr"] = np.log(s[comp]) - s.groupby("loc")[comp].transform(lambda x: np.log(x).mean())
    nloc = s["loc"].nunique()
    byq = s.groupby("q").lr.median()
    # FE regression with quarter + event order
    X = pd.get_dummies(s[["q", "ev"]].astype(str), drop_first=True).astype(float)
    y = s.lr.values
    Xd = X - X.groupby(s["loc"].values).transform("mean")
    beta, *_ = np.linalg.lstsq(Xd.values, y, rcond=None)
    coef = dict(zip(X.columns, np.round(np.exp(beta), 3)))
    byyr = s.groupby(["yr", "q"]).lr.median().unstack().round(3)
    res[comp] = {"n_loc": nloc, "median_ratio_by_q": np.round(np.exp(byq), 3).to_dict(), "fe_coef_vs_Q1": coef}
    print(comp, "n_loc", nloc, "median ratio by quarter", np.round(np.exp(byq), 3).to_dict())
    print("  FE model (ratio vs Q1 / vs SE1):", coef)
    print("  by year x quarter (exp):\n", np.exp(byyr).round(3).to_string())
out["season_sw"] = res
# event order vs quarter crosstab
print(pd.crosstab(df.ev, df.q).to_string())

def median_ci(lr, key, nb=2000, seed=11):
    """95% percentile interval of the median warm-to-cool ratio, from bootstrap replicates resampling locations."""
    lr = np.asarray(lr, float)
    rng = np.random.default_rng(seed)
    meds = [np.median(lr[rng.integers(0, len(lr), len(lr))]) for _ in range(nb)]
    ci = np.exp(np.percentile(meds, [2.5, 97.5])).tolist()
    print(f"  {key}: median 95% CI {ci[0]:.3f}-{ci[1]:.3f}; locations with equal halves {int((lr == 0).sum())}")
    out.setdefault("warm_cool_ci", {})[key] = ci


# GW negative control: 2-event GW locations, samples in warm (Q3/Q4) vs cool (Q1/Q2) halves
evg = ev[ev == 2].index
gw = evg[meta.FacilityWaterType.reindex(evg) == "GW"]
wg = wide.loc[gw]
dg = pd.to_datetime(dates.loc[gw], format="%m/%d/%Y")
g = pd.DataFrame({"loc": [i[0] for i in wg.index], "warm": dg.dt.quarter.isin([3, 4]).values,
                  "short": wg[["PFBA", "PFPeA", "PFHxA", "PFBS"]].sum(axis=1).values,
                  "long": wg[["PFOA", "PFOS", "PFHxS"]].sum(axis=1).values})
for comp in ["short", "long"]:
    piv = g.pivot_table(index="loc", columns="warm", values=comp, aggfunc="mean").dropna()
    piv = piv[(piv > 0).all(axis=1)]
    r = np.exp(np.median(np.log(piv[True] / piv[False])))
    w = stats.wilcoxon(np.log(piv[True]), np.log(piv[False]))
    print(f"GW control {comp}: n={len(piv)} median warm/cool ratio {r:.3f} p={w.pvalue:.3g}")
    out[f"gw_{comp}"] = [len(piv), float(r), float(w.pvalue)]
    median_ci(np.log(piv[True] / piv[False]), f"gw_{comp}")
# same for SW 4-event using warm/cool halves
for comp in ["short", "long"]:
    df["warm"] = df.q.isin([3, 4])
    piv = df.pivot_table(index="loc", columns="warm", values=comp, aggfunc="mean").dropna()
    piv = piv[(piv > 0).all(axis=1)]
    r = np.exp(np.median(np.log(piv[True] / piv[False])))
    w = stats.wilcoxon(np.log(piv[True]), np.log(piv[False]))
    print(f"SW {comp}: n={len(piv)} median warm/cool ratio {r:.3f} p={w.pvalue:.3g}")
    out[f"sw_{comp}"] = [len(piv), float(r), float(w.pvalue)]
    median_ci(np.log(piv[True] / piv[False]), f"sw_{comp}")
json.dump(out, open("deep_treat_season.json", "w"), indent=1, default=float)
