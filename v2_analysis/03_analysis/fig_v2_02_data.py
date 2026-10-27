"""Round-7 data for the redrawn Figures 2-4 of manuscript V2.02 and new SI Figures S14-S16.

Run from 03_analysis:  python3 fig_v2_02_data.py
Inputs : cache.pkl, pws_county.pkl, data/usdm/*.json (same as deep_drought.py and deep_treat_season.py),
         deep_drought.json, deep_treat_season.json, deep_family.json, deep_story.json (for checks)
Outputs: fig_v2_02_data.json, fig_v2_02_pairs.csv, fig_v2_02_data.log
Every quantity that also exists in an earlier output is recomputed here and checked against it.
"""
import glob
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from pilot_common import ALL29, M533, MA6, MW, NF, PFAS4, SIX, U3MRL, EU20_MEASURED

warnings.filterwarnings("ignore")
LOG = open("fig_v2_02_data.log", "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


C = pickle.load(open("cache.pkl", "rb"))
wide, meta, dates, codes = C["wide"], C["meta"], C["dates"], C["codes"]
lm = wide.groupby(level=0).mean()
out, checks = {}, []


def check(name, got, want, tol=1e-6):
    ok = abs(float(got) - float(want)) <= tol
    checks.append((name, got, want, ok))
    log(f"check {'ok  ' if ok else 'FAIL'} {name}: {got} vs {want}")


# ------------------------------------------------------------------ drought bins with location-bootstrap intervals
pws_cty = pickle.load(open("pws_county.pkl", "rb"))
rows = []
for f in glob.glob("data/usdm/*.json"):
    if os.path.basename(f).startswith("_"):
        continue
    try:
        j = json.load(open(f))
    except Exception:
        continue
    for r in j:
        rows.append((r["fips"], r["mapDate"][:10], r["dsci"]))
U = pd.DataFrame(rows, columns=["fips", "date", "dsci"])
U["date"] = pd.to_datetime(U.date)
U = U.sort_values("date")


def build(locs):
    w = wide.loc[locs]
    d = pd.to_datetime(dates.loc[locs], format="%m/%d/%Y")
    df = pd.DataFrame({"loc": [i[0] for i in w.index], "ev": [i[1] for i in w.index], "date": d.values,
                       "short": w[["PFBA", "PFPeA", "PFHxA", "PFBS"]].sum(axis=1).values,
                       "long": w[["PFOA", "PFOS", "PFHxS"]].sum(axis=1).values,
                       "pfbs": w["PFBS"].values})
    df["fips"] = meta.PWSID.reindex(df["loc"]).map(pws_cty).values
    df = df.dropna(subset=["fips"]).sort_values("date")
    df = pd.merge_asof(df, U.rename(columns={"date": "udate"}), left_on="date", right_on="udate",
                       by="fips", direction="backward")
    return df.dropna(subset=["dsci"])


ev = wide.groupby(level=0).size()
L4 = ev[ev == 4].index
L4sw = L4[meta.FacilityWaterType.reindex(L4).isin(["SW", "GU", "MX"])]
gw2 = ev[ev == 2].index
gw2 = gw2[meta.FacilityWaterType.reindex(gw2) == "GW"]
dsw, dgw = build(L4sw), build(gw2)
BINS = ["much wetter", "wetter", "typical", "drier", "much drier"]
ref = json.load(open("deep_drought.json"))
rng = np.random.default_rng(20261023)
NB = 2000


def bins(df, comp):
    ok = df.groupby("loc")[comp].transform(lambda s: (s > 0).all())
    s = df[ok].copy()
    s["lr"] = np.log(s[comp]) - s.groupby("loc")[comp].transform(lambda x: np.log(x).mean())
    s["anom"] = s.dsci - s.groupby("loc").dsci.transform("mean")
    s["bin"] = pd.cut(s.anom, [-1e9, -100, -25, 25, 100, 1e9], labels=BINS)
    return s


drought = {}
for key, df, comp in (("sw_short", dsw, "short"), ("sw_long", dsw, "long"), ("gw_short", dgw, "short"),
                      ("gw_long", dgw, "long")):
    s = bins(df, comp)
    locs = s["loc"].unique()
    li = pd.Series(np.arange(len(locs)), index=locs)
    s_loc = li.reindex(s["loc"]).values
    s_bin = s["bin"].cat.codes.values
    s_lr = s["lr"].values
    est = {b: float(np.exp(np.median(s_lr[s_bin == k]))) for k, b in enumerate(BINS)}
    boot = {b: [] for b in BINS}
    for _ in range(NB):
        wts = np.bincount(rng.integers(0, len(locs), len(locs)), minlength=len(locs))[s_loc]
        for k, b in enumerate(BINS):
            m = (s_bin == k) & (wts > 0)
            boot[b].append(np.exp(np.median(np.repeat(s_lr[m], wts[m]))))
    drought[key] = {"n_loc": int(len(locs)),
                    "bins": {b: {"ratio": round(est[b], 4), "lo": round(float(np.percentile(boot[b], 2.5)), 4),
                                 "hi": round(float(np.percentile(boot[b], 97.5)), 4), "n": int((s_bin == k).sum())}
                             for k, b in enumerate(BINS)}}
    log(key, "locations", len(locs), {b: (drought[key]["bins"][b]["ratio"], drought[key]["bins"][b]["lo"],
                                         drought[key]["bins"][b]["hi"], drought[key]["bins"][b]["n"]) for b in BINS})
    if key in ("sw_short", "sw_long"):
        rb = ref[f"bins_{comp}"]
        for b in BINS:
            check(f"drought {key} {b} ratio", round(est[b], 3), rb["ratio"][b], 1e-9)
            check(f"drought {key} {b} n", drought[key]["bins"][b]["n"], rb["n"][b], 0)
out["drought_bins"] = drought
out["drought_bins_bootstrap"] = f"{NB} replicates resampling locations; percentile 95% intervals of the bin median"

# Drought-analysis entry points: would their short-chain sum have been reported under the UCMR 3 rule?
s = bins(dsw, "short")
sw_locs = s["loc"].unique()
w4 = wide.loc[sw_locs]
u3 = w4[SIX].copy()
for c in SIX:
    u3.loc[u3[c] < U3MRL[c], c] = 0
short_u3_all = (u3["PFBS"] > 0).groupby(level=0).all()
any_u3 = (u3 > 0).groupby(level=0).any().any(axis=1)
out["drought_visibility"] = {"n": int(len(sw_locs)), "short_sum_reported_all_samples_ucmr3_rule": int(short_u3_all.sum()),
                             "any_pfas_reported_ucmr3_rule": int(any_u3.sum())}
log("drought entry points under UCMR 3 rule:", out["drought_visibility"])
check("drought SW short locations", len(sw_locs), ref["sw_short"]["n_loc"], 0)

# ------------------------------------------------------------------ treated vs untreated pairs (Fig. 4c)
pt, po = codes["PFASTreatment"], codes["PFASOccurrence"]
L = lm.index
treated = pd.Series([bool(pt.get(l, set()) & {"GAC", "IEX", "NRO"}) for l in L], index=L)
untreated = pd.Series([(not treated[l]) and bool(pt.get(l, set()) & {"NMT"} or po.get(l, set()) & {"No"}) for l in L], index=L)
pws = meta.PWSID.reindex(L)
tot = lm[ALL29].sum(axis=1)
short6 = lm[["PFBA", "PFPeA", "PFHxA", "PFBS", "PFPeS", "PFHpA"]].sum(axis=1)
oaos = lm[["PFOA", "PFOS"]].sum(axis=1)
exc = (lm.PFOA >= 4.05) | (lm.PFOS >= 4.05)
P = []
for p in sorted(set(pws[treated]) & set(pws[untreated])):
    t = L[(pws == p) & treated]
    u = L[(pws == p) & untreated]
    P.append({"pws": p, "n_treated_ep": len(t), "n_untreated_ep": len(u),
              "short_treated": short6[t].mean(), "short_untreated": short6[u].mean(),
              "total_treated": tot[t].mean(), "total_untreated": tot[u].mean(),
              "oaos_treated": oaos[t].mean(), "oaos_untreated": oaos[u].mean(),
              "exceed_share_treated": exc[t].mean(), "exceed_share_untreated": exc[u].mean()})
D = pd.DataFrame(P)
D["sf_treated"] = D.short_treated / D.total_treated.replace(0, np.nan)
D["sf_untreated"] = D.short_untreated / D.total_untreated.replace(0, np.nan)
D["both_reported"] = D[["sf_treated", "sf_untreated"]].notna().all(axis=1)
D.round(6).to_csv("fig_v2_02_pairs.csv", index=False)
dd = D[D.both_reported]
W = json.load(open("deep_treat_season.json"))["within_pws"]
check("pairs n systems", len(D), W["n"], 0)
check("pairs n both reported", len(dd), W["sf_n"], 0)
check("pairs median sf treated", dd.sf_treated.median(), W["sf_med"][0])
check("pairs median sf untreated", dd.sf_untreated.median(), W["sf_med"][1])
check("pairs mean exceed treated", D.exceed_share_treated.mean(), W["exc_mean"][0])
check("pairs mean short untreated", D.short_untreated.mean(), W["short_mean"][1])
out["pairs"] = {"n": int(len(D)), "n_both": int(len(dd)),
                "sf_median": [float(dd.sf_treated.median()), float(dd.sf_untreated.median())],
                "treated_sf_is_1": int((dd.sf_treated >= 0.999).sum()),
                "treated_higher": int((dd.sf_treated > dd.sf_untreated).sum()),
                "treated_lower": int((dd.sf_treated < dd.sf_untreated).sum()),
                "equal": int((dd.sf_treated == dd.sf_untreated).sum())}
log("pairs:", out["pairs"])

# ------------------------------------------------------------------ visibility of the signals under each rule (Fig. 4d)
S = json.load(open("deep_story.json"))
out["visibility"] = [
    {"row": "texas", "n": S["texas_canada_only"]["n_ep"], "ucmr3_rule": S["texas_canada_only"]["seen_under_ucmr3_rule"]},
    {"row": "short_only_treated", "n": S["treated_eps"]["short_only"],
     "ucmr3_rule": S["treated_eps"]["short_only_seen_under_ucmr3_rule"]},
    {"row": "drought_short", "n": out["drought_visibility"]["n"],
     "ucmr3_rule": out["drought_visibility"]["short_sum_reported_all_samples_ucmr3_rule"]},
]
log("visibility rows:", out["visibility"])

# ------------------------------------------------------------------ full Jaccard matrix of rankings (Fig. 3d)
RPF_UP = {"PFBA": 0.05, "PFPeA": 0.05, "PFHxA": 0.01, "PFHpA": 1, "PFOA": 1, "PFNA": 10, "PFDA": 10,
          "PFUnA": 4, "PFDoA": 3, "PFTrDA": 3, "PFTA": 0.3, "PFBS": 0.001, "PFPeS": 0.6, "PFHxS": 0.6,
          "PFHpS": 2, "PFOS": 2, "HFPO-DA": 0.06, "ADONA": 0.03}
RPF_LO = dict(RPF_UP, PFPeA=0.01, PFHpA=0.01, PFDA=4, PFTrDA=0.3, PFPeS=0.001, PFHpS=0.6)
panels = {
    "PFOA/PFOS max": lm[["PFOA", "PFOS"]].max(axis=1),
    "UCMR3 six sum": lm[SIX].sum(axis=1),
    "PFAS4 sum": lm[PFAS4].sum(axis=1),
    "MA6 sum": lm[MA6].sum(axis=1),
    "EU20(15) sum": lm[EU20_MEASURED].sum(axis=1),
    "Canada25 sum": lm[M533].sum(axis=1),
    "Total F (molar)": sum(lm[c] / MW[c] * NF[c] for c in ALL29),
    "RPF upper": sum(lm[k] * v for k, v in RPF_UP.items()),
    "RPF lower": sum(lm[k] * v for k, v in RPF_LO.items()),
}
Pn = pd.DataFrame(panels)
q = 0.0428
F = pd.DataFrame({k: Pn[k] > Pn[k].quantile(1 - q) for k in Pn})
names = list(Pn.columns)
M = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        M.loc[a, b] = (F[a] & F[b]).sum() / (F[a] | F[b]).sum()
fam = json.load(open("deep_family.json"))["quantile_sensitivity"]["0.0428"]
for k, v in fam.items():
    a, b = k.split(" vs ")
    check(f"jaccard {k}", round(M.loc[a, b], 3), v, 1e-9)
out["jaccard_matrix"] = {"q": q, "names": names, "values": M.round(4).values.tolist(),
                         "n_top": {k: int(F[k].sum()) for k in names}}
log("Jaccard matrix (top 4.28%):\n" + M.round(2).to_string())

n_fail = sum(not c[3] for c in checks)
out["checks"] = {"n": len(checks), "failed": n_fail}
json.dump(out, open("fig_v2_02_data.json", "w"), indent=1)
log(f"{len(checks)} checks, {n_fail} failed")
LOG.close()
sys.exit(1 if n_fail else 0)
