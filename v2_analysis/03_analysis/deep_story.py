"""Closure analyses for the V2.00 narrative (round 5).

Run: cd 03_analysis && python3 deep_story.py  ->  deep_story.json, deep_story.log
(a) Systems now exceeding the PFOA/PFOS standards that UCMR 3 missed: would today's water still be missed under the
    UCMR 3 rule, had they tested for PFAS before UCMR 5, did they treat, did they know of potential sources,
    and who are they (size, source, state, largest systems)?
(b) All UCMR 5 systems exceeding PFOA/PFOS: how many had never tested before UCMR 5?
(c) Would the Texas sum exceedances and the short-chain signal at treated entry points be visible under the UCMR 3 rule?
Definitions match deep_lost.py, deep_tables_extra.py and deep_treat_season.py.
"""
import json
import pickle
import warnings

import numpy as np
import pandas as pd

from pilot_common import M533, SIX, U3MRL

warnings.filterwarnings("ignore")
C = pickle.load(open("cache.pkl", "rb"))
wide, meta, pop, codes, wide3, meta3 = C["wide"], C["meta"], C["pop"], C["codes"], C["wide3"], C["meta3"]
sd = pd.read_csv("data/sdwa/SDWA_PUB_WATER_SYSTEMS.csv", usecols=["PWSID", "PWS_NAME", "SUBMISSIONYEARQUARTER"], dtype=str)
sdname = sd.sort_values("SUBMISSIONYEARQUARTER").drop_duplicates("PWSID", keep="last").set_index("PWSID").PWS_NAME
out = {}

lm5 = wide.groupby(level=0).mean()
L5 = lm5.index
pw5 = meta.PWSID.reindex(L5)
exc_ep = (lm5.PFOA >= 4.05) | (lm5.PFOS >= 4.05)
det3 = (wide3 > 0).groupby(level=0).any().any(axis=1)
pw3 = meta3.PWSID.reindex(det3.index)
w5c = wide[SIX].copy()
for c in SIX:
    w5c.loc[w5c[c] < U3MRL[c], c] = 0
seen_now_u3rule = (w5c > 0).groupby(level=0).any().any(axis=1).reindex(L5)

occ, trt, src = codes["PFASOccurrence"], codes["PFASTreatment"], codes["PotentialPFASSources"]
ep_occ = pd.Series([occ.get(l, set()) for l in L5], index=L5)
ep_trt = pd.Series([bool(trt.get(l, set()) & {"GAC", "IEX", "NRO"}) for l in L5], index=L5)
ep_src = pd.Series([src.get(l, set()) for l in L5], index=L5)


def sys_flags(pwsids):
    """System-level answers aggregated over entry points."""
    rows = {}
    for p in pwsids:
        L = L5[pw5 == p]
        o = set().union(*ep_occ[L]) if len(L) else set()
        s = set().union(*ep_src[L]) if len(L) else set()
        rows[p] = {"tested": "Yes" in o, "never": ("Yes" not in o) and ("No" in o), "dk_only": o == {"DK"} or not o,
                   "treated": bool(ep_trt[L].any()), "aware_src": "Yes" in s,
                   "seen_now_u3rule": bool(seen_now_u3rule[L].any()),
                   "max_oaos": float(lm5.loc[L, ["PFOA", "PFOS"]].max(axis=1).max()),
                   "any_sw": bool(meta.FacilityWaterType.reindex(L).isin(["SW", "GU", "MX"]).any()),
                   "size": meta.Size.reindex(L).iloc[0], "state": p[:2], "pop": float(pop.get(p, np.nan))}
    return pd.DataFrame(rows).T


def share(s):
    return round(float(s.astype(bool).mean()), 6)


# (a) the 699 systems in both cycles now exceeding
both = sorted(set(pw5) & set(pw3))
exc_sys = set(pw5[exc_ep])
seen3 = set(pw3[det3])
E = sys_flags([p for p in both if p in exc_sys])
E["seen3"] = [p in seen3 for p in E.index]
U, S = E[~E.seen3.astype(bool)], E[E.seen3.astype(bool)]
a = {}
for lab, g in (("unseen", U), ("seen", S)):
    a[lab] = {"n": len(g), "pop_m": round(float(g["pop"].sum()) / 1e6, 2),
              "tested_before_ucmr5": share(g.tested), "never_tested": share(g.never), "dk_or_blank": share(g.dk_only),
              "treated": share(g.treated), "aware_of_sources": share(g.aware_src),
              "seen_today_under_ucmr3_rule": share(g.seen_now_u3rule),
              "max_oaos_median": round(float(g.max_oaos.median()), 1),
              "max_oaos_below_20": share(g.max_oaos < 20),
              "large_share": share(g["size"] == "L"), "any_surface_water": share(g.any_sw),
              "pop_quartiles": [int(x) for x in g["pop"].quantile([0.25, 0.5, 0.75])],
              "n_pop_under_10000": int((g["pop"] < 10000).sum()),
              "top_states": g.state.value_counts().head(8).to_dict(),
              "n_states": int(g.state.nunique())}
never_pop = U.loc[U.never.astype(bool), "pop"].sum() / 1e6
a["unseen"]["never_tested_pop_m"] = round(float(never_pop), 2)
a["unseen"]["never_tested_n"] = int(U.never.astype(bool).sum())
a["unseen"]["untested_or_dk_and_untreated_n"] = int((~U.tested.astype(bool) & ~U.treated.astype(bool)).sum())
top = U.sort_values("pop", ascending=False).head(12)
a["unseen_largest"] = [{"PWSID": p, "name": sdname.get(p, ""), "state": r.state, "pop": int(r["pop"]),
                        "max_PFOA_or_PFOS": round(r.max_oaos, 1), "tested_before": bool(r.tested), "treated": bool(r.treated)}
                       for p, r in top.iterrows()]
out["unseen_systems"] = a

# entry-point level: currently exceeding entry points that UCMR 3 did not report
m = L5.intersection(det3.index)
ue = m[exc_ep[m] & ~det3[m]]
out["unseen_eps"] = {"n": int(len(ue)),
                     "mean_below_u3_mrls": share((lm5.loc[ue, "PFOA"] < 20) & (lm5.loc[ue, "PFOS"] < 40)),
                     "no_sample_at_u3_mrls_today": share(~seen_now_u3rule[ue])}

# (b) all UCMR 5 systems now exceeding PFOA/PFOS
A = sys_flags(sorted(exc_sys))
out["all_exceeding_systems"] = {"n": len(A), "pop_m": round(float(A["pop"].sum()) / 1e6, 2),
                                "tested_before_ucmr5": share(A.tested), "never_tested": share(A.never),
                                "never_tested_n": int(A.never.astype(bool).sum()),
                                "never_tested_pop_m": round(float(A.loc[A.never.astype(bool), "pop"].sum()) / 1e6, 2),
                                "dk_or_blank": share(A.dk_only), "treated": share(A.treated),
                                "aware_of_sources": share(A.aware_src)}

# (c1) Texas entry points exceeding Canada's sum while passing PFOA/PFOS
st = meta.State.reindex(L5)
tx = L5[(lm5[M533].sum(axis=1) > 30) & ~exc_ep & (st == "TX")]
out["texas_canada_only"] = {"n_ep": int(len(tx)), "seen_under_ucmr3_rule": int(seen_now_u3rule[tx].sum()),
                            "pfoa_pfos_max": round(float(lm5.loc[tx, ["PFOA", "PFOS"]].max(axis=1).max()), 2)}

# (c2) treated entry points in systems that also have untreated entry points
untreated = pd.Series([(not ep_trt[l]) and bool(trt.get(l, set()) & {"NMT"} or occ.get(l, set()) & {"No"}) for l in L5], index=L5)
both_t = set(pw5[ep_trt]) & set(pw5[untreated])
T = L5[ep_trt & pw5.isin(both_t)]
tot = lm5[M533 + ["NEtFOSAA", "NMeFOSAA", "PFTA", "PFTrDA"]].sum(axis=1)
short = lm5[["PFBA", "PFPeA", "PFHxA", "PFBS", "PFPeS", "PFHpA"]].sum(axis=1)
sf = (short / tot.replace(0, np.nan)).reindex(T)
T1 = T[(sf >= 0.999).values]
out["treated_eps"] = {"n": int(len(T)), "any_pfas": int((tot[T] > 0).sum()), "short_only": int(len(T1)),
                      "short_only_seen_under_ucmr3_rule": int(seen_now_u3rule[T1].sum()),
                      "short_only_median_short_sum": round(float(short[T1].median()), 2)}

# (d) is the treatment contrast explained by concentration? Current maximum is post-treatment, so this is descriptive.
import statsmodels.formula.api as smf  # noqa: E402

for c in ("tested", "never", "treated", "any_sw", "seen3"):
    E[c] = E[c].astype(bool)
E["max_oaos"], E["pop"] = E.max_oaos.astype(float), E["pop"].astype(float)
E["band"] = pd.cut(E.max_oaos, [4, 8, 20, np.inf], labels=["4-8", "8-20", ">20"], right=False)
bands = {f"{b}|{'seen' if s else 'unseen'}": {"treated": round(float(g.treated.mean()), 4), "n": int(len(g))}
         for (b, s), g in E.groupby(["band", "seen3"], observed=True)}
d = E.assign(y=E.treated.astype(int), s=E.seen3.astype(int), lc=np.log10(E.max_oaos),
             lp=np.log10(E["pop"].clip(lower=25)), sw=E.any_sw.astype(int))


def or_seen(df):
    m = smf.logit("y ~ s + lc + lp + sw", df).fit(disp=0)
    lo, hi = np.exp(m.conf_int().loc["s"])
    return {"or": round(float(np.exp(m.params["s"])), 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2),
            "p": float(m.pvalues["s"]), "n": int(len(df))}


out["treatment_vs_visibility"] = {"by_band": bands, "adjusted_or_all": or_seen(d),
                                  "adjusted_or_self_reported_testers": or_seen(d[d.tested])}
A["inu3"] = [p in set(pw3) for p in A.index]
A["never"], A["pop"] = A.never.astype(bool), A["pop"].astype(float)
out["never_tested_split"] = {
    lab: {"n": int(len(g)), "pop_m": round(float(g["pop"].sum()) / 1e6, 2),
          "never_n": int(g.never.sum()), "never_share": round(float(g.never.mean()), 4),
          "never_pop_m": round(float(g.loc[g.never, "pop"].sum()) / 1e6, 2), "small_n": int((g["size"] == "S").sum())}
    for lab, g in (("in_ucmr3", A[A.inu3]), ("not_in_ucmr3", A[~A.inu3]))}
out["unseen_systems"]["jurisdictions"] = {
    "states": int(sum(1 for s in U.state.unique() if s.isalpha() and s not in ("PR", "GU", "VI", "AS", "MP"))),
    "territories": sorted(s for s in U.state.unique() if s in ("PR", "GU", "VI", "AS", "MP")),
    "epa_region_tribal": sorted(s for s in U.state.unique() if not s.isalpha())}

json.dump(out, open("deep_story.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
