"""System-level table behind the round-6 draft figures (Fig. 1 redesign and TOC graphic).

Run: cd 03_analysis && python3 fig_r6_data.py  ->  fig_r6_systems.csv, fig_r6_data.json
One row per UCMR 5 system with an entry point whose 2023-2025 mean PFOA or PFOS is >= 4.05 ng/L (1,173 systems):
  cls        seen (any PFAS reported in UCMR 3), unseen (in UCMR 3, nothing reported), not_monitored (not in UCMR 3)
  max_oaos   highest entry-point mean of PFOA or PFOS, ng/L; max_pfoa and max_pfos separately
  pop        SDWIS population served (whole system, upper bound); pop_lower allocates it by the share of
             entry points that exceed (lower bound), as in deep_misc.py
  treated, tested, never, dk_only, seen_now_u3rule  as in deep_story.py
Definitions and inputs are those of deep_story.py; the script checks that it reproduces deep_story.json.
"""
import json
import pickle

import numpy as np
import pandas as pd

from pilot_common import SIX, U3MRL

C = pickle.load(open("cache.pkl", "rb"))
wide, meta, pop, codes, wide3, meta3 = C["wide"], C["meta"], C["pop"], C["codes"], C["wide3"], C["meta3"]

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
ep_trt = pd.Series([bool(trt.get(l, set()) & {"GAC", "IEX", "NRO"}) for l in L5], index=L5)

exc_sys = sorted(set(pw5[exc_ep]))
in3, seen3 = set(pw3), set(pw3[det3])
rows = []
for p in exc_sys:
    L = L5[pw5 == p]
    o = set().union(*[occ.get(l, set()) for l in L])
    s = set().union(*[src.get(l, set()) for l in L])
    P = float(pop.get(p, np.nan))
    rows.append({
        "PWSID": p, "state": p[:2],
        "cls": "seen" if p in seen3 else ("unseen" if p in in3 else "not_monitored"),
        "max_oaos": float(lm5.loc[L, ["PFOA", "PFOS"]].max(axis=1).max()),
        "max_pfoa": float(lm5.loc[L, "PFOA"].max()), "max_pfos": float(lm5.loc[L, "PFOS"].max()),
        "n_ep": int(len(L)), "n_ep_exc": int(exc_ep[L].sum()),
        "pop": P, "pop_lower": P * exc_ep[L].sum() / len(L),
        "size": meta.Size.reindex(L).iloc[0],
        "any_sw": bool(meta.FacilityWaterType.reindex(L).isin(["SW", "GU", "MX"]).any()),
        "treated": bool(ep_trt[L].any()), "tested": "Yes" in o, "never": ("Yes" not in o) and ("No" in o),
        "dk_only": o == {"DK"} or not o, "aware_src": "Yes" in s, "seen_now_u3rule": bool(seen_now_u3rule[L].any())})
D = pd.DataFrame(rows)
D.to_csv("fig_r6_systems.csv", index=False, float_format="%.4g")

S = json.load(open("deep_story.json"))
summ = {}
for c, g in D.groupby("cls"):
    summ[c] = {"n": int(len(g)), "pop_upper_m": round(g["pop"].sum() / 1e6, 2), "pop_lower_m": round(g.pop_lower.sum() / 1e6, 2),
               "max_oaos_median": round(float(g.max_oaos.median()), 1), "share_max_below_20": round(float((g.max_oaos < 20).mean()), 6),
               "share_pfoa_below_20_and_pfos_below_40": round(float(((g.max_pfoa < 20) & (g.max_pfos < 40)).mean()), 6),
               "treated": round(float(g.treated.mean()), 6), "tested": round(float(g.tested.mean()), 6),
               "never": round(float(g.never.mean()), 6), "dk_only": round(float(g.dk_only.mean()), 6),
               "aware_src": round(float(g.aware_src.mean()), 6), "seen_now_u3rule": round(float(g.seen_now_u3rule.mean()), 6)}
checks = {
    "unseen n": (summ["unseen"]["n"], S["unseen_systems"]["unseen"]["n"]),
    "seen n": (summ["seen"]["n"], S["unseen_systems"]["seen"]["n"]),
    "unseen median max": (summ["unseen"]["max_oaos_median"], S["unseen_systems"]["unseen"]["max_oaos_median"]),
    "unseen treated": (round(summ["unseen"]["treated"], 3), round(S["unseen_systems"]["unseen"]["treated"], 3)),
    "seen treated": (round(summ["seen"]["treated"], 3), round(S["unseen_systems"]["seen"]["treated"], 3)),
    "unseen seen-now": (round(summ["unseen"]["seen_now_u3rule"], 3), round(S["unseen_systems"]["unseen"]["seen_today_under_ucmr3_rule"], 3)),
    "unseen aware": (round(summ["unseen"]["aware_src"], 3), round(S["unseen_systems"]["unseen"]["aware_of_sources"], 3)),
    "seen aware": (round(summ["seen"]["aware_src"], 3), round(S["unseen_systems"]["seen"]["aware_of_sources"], 3)),
    "unseen tested": (round(summ["unseen"]["tested"], 3), round(S["unseen_systems"]["unseen"]["tested_before_ucmr5"], 3)),
    "not monitored n": (summ["not_monitored"]["n"], S["never_tested_split"]["not_in_ucmr3"]["n"]),
    "not monitored pop": (summ["not_monitored"]["pop_upper_m"], S["never_tested_split"]["not_in_ucmr3"]["pop_m"]),
    "all exceeding": (len(D), S["all_exceeding_systems"]["n"]),
}
summ["checks"] = {k: {"here": a, "deep_story": b, "ok": bool(a == b)} for k, (a, b) in checks.items()}
json.dump(summ, open("fig_r6_data.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
