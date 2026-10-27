"""Detail rows for SI Table S47 (V2.00 addendum).

Run: cd 03_analysis && python3 deep_tables_extra.py  ->  deep_tables_extra.json
(a) systems flagged by the 2024 U.S. rule but not by PFOA/PFOS alone (lost under the 2026 proposal);
(b) the largest systems flagged by the RIVM benchmark but not by the U.S. PFOA/PFOS standards;
(c) Texas entry points exceeding Canada's 30 ng/L sum while passing the U.S. PFOA/PFOS standards.
Definitions match deep_misc.py and deep_family.py (entry-point means, non-reports = 0, EPA rounding).
"""
import json
import pickle
import warnings

import numpy as np
import pandas as pd

from pilot_common import M533

warnings.filterwarnings("ignore")
C = pickle.load(open("cache.pkl", "rb"))
wide, meta, pop = C["wide"], C["meta"], C["pop"]
raw = pd.read_csv("data/u5/UCMR5_All.txt", sep="\t", encoding="latin1", dtype=str,
                  usecols=["PWSID", "PWSName", "FacilityID", "FacilityName", "SamplePointID"]).drop_duplicates()
raw["loc"] = raw.PWSID + "|" + raw.FacilityID + "|" + raw.SamplePointID
fac = raw.drop_duplicates("loc").set_index("loc")
pname = raw.drop_duplicates("PWSID").set_index("PWSID").PWSName
sd = pd.read_csv("data/sdwa/SDWA_PUB_WATER_SYSTEMS.csv", usecols=["PWSID", "PWS_NAME", "SUBMISSIONYEARQUARTER"], dtype=str)
sdname = sd.sort_values("SUBMISSIONYEARQUARTER").drop_duplicates("PWSID", keep="last").set_index("PWSID").PWS_NAME

lm = wide.groupby(level=0).mean()
cnt = (wide > 0).groupby(level=0).sum()
pw = meta.PWSID.reindex(lm.index)
st = meta.State.reindex(lm.index)
hi = lm.PFHxS / 10 + lm.PFNA / 10 + lm["HFPO-DA"] / 10 + lm.PFBS / 2000
ncomp = (cnt[["PFHxS", "PFNA", "HFPO-DA", "PFBS"]] > 0).sum(axis=1)
oaos = (lm.PFOA >= 4.05) | (lm.PFOS >= 4.05)
minor = pd.DataFrame({"PFHxS": lm.PFHxS >= 15, "PFNA": lm.PFNA >= 15, "HFPO-DA": lm["HFPO-DA"] >= 15,
                      "HI": (hi >= 1.5) & (ncomp >= 2)})
out = {}

# (a) lost under the 2026 proposal
us_full_pws = set(pw[oaos | minor.any(axis=1)])
us_res_pws = set(pw[oaos])
lost = sorted(us_full_pws - us_res_pws)
rows = []
for p in lost:
    L = lm.index[(pw == p) & minor.any(axis=1)]
    which = sorted({c for c in minor.columns for l in L if minor.loc[l, c]})
    rows.append({"PWSID": p, "name": sdname.get(p, pname.get(p)), "state": p[:2], "pop": int(pop.get(p, 0)),
                 "n_ep_flagged": int(len(L)), "standards": which,
                 "max_PFHxS": round(float(lm.loc[L, "PFHxS"].max()), 1), "max_PFNA": round(float(lm.loc[L, "PFNA"].max()), 1),
                 "max_HFPO_DA": round(float(lm.loc[L, "HFPO-DA"].max()), 1), "max_HI": round(float(hi[L].max()), 2),
                 "max_PFOA_PFOS": round(float(lm.loc[L, ["PFOA", "PFOS"]].max(axis=1).max()), 2)})
out["lost_under_proposal"] = sorted(rows, key=lambda r: -r["pop"])
print("lost under proposal:", len(rows), "systems,", round(sum(r["pop"] for r in rows) / 1e6, 2), "M people")
for r in out["lost_under_proposal"]:
    print("  ", r)

# (b) RIVM-only systems
RPF_UP = {"PFBA": 0.05, "PFPeA": 0.05, "PFHxA": 0.01, "PFHpA": 1, "PFOA": 1, "PFNA": 10, "PFDA": 10,
          "PFUnA": 4, "PFDoA": 3, "PFTrDA": 3, "PFTA": 0.3, "PFBS": 0.001, "PFPeS": 0.6, "PFHxS": 0.6,
          "PFHpS": 2, "PFOS": 2, "HFPO-DA": 0.06, "ADONA": 0.03}
peq_parts = pd.DataFrame({k: lm[k] * v for k, v in RPF_UP.items()})
peq = peq_parts.sum(axis=1)
riv = peq > 4.4
riv_pws = set(pw[riv]) - us_res_pws
rows = []
for p in riv_pws:
    L = lm.index[(pw == p) & riv]
    sh = peq_parts.loc[L].div(peq.loc[L], axis=0).mean()
    rows.append({"PWSID": p, "name": sdname.get(p, pname.get(p)), "state": p[:2], "pop": int(pop.get(p, 0)),
                 "n_ep_flagged": int(len(L)), "max_peq": round(float(peq[L].max()), 2),
                 "max_PFOA_PFOS": round(float(lm.loc[L, ["PFOA", "PFOS"]].max(axis=1).max()), 2),
                 "share_PFOS": round(float(sh["PFOS"]), 3), "share_PFHxS": round(float(sh["PFHxS"]), 3),
                 "share_PFOA": round(float(sh["PFOA"]), 3)})
rows = sorted(rows, key=lambda r: -r["pop"])
out["rivm_only_n_pws"] = len(rows)
out["rivm_only_top"] = rows[:12]
print("RIVM-only systems:", len(rows))
for r in rows[:12]:
    print("  ", r)

# (c) Texas Canada-only entry points, summarized by system
ca = lm[M533].sum(axis=1) > 30
tx = lm.index[ca & ~oaos & (st == "TX")]
t = pd.DataFrame({"PWSID": pw.reindex(tx).values, "facility": fac.reindex(tx).FacilityName.values,
                  "wt": meta.FacilityWaterType.reindex(tx).values, "sum25": lm.loc[tx, M533].sum(axis=1).values,
                  "PFBA": lm.loc[tx, "PFBA"].values, "PFPeA": lm.loc[tx, "PFPeA"].values,
                  "PFHxA": lm.loc[tx, "PFHxA"].values, "oaos_max": lm.loc[tx, ["PFOA", "PFOS"]].max(axis=1).values}, index=tx)
g = t.groupby("PWSID").agg(n=("sum25", "size"), sum25=("sum25", "median"), PFBA=("PFBA", "median"),
                            PFPeA=("PFPeA", "median"), PFHxA=("PFHxA", "median"), oaos_max=("oaos_max", "max"),
                            facility=("facility", "first"), wt=("wt", lambda s: "/".join(sorted(set(s)))))
g["pop"] = [int(pop.get(p, 0)) for p in g.index]
g["name"] = [sdname.get(p, pname.get(p)) for p in g.index]
g = g.sort_values("pop", ascending=False)
out["texas_canada_only"] = {"n_ep": int(len(tx)), "n_pws": int(g.shape[0]),
                            "rows": [{"PWSID": p, **{k: (round(float(v), 2 if k == "oaos_max" else 1)
                                                         if isinstance(v, (float, np.floating)) else v)
                                                     for k, v in r.items()}} for p, r in g.head(20).iterrows()]}
print("Texas Canada-only:", len(tx), "EPs in", g.shape[0], "systems")
print(g.head(20).to_string())
out["ucmr5_mrl_ng_L"] = {k: float(v) for k, v in C["mrl"].items()}
json.dump(out, open("deep_tables_extra.json", "w"), indent=1, default=int)
