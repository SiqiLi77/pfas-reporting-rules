"""Compare the regulatory convention used here with EPA's July 2026 UCMR 5 Data Summary, Table 4.

Run: cd 03_analysis && python3 calib_epa_table4.py  ->  calib_epa_table4.json
EPA counts are transcribed from data/u5/UCMR5_DataSummary_Jul2026_508.pdf (Table 4, page 12).
Size classes follow EPA: large >10,000 (UCMR 'L'); medium 3,300-10,000 and small <3,300 split by
SDWIS population among UCMR 'S' systems.
"""
import json
import warnings

import pandas as pd

from pilot_common import load_pop, load_u5

warnings.filterwarnings("ignore")

EPA = {  # Table 4: systems with full set(s) of results, and systems with averages > MCL
    "full_sets": {"large": 4277, "medium": 5077, "small": 775},
    "PFOS": {"large": 523, "medium": 307, "small": 45},
    "PFOA": {"large": 472, "medium": 273, "small": 34},
    "HFPO-DA": {"large": 3, "medium": 5, "small": 0},
    "PFHxS": {"large": 40, "medium": 22, "small": 4},
    "PFNA": {"large": 3, "medium": 6, "small": 0},
    "HI": {"large": 45, "medium": 31, "small": 4},
    "any": {"large": 662, "medium": 419, "small": 55},
}

u5, wide, meta, dates, mrl = load_u5()
pop = load_pop()
ev = wide.groupby(level=0).size()
wt = meta.FacilityWaterType.reindex(ev.index)
full = ev[((wt == "GW") & (ev == 2)) | ((wt != "GW") & (ev == 4))].index
lm = wide.groupby(level=0).mean().loc[full]
cnt = (wide > 0).groupby(level=0).sum().loc[full]
hi = lm.PFHxS / 10 + lm.PFNA / 10 + lm["HFPO-DA"] / 10 + lm.PFBS / 2000
ncomp = (cnt[["PFHxS", "PFNA", "HFPO-DA", "PFBS"]] > 0).sum(axis=1)
F = pd.DataFrame({
    "PFOS": lm.PFOS >= 4.05, "PFOA": lm.PFOA >= 4.05, "HFPO-DA": lm["HFPO-DA"] >= 15,
    "PFHxS": lm.PFHxS >= 15, "PFNA": lm.PFNA >= 15, "HI": (hi >= 1.5) & (ncomp >= 2),
})
F["any"] = F.any(axis=1)
pws = meta.loc[full, "PWSID"]
size_u = meta.loc[full].drop_duplicates("PWSID").set_index("PWSID")["Size"]
ppop = pop.reindex(size_u.index)
cls = pd.Series("medium", index=size_u.index)
cls[size_u == "L"] = "large"
cls[(size_u != "L") & (ppop < 3300)] = "small"

out = {"full_set_locations": int(len(full)), "full_set_pws": int(pws.nunique()),
       "full_sets": cls.value_counts().to_dict(), "rows": {}}
for k in ["PFOS", "PFOA", "HFPO-DA", "PFHxS", "PFNA", "HI", "any"]:
    flagged = pd.Index(pws[F[k]].unique())
    here = cls.reindex(flagged).value_counts().to_dict()
    out["rows"][k] = {"here_total": int(len(flagged)), "epa_total": int(sum(EPA[k].values())),
                      "here": {c: int(here.get(c, 0)) for c in ["large", "medium", "small"]}, "epa": EPA[k]}
    print(k, out["rows"][k])
print("full-set locations", out["full_set_locations"], "PWS", out["full_set_pws"], out["full_sets"],
      "| EPA:", EPA["full_sets"])
json.dump(out, open("calib_epa_table4.json", "w"), indent=1)
