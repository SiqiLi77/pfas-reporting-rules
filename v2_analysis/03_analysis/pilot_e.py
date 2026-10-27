import json
import warnings
import pandas as pd
from pilot_common import *

warnings.filterwarnings("ignore")
u5, wide, meta, dates, mrl = load_u5()
pop = load_pop()
ev = wide.groupby(level=0).size()
wt = meta.FacilityWaterType.reindex(ev.index)
full = ev[((wt == "GW") & (ev == 2)) | ((wt != "GW") & (ev == 4))].index
lm = wide.groupby(level=0).mean().loc[full]
cnt = (wide > 0).groupby(level=0).sum().loc[full]
print("full-set locs", len(full), "PWS", meta.loc[full, "PWSID"].nunique())

def flags(lm, cnt, rule):
    if rule == "epa_rounding":
        t4, t10, thi = 4.05, 15, 1.5
    else:
        t4, t10, thi = 4.0, 10, 1.0
    hi = lm.PFHxS / 10 + lm.PFNA / 10 + lm["HFPO-DA"] / 10 + lm.PFBS / 2000
    ncomp = (cnt[["PFHxS", "PFNA", "HFPO-DA", "PFBS"]] > 0).sum(axis=1)
    return pd.DataFrame({
        "PFOS": lm.PFOS >= t4 if rule == "epa_rounding" else lm.PFOS > t4,
        "PFOA": lm.PFOA >= t4 if rule == "epa_rounding" else lm.PFOA > t4,
        "HFPO-DA": lm["HFPO-DA"] >= t10 if rule == "epa_rounding" else lm["HFPO-DA"] > t10,
        "PFHxS": lm.PFHxS >= t10 if rule == "epa_rounding" else lm.PFHxS > t10,
        "PFNA": lm.PFNA >= t10 if rule == "epa_rounding" else lm.PFNA > t10,
        "HI": ((hi >= thi) & (ncomp >= 2)) if rule == "epa_rounding" else (hi > thi),
    })

res = {}
for rule in ["epa_rounding", "literal"]:
    F = flags(lm, cnt, rule)
    pws = meta.loc[full, "PWSID"]
    per = {c: int(pws[F[c]].nunique()) for c in F}
    anyp = set(pws[F.any(axis=1)])
    keep = set(pws[(F.PFOA | F.PFOS)])
    lost = anyp - keep
    res[rule] = {"per_mcl_pws": per, "any_pws": len(anyp), "pfoa_pfos_pws": len(keep),
                 "lost_pws": len(lost), "lost_pop_m": round(pop.reindex(list(lost)).sum() / 1e6, 2),
                 "any_pop_m": round(pop.reindex(list(anyp)).sum() / 1e6, 1)}
    print(rule, res[rule])
json.dump(res, open("pilot_e.json", "w"), indent=1)
