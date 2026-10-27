import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pilot_common import *

warnings.filterwarnings("ignore")
C = pickle.load(open("cache.pkl", "rb"))
wide, meta, pop, codes, wide3, meta3 = C["wide"], C["meta"], C["pop"], C["codes"], C["wide3"], C["meta3"]
out = {}

lm = wide.groupby(level=0).mean()
cnt = (wide > 0).groupby(level=0).sum()
pw = meta.PWSID.reindex(lm.index)
n_ep = pw.groupby(pw).size()
pt = codes["PFASTreatment"]
treated = pd.Series([bool(pt.get(l, set()) & {"GAC", "IEX", "NRO"}) for l in lm.index], index=lm.index)

# ---------- 1. short-chain hazard quotients ----------
DWI, RSC = 0.0469, 0.2
RFD = {"PFBA": 1e-3, "PFHxA": 5e-4, "PFBS": 3e-4}
HBWC_EPA = {k: v * 1e6 * RSC / DWI for k, v in RFD.items()}
HBWC_MDH = {"PFBA": 7000, "PFHxA": 200, "PFBS": 100}
mx = wide.groupby(level=0).max()
hq = {}
for lab, H in [("EPA-style (IRIS RfD, RSC 0.2, DWI 0.0469)", HBWC_EPA), ("Minnesota HBV", HBWC_MDH)]:
    hi_mean = sum(lm[k] / v for k, v in H.items())
    hi_max = sum(mx[k] / v for k, v in H.items())
    r = {"HBWC_ng_L": {k: round(v) for k, v in H.items()},
         "max_HI_mean": round(float(hi_mean.max()), 3),
         "EP_HI_gt_1": int((hi_mean > 1).sum()), "EP_HI_gt_0.1": int((hi_mean > 0.1).sum()),
         "EP_HI_gt_1_any_sample": int((hi_max > 1).sum()),
         "PWS_HI_gt_1": int(pw[hi_mean > 1].nunique()),
         "pop_HI_gt_1_m": round(float(pop.reindex(list(set(pw[hi_mean > 1]))).sum() / 1e6), 2),
         "treated_EP_HI_gt_0.1": int((hi_mean[treated] > 0.1).sum()), "treated_EP_n": int(treated.sum()),
         "states_HI_gt_1": meta.State.reindex(hi_mean[hi_mean > 1].index).value_counts().head(6).to_dict(),
         "driver_share": {k: round(float((lm[k] / v)[hi_mean > 1].sum() / hi_mean[hi_mean > 1].sum()), 3)
                          for k, v in H.items()} if (hi_mean > 1).any() else {}}
    hq[lab] = r
    print(lab, r)
out["short_chain_hq"] = hq
for k in ["PFBA", "PFHxA", "PFBS"]:
    print(k, "EP mean p99", round(lm[k].quantile(0.99), 1), "max", round(lm[k].max(), 1),
          "| treated p99", round(lm[k][treated].quantile(0.99), 1))

# ---------- 2. regime flags with population bounds ----------
RPF = {"PFBA": 0.05, "PFPeA": 0.05, "PFHxA": 0.01, "PFHpA": 1, "PFOA": 1, "PFNA": 10, "PFDA": 10,
       "PFUnA": 4, "PFDoA": 3, "PFTrDA": 3, "PFTA": 0.3, "PFBS": 0.001, "PFPeS": 0.6, "PFHxS": 0.6,
       "PFHpS": 2, "PFOS": 2, "HFPO-DA": 0.06, "ADONA": 0.03}
hi4 = lm.PFHxS / 10 + lm.PFNA / 10 + lm["HFPO-DA"] / 10 + lm.PFBS / 2000
ncomp = (cnt[["PFHxS", "PFNA", "HFPO-DA", "PFBS"]] > 0).sum(axis=1)
us_res = (lm.PFOA >= 4.05) | (lm.PFOS >= 4.05)
us_full = us_res | (lm.PFHxS >= 15) | (lm.PFNA >= 15) | (lm["HFPO-DA"] >= 15) | ((hi4 >= 1.5) & (ncomp >= 2))
p4 = lm[PFAS4].sum(axis=1)
R = {
    "US 2024 NPDWR (6 MCLs, EPA rounding)": us_full,
    "US after 2026 rescission (PFOA/PFOS)": us_res,
    "Canada sum-25 > 30": lm[M533].sum(axis=1) > 30,
    "EU sum-20 (15 measured) > 100": lm[EU20_MEASURED].sum(axis=1) > 100,
    "Germany PFAS-4 > 20": p4 > 20,
    "Sweden PFAS-4 > 4": p4 > 4,
    "Denmark PFAS-4 > 2": p4 > 2,
    "RIVM > 4.4 ng PEQ/L": sum(lm[k] * v for k, v in RPF.items()) > 4.4,
    "Massachusetts PFAS6 > 20": lm[MA6].sum(axis=1) > 20,
    "Vermont PFAS5 > 20": lm[VT5].sum(axis=1) > 20,
    "New Jersey (PFOA 14/PFOS 13/PFNA 13)": (lm.PFOA > 14) | (lm.PFOS > 13) | (lm.PFNA > 13),
    "China GB 5749 (PFOA 80/PFOS 40)": (lm.PFOA > 80) | (lm.PFOS > 40),
}
us_pws = set(pw[us_res])

def bounds(flag):
    f = flag.groupby(pw).sum()
    f = f[f > 0]
    upper = pop.reindex(f.index).sum()
    lower = (pop.reindex(f.index) * f / n_ep.reindex(f.index)).sum()
    return f.index, upper, lower

tab = {}
for k, f in R.items():
    ids, up, lo = bounds(f)
    extra = [p for p in ids if p not in us_pws]
    tab[k] = {"EP": int(f.sum()), "PWS": len(ids), "pop_upper_m": round(up / 1e6, 1), "pop_lower_m": round(lo / 1e6, 1),
              "PWS_not_flagged_by_US_rescinded": len(extra),
              "pop_not_flagged_by_US_rescinded_m": round(pop.reindex(extra).sum() / 1e6, 1)}
    print(f"{k:42s}", tab[k])
out["regimes"] = tab
tot = pop.reindex(pw.unique()).sum()
out["ucmr5_total_pop_m"] = round(tot / 1e6, 1)
print("UCMR5 total pop served (M):", round(tot / 1e6, 1))

# ---------- 3. lost-decade population bounds ----------
det3 = (wide3 > 0).groupby(level=0).any().any(axis=1)
p3 = meta3.PWSID.reindex(det3.index)
pd3 = det3.groupby(p3).any()
both = sorted(set(pw) & set(pd3.index))
f = us_res.groupby(pw).sum().reindex(both)
f = f[f > 0]
inv = [p for p in f.index if not pd3[p]]
up = pop.reindex(inv).sum()
lo = (pop.reindex(inv) * f.reindex(inv) / n_ep.reindex(inv)).sum()
up_all = pop.reindex(f.index).sum()
lo_all = (pop.reindex(f.index) * f / n_ep.reindex(f.index)).sum()
out["lost_decade_bounds"] = {"PWS": len(inv), "pop_upper_m": round(up / 1e6, 1), "pop_lower_m": round(lo / 1e6, 1),
                             "share_upper": round(up / up_all, 3), "share_lower": round(lo / lo_all, 3)}
print("lost decade:", out["lost_decade_bounds"])

# ---------- 4. RIVM-only PWS: what drives them ----------
riv = R["RIVM > 4.4 ng PEQ/L"]
rp = set(pw[riv]) - us_pws
drv = pd.DataFrame({k: lm[k] * v for k, v in RPF.items()})[riv & ~us_res & pw.isin(rp)]
share = drv.div(drv.sum(axis=1), axis=0).mean().sort_values(ascending=False).head(6).round(3).to_dict()
big = pop.reindex(list(rp)).sort_values(ascending=False).head(8)
out["rivm_only"] = {"PWS": len(rp), "pop_m": round(pop.reindex(list(rp)).sum() / 1e6, 1), "driver_share": share,
                    "largest": {k: int(v) for k, v in big.items()}}
print("RIVM-only:", out["rivm_only"])
json.dump(out, open("deep_misc.json", "w"), indent=1, default=float)
