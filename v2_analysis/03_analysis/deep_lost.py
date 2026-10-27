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

lm5 = wide.groupby(level=0).mean()
cnt5 = (wide > 0).groupby(level=0).sum()
exc5 = (lm5.PFOA >= 4.05) | (lm5.PFOS >= 4.05)
det5 = (wide > 0).groupby(level=0).any().any(axis=1)
det3 = (wide3 > 0).groupby(level=0).any().any(axis=1)
lm3 = wide3.groupby(level=0).mean()

m = lm5.index.intersection(lm3.index)
print("matched EPs:", len(m), "of UCMR5", len(lm5), "UCMR3", len(lm3))
st5 = pd.Series(np.where(exc5[m], "UCMR5 超标", np.where(det5[m], "UCMR5 有检出未超标", "UCMR5 未检出")), index=m)
st3 = pd.Series(np.where(det3[m], "UCMR3 有检出", "UCMR3 未检出"), index=m)
T = pd.crosstab(st3, st5)
print(T)
out["transition"] = T.to_dict()
sens = (exc5[m] & det3[m]).sum() / exc5[m].sum()
ppv = (exc5[m] & det3[m]).sum() / det3[m].sum()
print(f"sensitivity of UCMR3 detection for current exceedance {sens:.3f}; PPV {ppv:.3f}")
out["sens"] = float(sens); out["ppv"] = float(ppv)

# within exceeders: by concentration band
mx = lm5[["PFOA", "PFOS"]].max(axis=1)
for lo, hi in [(4.05, 8), (8, 20), (20, 1e9)]:
    s = m[(exc5[m]) & (mx[m] >= lo) & (mx[m] < hi)]
    print(f"band {lo}-{hi}: n={len(s)} UCMR3 detected {det3[s].mean():.3f}")
    out[f"band_{lo}"] = [len(s), float(det3[s].mean())]

# PWS level (robust to facility/sample point ID changes)
p5 = meta.PWSID.reindex(lm5.index)
p3 = meta3.PWSID.reindex(lm3.index)
pws_both = sorted(set(p5) & set(p3))
pe5 = exc5.groupby(p5).any().reindex(pws_both)
pd3 = det3.groupby(p3).any().reindex(pws_both)
pp = pop.reindex(pws_both)
print("PWS in both cycles:", len(pws_both), "pop M", round(pp.sum() / 1e6, 1))
print("currently exceeding PWS:", int(pe5.sum()), "pop M", round(pp[pe5].sum() / 1e6, 1))
inv = pe5 & ~pd3
print("  of which no UCMR3 detection at any EP:", int(inv.sum()), "pop M", round(pp[inv].sum() / 1e6, 1),
      f"({pp[inv].sum() / pp[pe5].sum():.3f} of exceeding pop)")
out["pws"] = {"both": len(pws_both), "exc": int(pe5.sum()), "exc_pop_m": float(pp[pe5].sum() / 1e6),
              "invisible": int(inv.sum()), "invisible_pop_m": float(pp[inv].sum() / 1e6)}
# Hu et al. 2016-style: PWS with UCMR3 detection
print("PWS with UCMR3 detection:", int(pd3.sum()), "pop M", round(pp[pd3].sum() / 1e6, 1))

# accountability: did UCMR3-detected EPs install PFAS treatment, and did they resolve?
pt = codes["PFASTreatment"]
treated = pd.Series([bool(pt.get(l, set()) & {"GAC", "IEX", "NRO"}) for l in m], index=m)
print("PFAS treatment installed: UCMR3-detected EPs", round(treated[det3[m]].mean(), 3), f"(n={int(det3[m].sum())})",
      "vs UCMR3-ND EPs", round(treated[~det3[m]].mean(), 3))
out["treated_share"] = {"det3": float(treated[det3[m]].mean()), "nd3": float(treated[~det3[m]].mean()),
                        "n_det3": int(det3[m].sum())}
# shared-cutoff persistence among UCMR3-detected EPs
w5c = wide[SIX].copy()
for c in SIX:
    w5c.loc[w5c[c] < U3MRL[c], c] = 0
det5_at3 = (w5c > 0).groupby(level=0).any().any(axis=1)
d3m = m[det3[m]]
persist = det5_at3[d3m]
print("UCMR3-detected EPs still detected at UCMR3 MRLs in UCMR5:", round(persist.mean(), 3))
print("  treated among persisted:", round(treated[d3m][persist].mean(), 3), "treated among resolved:", round(treated[d3m][~persist].mean(), 3))
print("  current MCL exceed among UCMR3-detected:", round(exc5[d3m].mean(), 3), "| treated:", round(exc5[d3m][treated[d3m]].mean(), 3),
      "untreated:", round(exc5[d3m][~treated[d3m]].mean(), 3))
out["persist"] = {"persist": float(persist.mean()), "treated_persist": float(treated[d3m][persist].mean()),
                  "treated_resolved": float(treated[d3m][~persist].mean()),
                  "exc_among_det3": float(exc5[d3m].mean())}
# new appearances: EPs detected at UCMR3 MRLs in UCMR5 but not in UCMR3
new = m[(~det3[m]) & det5_at3[m]]
print("new above-UCMR3-MRL appearances in UCMR5:", len(new), "vs persisted:", int(persist.sum()), "resolved:", int((~persist).sum()))
out["flows"] = {"new": len(new), "persist": int(persist.sum()), "resolved": int((~persist).sum())}
json.dump(out, open("deep_lost.json", "w"), indent=1, default=float)
