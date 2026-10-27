import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pilot_common import *

warnings.filterwarnings("ignore")
C = pickle.load(open("cache.pkl", "rb"))
wide, meta, pop = C["wide"], C["meta"], C["pop"]
lm = wide.groupby(level=0).mean()
pws = meta.PWSID.reindex(lm.index)
RPF_UP = {"PFBA": 0.05, "PFPeA": 0.05, "PFHxA": 0.01, "PFHpA": 1, "PFOA": 1, "PFNA": 10, "PFDA": 10,
          "PFUnA": 4, "PFDoA": 3, "PFTrDA": 3, "PFTA": 0.3, "PFBS": 0.001, "PFPeS": 0.6, "PFHxS": 0.6,
          "PFHpS": 2, "PFOS": 2, "HFPO-DA": 0.06, "ADONA": 0.03}
RPF_LO = dict(RPF_UP, PFPeA=0.01, PFHpA=0.01, PFDA=4, PFTrDA=0.3, PFPeS=0.001, PFHpS=0.6)
mol = sum(lm[c] / MW[c] * NF[c] for c in ALL29)
panels = {
    "PFOA/PFOS max": lm[["PFOA", "PFOS"]].max(axis=1),
    "UCMR3 six sum": lm[SIX].sum(axis=1),
    "PFAS4 sum": lm[PFAS4].sum(axis=1),
    "MA6 sum": lm[MA6].sum(axis=1),
    "EU20(15) sum": lm[EU20_MEASURED].sum(axis=1),
    "Canada25 sum": lm[M533].sum(axis=1),
    "Total F (molar)": mol,
    "RPF upper": sum(lm[k] * v for k, v in RPF_UP.items()),
    "RPF lower": sum(lm[k] * v for k, v in RPF_LO.items()),
}
P = pd.DataFrame(panels)
out = {}

def jacc(F, a, b):
    u = (F[a] | F[b]).sum()
    return (F[a] & F[b]).sum() / u if u else np.nan

pairs = [("PFOA/PFOS max", "Canada25 sum"), ("PFOA/PFOS max", "PFAS4 sum"), ("RPF upper", "Canada25 sum"),
         ("RPF upper", "PFAS4 sum"), ("RPF lower", "PFAS4 sum"), ("RPF lower", "Canada25 sum"),
         ("Canada25 sum", "Total F (molar)"), ("UCMR3 six sum", "Canada25 sum"), ("RPF upper", "RPF lower")]
qs = [0.01, 0.02, 0.0428, 0.0786, 0.10]
sens = {}
for q in qs:
    F = pd.DataFrame({k: P[k] > P[k].quantile(1 - q) for k in P})
    sens[q] = {f"{a} vs {b}": round(jacc(F, a, b), 3) for a, b in pairs}
S = pd.DataFrame(sens)
print("Jaccard by top-q share:\n", S.to_string())
out["quantile_sensitivity"] = S.to_dict()

# PWS cluster bootstrap at q = 4.28%
q = 0.0428
F = pd.DataFrame({k: P[k] > P[k].quantile(1 - q) for k in P})
upws = pws.unique()
groups = pd.Series(np.arange(len(lm)), index=lm.index).groupby(pws.values).apply(list)
rng = np.random.default_rng(1)
boot = {f"{a} vs {b}": [] for a, b in pairs}
Fv = {k: F[k].values for k in F}
for r in range(300):
    take = rng.choice(len(groups), len(groups), replace=True)
    ix = np.concatenate([groups.iloc[t] for t in take])
    for a, b in pairs:
        A, B = Fv[a][ix], Fv[b][ix]
        boot[f"{a} vs {b}"].append((A & B).sum() / max((A | B).sum(), 1))
ci = {k: [round(np.percentile(v, 2.5), 3), round(np.percentile(v, 97.5), 3)] for k, v in boot.items()}
print("bootstrap 95% CI (q=4.28%):", ci)
out["boot_ci"] = ci

# a real toxicity-weighted guideline: RIVM 4.4 ng PEQ/L
for k in ["RPF upper", "RPF lower"]:
    f = P[k] > 4.4
    pw = set(pws[f])
    print(k, "> 4.4 ng PEQ/L:", int(f.sum()), "EPs", len(pw), "PWS", round(pop.reindex(list(pw)).sum() / 1e6, 1), "M")
    out[f"rivm_{k}"] = [int(f.sum()), len(pw), float(pop.reindex(list(pw)).sum() / 1e6)]
# RIVM lists no range for PFPeS; the lower set applies the adjacent-chain rule (PFBS factor), this keeps RIVM's 0.6.
f = sum(lm[k] * v for k, v in dict(RPF_LO, PFPeS=RPF_UP["PFPeS"]).items()) > 4.4
pw = set(pws[f])
print("RPF lower, PFPeS at RIVM's 0.6 > 4.4 ng PEQ/L:", int(f.sum()), "EPs", len(pw), "PWS",
      round(pop.reindex(list(pw)).sum() / 1e6, 1), "M")
out["rivm_RPF lower, PFPeS 0.6"] = [int(f.sum()), len(pw), float(pop.reindex(list(pw)).sum() / 1e6)]
oaos = (lm.PFOA >= 4.05) | (lm.PFOS >= 4.05)
f = P["RPF upper"] > 4.4
print("RIVM-flagged but pass US PFOA/PFOS:", int((f & ~oaos).sum()), "; US-flagged but pass RIVM:", int((oaos & ~f).sum()))
drivers = (pd.DataFrame({k: lm[k] * v for k, v in RPF_UP.items()})[f & ~oaos])
print("PEQ share drivers among RIVM-only:", (drivers.div(drivers.sum(axis=1), axis=0)).mean().sort_values(ascending=False).head(5).round(3).to_dict())
out["rivm_vs_us"] = {"rivm_only": int((f & ~oaos).sum()), "us_only": int((oaos & ~f).sum())}
json.dump(out, open("deep_family.json", "w"), indent=1, default=float)
