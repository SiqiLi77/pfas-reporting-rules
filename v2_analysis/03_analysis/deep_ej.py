import json
import pickle
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pilot_common import *

warnings.filterwarnings("ignore")
C = pickle.load(open("cache.pkl", "rb"))
wide, meta, pop, wide3, meta3 = C["wide"], C["meta"], C["pop"], C["wide3"], C["meta3"]

def read_acs(path, cols):
    it = pd.read_csv(path, sep="|", dtype=str, usecols=["GEO_ID"] + cols, chunksize=200000)
    df = pd.concat([c[c.GEO_ID.str.startswith("860Z200US")] for c in it])
    df["zcta"] = df.GEO_ID.str[-5:]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("zcta")[cols]

r = read_acs("data/b03002.dat", ["B03002_E001", "B03002_E003", "B03002_E004", "B03002_E012"])
inc = read_acs("data/b19013.dat", ["B19013_E001"])
pv = read_acs("data/b17001.dat", ["B17001_E001", "B17001_E002"])
z = r.join(inc).join(pv)
z.columns = ["pop", "nhw", "nhb", "hisp", "mhi", "pov_den", "pov_num"]
z.loc[z.mhi < 0, "mhi"] = np.nan
print("ZCTAs", len(z))

zips = pd.read_csv("data/u5/UCMR5_ZIPCodes.txt", sep="\t", dtype=str, encoding="latin1")
zz = zips.merge(z, left_on="ZIPCODE", right_index=True, how="inner")
agg = zz.groupby("PWSID").agg(zpop=("pop", "sum"), nhb=("nhb", "sum"), hisp=("hisp", "sum"),
                              pov_num=("pov_num", "sum"), pov_den=("pov_den", "sum"))
zz["mhi_w"] = zz.mhi * zz["pop"]
agg["mhi"] = zz.groupby("PWSID").mhi_w.sum() / zz.dropna(subset=["mhi"]).groupby("PWSID")["pop"].sum()
agg = agg[agg.zpop > 0]
agg["pct_black"] = 100 * agg.nhb / agg.zpop
agg["pct_hisp"] = 100 * agg.hisp / agg.zpop
agg["pct_pov"] = 100 * agg.pov_num / agg.pov_den
print("PWS with demographics", len(agg))

lm = wide.groupby(level=0).mean()
pw = meta.PWSID.reindex(lm.index)
w3c = wide[SIX].copy()
for c in SIX:
    w3c.loc[w3c[c] < U3MRL[c], c] = 0
det_u3rule = (w3c > 0).groupby(level=0).any().any(axis=1)
RPF = {"PFBA": 0.05, "PFPeA": 0.05, "PFHxA": 0.01, "PFHpA": 1, "PFOA": 1, "PFNA": 10, "PFDA": 10,
       "PFUnA": 4, "PFDoA": 3, "PFTrDA": 3, "PFTA": 0.3, "PFBS": 0.001, "PFPeS": 0.6, "PFHxS": 0.6,
       "PFHpS": 2, "PFOS": 2, "HFPO-DA": 0.06, "ADONA": 0.03}
peq = sum(lm[k] * v for k, v in RPF.items())
outcomes = {
    "Any detection, UCMR3 rule (6 @ old MRLs)": det_u3rule,
    "Any detection, UCMR5 rule (29 @ new MRLs)": (wide > 0).groupby(level=0).any().any(axis=1),
    "US PFOA/PFOS MCL": (lm.PFOA >= 4.05) | (lm.PFOS >= 4.05),
    "Canada sum-25 > 30": lm[M533].sum(axis=1) > 30,
    "RIVM > 4.4 ng PEQ/L": peq > 4.4,
}
Y = pd.DataFrame({k: v.groupby(pw).any() for k, v in outcomes.items()})
d3 = (wide3 > 0).groupby(level=0).any().any(axis=1)
Y3 = d3.groupby(meta3.PWSID.reindex(d3.index)).any().rename("Actual UCMR3 detection (2013-15)")
anysw = meta.FacilityWaterType.isin(["SW", "GU", "MX"]).groupby(meta.PWSID).any()
st = meta.drop_duplicates("PWSID").set_index("PWSID").State
X = agg[["pct_black", "pct_hisp", "pct_pov"]].copy()
X["log_pop"] = np.log10(pop.reindex(X.index).clip(lower=25))
X["any_sw"] = anysw.reindex(X.index).astype(float)
X["state"] = st.reindex(X.index)
D = X.join(Y).join(Y3).dropna(subset=["pct_black", "pct_hisp", "pct_pov", "log_pop", "state"])
print("analysis PWS", len(D))

res = {}
def fit(ycol, data, fe=True):
    dd = data.dropna(subset=[ycol])
    Xm = dd[["pct_black", "pct_hisp", "pct_pov", "log_pop", "any_sw"]].copy()
    Xm[["pct_black", "pct_hisp", "pct_pov"]] /= 10.0
    if fe:
        keep = dd.groupby("state")[ycol].transform(lambda s: s.nunique() > 1)
        dd, Xm = dd[keep], Xm[keep]
        Xm = pd.concat([Xm, pd.get_dummies(dd.state, prefix="st", drop_first=True).astype(float)], axis=1)
    Xm = sm.add_constant(Xm)
    m = sm.GLM(dd[ycol].astype(float), Xm, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.state)[0]})
    o = {}
    for v in ["pct_black", "pct_hisp", "pct_pov"]:
        b, se = m.params[v], m.bse[v]
        o[v] = [round(np.exp(b), 3), round(np.exp(b - 1.96 * se), 3), round(np.exp(b + 1.96 * se), 3), round(m.pvalues[v], 4)]
    return o, int(dd[ycol].sum()), len(dd)

for ycol in list(Y.columns) + ["Actual UCMR3 detection (2013-15)"]:
    for fe in [True, False]:
        o, npos, n = fit(ycol, D, fe)
        res[f"{ycol} | {'stateFE' if fe else 'noFE'}"] = {"npos": npos, "n": n, **o}
        print(f"{ycol:45s} {'FE ' if fe else 'raw'} n={n} pos={npos}  OR/10pp Black {o['pct_black'][:3]}  Hisp {o['pct_hisp'][:3]}  Pov {o['pct_pov'][:3]}")
json.dump(res, open("deep_ej.json", "w"), indent=1, default=float)
