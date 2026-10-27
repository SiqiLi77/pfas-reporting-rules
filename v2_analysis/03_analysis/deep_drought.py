import glob
import json
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from pilot_common import *

warnings.filterwarnings("ignore")
C = pickle.load(open("cache.pkl", "rb"))
wide, meta, dates = C["wide"], C["meta"], C["dates"]
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
print("USDM rows", len(U), "counties", U.fips.nunique())

out = {}
def build(locs):
    w = wide.loc[locs]
    d = pd.to_datetime(dates.loc[locs], format="%m/%d/%Y")
    df = pd.DataFrame({"loc": [i[0] for i in w.index], "ev": [i[1] for i in w.index], "date": d.values,
                       "short": w[["PFBA", "PFPeA", "PFHxA", "PFBS"]].sum(axis=1).values,
                       "long": w[["PFOA", "PFOS", "PFHxS"]].sum(axis=1).values})
    df["fips"] = meta.PWSID.reindex(df["loc"]).map(pws_cty).values
    df = df.dropna(subset=["fips"]).sort_values("date")
    df = pd.merge_asof(df, U.rename(columns={"date": "udate"}), left_on="date", right_on="udate",
                       by="fips", direction="backward")
    df["q"] = df.date.dt.quarter
    return df.dropna(subset=["dsci"])

def fe_model(df, comp, label):
    ok = df.groupby("loc")[comp].transform(lambda s: (s > 0).all()) & (df.groupby("loc")[comp].transform("size") >= 2)
    s = df[ok].copy()
    s["y"] = np.log(s[comp])
    X = pd.get_dummies(s[["q", "ev"]].astype(str), drop_first=True).astype(float)
    X["dsci100"] = s.dsci / 100.0
    g = s["loc"].values
    Xd = X - X.groupby(g).transform("mean")
    yd = s.y - s.y.groupby(g).transform("mean")
    beta, *_ = np.linalg.lstsq(Xd.values, yd.values, rcond=None)
    resid = yd.values - Xd.values @ beta
    # cluster-robust SE by location
    XtX_inv = np.linalg.pinv(Xd.values.T @ Xd.values)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in pd.Series(np.arange(len(s))).groupby(g):
        xi = Xd.values[idx.values]; ei = resid[idx.values]
        sc = xi.T @ ei
        meat += np.outer(sc, sc)
    V = XtX_inv @ meat @ XtX_inv
    j = list(X.columns).index("dsci100")
    b, se = beta[j], np.sqrt(V[j, j])
    r = {"n_loc": int(s["loc"].nunique()), "n": len(s), "ratio_per_100dsci": round(float(np.exp(b)), 3),
         "ci": [round(float(np.exp(b - 1.96 * se)), 3), round(float(np.exp(b + 1.96 * se)), 3)],
         "sd_within_dsci": round(float((s.dsci - s.groupby("loc").dsci.transform("mean")).std()), 1)}
    print(label, comp, r)
    return r

ev = wide.groupby(level=0).size()
L4 = ev[ev == 4].index
L4sw = L4[meta.FacilityWaterType.reindex(L4).isin(["SW", "GU", "MX"])]
gw2 = ev[ev == 2].index
gw2 = gw2[meta.FacilityWaterType.reindex(gw2) == "GW"]
dsw = build(L4sw)
dgw = build(gw2)
for comp in ["short", "long"]:
    out[f"sw_{comp}"] = fe_model(dsw, comp, "SW 4-event")
    out[f"gw_{comp}"] = fe_model(dgw, comp, "GW 2-event (control)")
# median-split illustration within SW locations: samples in drier vs wetter weeks than the location mean
for comp in ["short", "long"]:
    ok = dsw.groupby("loc")[comp].transform(lambda s: (s > 0).all())
    s = dsw[ok].copy()
    s["lr"] = np.log(s[comp]) - s.groupby("loc")[comp].transform(lambda x: np.log(x).mean())
    s["anom"] = s.dsci - s.groupby("loc").dsci.transform("mean")
    s["bin"] = pd.cut(s.anom, [-1e9, -100, -25, 25, 100, 1e9], labels=["much wetter", "wetter", "typical", "drier", "much drier"])
    b = np.exp(s.groupby("bin").lr.median()).round(3)
    print(comp, "ratio to location mean by drought anomaly bin:", b.to_dict(), s.groupby("bin").size().to_dict())
    out[f"bins_{comp}"] = {"ratio": b.to_dict(), "n": s.groupby("bin").size().to_dict()}
json.dump(out, open("deep_drought.json", "w"), indent=1, default=str)
