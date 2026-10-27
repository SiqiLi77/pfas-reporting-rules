import pickle
import re
import warnings
import pandas as pd
from pilot_common import *

warnings.filterwarnings("ignore")
C = pickle.load(open("cache.pkl", "rb"))
wide, meta, pop = C["wide"], C["meta"], C["pop"]
u5 = pd.read_parquet("u5_pfas.parquet")
u5["loc"] = u5.PWSID + "|" + u5.FacilityID + "|" + u5.SamplePointID
names = u5.drop_duplicates("PWSID").set_index("PWSID")
raw = pd.read_csv("data/u5/UCMR5_All.txt", sep="\t", encoding="latin1", dtype=str,
                  usecols=["PWSID", "PWSName", "FacilityID", "FacilityName", "SamplePointID"]).drop_duplicates()
raw["loc"] = raw.PWSID + "|" + raw.FacilityID + "|" + raw.SamplePointID
fac = raw.drop_duplicates("loc").set_index("loc")
lm = wide.groupby(level=0).mean()
oaos = (lm.PFOA >= 4.05) | (lm.PFOS >= 4.05)
ca = lm[M533].sum(axis=1) > 30
st = meta.State.reindex(lm.index)

tx = lm.index[ca & ~oaos & (st == "TX")]
t = pd.DataFrame({"pws": fac.loc[tx, "PWSName"], "facility": fac.loc[tx, "FacilityName"],
                  "wt": meta.FacilityWaterType.reindex(tx), "sum25": lm.loc[tx, M533].sum(axis=1).round(1),
                  "PFBA": lm.loc[tx, "PFBA"].round(1), "PFPeA": lm.loc[tx, "PFPeA"].round(1)})
t["pop"] = meta.PWSID.reindex(tx).map(pop).values
print("TX Canada-only EPs by PWS (top 20 by pop):")
print(t.groupby("pws").agg(n=("sum25", "size"), sum25=("sum25", "median"), PFBA=("PFBA", "median"),
                            PFPeA=("PFPeA", "median"), pop=("pop", "first"), fac=("facility", "first"))
      .sort_values("pop", ascending=False).head(20).to_string())

gx = lm.index[lm["HFPO-DA"] > 0]
g = pd.DataFrame({"state": st.reindex(gx), "pws": fac.loc[gx, "PWSName"], "fac": fac.loc[gx, "FacilityName"],
                  "HFPO-DA": lm.loc[gx, "HFPO-DA"].round(1), "PFMPA": lm.loc[gx, "PFMPA"].round(1),
                  "PFMBA": lm.loc[gx, "PFMBA"].round(1), "PFEESA": lm.loc[gx, "PFEESA"].round(1)})
print("\nHFPO-DA detected EPs by PWS:")
print(g.groupby(["state", "pws"]).agg(n=("HFPO-DA", "size"), hfpo=("HFPO-DA", "max"), pfmpa=("PFMPA", "max"),
                                       pfeesa=("PFEESA", "max"), fac=("fac", "first")).sort_values("n", ascending=False).head(15).to_string())

s = u5[u5.Contaminant.isin(["PFBA", "PFPeA", "PFHxA", "PFOS"])].copy()
s["fmt"] = s.SampleID.map(lambda x: re.sub(r"[0-9]", "9", re.sub(r"[A-Za-z]", "A", str(x)))[:12])
s["det"] = s.val.notna()
f11 = s[s.fmt == "99999999999"]
print("\n11-digit SampleID format: states", f11.drop_duplicates("SampleID").State.value_counts().head(8).to_dict(),
      "PWS", f11.PWSID.nunique())
print(f11.pivot_table(index="State", columns="Contaminant", values="det", aggfunc="mean").round(3).assign(
    n=f11.groupby("State").SampleID.nunique()).sort_values("n", ascending=False).head(6).to_string())
others = s[(s.fmt != "99999999999") & s.State.isin(f11.State.unique())]
print("same states, other formats:\n", others.pivot_table(index="State", columns="Contaminant", values="det", aggfunc="mean").round(3).head(6).to_string())
mx = u5[(u5.Contaminant == "PFBA")].sort_values("val", ascending=False).head(5)[["PWSID", "State", "FacilityWaterType", "CollectionDate", "val"]]
print("\nTop PFBA results:\n", mx.assign(name=mx.PWSID.map(names.PWSName) if "PWSName" in names else None).to_string())
