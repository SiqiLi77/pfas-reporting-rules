import pandas as pd, numpy as np
from pilot_common import *
u5 = pd.read_parquet("u5_pfas.parquet")
print("rows", len(u5), "NaN SamplePointID", u5.SamplePointID.isna().sum(), "NaN FacilityID", u5.FacilityID.isna().sum())
print(u5[u5.SamplePointID.isna()][["PWSID","FacilityID","State"]].drop_duplicates().head())
u5["loc"] = u5.PWSID + "|" + u5.FacilityID + "|" + u5.SamplePointID
g = u5.groupby(["loc","SampleEventCode","Contaminant"]).size()
print("dup loc-event-analyte:", (g>1).sum(), "max", g.max())
d = g[g>1].reset_index().head(3); print(d)
if (g>1).sum():
    k = g[g>1].index[0]
    print(u5.set_index(["loc","SampleEventCode","Contaminant"]).loc[k][["SampleID","CollectionDate","val","AnalyticalResultsSign"]])
det = u5[u5.AnalyticalResultsSign=="="]
chk = det.groupby("Contaminant").apply(lambda x: (x.val < x.mrl - 1e-9).sum())
print("detections below MRL per analyte:", chk[chk>0].to_dict())
print("detect with NaN val:", det.val.isna().sum(), "ND with value:", u5[(u5.AnalyticalResultsSign=="<") & u5.val.notna()].shape[0])
print("max values ng/L:", det.groupby("Contaminant").val.max().sort_values(ascending=False).head(8).round(1).to_dict())
print("Units:", u5.columns.tolist())
u3 = pd.read_parquet("u3_pfas.parquet")
u3["loc"] = u3.PWSID + "|" + u3.FacilityID + "|" + u3.SamplePointID
print("u3 locs", u3["loc"].nunique(), "SPtype", u3.drop_duplicates("loc").SamplePointType.value_counts().to_dict())
g3 = u3.groupby(["loc","SampleEventCode","Contaminant"]).size(); print("u3 dup", (g3>1).sum())
d3 = u3[u3.AnalyticalResultsSign=="="]
print("u3 det below MRL:", (d3.val < d3.mrl - 1e-9).sum(), "u3 det count by analyte", d3.Contaminant.value_counts().to_dict())
