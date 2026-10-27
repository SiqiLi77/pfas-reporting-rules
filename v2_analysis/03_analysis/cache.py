import pickle
import warnings
import pandas as pd
from pilot_common import *

warnings.filterwarnings("ignore")
u5, wide, meta, dates, mrl = load_u5()
pop = load_pop()
a = pd.read_csv("data/u5/UCMR5_AddtlDataElem.txt", sep="\t", encoding="latin1", dtype=str)
a["loc"] = a.PWSID + "|" + a.FacilityID + "|" + a.SamplePointID
a["code"] = a.Response.str.strip()
codes = {k: g.groupby("loc").code.agg(lambda s: set(s.dropna())) for k, g in a.groupby("AdditionalDataElement")}

u3 = pd.read_parquet("u3_pfas.parquet")
u3["loc"] = u3.PWSID + "|" + u3.FacilityID + "|" + u3.SamplePointID
u3["v0"] = u3.val.fillna(0.0)
wide3 = u3.pivot_table(index=["loc", "SampleEventCode"], columns="Contaminant", values="v0", aggfunc="max")
meta3 = u3.dropna(subset=["loc"]).drop_duplicates("loc").set_index("loc")[["PWSID", "Size", "FacilityWaterType", "State"]]

pickle.dump({"wide": wide, "meta": meta, "dates": dates, "mrl": mrl, "pop": pop, "codes": codes,
             "wide3": wide3, "meta3": meta3}, open("cache.pkl", "wb"))
print("cached", wide.shape, wide3.shape)
