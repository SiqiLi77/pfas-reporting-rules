import numpy as np
import pandas as pd

M533 = ["4:2 FTS", "6:2 FTS", "8:2 FTS", "9Cl-PF3ONS", "11Cl-PF3OUdS", "ADONA",
        "HFPO-DA", "NFDHA", "PFBA", "PFBS", "PFDA", "PFDoA", "PFEESA", "PFHpA",
        "PFHpS", "PFHxA", "PFHxS", "PFMBA", "PFMPA", "PFNA", "PFOA", "PFOS",
        "PFPeA", "PFPeS", "PFUnA"]
M5371_ONLY = ["NEtFOSAA", "NMeFOSAA", "PFTA", "PFTrDA"]
ALL29 = M533 + M5371_ONLY
SIX = ["PFOA", "PFOS", "PFHxS", "PFNA", "PFHpA", "PFBS"]
EU20_MEASURED = ["PFBA", "PFPeA", "PFHxA", "PFHpA", "PFOA", "PFNA", "PFDA", "PFUnA",
                 "PFDoA", "PFTrDA", "PFBS", "PFPeS", "PFHxS", "PFHpS", "PFOS"]
PFAS4 = ["PFOA", "PFNA", "PFHxS", "PFOS"]
MA6 = ["PFOS", "PFOA", "PFHxS", "PFNA", "PFHpA", "PFDA"]
VT5 = ["PFOA", "PFOS", "PFHxS", "PFNA", "PFHpA"]
U3MRL = {"PFOA": 20, "PFOS": 40, "PFHxS": 30, "PFNA": 20, "PFHpA": 10, "PFBS": 90}
SHORT = ["PFBA", "PFPeA", "PFHxA", "PFBS", "PFPeS", "PFHpA"]
LONG = ["PFOA", "PFNA", "PFDA", "PFUnA", "PFDoA", "PFTrDA", "PFTA", "PFHxS",
        "PFHpS", "PFOS"]
ETHER = ["HFPO-DA", "ADONA", "PFMPA", "PFMBA", "PFEESA", "NFDHA", "9Cl-PF3ONS",
         "11Cl-PF3OUdS"]
FTS = ["4:2 FTS", "6:2 FTS", "8:2 FTS"]

MW = {"PFBA": 214.04, "PFPeA": 264.05, "PFHxA": 314.05, "PFHpA": 364.06,
      "PFOA": 414.07, "PFNA": 464.08, "PFDA": 514.08, "PFUnA": 564.09,
      "PFDoA": 614.1, "PFTrDA": 664.11, "PFTA": 714.11, "PFBS": 300.1,
      "PFPeS": 350.11, "PFHxS": 400.12, "PFHpS": 450.12, "PFOS": 500.13,
      "4:2 FTS": 328.18, "6:2 FTS": 428.17, "8:2 FTS": 528.18, "HFPO-DA": 330.05,
      "ADONA": 378.08, "PFMPA": 230.03, "PFMBA": 280.04, "PFEESA": 316.1,
      "NFDHA": 296.05, "9Cl-PF3ONS": 532.58, "11Cl-PF3OUdS": 632.6,
      "NEtFOSAA": 585.24, "NMeFOSAA": 571.21}
NF = {"PFBA": 7, "PFPeA": 9, "PFHxA": 11, "PFHpA": 13, "PFOA": 15, "PFNA": 17,
      "PFDA": 19, "PFUnA": 21, "PFDoA": 23, "PFTrDA": 25, "PFTA": 27, "PFBS": 9,
      "PFPeS": 11, "PFHxS": 13, "PFHpS": 15, "PFOS": 17, "4:2 FTS": 9,
      "6:2 FTS": 13, "8:2 FTS": 17, "HFPO-DA": 11, "ADONA": 12, "PFMPA": 7,
      "PFMBA": 9, "PFEESA": 9, "NFDHA": 9, "9Cl-PF3ONS": 16, "11Cl-PF3OUdS": 20,
      "NEtFOSAA": 17, "NMeFOSAA": 17}


def load_u5():
    u5 = pd.read_parquet("u5_pfas.parquet")
    u5["loc"] = u5.PWSID + "|" + u5.FacilityID + "|" + u5.SamplePointID
    u5["v0"] = u5.val.fillna(0.0)
    wide = u5.pivot_table(index=["loc", "SampleEventCode"], columns="Contaminant",
                          values="v0", aggfunc="max")
    wide = wide.reindex(columns=ALL29)
    meta = u5.dropna(subset=["loc"]).drop_duplicates("loc").set_index("loc")[
        ["PWSID", "Size", "FacilityWaterType", "State"]]
    # An event's Method 533 and 537.1 samples can be months apart. Date each event by its earliest Method 533 sample,
    # which carries 25 of the 29 PFAS, so that the date does not depend on the row order of the EPA file.
    o = u5.dropna(subset=["loc"]).assign(_n=u5.MethodID != "EPA 533",
                                         _d=pd.to_datetime(u5.CollectionDate, format="%m/%d/%Y"))
    dates = (o.sort_values(["loc", "SampleEventCode", "_n", "_d"], kind="stable")
             .drop_duplicates(["loc", "SampleEventCode"]).set_index(["loc", "SampleEventCode"]).CollectionDate)
    mrl = u5.groupby("Contaminant").mrl.first().reindex(ALL29)
    return u5, wide, meta, dates, mrl


def load_pop():
    p = pd.read_csv("data/sdwa/SDWA_PUB_WATER_SYSTEMS.csv",
                    usecols=["PWSID", "POPULATION_SERVED_COUNT", "SUBMISSIONYEARQUARTER",
                             "PWS_TYPE_CODE"], dtype=str)
    p["pop"] = pd.to_numeric(p.POPULATION_SERVED_COUNT, errors="coerce")
    p = p.sort_values("SUBMISSIONYEARQUARTER").drop_duplicates("PWSID", keep="last")
    return p.set_index("PWSID")["pop"]
