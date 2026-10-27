import pandas as pd

cols = ["PWSID", "Size", "FacilityID", "FacilityWaterType", "SamplePointID",
        "SamplePointType", "CollectionDate", "SampleID", "Contaminant", "MRL",
        "MethodID", "AnalyticalResultsSign", "AnalyticalResultValue",
        "SampleEventCode", "MonitoringRequirement", "State"]

u5 = pd.read_csv("data/u5/UCMR5_All.txt", sep="\t", encoding="latin1",
                 usecols=cols, dtype=str)
print(u5.shape)
print(u5.MethodID.value_counts())
print(u5.Contaminant.value_counts().to_string())
u5 = u5[u5.MethodID.isin(["EPA 533", "EPA 537.1"])].copy()
u5["val"] = pd.to_numeric(u5.AnalyticalResultValue, errors="coerce") * 1000
u5["mrl"] = pd.to_numeric(u5.MRL, errors="coerce") * 1000
u5.to_parquet("u5_pfas.parquet")

u3 = pd.read_csv("data/u3/UCMR3_All.txt", sep="\t", encoding="latin1",
                 usecols=cols, dtype=str)
print(u3.MethodID.value_counts())
u3 = u3[u3.MethodID == "EPA 537"].copy()
print(u3.Contaminant.value_counts())
u3["val"] = pd.to_numeric(u3.AnalyticalResultValue, errors="coerce") * 1000
u3["mrl"] = pd.to_numeric(u3.MRL, errors="coerce") * 1000
u3.to_parquet("u3_pfas.parquet")
print(u5.AnalyticalResultsSign.value_counts(), u3.AnalyticalResultsSign.value_counts())
print(u5.groupby("Contaminant").mrl.unique())
print(u3.groupby("Contaminant").mrl.unique())
