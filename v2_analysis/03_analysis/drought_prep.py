import pickle
import pandas as pd

C = pickle.load(open("cache.pkl", "rb"))
wide, meta = C["wide"], C["meta"]
ev = wide.groupby(level=0).size()
L4 = ev[ev == 4].index
L4sw = L4[meta.FacilityWaterType.reindex(L4).isin(["SW", "GU", "MX"])]
gw2 = ev[ev == 2].index
gw2 = gw2[meta.FacilityWaterType.reindex(gw2) == "GW"]
zips = pd.read_csv("data/u5/UCMR5_ZIPCodes.txt", sep="\t", dtype=str, encoding="latin1")
zc = pd.read_csv("data/zcta_county.txt", sep="|", dtype=str)
zc = zc[zc.GEOID_ZCTA5_20.notna()]
zc["area"] = pd.to_numeric(zc.AREALAND_PART)
zc = zc.sort_values("area", ascending=False).drop_duplicates("GEOID_ZCTA5_20")[["GEOID_ZCTA5_20", "GEOID_COUNTY_20"]]

# The Drought Monitor reports Connecticut by the 2022 planning regions, not the 2020 counties:
# ZCTA -> town with the largest land area -> planning region.
zs = pd.read_csv("data/tab20_zcta520_cousub20_natl.txt", sep="|", dtype=str)
zs = zs[zs.GEOID_ZCTA5_20.notna() & zs.GEOID_COUSUB_20.str.startswith("09")]
zs["area"] = pd.to_numeric(zs.AREALAND_PART)
zs = zs.sort_values("area", ascending=False).drop_duplicates("GEOID_ZCTA5_20")
cw = pd.read_csv("data/ct_cou_to_cousub_crosswalk.txt", sep="|", dtype=str)
cw.columns = [c.split("\n")[0] for c in cw.columns]
region = dict(zip(cw.OLD_COUSUB_GEOID, "09" + cw.NEW_COUNTYFP))
ct_region = zs.set_index("GEOID_ZCTA5_20").GEOID_COUSUB_20.map(region).dropna()
is_ct = zc.GEOID_COUNTY_20.str.startswith("09")
zc.loc[is_ct, "GEOID_COUNTY_20"] = zc.loc[is_ct, "GEOID_ZCTA5_20"].map(ct_region).fillna(zc.loc[is_ct, "GEOID_COUNTY_20"])
print("CT ZCTAs", int(is_ct.sum()), "assigned a planning region", int(zc.loc[is_ct, "GEOID_COUNTY_20"].str[2:].astype(int).ge(110).sum()))

z = zips.merge(zc, left_on="ZIPCODE", right_on="GEOID_ZCTA5_20", how="left")
pws_cty = z.dropna(subset=["GEOID_COUNTY_20"]).groupby("PWSID").GEOID_COUNTY_20.agg(lambda s: s.value_counts().index[0])
locs = L4sw.append(gw2)
cty = meta.PWSID.reindex(locs).map(pws_cty)
print("SW 4-event EPs", len(L4sw), "GW 2-event EPs", len(gw2), "with county", int(cty.notna().sum()),
      "unique counties", cty.nunique())
cty.dropna().to_csv("ep_county.csv", header=["county"])
pd.Series(sorted(cty.dropna().unique())).to_csv("counties.txt", index=False, header=False)
pickle.dump(pws_cty, open("pws_county.pkl", "wb"))
