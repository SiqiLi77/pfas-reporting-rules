"""Record size and SHA-256 of the raw input files used by the analyses.

Run: cd 03_analysis && python3 hash_inputs.py  ->  input_checksums.json (read by the SI builder, Text S19).
"""
import glob
import hashlib
import json
import os

FILES = [
    ("UCMR 5 occurrence data, archive retrieved September 13, 2026 (ZIP)", "data/ucmr5.zip"),
    ("UCMR 5 occurrence data, file of October 15, 2026, sensitivity check only (ZIP)", "data/ucmr5_20261015.zip"),
    ("UCMR 3 occurrence data (ZIP)", "data/ucmr3.zip"),
    ("SDWIS public water systems (ECHO download)", "data/sdwa/SDWA_PUB_WATER_SYSTEMS.csv"),
    ("ACS 2018–2022 table B03002", "data/b03002.dat"),
    ("ACS 2018–2022 table B19013", "data/b19013.dat"),
    ("ACS 2018–2022 table B17001", "data/b17001.dat"),
    ("Census 2020 ZCTA–county relationship file", "data/zcta_county.txt"),
    ("Census 2020 ZCTA–county subdivision relationship file", "data/tab20_zcta520_cousub20_natl.txt"),
    ("Census Connecticut county-to-planning-region crosswalk", "data/ct_cou_to_cousub_crosswalk.txt"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    out = []
    for label, rel in FILES:
        out.append({"label": label, "file": rel, "bytes": os.path.getsize(rel), "sha256": sha256(rel)})
    usdm = sorted(glob.glob("data/usdm/*.json"))
    h = hashlib.sha256()
    total = 0
    for p in usdm:
        with open(p, "rb") as f:
            b = f.read()
        total += len(b)
        h.update(os.path.basename(p).encode() + b"\0" + b)
    out.append({"label": f"U.S. Drought Monitor county DSCI ({len(usdm):,} JSON files; hash over sorted name+content)",
                "file": "data/usdm/*.json", "bytes": total, "sha256": h.hexdigest()})
    json.dump(out, open("input_checksums.json", "w"), indent=1)
    for r in out:
        print(f"{r['bytes']:>12,}  {r['sha256'][:16]}…  {r['file']}")


if __name__ == "__main__":
    main()
