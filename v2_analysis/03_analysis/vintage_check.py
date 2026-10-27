"""Effect of the UCMR 5 file vintage: archive retrieved September 13, 2026 (results file of August 28) vs EPA's October 15 file.

1. Row-level difference between the two UCMR5_All.txt files (data/u5 and data/u5_20261015): rows present in only one
   file, grouped by sample, and rows that differ only in bytes that are not printable ASCII.
2. Sample-event dates on each file: events whose samples carry more than one date, and the dates given by the rule of
   pilot_common.load_u5 (earliest Method 533 sample, else earliest sample) and by the earlier rule (first date in file
   order), which depended on the row order of the file.
3. Every numeric value in the analysis outputs computed with the same code on each file, stored in
   vintage_check/oct2026/03_analysis and vintage_check/aug2026/03_analysis: values that differ, with old and new value.
4. The August snapshot against the current outputs in 03_analysis: a snapshot file that differs was made with code
   that no longer reproduces the current results, and both of its snapshots must be remade.

Run from 03_analysis/ after the analyses:  python3 vintage_check.py
Output: vintage_check.json, vintage_check.log
"""
import collections
import glob
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"aug": os.path.join(HERE, "data", "u5", "UCMR5_All.txt"),
         "oct": os.path.join(HERE, "data", "u5_20261015", "UCMR5_All.txt")}
SNAP = {k: os.path.join(HERE, "vintage_check", k + "2026", "03_analysis") for k in ("oct", "aug")}
EVENT = ["loc", "SampleEventCode"]
SAMPLE = ["PWSID", "PWSName", "FacilityID", "SamplePointID", "CollectionDate", "SampleID", "MethodID"]
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def lines(path):
    with open(path, "rb") as f:
        return f.read().split(b"\n")


def data_difference():
    raw = {k: lines(p) for k, p in FILES.items()}
    header = raw["aug"][0].decode("latin-1").rstrip("\r").split("\t")
    assert raw["oct"][0] == raw["aug"][0]
    hashes = {k: np.fromiter((hash(x) for x in v[1:]), dtype=np.int64, count=len(v) - 1) for k, v in raw.items()}
    cnt = {k: collections.Counter(h.tolist()) for k, h in hashes.items()}
    only = {}
    for a, b in (("aug", "oct"), ("oct", "aug")):
        extra = {h: n - cnt[b].get(h, 0) for h, n in cnt[a].items() if n > cnt[b].get(h, 0)}
        rows = []
        for h, x in zip(hashes[a].tolist(), raw[a][1:]):
            if extra.get(h, 0) > 0:
                extra[h] -= 1
                rows.append(x)
        only[a] = rows
    norm = lambda x: re.sub(rb"[^\x21-\x7e]", b"", x)  # noqa: E731
    in_oct = collections.Counter(norm(x) for x in only["oct"])
    invisible, withdrawn = [], []
    for x in only["aug"]:
        if in_oct[norm(x)] > 0:
            in_oct[norm(x)] -= 1
            invisible.append(x)
        else:
            withdrawn.append(x)
    added = [x for x in only["oct"] if norm(x) not in {norm(y) for y in invisible}]

    def rec(x):
        return dict(zip(header, x.decode("latin-1").rstrip("\r").split("\t")))

    groups = collections.OrderedDict()
    for x in withdrawn:
        r = rec(x)
        g = groups.setdefault(" | ".join(r[c] for c in SAMPLE), {"rows": 0, "reported": 0, "contaminants": []})
        g["rows"] += 1
        g["reported"] += r["AnalyticalResultsSign"] == "="
        g["contaminants"].append(r["Contaminant"])
    out = {"rows": {k: len(v) - 1 - (v[-1] == b"") for k, v in raw.items()},
           "only_in_aug_file": len(only["aug"]), "only_in_oct_file": len(only["oct"]),
           "withdrawn_in_oct_file": {"rows": len(withdrawn), "samples": {k: {"rows": g["rows"], "reported": g["reported"]}
                                                                        for k, g in groups.items()}},
           "added_in_oct_file": len(added),
           "differ_only_in_non_printing_bytes": [" | ".join(rec(x)[c] for c in SAMPLE + ["Contaminant"]) for x in invisible]}
    log("UCMR5_All.txt rows: August file", f"{out['rows']['aug']:,};", "October file", f"{out['rows']['oct']:,}")
    log("rows only in the August file:", len(only["aug"]), "; only in the October file:", len(only["oct"]))
    for k, g in groups.items():
        log(f"  withdrawn in October: {k}: {g['rows']} rows, {g['reported']} reported")
    for x in out["differ_only_in_non_printing_bytes"]:
        log("  same row up to non-printing bytes:", x)
    log("rows added in the October file:", len(added))
    return out


def event_dates():
    rules, out = {}, {}
    for k, p in FILES.items():
        u = pd.read_csv(p, sep="\t", encoding="latin1", dtype=str,
                        usecols=["PWSID", "FacilityID", "SamplePointID", "CollectionDate", "MethodID", "SampleEventCode"])
        u = u[u.MethodID.isin(["EPA 533", "EPA 537.1"])]
        u = u.assign(loc=u.PWSID + "|" + u.FacilityID + "|" + u.SamplePointID).dropna(subset=["loc"])
        u = u.assign(_n=u.MethodID != "EPA 533", _d=pd.to_datetime(u.CollectionDate, format="%m/%d/%Y"))
        g = u.groupby(EVENT)._d
        span = (g.max() - g.min()).dt.days
        n533 = u[~u._n].groupby(EVENT)._d.nunique().reindex(span.index, fill_value=0)
        new = u.sort_values(EVENT + ["_n", "_d"], kind="stable").drop_duplicates(EVENT).set_index(EVENT)._d
        old = u.drop_duplicates(EVENT).set_index(EVENT)._d.reindex(new.index)
        rules[k] = {"new": new, "old": old}
        multi = span[span > 0]
        out[k] = {"events": len(span), "events_with_more_than_one_date": len(multi),
                  "median_days_between_first_and_last_date": float(multi.median()), "max_days": int(multi.max()),
                  "events_with_more_than_one_method_533_date": int((n533 > 1).sum()),
                  "events_without_method_533_sample": int((n533 == 0).sum()),
                  "dates_changed_by_the_rule": int((old != new).sum())}
        log(f"{k} file: {len(span):,} events, {len(multi):,} with more than one date (median {multi.median():.0f} days "
            f"between first and last, maximum {multi.max():,}); {out[k]['events_with_more_than_one_method_533_date']} with "
            f"more than one Method 533 date, {out[k]['events_without_method_533_sample']} without a Method 533 sample; the "
            f"rule changes {out[k]['dates_changed_by_the_rule']:,} dates from those in file order")
    common = rules["aug"]["new"].index.intersection(rules["oct"]["new"].index)
    for r in ("new", "old"):
        d = common[rules["aug"][r].reindex(common).values != rules["oct"][r].reindex(common).values]
        out[f"common_events_dated_differently_by_the_{r}_rule"] = {"n": len(d), "events": [" | ".join(e) for e in d[:20]]}
        log(f"events in both files dated differently by the {r} rule: {len(d):,}"
            + (f" ({'; '.join(' | '.join(e) for e in d[:5])})" if 0 < len(d) <= 5 else ""))
    out["events_only_in_one_file"] = {k: len(rules[k]["new"].index.difference(common)) for k in rules}
    return out


def code_check():
    stale = []
    for n in sorted(map(os.path.basename, glob.glob(os.path.join(SNAP["aug"], "*.json")))):
        cur = os.path.join(HERE, n)
        s, c = (dict(leaves(json.load(open(p)))) for p in (os.path.join(SNAP["aug"], n), cur))
        if set(s) != set(c) or any(not same(s[k], c[k]) for k in s):
            stale.append(n)
    log(f"\nAugust snapshot against the current outputs: {len(stale)} of "
        f"{len(glob.glob(os.path.join(SNAP['aug'], '*.json')))} files differ" + (f" ({', '.join(stale)})" if stale else ""))
    return stale


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield path, float(x)


def same(a, b):
    if np.isnan(a) and np.isnan(b):
        return True
    return a == b or abs(a - b) <= 1e-9 * max(abs(a), abs(b))


def output_difference():
    out = {}
    names = sorted(set(map(os.path.basename, glob.glob(os.path.join(SNAP["oct"], "*.json")))) &
                   set(map(os.path.basename, glob.glob(os.path.join(SNAP["aug"], "*.json")))))
    log(f"\noutputs compared: {len(names)} JSON files")
    for n in names:
        o, a = (dict(leaves(json.load(open(os.path.join(SNAP[k], n))))) for k in ("oct", "aug"))
        common = sorted(set(o) & set(a))
        diff = [[p, o[p], a[p]] for p in common if not same(o[p], a[p])]
        rel = [abs(x[2] - x[1]) / max(abs(x[1]), abs(x[2])) for x in diff if max(abs(x[1]), abs(x[2])) > 0]
        out[n] = {"numbers": len(common), "changed": len(diff), "max_relative_change": max(rel, default=0.0),
                  "only_oct": sorted(set(o) - set(a)), "only_aug": sorted(set(a) - set(o)), "values": diff}
        log(f"{n:28s} {len(common):>7,} numbers, {len(diff):>6,} changed, largest relative change {max(rel, default=0.0):.2%}"
            + (f"; keys only in one run: {len(out[n]['only_oct']) + len(out[n]['only_aug'])}" if out[n]["only_oct"] or out[n]["only_aug"] else ""))
    return out


def main():
    res = {"files": {k: os.path.relpath(p, HERE) for k, p in FILES.items()},
           "snapshots": {k: os.path.relpath(p, HERE) for k, p in SNAP.items()},
           "data_difference": data_difference(), "event_dates": event_dates(), "outputs": output_difference(),
           "stale_snapshots": code_check()}
    tot = sum(v["numbers"] for v in res["outputs"].values())
    ch = sum(v["changed"] for v in res["outputs"].values())
    res["summary"] = {"numbers": tot, "changed": ch,
                      "files_changed": sorted(n for n, v in res["outputs"].items() if v["changed"])}
    log(f"\n{ch:,} of {tot:,} output numbers differ between the two files")
    json.dump(res, open(os.path.join(HERE, "vintage_check.json"), "w"), indent=1)
    with open(os.path.join(HERE, "vintage_check.log"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
