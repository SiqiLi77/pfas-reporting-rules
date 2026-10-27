"""Firth-penalized refits of the environmental-justice models on one harmonized sample.

deep_ej.py, deep_ej2.py and deep_ej3.py drop states without outcome variation, so their samples differ by outcome and
specification, and in the unpenalized all-state refits of deep_ej4.py the fixed effects of states without a UCMR 3-rule
positive tend to minus infinity (quasi-complete separation). Here every model uses the same systems and all states. Firth's penalty (Jeffreys prior,
half the log determinant of the Fisher information) keeps all estimates finite.

Run: cd 03_analysis && python3 deep_ej5_firth.py > deep_ej5_firth.log   ->  deep_ej5_firth.json
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from pilot_common import *

warnings.filterwarnings("ignore")
exec(open("deep_ej.py").read().split("res = {}")[0])

DEM = ["pct_black", "pct_hisp", "pct_pov"]
y_old = "Any detection, UCMR3 rule (6 @ old MRLs)"
y_new = "Any detection, UCMR5 rule (29 @ new MRLs)"
n_ep = meta.groupby("PWSID").size()
D["log_nep"] = np.log(n_ep.reindex(D.index).fillna(1))
H = D.dropna(subset=[y_old, y_new]).copy()
print("harmonized sample: systems", len(H), "states", H.state.nunique(),
      "UCMR 3-rule positives", int(H[y_old].sum()), "UCMR 5-rule positives", int(H[y_new].sum()))


def penalized_loglik(b, X, y):
    eta = X @ b
    w = expit(eta) * (1 - expit(eta))
    return float(np.sum(y * eta - np.logaddexp(0, eta)) + 0.5 * np.linalg.slogdet(X.T @ (X * w[:, None]))[1])


def firth(X, y, maxit=100, tol=1e-5):
    """Firth logistic regression by modified-score Newton steps with step halving on the penalized log-likelihood.
    Converged when every component of the modified score is below tol. A step criterion does not work here: near the
    optimum, state effects with few positive systems oscillate at ~1e-7, and the score stalls at ~1e-6, because the
    information matrix is ill-conditioned."""
    b = np.zeros(X.shape[1])
    cur = penalized_loglik(b, X, y)
    for it in range(1, maxit + 1):
        pi = expit(X @ b)
        w = pi * (1 - pi)
        Iinv = np.linalg.inv(X.T @ (X * w[:, None]))
        h = w * np.sum((X @ Iinv) * X, axis=1)
        U = X.T @ (y - pi + h * (0.5 - pi))
        if np.max(np.abs(U)) < tol:
            break
        step = Iinv @ U
        t = 1.0
        while True:
            nb = b + t * step
            new = penalized_loglik(nb, X, y)
            if new >= cur - 1e-10 or t < 1e-8:
                break
            t /= 2
        b, cur = nb, new
    pi = expit(X @ b)
    w = pi * (1 - pi)
    Iinv = np.linalg.inv(X.T @ (X * w[:, None]))
    h = w * np.sum((X @ Iinv) * X, axis=1)
    score = X * (y - pi + h * (0.5 - pi))[:, None]
    return {"b": b, "Iinv": Iinv, "score": score, "iter": it, "max_abs_score": float(np.max(np.abs(score.sum(axis=0))))}


def cluster_se(fit, groups):
    G = pd.DataFrame(fit["score"]).groupby(groups).sum().values
    g = len(G)
    V = fit["Iinv"] @ (G.T @ G) @ fit["Iinv"] * g / (g - 1)
    return np.sqrt(np.diag(V))


def dummies(labels):
    return pd.get_dummies(pd.Series(labels), drop_first=True).astype(float).values


def single_design(dd, ycol):
    Xm = dd[DEM + ["log_pop", "any_sw"]].copy()
    Xm[DEM] /= 10.0
    X = np.column_stack([np.ones(len(dd)), Xm.values, dummies(dd.state.values)])
    return X, dd[ycol].astype(float).values, list(Xm.columns)


def stacked_design(dd, covs, common, fe="state"):
    """Each system twice (UCMR 3 rule, UCMR 5 rule). Shares and fixed effects interact with the rule; covariates in
    `common` enter once, the others also interact. The rule main effect is absorbed by the fixed effect x rule terms."""
    S = pd.concat([dd.assign(y=dd[y_old].astype(float), new=0.0), dd.assign(y=dd[y_new].astype(float), new=1.0)])
    Xm = S[DEM + covs].copy()
    Xm[DEM] /= 10.0
    for c in DEM + [c for c in covs if c not in common]:
        Xm[c + "_x_new"] = Xm[c] * S.new
    fd = dummies((S[fe] + "_" + S.new.astype(int).astype(str)).values)
    X = np.column_stack([np.ones(len(S)), Xm.values, fd])
    return X, S.y.values, list(Xm.columns), np.asarray(pd.factorize(S.index)[0])


def numeric_gradient_check(X, y, b, cols, names):
    """The modified score is the gradient of the penalized log-likelihood: check it is ~0 at the solution."""
    g = []
    for name in names:
        j = 1 + cols.index(name)
        e = np.zeros_like(b)
        e[j] = 1e-5
        g.append((penalized_loglik(b + e, X, y) - penalized_loglik(b - e, X, y)) / 2e-5)
    return float(np.max(np.abs(g)))


out = {"sample": {"systems": len(H), "states": int(H.state.nunique()), "pos_ucmr3_rule": int(H[y_old].sum()),
                  "pos_ucmr5_rule": int(H[y_new].sum())}, "single": {}, "stacked": {}}

for label, ycol in (("UCMR 3 rule", y_old), ("UCMR 5 rule", y_new)):
    X, y, cols = single_design(H, ycol)
    f = firth(X, y)
    se = cluster_se(f, H.state.values)
    r = {"n": len(y), "npos": int(y.sum()), "iter": f["iter"], "max_abs_score": f["max_abs_score"],
         "grad_check": numeric_gradient_check(X, y, f["b"], cols, DEM)}
    for v in DEM:
        j = 1 + cols.index(v)
        r[v] = [float(np.exp(f["b"][j])), float(np.exp(f["b"][j] - 1.96 * se[j])), float(np.exp(f["b"][j] + 1.96 * se[j]))]
    out["single"][label] = r
    print(label, {k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in r.items()})

base = ["log_pop", "any_sw"]
SPECS = {"shares x rule": (base, base, "state"),
         "all covariates x rule": (base, [], "state"),
         "all covariates x rule + log n EP": (base + ["log_nep"], [], "state"),
         "all covariates x rule, division FE": (base, [], "div")}
DIV = {"New England": "CT ME MA NH RI VT", "Mid-Atlantic": "NJ NY PA", "ENC": "IL IN MI OH WI",
       "WNC": "IA KS MN MO NE ND SD", "South Atlantic": "DE DC FL GA MD NC SC VA WV",
       "ESC": "AL KY MS TN", "WSC": "AR LA OK TX", "Mountain": "AZ CO ID MT NV NM UT WY",
       "Pacific": "AK CA HI OR WA"}
H["div"] = H.state.map({s: d for d, ss in DIV.items() for s in ss.split()}).fillna("Territory/Tribal")
for name, (covs, common, fe) in SPECS.items():
    X, y, cols, grp = stacked_design(H, covs, common, fe)
    f = firth(X, y)
    se = cluster_se(f, grp)
    r = {"iter": f["iter"], "max_abs_score": f["max_abs_score"],
         "grad_check": numeric_gradient_check(X, y, f["b"], cols, ["pct_black", "pct_black_x_new"])}
    for v in DEM:
        j, k = 1 + cols.index(v), 1 + cols.index(v + "_x_new")
        r[v] = {"OR_old": float(np.exp(f["b"][j])), "OR_new": float(np.exp(f["b"][j] + f["b"][k])),
                "ratio": float(np.exp(f["b"][k])),
                "ratio_ci_cluster": [float(np.exp(f["b"][k] - 1.96 * se[k])), float(np.exp(f["b"][k] + 1.96 * se[k]))],
                "p_cluster": float(2 * stats.norm.sf(abs(f["b"][k]) / se[k]))}
    out["stacked"][name] = r
    print(name, "iter", f["iter"], {v: {a: np.round(b, 3).tolist() for a, b in r[v].items()} for v in DEM})

# system bootstrap of the two main specifications: resample systems, refit both rules on all states
rng = np.random.default_rng(11)
B = 200
idx = np.arange(len(H))
for name in ("shares x rule", "all covariates x rule"):
    covs, common, fe = SPECS[name]
    bs, iters, scores = [], [], []
    for _ in range(B):
        sub = H.iloc[rng.choice(idx, len(idx), replace=True)].copy()
        sub.index = [f"b{i}" for i in range(len(sub))]
        X, y, cols, _g = stacked_design(sub, covs, common, fe)
        f = firth(X, y)
        bs.append(f["b"][1 + cols.index("pct_black_x_new")])
        iters.append(f["iter"])
        scores.append(f["max_abs_score"])
    bs, scores = np.array(bs), np.array(scores)
    out["stacked"][name]["boot"] = {"n_boot": int(np.isfinite(bs).sum()), "max_iter": int(max(iters)),
                                    "n_hit_maxit": int(sum(i == 100 for i in iters)), "max_abs_score": float(scores.max()),
                                    "ratio_ci": [float(np.exp(np.percentile(bs, 2.5))), float(np.exp(np.percentile(bs, 97.5)))],
                                    "share_le_1": float((bs <= 0).mean())}
    print(name, "bootstrap", out["stacked"][name]["boot"])

out["checks"] = {"converged": all(r["max_abs_score"] < 1e-5 for r in list(out["single"].values()) + list(out["stacked"].values())),
                 "bootstrap_converged": all(out["stacked"][k]["boot"]["max_abs_score"] < 1e-5 and out["stacked"][k]["boot"]["n_hit_maxit"] == 0
                                            for k in ("shares x rule", "all covariates x rule")),
                 "gradient_zero": all(r["grad_check"] < 1e-3 for r in list(out["single"].values()) + list(out["stacked"].values()))}
print("checks", out["checks"])
assert all(out["checks"].values())
json.dump(out, open("deep_ej5_firth.json", "w"), indent=1)
