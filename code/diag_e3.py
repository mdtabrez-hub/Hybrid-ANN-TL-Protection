"""Post-hoc diagnostic (not pre-registered): where the E3 tail errors of the random forest lie. Writes results/phase3c_E3_diagnostic.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import json, numpy as np
import run_phase3c as P
from sklearn.ensemble import RandomForestClassifier
keep, yy = P.variant('E3'); relab = yy != P.y; out = {}
for part in ('tail', 'random'):
    acc_rel, acc_oth, n_rel, err_rel, err_tot = [], [], [], [], []
    for seed in range(5):
        tr, te, _ = P.RE.split(part, seed, len(P.c), P.y, P.pos); tr, te = tr[keep[tr]], te[keep[te]]
        p = RandomForestClassifier(200, random_state=seed, n_jobs=2).fit(P.X_raw[tr], yy[tr]).predict(P.X_raw[te])
        r = relab[te]; ok = p == yy[te]
        acc_rel.append(ok[r].mean()); acc_oth.append(ok[~r].mean()); n_rel.append(int(r.sum())); err_rel.append(int((~ok & r).sum())); err_tot.append(int((~ok).sum()))
        if part == 'tail' and seed == 0: train_rel = int(relab[tr].sum())
    out[part] = dict(acc_relabelled=float(np.mean(acc_rel)), acc_other=float(np.mean(acc_oth)), n_relabelled_test=float(np.mean(n_rel)),
                     frac_errors_on_relabelled=float(np.sum(err_rel) / max(1, np.sum(err_tot))))
out['tail']['n_relabelled_train'] = train_rel; out['n_relabelled_total'] = int(relab[keep].sum())
json.dump(out, open(ROOT+'/results/phase3c_E3_diagnostic.json', 'w'), indent=1); print(json.dumps(out, indent=1))
