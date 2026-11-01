"""Categorise detection errors of B4 on the simulation timeline (results/phase3b_B4_errors.csv; keys added to phase3b_summary.json['B4_cat'])."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import json, glob, pandas as pd, numpy as np
import run_experiments as RE
D = RE.d; T = json.load(open(ROOT+'/results/data_facts.json'))['timeline']
PH = {'1001': ['Ia'], '1011': ['Ia', 'Ib'], '0110': ['Ib', 'Ic'], '0111': ['Ia', 'Ib', 'Ic'], '1111': ['Ia', 'Ib', 'Ic']}
rows = []
for f in sorted(glob.glob(ROOT+'/results/phase3b/B4_*.json')):
    r = json.load(open(f))
    for i, d, reg, y in zip(r['err_rows'], r['err_dist_ms'], r['err_regime'], r['err_true']):
        t = i * 0.05; cat, ev, imax = 'other', '', np.nan
        for k, v in T.items():
            if v['label_start_ms'] <= t <= v['label_end_ms']:
                ev = k; cat = 'fault row missed'; imax = float(D.iloc[i][PH[k]].abs().max())
                peak = float(D.iloc[int(v['label_start_ms'] / 0.05):int(v['label_end_ms'] / 0.05)][PH[k]].abs().max().max())
            elif v['label_end_ms'] < t <= v['label_end_ms'] + 2.0: ev = k; cat = 'healthy row just after label end'
            elif v['switch_on_ms'] <= t < v['label_start_ms']: ev = k; cat = 'healthy row before label start'
        if cat == 'other' and t < 1.0: cat = 'simulation start'
        rows.append(dict(seed=r['seed'], row=i, t_ms=t, regime=reg, label=y, category=cat, event=ev, dist_ms=d,
                         faulted_phase_absI=imax, event_faulted_peak=peak if cat == 'fault row missed' else np.nan))
E4 = pd.DataFrame(rows); E4.to_csv(ROOT+'/results/phase3b_B4_errors.csv', index=False)
S = json.load(open(ROOT+'/results/phase3b_summary.json'))
after = E4[E4.category == 'healthy row just after label end']
miss = E4[E4.category == 'fault row missed']
S['B4_cat'] = dict(total=int(len(E4)), counts=E4.category.value_counts().to_dict(),
                   after_end_max_ms=float((after.t_ms - after.event.map(lambda k: T[k]['label_end_ms'])).max()) if len(after) else None,
                   miss_absI_max=float(miss.faulted_phase_absI.max()) if len(miss) else None,
                   miss_peak_min=float(miss.event_faulted_peak.min()) if len(miss) else None,
                   miss_events=sorted(miss.event.unique().tolist()))
json.dump(S, open(ROOT+'/results/phase3b_summary.json', 'w'), indent=1)
print(json.dumps(S['B4_cat'], indent=0))
