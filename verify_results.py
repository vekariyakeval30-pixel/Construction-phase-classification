"""Check that published metrics match predictions and split hashes do not overlap."""
import csv
import json
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from features import ROOT,LABELS

rows=list(csv.DictReader((ROOT/'results/split_manifest.csv').open()))
for key in ['sha256','group','path']:
    sets={s:{r[key] for r in rows if r['split']==s} for s in ['train','validation','test']}
    for a,b in combinations(sets,2):
        assert not sets[a]&sets[b],f'{key} overlap: {a}/{b}'
pred=list(csv.DictReader((ROOT/'results/test_predictions.csv').open()))
metrics=json.loads((ROOT/'results/metrics.json').read_text())
y=[LABELS.index(r['true_label']) for r in pred]
assert {r['path'] for r in pred}=={r['path'] for r in rows if r['split']=='test'}
for name, expected in metrics.items():
    p=[LABELS.index(r[name]) for r in pred]
    assert np.isclose(accuracy_score(y,p),expected['accuracy'])
    assert np.isclose(f1_score(y,p,average='macro'),expected['macro_f1'])
    assert confusion_matrix(y,p,labels=range(4)).tolist()==expected['confusion_matrix']
print('PASS: metrics match predictions; no exact-hash, path or perceptual-group overlap.')
print('This check cannot certify site independence without geographic site identifiers.')
