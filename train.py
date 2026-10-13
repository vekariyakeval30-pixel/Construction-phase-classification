"""Reproducible baseline versus frozen ResNet-18 transfer-learning experiment.

Published val is the locked test set. Validation is made from published train.
No test metrics are computed until both model configurations are fixed.
"""
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import dctn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from features import ROOT, LABELS, NAMES, open_rgb, simple_features, encode_paths

SEED = 42
OUT = ROOT / 'results'

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')

def phash(image):
    pixels = np.asarray(image.convert('L').resize((32,32), Image.Resampling.LANCZOS), dtype=float)
    low = dctn(pixels, type=2, norm='ortho')[:8,:8].ravel()
    bits = low > np.median(low[1:])
    return sum(int(v) << i for i, v in enumerate(bits))

def prepare():
    records = json.loads((ROOT/'data/source_manifest.json').read_text(encoding='utf-8'))
    records.sort(key=lambda r: r['path'])
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(i,j):
        parents[find(j)] = find(i)
    hashes = []
    for r in records:
        path = ROOT/'data'/r['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != r['sha256']:
            raise ValueError(f'File hash mismatch: {path}')
        im = open_rgb(path)
        # Match rotation and mirror variants, as repeated views may be augmented.
        variants = [im, *[im.transpose(t) for t in (
            Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
            Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
            Image.Transpose.TRANSPOSE, Image.Transpose.TRANSVERSE)]]
        hashes.append([phash(v) for v in variants])
        r['width'], r['height'] = im.size
    matches = []
    for i in range(len(records)):
        for j in range(i):
            distance = min((hashes[i][0] ^ h).bit_count() for h in hashes[j])
            if records[i]['sha256'] == records[j]['sha256'] or distance <= 4:
                union(i,j)
                matches.append({'first': records[j]['path'], 'second': records[i]['path'], 'phash_distance': distance})
    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    selected, removed, conflicts = [], [], []
    for group_id, indices in groups.items():
        if len({records[i]['label'] for i in indices}) > 1:
            conflicts.extend(records[i]['path'] for i in indices)
            removed.extend(indices)
            continue
        # Prefer a test representative; this prevents similar train views of test images.
        representative = sorted(indices, key=lambda i: (records[i]['source_split'] != 'val', records[i]['path']))[0]
        row = dict(records[representative], group=f'g{group_id:04d}')
        selected.append(row)
        removed.extend(i for i in indices if i != representative)
    pool = [r for r in selected if r['source_split'] == 'train']
    test = [r for r in selected if r['source_split'] == 'val']
    y = [LABELS.index(r['label']) for r in pool]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(splitter.split(np.zeros(len(pool)), y, [r['group'] for r in pool]))
    for i in tr: pool[i]['split'] = 'train'
    for i in va: pool[i]['split'] = 'validation'
    for r in test: r['split'] = 'test'
    rows = sorted(pool+test, key=lambda r: (r['split'], r['label'], r['path']))
    for a in ('train','validation','test'):
        assert {r['label'] for r in rows if r['split']==a} == set(LABELS), f'Missing class in {a}'
    with (OUT/'split_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path','label','source_split','split','group','sha256','width','height'])
        writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
    audit = dict(source_counts=dict(Counter(r['source_split']+'/'+r['label'] for r in records)),
        retained_counts=dict(Counter(r['split']+'/'+r['label'] for r in rows)),
        original_images=len(records), retained_images=len(rows), removed_images=len(removed),
        duplicate_pairs=len(matches), conflicting_label_images=conflicts,
        removed_paths=[records[i]['path'] for i in removed],
        method='SHA256 and 64-bit pHash distance <=4 over 8 dihedral orientations; connected components; one representative per component; exclude mixed-label components; prefer published val representative.',
        limitation='No geographic site IDs or camera IDs are published. Perceptual deduplication does not establish site independence. Published val is repurposed as test; no separate Test Set B folder was available.')
    write_json(OUT/'data_audit.json', audit)
    write_json(OUT/'duplicate_pairs.json', matches)
    print('Split counts:', audit['retained_counts'], flush=True)
    return rows, audit

def tune(name, x, y, train, valid):
    trials, best, score = [], None, -1
    for c in (0.01, 0.1, 1.0, 10.0):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000,
                    class_weight='balanced', solver='lbfgs', random_state=SEED))
        with threadpool_limits(limits=4):
            model.fit(x[train], y[train])
        prediction = model.predict(x[valid])
        f1 = float(f1_score(y[valid], prediction, average='macro', zero_division=0))
        trials.append(dict(model=name, C=c, validation_macro_f1=f1))
        print(name, 'C', c, 'validation macro F1', round(f1,4), flush=True)
        if f1 > score:
            best, score = model, f1
    joblib.dump(best, ROOT/'models'/f'{name}.joblib')
    return best, trials

def evaluate(model, x, y):
    prediction = model.predict(x)
    return prediction, dict(accuracy=float(accuracy_score(y,prediction)),
        macro_f1=float(f1_score(y,prediction,average='macro',zero_division=0)),
        report=classification_report(y,prediction, labels=range(4), target_names=LABELS, output_dict=True,zero_division=0),
        confusion_matrix=confusion_matrix(y,prediction,labels=range(4)).tolist())

def gallery(paths, y, prediction, indices, filename, title):
    fig, axes = plt.subplots(2,3,figsize=(11,7))
    for ax, i in zip(axes.flat,indices):
        ax.imshow(open_rgb(paths[i]))
        ax.set_title(f'True: {NAMES[y[i]]} | Predicted: {NAMES[prediction[i]]}', fontsize=10)
        ax.set_xlabel(paths[i].name, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(indices):]: ax.axis('off')
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT/filename,dpi=150,bbox_inches='tight'); plt.close(fig)

def main():
    start = time.perf_counter()
    OUT.mkdir(exist_ok=True); (ROOT/'models').mkdir(exist_ok=True)
    rows, audit = prepare()
    paths = [ROOT/'data'/r['path'] for r in rows]
    y = np.array([LABELS.index(r['label']) for r in rows])
    masks = {s:np.array([r['split']==s for r in rows]) for s in ('train','validation','test')}
    print('Extracting 153 color/spatial/edge features',flush=True)
    base_x = np.stack([simple_features(open_rgb(p)) for p in paths])
    encoded_start = time.perf_counter()
    transfer_x = encode_paths(paths)
    feature_seconds = time.perf_counter()-encoded_start
    np.savez_compressed(OUT/'feature_cache.npz', baseline=base_x, transfer=transfer_x, labels=y)
    baseline, trials_b = tune('image_baseline',base_x,y,masks['train'],masks['validation'])
    transfer, trials_t = tune('transfer_resnet18',transfer_x,y,masks['train'],masks['validation'])
    write_json(OUT/'validation_trials.json', trials_b+trials_t)
    # Selection is finished before test predictions or test metrics are computed.
    selected = max([('image_baseline',max(t['validation_macro_f1'] for t in trials_b)),
                    ('transfer_resnet18',max(t['validation_macro_f1'] for t in trials_t))],key=lambda p:p[1])[0]
    write_json(ROOT/'models/model_config.json',dict(labels=LABELS, selected_model=selected, seed=SEED,
        encoder='ResNet18_Weights.IMAGENET1K_V1', baseline_features=153, transfer_features=512,
        note='Original folder labels are authoritative; probabilities are uncalibrated.'))
    dummy = DummyClassifier(strategy='most_frequent').fit(base_x[masks['train']],y[masks['train']])
    results, predictions = {}, {}
    for name,model,x in [('majority',dummy,base_x),('image_baseline',baseline,base_x),('transfer_resnet18',transfer,transfer_x)]:
        pred, metric = evaluate(model,x[masks['test']],y[masks['test']])
        results[name] = metric; predictions[name] = pred
        print('FINAL TEST',name,metric['accuracy'],metric['macro_f1'],flush=True)
        if name != 'majority':
            fig,ax=plt.subplots(figsize=(6,5))
            ConfusionMatrixDisplay(np.array(metric['confusion_matrix']),display_labels=NAMES).plot(ax=ax,cmap='Blues',colorbar=False)
            ax.set_title(name.replace('_',' ')); fig.tight_layout()
            fig.savefig(OUT/f'confusion_{name}.png',dpi=160); plt.close(fig)
    write_json(OUT/'metrics.json', results)
    test_rows = [r for r in rows if r['split']=='test']
    with (OUT/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['path','true_label',*predictions])
        for i,r in enumerate(test_rows): writer.writerow([r['path'],r['label'],*[LABELS[p[i]] for p in predictions.values()]])
    test_y=y[masks['test']]; test_paths=[p for p,m in zip(paths,masks['test']) if m]
    wrong=np.flatnonzero(test_y!=predictions['transfer_resnet18'])
    correct=np.flatnonzero(test_y==predictions['transfer_resnet18'])
    gallery(test_paths,test_y,predictions['transfer_resnet18'],wrong[:6], 'errors.png','Frozen ResNet-18: first six test errors in filename order')
    gallery(test_paths,test_y,predictions['transfer_resnet18'],correct[:6], 'correct.png','Frozen ResNet-18: correctly classified test examples')
    fig,ax=plt.subplots(figsize=(8,4))
    names=list(results); positions=np.arange(3)
    ax.bar(positions-.17,[results[n]['accuracy'] for n in names],.34,label='Accuracy',color='#547a8c')
    ax.bar(positions+.17,[results[n]['macro_f1'] for n in names],.34,label='Macro F1',color='#d08b36')
    ax.set_xticks(positions,['Majority','Image baseline','Frozen ResNet-18']); ax.set_ylim(0,1)
    ax.legend(); ax.set_title('Held-out test comparison'); fig.tight_layout()
    fig.savefig(OUT/'comparison.png',dpi=160); plt.close(fig)
    environment = dict(python=platform.python_version(), platform=platform.platform(),processor=platform.processor(),
        torch_threads=4, device='CPU', seed=SEED, resnet_feature_seconds=feature_seconds,
        total_seconds=time.perf_counter()-start,
        packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','scikit-learn','numpy','scipy','Pillow','matplotlib','joblib']})
    write_json(OUT/'environment.json',environment)
    print('Saved all results and model files.',flush=True)

if __name__=='__main__': main()
