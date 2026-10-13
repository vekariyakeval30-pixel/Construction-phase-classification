"""A short, reproducible console demonstration, with actual test predictions."""
import csv
import json
from features import ROOT
from predict import predict

def main():
    print('\nConstruction Phase Classification from Site Images')
    print('Pattern Recognition | Aerial-image classifier\n')
    results=json.loads((ROOT/'results/metrics.json').read_text())
    print('HELD-OUT TEST RESULTS')
    for name,metrics in results.items():
        print(f'{name:22s} Accuracy={metrics["accuracy"]:.3f}  Macro F1={metrics["macro_f1"]:.3f}')
    rows=list(csv.DictReader((ROOT/'results/test_predictions.csv').open()))
    # Use the first test image of each class, irrespective of prediction correctness.
    for phase in ['stage_1','stage_2','stage_3','stage_4']:
        row=next(r for r in rows if r['true_label']==phase)
        path=ROOT/'data'/row['path']
        if not path.exists():
            path=ROOT/'examples'/phase/path.name
        output=predict(path)
        print('\nImage:',row['path'])
        print('Dataset label:',phase)
        print('Predicted label:',output['predicted_phase'])
        print('Class probabilities:',output['probabilities'])
    print('\nLimitations: aerial imagery, unverified site independence, uncalibrated probabilities.')
    print('The output is a phase category, not a completion percentage or safety decision.')

if __name__=='__main__': main()
