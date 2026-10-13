"""Predict an image using the saved, validation-selected classifier."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from features import ROOT, open_rgb, simple_features, encode_paths

def predict(path, model_name=None):
    config=json.loads((ROOT/'models/model_config.json').read_text())
    name=model_name or config['selected_model']
    model=joblib.load(ROOT/'models'/f'{name}.joblib')
    x = encode_paths([Path(path)]) if name=='transfer_resnet18' else np.array([simple_features(open_rgb(path))])
    probabilities=model.predict_proba(x)[0]
    winner=int(model.classes_[np.argmax(probabilities)])
    return dict(image=Path(path).name, model=name, predicted_phase=config['labels'][winner],
        probabilities={config['labels'][int(c)]:round(float(p),4) for c,p in zip(model.classes_,probabilities)},
        note='Probabilities are uncalibrated. A phase label is not percentage completion. Aerial imagery only.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--model',choices=['image_baseline','transfer_resnet18'])
    args=parser.parse_args()
    print(json.dumps(predict(args.image,args.model),indent=2))
