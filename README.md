# Construction Phase Classification from Site Images

Pattern Recognition project materials for Keval Sanjaybhai Vekariya.

This is an executed, reproducible aerial-image experiment comparing a simple image
baseline with frozen ResNet-18 transfer learning. The implementation, experiment
execution and draft materials were prepared using OpenAI Codex. See
`DATA_AND_MODEL_CREDITS.md` for attribution and authorship disclosure.

## Actual results

| Model | Test accuracy | Test macro F1 |
|---|---:|---:|
| Most-frequent-class reference | 0.2778 | 0.1087 |
| Colour, spatial and edge features + logistic regression | 0.2389 | 0.1443 |
| Frozen ImageNet ResNet-18 + logistic regression | 0.5778 | 0.5743 |

The preferred model is selected using validation macro F1, not these test scores.
Transfer learning selected C=0.01 (validation macro F1=0.8550); the image baseline
selected C=0.1 (validation macro F1=0.7140). The large validation-to-test drop is
a central finding. This model is a proof of concept, not a reliable progress monitor.

## Quick demonstration

On the computer where this package was prepared, double-click `RUN_DEMO.cmd`.
It uses the isolated environment already created in the workspace. The full
download package includes all data and the encoder weights, so that demo requires
no additional downloads on that computer.

For another computer, use Python 3.12 and install the dependencies first:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python demo.py
```

On macOS/Linux, replace `.venv\Scripts\python` with `.venv/bin/python` and install
PyTorch from the normal platform-appropriate package source. The saved experiment
was tested on Windows/Python 3.12; other platforms were not tested.

The compact GitHub package includes four attributed example images and the trained
logistic classifiers. Its first inference downloads the official ResNet-18 weights
(about 45 MB). The full package already contains those weights.

## Reproduce the experiment

Run from this directory, using the environment's Python:

```powershell
python download_data.py
python train.py
python verify_results.py
python demo.py
```

The download is approximately 154 MB and verifies every image's SHA-256 against
the source manifest. Training uses CPU, up to four PyTorch threads, and seed 42.
The measured first run took 86.0 seconds on the preparation machine, including the
initial pretrained-model retrieval; your runtime will vary.

`project.ipynb` offers the same workflow if opened in a notebook environment after
installing the requirements. Its cells have no prefilled or invented outputs.

To classify one image:

```powershell
python predict.py examples/stage_4/stage_4_000001.JPEG
```

The command prints the selected model, predicted source label and four class
probabilities. These probabilities are uncalibrated. Stage labels do not measure
percentage completion, schedule adherence or safety.

## Dataset and split

Shuchang Xu (2026), **HZNU-BuildStage-4 Dataset v1**, Mendeley Data, CC BY 4.0.
Source and DOI: https://doi.org/10.17632/wy8jnypms4.1

The publisher lists excavation, main structure construction, topping-out and
completion. All evaluation uses the original stage_1 through stage_4 folder
labels. Descriptive names follow the source page's order, not an independently
verified mapping file.

The public download contains `train` (1,200 images) and `val` (400). Although its
description mentions a separate Test Set B, no third public split was present.
This project reserves source `val` as the final test pool and forms an internal
validation set from source `train`.

SHA-256 and rotation/reflection-aware perceptual hashes screen for near copies.
One representative is retained per connected similarity group. Of 1,600 source
images, 42 are removed, leaving 958 training, 240 validation and 360 test images.
The 47 matching pairs and removed paths are recorded. No conflicting-label group
was found. A 5-fold StratifiedGroupKFold, seed 42, supplies the first internal
validation fold; after deduplication the perceptual groups are singletons.

**Site independence remains unverified.** Perceptual matching is not geographic
grouping. Site/camera identifiers were not provided, and repeated sites or distinct
views may remain. Do not claim that these results prove performance on unseen sites.

## Methods

- Baseline: direct resize to 224x224; 48 normalized RGB histogram bins, 96 spatial
  colour moments and 9 unsigned gradient orientation bins; StandardScaler and a
  balanced multinomial logistic classifier.
- Transfer learning: frozen ResNet-18, official ImageNet-1K V1 weights, default
  resize/crop/normalization, 512 features; a training-fitted scaler and balanced
  logistic classifier. This is feature-extraction transfer learning, not fine-tuning.
- Search: C in 0.01, 0.1, 1, 10; maximum 2,000 LBFGS iterations. Select on validation
  macro F1 and retain training-only fitted models. Test scores do not select C.

The two geometric preprocessing procedures differ. No augmentation, repeated-seed
study or camera-angle experiment was run. Native source labels are retained without
relabelling based on predictions.

## Files

| Path | Purpose |
|---|---|
| `download_data.py` | Public data download and checksum verification |
| `features.py` | Shared simple descriptors and frozen encoder |
| `train.py` | Deduplication, split, tuning, final evaluation and plots |
| `predict.py`, `demo.py` | Saved-model inference and short demonstration |
| `verify_results.py` | Recompute metrics and check split overlaps |
| `models/*.joblib` | Fitted classifiers and their training-only scalers |
| `models/model_config.json` | Class order and validation-selected pipeline |
| `results/split_manifest.csv` | Exact image membership, hashes and dimensions |
| `results/validation_trials.json` | All eight tuning runs |
| `results/metrics.json` | Accuracy, macro F1, per-class metrics and matrices |
| `results/test_predictions.csv` | Every test prediction for all models |
| `results/data_audit.json` | Counts, removed files and audit limitations |
| `results/environment.json` | Runtime and package versions |
| `environment-lock.txt` | Full environment freeze for the actual run |

`results/errors_by_phase.png` is a post-evaluation explanatory figure containing
the first error per class. `train.py` independently regenerates `errors.png` using
the first six errors overall. Neither selection is used for tuning.

## Preparing a GitHub repository

The supplied `github-upload.zip` contains the source, results, examples and fitted
small classifiers. Extract it and upload its contents to a new repository. Keep
`.gitignore`. Do not upload the full `data` directories, `.cache`, Python environment
or the large encoder file. They can be downloaded using the included instructions.
Check that README.md appears on the repository home page, then check the repository
link in a private browser window. No remote GitHub repository has been created by
this package.

## Interpretation and authorship

Stage 4 is frequently predicted as Stage 3 (44/79). Transfer Stage 3 recall is 0.810
but precision is 0.436. Ground-level imagery, calibrated confidence, safety and
completion percentages are outside this study. A future improvement after seeing
these test results requires a fresh untouched test set.

The report and slides are AI-assisted reference drafts. They do not by themselves
satisfy the assignment's individual-authorship rule. The student must write their
final report, explain their actual contributions and record their own videos.
