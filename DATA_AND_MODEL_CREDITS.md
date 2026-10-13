# Data and model credits

## Image dataset

Shuchang Xu (2026). HZNU-BuildStage-4 Dataset, version 1. Mendeley Data.
DOI: https://doi.org/10.17632/wy8jnypms4.1
Source: https://data.mendeley.com/datasets/wy8jnypms4/1
License: Creative Commons Attribution 4.0 International.
License text: https://creativecommons.org/licenses/by/4.0/

The downloaded images retain their source labels. This study creates a duplicate-screened
subset and a new internal validation split. Images displayed in figures are resized and
captioned. The source folder named `val` is used as the final test set in this study.
The source webpage mentions Test Set B, but the v1 public folder tree exposes only
`train` and `val`. This project does not claim to reproduce the publisher's full protocol.
Source hashes, URLs, file IDs, selection and removed-image records are saved by the scripts.
Accessed 13-14 October 2026, Europe/Berlin.

Original class identifiers: stage_1, stage_2, stage_3, stage_4.
The publisher lists foundation excavation, main structure construction, topping-out,
and completion in that order. These are descriptive interpretations of the ordered
stage labels; no separate machine-readable semantic mapping was supplied.
Predictions and all numerical evaluations use the original folder identifiers.

## Transfer-learning encoder

Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun (2016).
Deep Residual Learning for Image Recognition. CVPR.
Paper: https://arxiv.org/abs/1512.03385

Torchvision ResNet-18, ResNet18_Weights.IMAGENET1K_V1.
Official model documentation:
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
Official weights: https://download.pytorch.org/models/resnet18-f37072fd.pth
The network is a frozen feature extractor, not a network trained from scratch in this study.
Only a new logistic classifier is trained on construction images.
Torchvision source license: https://github.com/pytorch/vision/blob/main/LICENSE
Dataset and pretrained-model rights remain with their respective owners.

## Software and assistance

PyTorch, torchvision, NumPy, SciPy, scikit-learn, Pillow, Matplotlib and joblib.
StratifiedGroupKFold documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

OpenAI Codex generated this project code, selected and downloaded the dataset,
executed the experiments, and prepared report and presentation materials.
Do not describe these activities as unaided student work. Record the student's actual
subsequent review, modifications, reruns and interpretation separately.
