# MaterialNet — controlled GPT-synthetic training comparison

**Date:** 2026-11-18  
**GitHub Actions:** https://github.com/exmachina100/kumo-material-intelligence/actions/runs/44342564697  
**Image bundle:** https://github.com/exmachina100/kumo-material-intelligence/releases/tag/gpt-cmf-approved-seed-2026-11-18  
**Status:** RESEARCH ONLY — no production model change.

## Purpose

Test whether adding explicitly approved GPT-generated synthetic texture samples to the training split improves recognition of **real-source catalog images**. This is a model-training experiment, not an Allure/Floot rendering experiment.

## Dataset and controls

- 315 Poly Haven / ambientCG CC0 reference previews with **weak catalog labels**.
- Source-grouped split: **204 train / 48 validation / 63 held-out test**.
- 10 original GPT material texture images imported from a pinned, SHA-256-checked GitHub Release.
- **8 GPT samples** matched the experimental ten-class classifier and were added **only** to training.
- Two classes were excluded without relabeling: **synthetic-leather** and **composite**. These are not separate output classes in the current classifier.
- Synthetic sample weights were capped relative to each real class. Real training, validation, and test splits were identical across both arms.
- Each arm independently selected its regularization using only the real-source validation split. Synthetic images were never included in validation or test.
- Both arms trained fresh scikit-learn classifiers over ImageNet-pretrained MobileNetV3-Small visual embeddings, with identical feature extraction and normalization.

## Held-out results — catalog-label agreement, not physical-material accuracy

| Arm | Held-out macro F1 |
| --- | ---: |
| Real-source training only | **0.7836** |
| Real-source training + eight synthetic GPT samples | 0.7647 |
| Delta after adding synthetic samples | **−0.0189** |

Adding the small GPT seed set did **not** improve this held-out benchmark. Neither model is approved for production, and neither result demonstrates physical substance identification on independent automotive camera photographs.

## Decision

**Do not promote the GPT-augmented model.** Retain the current deployed models. The synthetic images remain labeled and reproducible, available for future controlled experiments.

## Next experimental changes

1. Grow genuine camera-captured, expert-reviewed examples of rubber, plastics, textile, coating, glass, and composites.
2. Expand synthetic training families selectively, with multiple genuinely different lighting, scale, camera, and imperfection cases. Reject near-duplicates.
3. Track per-family errors, false positives on unknown materials, and source-domain shift.
4. Repeat the same grouped real-only holdout with controlled synthetic ratios (0%, 5%, 10%, 15%) and multiple seeds. Select on validation only.
5. Move to a real-world test set before recommending any production promotion.

## Provenance

The ten GPT textures were previously created and named by the user/assistant; their visual-family labels were recorded as **approved synthetic training intent**. Their actual physical composition is not verified because these are generated images.

The batch ZIP includes image files and manifest, and GitHub Actions verified each image byte length and SHA-256 before packaging. The full JSON evaluation report remains available as a GitHub Actions artifact on the run linked above.
